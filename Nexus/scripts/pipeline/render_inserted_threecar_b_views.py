"""Render synchronized virtual perimeter cameras rigidly attached to car B.

Replay the frozen insertion run's exact actor transforms. No Nexus rerun,
trajectory retargeting, ground recalibration, or edits to the previous renderer.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/nexus_3dgs/asset_insert_20261028'
sys.path[:0] = [str(ROOT), str(ROOT / 'Omniscenegraph'), str(ROOT / 'scripts/pipeline')]
from render_inserted_threecar import sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    print('B_VIEW_INIT imports', flush=True)

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from nexus_3dgs.geometry import matrix_quaternion
    from nexus_3dgs.inserted_vehicle import GaussianVehicle
    from nexus_3dgs.replay import camera_inputs, load_trainer

    torch.set_num_threads(6)
    source_path = OUT / 'render_final/run.json'
    source = json.loads(source_path.read_text())
    assert source['status'] == 'complete' and not source['probe_only']
    protocol = json.loads((OUT / 'insertion_protocol.json').read_text())
    scene = Path(protocol['background_scene'])
    metadata = json.loads((scene / 'nexus_scene.json').read_text())
    world_from_scene = np.array(metadata['frames'][0]['ego_to_world'])
    scene_from_world = np.linalg.inv(world_from_scene)
    checkpoint = Path(source['checkpoint'])
    assert sha(checkpoint) == source['checkpoint_sha256']
    source_hashes = {str(source_path): sha(source_path),
                     str(checkpoint): source['checkpoint_sha256']}
    source_hashes.update(protocol['source_prediction_sha256'])
    source_hashes.update({a['source']: a['source_sha256'] for a in source['asset_info'].values()})
    geometry = json.loads((checkpoint.parent / 'scene_geometry.json').read_text())
    print('B_VIEW_INIT background', flush=True)
    trainer, cfg = load_trainer(checkpoint, scene_aabb=geometry['scene_aabb'])
    print('B_VIEW_INIT background_loaded', flush=True)
    trainer.render_each_class = False
    rigid = trainer.models['RigidNodes']
    original_get = rigid.get_gaussians
    versions = [(parameter, parameter._version)
                for model in trainer.models.values() for parameter in model.parameters()]
    vehicles = {role: GaussianVehicle(asset['source'], asset['target_original_lw'])
                for role, asset in protocol['asset_assignment'].items()}
    for role, vehicle in vehicles.items():
        assert np.allclose(vehicle.size, source['asset_info'][role]['render_extent_lwh'])

    # A co-located roof camera rig avoids near-field gaps between four
    # perimeter camera origins. It sits 15cm above the B asset's roof.
    # Omit the host from its own cameras; no cabin reconstruction is available.
    length, width, height = vehicles['B'].size
    offset = [0., 0., height / 2 + .15]
    definitions = {
        'front': (0., offset),
        'rear': (180., offset),
        'left': (90., offset),
        'right': (-90., offset),
    }
    mounts = {}
    pitch = np.deg2rad(3.)
    for name, (angle, offset) in definitions.items():
        yaw = np.deg2rad(angle)
        forward = np.array([np.cos(yaw) * np.cos(pitch),
                            np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        mount = np.eye(4)
        mount[:3, :3] = np.column_stack([right, down, forward])
        mount[:3, 3] = offset
        assert np.allclose(mount[:3, :3].T @ mount[:3, :3], np.eye(3))
        assert np.linalg.det(mount[:3, :3]) > .99999
        mounts[name] = mount
    width_px, height_px = 960, 540
    fov = 105.
    focal = width_px / (2 * np.tan(np.deg2rad(fov / 2)))
    intrinsic = np.array([[focal, 0., width_px / 2],
                          [0., focal, height_px / 2], [0., 0., 1.]])
    intrinsic_cuda = torch.tensor(intrinsic, dtype=torch.float32, device='cuda')
    egos = np.array([frame['ego_to_world'] for frame in metadata['frames']])
    state = {}

    def inserted_get(camera):
        entries = []
        for role in ('A', 'C'):
            pose = state['poses'][role]
            entries.append(vehicles[role].gaussians(pose, camera.camtoworlds[:3, 3]))
            size = vehicles[role].size
            shadow = pose.copy()
            shadow[:3, 3] -= pose[:3, 2] * (size[2] / 2 - .035)
            entries.append({
                '_means': torch.tensor(shadow[:3, 3], dtype=torch.float32, device='cuda')[None],
                '_quats': torch.tensor(matrix_quaternion(shadow[:3, :3]), dtype=torch.float32, device='cuda')[None],
                '_scales': torch.tensor([[size[0] * .34, size[1] * .36, .008]], dtype=torch.float32, device='cuda'),
                '_opacities': torch.tensor([[.28]], device='cuda'),
                '_rgbs': torch.tensor([[.04, .04, .04]], device='cuda'),
            })
        return {key: torch.cat([entry[key] for entry in entries], 0) for key in entries[0]}

    rigid.get_gaussians = inserted_get
    indices = [0, 20, 40, 60, 80] if args.probe else list(range(81))
    videos = {}
    frames = {}
    started = time.monotonic()
    try:
        for branch in ('normal', 'event'):
            videos[branch] = {}
            frames[branch] = []
            folder = args.output / branch
            folder.mkdir()
            writers = {}
            try:
                if not args.probe:
                    writers = {view: imageio.get_writer(folder / f'{view}.mp4', fps=10,
                               macro_block_size=2, ffmpeg_params=['-threads', '2', '-crf', '18'])
                               for view in mounts}
                for index in indices:
                    record = source['frames'][branch][index]
                    actors = {actor['role']: np.array(actor['body_world']) for actor in record['actors']}
                    state['poses'] = {role: scene_from_world @ pose for role, pose in actors.items()}
                    nearest = int(np.argmin(np.linalg.norm(egos[:, :2, 3] - actors['B'][:2, 3], axis=-1)))
                    row = {'frame': index, 'time_s': record['time_s'], 'views': {}}
                    for view, mount in mounts.items():
                        camera_world = actors['B'] @ mount
                        camera_scene = scene_from_world @ camera_world
                        # Audit attachment in world coordinates, without the
                        # float32 scene conversion used only for GPU rasterization.
                        assert np.allclose(np.linalg.inv(actors['B']) @ camera_world, mount, atol=1e-8)
                        with torch.no_grad():
                            image, camera = camera_inputs(
                                torch.tensor(camera_scene, dtype=torch.float32, device='cuda'),
                                intrinsic_cuda, nearest, height_px, width_px,
                                rigid.num_frames, len(cfg.data.pixel_source.cameras))
                            image['img_idx'] = torch.tensor(
                                [protocol['background_camera_anchor_frame'] * len(cfg.data.pixel_source.cameras)],
                                device='cuda', dtype=torch.long)
                            result = trainer(image, camera, novel_view=True)
                            result = result[0] if isinstance(result, tuple) else result
                            assert torch.isfinite(result['rgb']).all()
                            rgb = (result['rgb'].clamp(0, 1).cpu().numpy() * 255).round().astype('uint8')
                        if writers:
                            writers[view].append_data(rgb)
                        if index in (0, 20, 40, 60, 80):
                            imageio.imwrite(folder / f'{view}_{index:03d}.jpg', rgb)
                        visible = {}
                        for role in ('A', 'C'):
                            point = np.linalg.inv(camera_world) @ actors[role][:, 3]
                            uv = intrinsic @ point[:3]
                            uv = uv[:2] / uv[2] if point[2] > 0 else np.array([-1., -1.])
                            visible[role] = {'depth_m': float(point[2]), 'center_in_frustum': bool(
                                point[2] > 0 and 0 <= uv[0] < width_px and 0 <= uv[1] < height_px),
                                'uv_px': uv.tolist()}
                        row['views'][view] = {'camera_world': camera_world.tolist(),
                                              'camera_scene': camera_scene.tolist(),
                                              'actor_center_frustum': visible}
                    frames[branch].append(row)
                    if index % 20 == 0:
                        print('B_VIEW_RENDER', branch, index, flush=True)
            finally:
                for writer in writers.values():
                    writer.close()
            if not args.probe:
                for view in mounts:
                    path = folder / f'{view}.mp4'
                    with imageio.get_reader(path) as reader:
                        count = sum(1 for _ in reader)
                        fps = reader.get_meta_data()['fps']
                    assert count == 81 and fps == 10
                    videos[branch][view] = {'frames': count, 'fps': fps, 'sha256': sha(path)}
    finally:
        rigid.get_gaussians = original_get
    assert all(parameter._version == version for parameter, version in versions)
    for path, expected in source_hashes.items():
        assert sha(path) == expected, path
    for branch in frames:
        for row in frames[branch]:
            assert any(v['actor_center_frustum']['A']['center_in_frustum'] for v in row['views'].values())
            assert any(v['actor_center_frustum']['C']['center_in_frustum'] for v in row['views'].values())

    report = {
        'status': 'complete', 'probe_only': args.probe,
        'source_render_run': str(source_path), 'source_hashes_unchanged': source_hashes,
        'actor_transforms_replayed_exactly': True, 'camera_host': 'B',
        'camera_mounts_B_body': {key: value.tolist() for key, value in mounts.items()},
        'camera_horizontal_fov_deg': fov, 'camera_downward_pitch_deg': 3.,
        'intrinsics': intrinsic.tolist(), 'size': [width_px, height_px],
        'host_B_omitted_for_self_occlusion': True,
        'mount_policy': 'co-located virtual roof cameras 15cm above B roof; rigidly follow B translation, yaw, pitch and roll; not calibrated factory cameras or cabin reconstruction',
        'constant_intrinsics_and_exposure': True,
        'each_external_actor_center_in_at_least_one_camera_every_sample': True,
        'frustum_checks_do_not_prove_visibility_without_occlusion': True,
        'native_gaussian_depth_compositing': True, 'Nexus_updates': 0,
        'source_scope': source['scope'], 'videos': videos, 'frames': frames,
        'renderer_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - started,
        'quality_limit': 'background trained from front and front-side imagery; rear/side novel views may be incomplete',
    }
    write(args.output / 'run.json', report)
    if args.probe:
        font = ImageFont.truetype(str(ROOT / 'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'), 22)
        board = Image.new('RGB', (1280, 5 * 780), 'white')
        draw = ImageDraw.Draw(board)
        for row, index in enumerate(indices):
            for col, view in enumerate(mounts):
                x = (col % 2) * 640
                y = row * 780 + (col // 2) * 390
                board.paste(Image.open(args.output / f'event/{view}_{index:03d}.jpg').resize((640, 360)), (x, y + 30))
                draw.text((x + 10, y + 2), f'B {view} | {index / 10:.1f}s', fill='black', font=font)
        board.save(args.output / 'probe_contact.jpg')
    print('B_VIEW_RENDER_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
