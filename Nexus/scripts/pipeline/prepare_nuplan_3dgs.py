"""Prepare one existing nuPlan/Nexus clip for Omniscenegraph reconstruction.

Images remain symlinks into shared storage. Only selected LiDAR scans are
converted. Labels/identities/timestamps come from the same read-only log DB.
No semantic masks or Gaussian appearance assets are fabricated.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from nexus_3dgs.bundle import load_bundle
from nexus_3dgs import nuplan


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() or destination.resolve() != source.resolve():
            raise ValueError(f'Conflicting existing file: {destination}')
    else:
        destination.symlink_to(source.resolve())


def convert_lidar(pair):
    source, destination = pair
    from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
    # SDK from_file() takes its legacy ASCII-only path for .pcd. nuPlan sensor
    # blobs use binary PCD; from_buffer delegates to the official binary parser.
    with source.open('rb') as stream:
        points = LidarPointCloud.from_buffer(stream, content_type='pcd').points.T
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError(f'Expected nuPlan v1.1 x,y,z,intensity,ring,lidar_id: {source} {points.shape}')
    temporary = destination.with_suffix('.bin.tmp')
    points.astype(np.float32).tofile(temporary)
    temporary.replace(destination)
    return len(points)


def training_config(scene, camera_ids, output):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(REPO/'Omniscenegraph/configs/original/omnire.yaml')
    cfg.pop('dataset', None)
    data = OmegaConf.load(REPO/'Omniscenegraph/configs/datasets/nuplan/1cams.yaml').data
    data.data_root = str(scene.parent.resolve())
    data.scene_idx = scene.name
    data.pixel_source.cameras = camera_ids
    data.pixel_source.downscale_when_loading = [2]*len(camera_ids)
    data.pixel_source.undistort = True
    data.pixel_source.load_smpl = False
    data.pixel_source.load_dynamic_mask = False
    data.pixel_source.load_sky_mask = False
    data.pixel_source.load_road_mask = False
    data.lidar_source.depth_mode = 'sf_raw'
    data.lidar_source.projection_policy = 'zbuffer'
    data.lidar_source.occlusion_tolerance = .2
    cfg.data = data
    for key in ('SMPLNodes', 'DeformableNodes'):
        cfg.model.pop(key, None)
    cfg.model.RigidNodes.init.only_moving = False
    cfg.model.RigidNodes.init.instance_max_pts = 2000
    cfg.model.Background.init.from_lidar.num_samples = 200000
    cfg.model.Background.init.near_randoms = 20000
    cfg.model.Background.init.far_randoms = 20000
    # This adapter uses the full background, including road. Road decomposition
    # and sky-mask supervision can be enabled when real masks are supplied.
    cfg.model.Background.init.seg_road_type = 'NONE'
    cfg.model.Sky.params.bounded = True
    # Without real sky masks, a high-resolution environment texture can learn
    # foreground residuals and reveal colored stripes when actors move away.
    cfg.model.Sky.params.resolution = 64
    cfg.trainer.losses.pop('mask', None)
    cfg.render.render_novel = None
    cfg.render.render_test = False
    cfg.render.vis_error = False
    OmegaConf.save(cfg, output)


def prepare(args):
    bundle, manifest = load_bundle(args.bundle)
    scene_name = Path(manifest['database']).stem+'__'+manifest['scenario_token']
    scene = args.output_root.resolve()/scene_name
    camera_ids = sorted(set(args.cameras))
    if not camera_ids or any(i < 0 or i >= len(nuplan.CAMERAS) for i in camera_ids):
        raise ValueError('Camera IDs must be in 0..7')
    if args.fps <= 0 or args.fps > 20:
        raise ValueError('nuPlan preparation supports 0 < fps <= 20')
    first, last = map(int, (bundle['timestamps_us'][0], bundle['timestamps_us'][-1]))
    reconstruction_tail = getattr(args, 'reconstruction_tail_seconds', 0.)
    if not 0 <= reconstruction_tail <= 60:
        raise ValueError('Reconstruction tail must be in 0..60 seconds')
    last += int(round(reconstruction_tail * 1e6))
    targets = np.rint(np.arange(first, last+.5, 1e6/args.fps)).astype(np.int64)
    locator = nuplan.SensorLocator(args.sensor_roots)
    missing, prepared, all_boxes = [], [], []
    with nuplan.connect(manifest['database']) as db:
        frames = nuplan.nearest_frames(db, targets)
        calibrations = nuplan.cameras(db, [nuplan.CAMERAS[i] for i in camera_ids])
        for index, frame in enumerate(frames):
            lidar_path = locator.resolve(frame['filename'])
            if not lidar_path:
                missing.append(frame['filename'])
            image_records = []
            for channel, calibration in calibrations.items():
                image = nuplan.nearest_image(db, calibration['token'], frame['lidar_timestamp'])
                image_path = locator.resolve(image['filename_jpg'])
                if not image_path:
                    missing.append(image['filename_jpg'])
                image_records.append({'camera_id': calibration['id'], 'channel': channel,
                    'timestamp_us': int(image['image_timestamp']), 'ego_to_world': nuplan.ego_pose(image).tolist(),
                    'source': str(image_path) if image_path else None})
            prepared.append({'frame_idx': index, 'timestamp_us': int(frame['lidar_timestamp']),
                'lidar_token': frame['lidar_token'].hex(), 'lidar_source': str(lidar_path) if lidar_path else None,
                'ego_to_world': nuplan.ego_pose(frame).tolist(), 'images': image_records})
            all_boxes.append([o for o in nuplan.boxes(db, frame['lidar_token']) if o['category_name'] in ('vehicle', 'pedestrian', 'bicycle')])
    # Preflight all inputs before emitting a partially usable training dataset.
    if missing:
        nuplan.write_json(args.output_root/'missing_sensor_files.json', {'status': 'missing_sensor_files', 'files': missing})
        raise FileNotFoundError(f'{len(missing)} missing sensor files; see {args.output_root}/missing_sensor_files.json')
    for name in ('images', 'ego_pose', 'camera_ego_pose', 'intrinsics', 'extrinsics', 'instances', 'lidar'):
        (scene/name).mkdir(parents=True, exist_ok=True)
    identity = {'database': manifest['database'], 'scenario_token': manifest['scenario_token'],
        'bundle_sha256': manifest['bundle_sha256'], 'fps': args.fps, 'cameras': camera_ids}
    if reconstruction_tail:
        identity['reconstruction_tail_seconds'] = reconstruction_tail
    state_path = scene/'nexus_scene.json'
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if old['identity'] != identity:
            raise ValueError('Existing prepared scene uses different inputs or sampling')
    state = {'schema_version': 1, 'dataset': 'nuplan', 'status': 'preparing', 'identity': identity,
        'world_from_scene': prepared[0]['ego_to_world'], 'frames': prepared,
        'semantic_masks': 'not supplied; reconstruction config disables semantic supervision'}
    nuplan.write_json(state_path, state)
    for calibration in calibrations.values():
        cam_id = calibration['id']
        K = calibration['intrinsics']
        np.savetxt(scene/'extrinsics'/f'{cam_id}.txt', calibration['extrinsics'])
        np.savetxt(scene/'intrinsics'/f'{cam_id}.txt', np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], calibration['distortion']])
    instances = {}
    tokens = sorted({o['track_token'] for objects in all_boxes for o in objects})
    id_map = {token: index for index, token in enumerate(tokens)}
    frame_instances = {}
    lidar_tasks = []
    for frame, objects in zip(prepared, all_boxes):
        index = frame['frame_idx']
        np.savetxt(scene/'ego_pose'/f'{index:03d}.txt', frame['ego_to_world'])
        for image in frame['images']:
            cam_id = image['camera_id']
            link(Path(image['source']), scene/'images'/f'{index:03d}_{cam_id}.jpg')
            np.savetxt(scene/'camera_ego_pose'/f'{index:03d}_{cam_id}.txt', image['ego_to_world'])
        frame_instances[index] = []
        for obj in objects:
            instance_id = id_map[obj['track_token']]
            record = instances.setdefault(instance_id, {'id': obj['track_token'], 'class_name': obj['category_name'],
                'frame_annotations': {'frame_idx': [], 'obj_to_world': [], 'box_size': []}})
            annotations = record['frame_annotations']
            annotations['frame_idx'].append(index)
            annotations['obj_to_world'].append(nuplan.box_pose(obj).tolist())
            annotations['box_size'].append([obj['length'], obj['width'], obj['height']])
            frame_instances[index].append(instance_id)
        lidar_tasks.append((Path(frame['lidar_source']), scene/'lidar'/f'{index:03d}.bin'))
    nuplan.write_json(scene/'instances/instances_info.json', instances)
    nuplan.write_json(scene/'instances/frame_instances.json', frame_instances)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        point_counts = list(executor.map(convert_lidar, lidar_tasks))
    training_config(scene, camera_ids, scene/'train_config.yaml')
    state.update(status='prepared', point_counts=point_counts,
        num_images=sum(len(frame['images']) for frame in prepared), num_instances=len(instances))
    nuplan.write_json(state_path, state)
    print(f'NUPLAN_3DGS_PREPARED scene={scene} frames={len(prepared)} images={state["num_images"]} '
          f'instances={len(instances)} points={sum(point_counts)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('artifacts/nexus_3dgs/processed'))
    parser.add_argument('--sensor-roots', type=Path, nargs='+', default=[
        Path('/inspire/hdd/global_public/public_datas/nuPlan/sensor_blobs_full'),
        Path('/inspire/hdd/global_public/public_datas/nuPlan/sensor_blobs_mini')])
    parser.add_argument('--fps', type=float, default=10.)
    parser.add_argument('--cameras', type=int, nargs='+', default=[0])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--reconstruction-tail-seconds', type=float, default=0.,
        help='Extra real observations for offline reconstruction only; never extend the Nexus prediction or its validity')
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
