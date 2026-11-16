"""Replay the saved Nexus prediction in the separately registered 3DGS scene."""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'Omniscenegraph')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint', 'scene', 'bundle', 'asset-registration', 'original-scene', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--frames', type=int, nargs='+', default=list(range(20, 100)))
    p.add_argument('--save-actor-metadata', action='store_true', help='Save actual temporary actor/camera poses and physical-box diagnostics')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    import imageio.v2 as imageio, numpy as np, torch
    from nexus_3dgs.bundle import load_bundle
    from nexus_3dgs.geometry import resample_poses, matrix_quaternion, quaternion_matrix
    from nexus_3dgs.replay import load_trainer, camera_inputs, RecordedStates
    from nexus_3dgs.trajectory_control import planar, register_measured_actor
    from nexus_3dgs.scene_projection import pair_separations
    torch.set_num_threads(8)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    checksum = digest(a.checkpoint)
    trainer, cfg = load_trainer(a.checkpoint); rigid = trainer.models['RigidNodes']
    original = {key: getattr(rigid, key).clone() for key in ['instances_trans', 'instances_quats', 'instances_fv']}
    immutable = [(param, param._version) for name, model in trainer.models.items() for key, param in model.named_parameters()
                 if not (name == 'RigidNodes' and key in ['instances_trans', 'instances_quats'])]
    meta = json.loads((a.scene / 'nexus_scene.json').read_text()); S = np.linalg.inv(meta['frames'][0]['ego_to_world']); W = np.linalg.inv(S)
    info = json.loads((a.scene / 'instances/instances_info.json').read_text())
    bytoken = {info[str(i)]['id']: m for m, i in enumerate(rigid.instance_ids.cpu().tolist())}
    states = RecordedStates(a.asset_registration, a.original_scene)
    source_tokens={body['model_index']:body['track_token'] for body in states.run['frames'][0]['images'][0]['scene_metadata']['actors'] if body['model_index']>=0}
    for index, dataset_id in enumerate(rigid.instance_ids.cpu().tolist()):
        assert info[str(dataset_id)]['id'] == source_tokens[index]
    ground = states.run['ground_registration']['ground_model']
    groundS = np.linalg.inv(ground.get('world_from_scene', states.scene['frames'][0]['ego_to_world']))
    aa, bb, cc = ground['coefficients_z_ax_by_c']; plane = groundS.T @ np.array([-aa, -bb, 1., -cc])
    frames = a.frames
    assert frames == sorted(set(frames)) and all(20 <= f < 100 for f in frames)
    cameras = [next(c for c in meta['frames'][f]['images'] if c['camera_id'] == 0) for f in frames]
    arrays, manifest = load_bundle(a.bundle)
    assert manifest['scenario_token'] == meta['identity']['scenario_token']
    poses, valid = resample_poses(arrays['timestamps_us'], arrays['predicted_poses_world'], arrays['valid'], [c['timestamp_us'] for c in cameras])
    assert valid[:, 0].all()
    bindings = {slot: bytoken[str(token)] for slot, token in enumerate(arrays['track_tokens']) if slot and str(token) in bytoken}
    assert len(bindings) == len(arrays['track_tokens']) - 1
    intr = np.loadtxt(a.scene / 'intrinsics/0.txt'); K = torch.tensor([[intr[0]/2, 0, intr[2]/2], [0, intr[1]/2, intr[3]/2], [0, 0, 1]], device='cuda', dtype=torch.float32)
    extr = np.loadtxt(a.scene / 'extrinsics/0.txt'); records = []
    try:
        with imageio.get_writer(str(a.output / 'camera_0.mp4'), fps=10, macro_block_size=2, ffmpeg_params=['-threads', '2', '-crf', '18']) as writer:
            for j, (f, camera) in enumerate(zip(frames, cameras)):
                E = np.array(camera['ego_to_world'])
                with torch.no_grad():
                    real = trainer.models['CamPose'](torch.tensor(S @ E @ extr, device='cuda', dtype=torch.float32), torch.tensor(f*len(cfg.data.pixel_source.cameras), device='cuda'))
                    real_world = W @ real.cpu().numpy()
                    generated = planar(poses[j, 0]) @ np.linalg.inv(planar(E)) @ real_world
                    for slot, model_index in bindings.items():
                        rigid.instances_fv[f, model_index] = bool(valid[j, slot])
                        if not valid[j, slot]: continue
                        key = str(model_index); flat = key in states.flat
                        mount = np.array(states.flat[key] if flat else states.mounts[key])
                        T = S @ register_measured_actor(poses[j, slot], mount, plane, flat)
                        rigid.instances_trans[f, model_index].copy_(torch.tensor(T[:3, 3], device='cuda', dtype=torch.float32))
                        rigid.instances_quats[f, model_index].copy_(torch.tensor(matrix_quaternion(T[:3, :3]), device='cuda', dtype=torch.float32))
                    C = torch.tensor(S @ generated, device='cuda', dtype=torch.float32)
                    image, cam = camera_inputs(C, K, f, 540, 960, rigid.num_frames, len(cfg.data.pixel_source.cameras))
                    result = trainer(image, cam, novel_view=True); result = result[0] if isinstance(result, tuple) else result
                    rgb = (result['rgb'].clamp(0, 1).cpu().numpy()*255).round().astype(np.uint8)
                    imageio.imwrite(a.output / f'{f:03d}_0.png', rgb); writer.append_data(rgb)
                    record = {'frame': f, 'timestamp_us': camera['timestamp_us'], 'camera_world': generated.tolist(), 'real_camera_world': real_world.tolist(), 'raw_predicted_ego_world': poses[j, 0].tolist()}
                    if a.save_actor_metadata:
                        ego = poses[j, 0]
                        yaw = float(np.arctan2(ego[1, 0], ego[0, 0]))
                        center = ego[:3, 3].copy()
                        center[:2] += 1.461 * np.array([np.cos(yaw), np.sin(yaw)])
                        actors = [{'model_index': -1, 'track_token': 'ego', 'nexus_slot': 0, 'valid': True,
                                   'center_world': center.tolist(), 'yaw_world_rad': yaw, 'size_lwh': [5.176, 2.297, 1.777]}]
                        inverse_bindings = {m: slot for slot, m in bindings.items()}
                        for m in range(rigid.num_instances):
                            body = np.eye(4)
                            body[:3, :3] = quaternion_matrix(rigid.instances_quats[f, m].cpu().numpy())
                            body[:3, 3] = rigid.instances_trans[f, m].cpu().numpy()
                            body = W @ body
                            slot = inverse_bindings.get(m)
                            actors.append({'model_index': m, 'track_token': source_tokens[m],
                                           'nexus_slot': int(arrays['nexus_slots'][slot]) if slot is not None else None,
                                           'valid': bool(rigid.instances_fv[f, m]), 'center_world': body[:3, 3].tolist(),
                                           'yaw_world_rad': float(np.arctan2(body[1, 0], body[0, 0])),
                                           'size_lwh': rigid.instances_size[m].cpu().tolist(), 'body_world': body.tolist()})
                        boxes = np.array([[*actor['center_world'][:2], actor['yaw_world_rad']] for actor in actors])
                        sizes = np.array([actor['size_lwh'][:2] for actor in actors])
                        visible = np.array([actor['valid'] for actor in actors])
                        sep, pairs = pair_separations(boxes[None], visible[None], sizes)
                        overlaps = np.argwhere((sep[0] <= 0) & pairs[0]).tolist()
                        ego_bus = next(i for i, actor in enumerate(actors) if actor['nexus_slot'] == 1)
                        record['scene_metadata'] = {'actors': actors, 'camera_world': generated.tolist(),
                            'intrinsics': K.cpu().tolist(), 'width': 960, 'height': 540,
                            'minimum_sat_separation_m': float(sep[pairs].min()),
                            'ego_bus_sat_separation_m': float(sep[0, 0, ego_bus]),
                            'overlapping_actor_pairs': overlaps,
                            'geometry_scope': 'physical oriented 2D boxes at actual camera exposure samples; ego rear-axle offset 1.461m; no continuous-time guarantee'}
                    records.append(record)
                if j % 10 == 0: print(f'REGISTERED_NEXUS_RENDER {j+1}/{len(frames)}', flush=True)
        assert all(param._version == version for param, version in immutable)
    finally:
        with torch.no_grad():
            for key, value in original.items():getattr(rigid, key).copy_(value)
        for key, value in original.items():assert torch.equal(value, getattr(rigid, key))
    assert digest(a.checkpoint) == checksum
    postprocessing={key:manifest[key] for key in ['heading_repair','path_stabilization'] if key in manifest}
    report = {'status': 'passed', 'checkpoint_sha256': checksum, 'bundle_sha256': manifest['bundle_sha256'],
              'checkpoint': str(a.checkpoint), 'scene': str(a.scene), 'bundle': str(a.bundle),
              'renderer_sha256': digest(Path(__file__)), 'actual_actor_metadata_saved': a.save_actor_metadata,
              'camera_policy': 'raw predicted ego XY/yaw with registered measured height/tilt and shared small camera residual',
              'actor_policy': ('audited derived Nexus bundle' if postprocessing else 'saved original Nexus XY/yaw')+' with existing measured-road asset mounting; unpredicted assets retain registered actual poses',
              'later_offline_nexus_trajectory_edits': bool(postprocessing), 'trajectory_postprocessing':postprocessing,'source_prediction_sha256':manifest['prediction_sha256'],'nexus_trained': False,
              'appearance_and_checkpoint_unchanged': True, 'temporary_actor_poses_restored_exactly': True,
              'generated_view_psnr_computed': False, 'frames': records}
    (a.output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print('REGISTERED_NEXUS_RENDER_COMPLETE', flush=True)


if __name__ == '__main__':main()
