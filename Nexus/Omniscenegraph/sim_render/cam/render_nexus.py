"""Render Nexus-controlled traffic using a trained Omniscenegraph scene.

Run from the Omniscenegraph directory, with its 3DGS Python environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

OMNI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OMNI.parent))
sys.path.insert(0, str(OMNI))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instance-map', type=Path, help='Legacy checkpoint: JSON track token -> rigid model index')
    parser.add_argument('--allow-uncontrolled', action='store_true', help='Explicitly allow Nexus tracks with no trained appearance asset')
    parser.add_argument('--camera-mode', choices=('generated_ego', 'logged'), default='generated_ego')
    parser.add_argument('--pose-policy', choices=('planar_delta', 'absolute'), default='planar_delta',
        help='Preserve trained height/tilt and apply Nexus XY/yaw deltas; absolute reproduces the original adapter')
    parser.add_argument('--camera-refinement', choices=('checkpoint', 'none'), default='checkpoint')
    parser.add_argument('--ground-model', type=Path, help='Validated local LiDAR plane: fixed rigid registration for unobserved height/tilt')
    parser.add_argument('--stabilize-all-rigid', action='store_true',
        help='Stabilize every rigid instance; outside plane use fixed track height/tilt instead of noisy per-frame poses')
    parser.add_argument('--rigid-max-std', type=float, help='Physical maximum Gaussian standard deviation in metres (appearance ablation)')
    parser.add_argument('--rigid-support-sigma', type=float, help='Confine this many Gaussian standard deviations to the vehicle box + 0.08 m')
    parser.add_argument('--rigid-sh-degree', type=int, choices=(0,1,2,3), help='Appearance ablation: maximum rendered vehicle SH degree')
    parser.add_argument('--reference', action='store_true', help='Render logged trajectories to validate registration')
    parser.add_argument('--include-history', action='store_true')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--start-frame', type=int, help='First prepared-scene frame index, for targeted diagnostics')
    parser.add_argument('--frame-indices', type=int, nargs='+',help='Explicit diagnostic frame indices; requires --images-only')
    parser.add_argument('--images-only',action='store_true',help='Write PNG frames and metadata without a video')
    parser.add_argument('--diagnostic-frames', type=int, nargs='*', default=[],
        help='Also save background/rigid RGB and dynamic opacity on these prepared frames')
    parser.add_argument('--render-cameras', type=int, nargs='+',
        help='Subset to render; training camera indices and embeddings are preserved')
    parser.add_argument('--save-actor-metadata', action='store_true',
        help='Save actual temporary poses and camera projection for a labelled diagram')
    parser.add_argument('--save-geometry', action='store_true',
        help='Save static depth/opacity and dynamic opacity for novel-view geometry checks')
    parser.add_argument('--camera-poses-from-run', type=Path,
        help='Use actual world cameras from a complete render of the identical Nexus bundle, for reconstruction A/B diagnostics')
    args = parser.parse_args()
    if args.frame_indices is not None and not args.images_only:
        parser.error('--frame-indices requires --images-only to avoid compressing nonuniform times into a misleading video')
    if not args.resume_from.is_file():
        parser.error(f'Missing trained 3DGS checkpoint: {args.resume_from}')

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str
    from nexus_3dgs.runtime import TrajectoryController, retarget_camera, apply_planar_motion
    from nexus_3dgs.nuplan import write_json

    if not torch.cuda.is_available():
        raise RuntimeError('Omniscenegraph rendering requires a CUDA-enabled 3DGS environment')
    cfg = OmegaConf.load(args.resume_from.parent/'config.yaml')
    save_geometry = (args.save_geometry or 'RoadNodes' in cfg.model
                     or cfg.trainer.losses.get('scene_reconstruction', None) is not None)
    if cfg.data.dataset != 'nuplan':
        raise ValueError('This adapter requires a nuPlan scene checkpoint')
    scene_dir = Path(cfg.data.data_root)/str(cfg.data.scene_idx)
    dataset = DrivingDataset(data_cfg=cfg.data)
    render_cameras = set(dataset.pixel_source.camera_list if args.render_cameras is None else args.render_cameras)
    if not render_cameras or not render_cameras.issubset(dataset.pixel_source.camera_list):
        raise ValueError('Render cameras must be a nonempty subset of training cameras')
    trainer = import_str(cfg.trainer.type)(**cfg.trainer, num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model, num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set), test_set_indices=[],
        scene_aabb=dataset.get_aabb().reshape(2, 3), device=torch.device('cuda'))
    trainer.resume_from_checkpoint(ckpt_path=str(args.resume_from), load_only_model=True)
    trainer.set_eval()
    if args.stabilize_all_rigid and not args.ground_model:
        raise ValueError('All-rigid stabilization requires a validated ground model')
    if args.rigid_max_std is not None and args.rigid_max_std <= 0:
        raise ValueError('Gaussian standard deviation must be positive')
    if args.rigid_support_sigma is not None and args.rigid_support_sigma <= 0:
        raise ValueError('Gaussian support sigma must be positive')
    trainer.models['RigidNodes'].render_max_std = args.rigid_max_std
    trainer.models['RigidNodes'].render_support_sigma = args.rigid_support_sigma
    trainer.models['RigidNodes'].render_sh_degree = args.rigid_sh_degree
    # The renderer writes controlled per-frame poses; held-out-neighbour smoothing
    # would overwrite them in RigidNodes.transform_means.
    trainer.test_set_indices = []
    explicit = json.loads(args.instance_map.read_text()) if args.instance_map else None
    controller = TrajectoryController(args.bundle, scene_dir, trainer.models.get('RigidNodes'),
        start_frame=int(cfg.data.start_timestep), explicit_map=explicit, strict=not args.allow_uncontrolled,
        pose_policy=args.pose_policy)
    camera_overrides = None
    if args.camera_poses_from_run:
        source = json.loads(args.camera_poses_from_run.read_text())
        if args.reference or args.camera_mode != 'generated_ego':
            raise ValueError('Camera A/B overrides require generated-ego mode')
        from nexus_3dgs.render_metadata import matched_camera_overrides
        camera_overrides = matched_camera_overrides(source,controller.manifest['bundle_sha256'])
    if args.ground_model:
        if args.pose_policy != 'planar_delta':
            raise ValueError('Ground registration requires planar_delta policy')
        from nexus_3dgs.grounding import GroundedRegistration
        controller.grounding = GroundedRegistration(controller, args.ground_model, stabilize_all=args.stabilize_all_rigid)
        for camera_position, camera_id in enumerate(dataset.pixel_source.camera_list):
            cameras = dataset.pixel_source.camera_data[camera_id].cam_to_worlds
            image_ids = torch.arange(len(cameras), device='cuda')*dataset.pixel_source.num_cams+camera_position
            with torch.no_grad():
                if args.camera_refinement == 'checkpoint':
                    for key in ('CamPosePerturb', 'CamPose'):
                        if key in trainer.models:
                            cameras = trainer.models[key](cameras, image_ids)
            controller.grounding.register_camera(camera_id, cameras.detach().cpu().numpy())
    args.output.mkdir(parents=True, exist_ok=True)
    run = {'status': 'started', 'renderer': 'Omniscenegraph_3DGS', 'camera_mode': args.camera_mode,
        'pose_policy': args.pose_policy, 'camera_refinement': args.camera_refinement,
        'lens': 'undistorted_pinhole' if cfg.data.pixel_source.undistort else 'pinhole_without_distortion_correction',
        'reference': args.reference, 'bundle': str(args.bundle.resolve()),
        'bundle_sha256': controller.manifest['bundle_sha256'], 'checkpoint': str(args.resume_from.resolve()),
        'checkpoint_sha256': hashlib.sha256(args.resume_from.read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256((args.resume_from.parent/'config.yaml').read_bytes()).hexdigest(),
        'lidar_projection_policy': cfg.data.lidar_source.get('projection_policy', 'legacy'),
        'sky_bounded': bool(cfg.model.Sky.get('params', {}).get('bounded', False)),
        'training_cameras': list(dataset.pixel_source.camera_list), 'render_cameras': sorted(render_cameras),
        'controlled_track_count': len(controller.bindings), 'uncontrolled_nexus_tracks': controller.missing,
        'ground_registration': controller.grounding.metadata() if controller.grounding is not None else None,
        'asset_size_policy': 'preserve reconstructed Gaussian geometry', 'frames': []}
    run['static_geometry_saved'] = bool(save_geometry)
    run['images_only'] = bool(args.images_only)
    if args.frame_indices is not None:run['requested_frame_indices'] = sorted(set(args.frame_indices))
    if args.camera_poses_from_run:
        run['camera_poses_from_run'] = str(args.camera_poses_from_run.resolve())
        run['camera_poses_source_sha256'] = hashlib.sha256(args.camera_poses_from_run.read_bytes()).hexdigest()
    run['rigid_geometry_constraints'] = {'max_std_m':args.rigid_max_std if args.rigid_max_std is not None else cfg.model.RigidNodes.ctrl.get('physical_max_std_m',None),
                                         'support_sigma':args.rigid_support_sigma if args.rigid_support_sigma is not None else cfg.model.RigidNodes.ctrl.get('physical_support_sigma',None),
                                         'box_margin_m':.08,'trajectory_xy_yaw_modified':False}
    run['rigid_render_sh_degree'] = args.rigid_sh_degree
    if any(run['rigid_geometry_constraints'][key] is not None for key in ('max_std_m','support_sigma')):
        run['asset_size_policy'] = 'Gaussian ellipsoid support constrained to physical vehicle boxes'
    write_json(args.output/'run.json', run)
    writers = {}
    try:
        for frame_index in range(dataset.num_img_timesteps):
            record = controller.scene['frames'][controller.start_frame+frame_index]
            if args.frame_indices is not None and record['frame_idx'] not in args.frame_indices:
                continue
            if args.start_frame is not None and record['frame_idx'] < args.start_frame:
                continue
            if not args.include_history and record['timestamp_us'] < int(controller.bundle['timestamps_us'][controller.manifest['current_index']]):
                continue
            if args.max_frames is not None and len(run['frames']) >= args.max_frames:
                break
            frame_files = []
            for camera_position, camera_id in enumerate(dataset.pixel_source.camera_list):
                if camera_id not in render_cameras:
                    continue
                camera = next(i for i in record['images'] if i['camera_id'] == camera_id)
                timestamp = camera['timestamp_us']
                if not controller.bundle['timestamps_us'][0] <= timestamp <= controller.bundle['timestamps_us'][-1]:
                    continue
                image_infos, camera_infos = dataset.full_image_set.get_image(
                    frame_index*dataset.pixel_source.num_cams+camera_position, 1)
                image_infos = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in image_infos.items()}
                camera_infos = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in camera_infos.items()}
                base_camera = camera_infos['camera_to_world']
                if args.camera_refinement == 'checkpoint':
                    with torch.no_grad():
                        base_camera = trainer.process_camera(camera_infos,
                            image_infos['img_idx'].flatten()[0], novel_view=False).camtoworlds
                with controller.edit_frame(frame_index, timestamp, args.reference) as ego:
                    if controller.grounding is not None:
                        target_ego = ego if args.camera_mode == 'generated_ego' and not args.reference else np.asarray(camera['ego_to_world'])
                        if not controller.grounding.supports(target_ego):
                            raise ValueError('Ego trajectory leaves validated local ground coverage')
                        c2w = controller.grounding.camera_scene_pose(camera_id, target_ego)
                        base_camera = torch.as_tensor(c2w, dtype=torch.float32, device='cuda')
                    elif args.camera_mode == 'generated_ego':
                        if args.pose_policy == 'planar_delta':
                            reference_ego = controller.sample(timestamp, True)[0][0]
                            c2w = apply_planar_motion(base_camera.detach().cpu().numpy(),
                                ego, reference_ego, controller.scene_from_world)
                        else:
                            logged_ego = np.asarray(camera['ego_to_world'])
                            c2w = controller.scene_from_world @ ego @ np.linalg.inv(logged_ego) @ np.linalg.inv(controller.scene_from_world) @ base_camera.detach().cpu().numpy()
                        base_camera = torch.as_tensor(c2w, dtype=torch.float32, device='cuda')
                    if camera_overrides is not None:
                        key = (record['frame_idx'],camera_id,timestamp)
                        if key not in camera_overrides:
                            raise ValueError(f'Missing camera A/B exposure: {key}')
                        matched = camera_overrides[key]
                        np.testing.assert_allclose(camera_infos['intrinsics'].detach().cpu().numpy(),matched['intrinsics'],atol=1e-6,rtol=0)
                        base_camera = torch.as_tensor(controller.scene_from_world @ np.asarray(matched['camera_world']),dtype=torch.float32,device='cuda')
                    retarget_camera(image_infos, camera_infos, base_camera)
                    with torch.no_grad():
                        result = trainer(image_infos, camera_infos, novel_view=True)
                    outputs = result[0] if isinstance(result, tuple) else result
                    actor_record = None
                    if args.save_actor_metadata:
                        from nexus_3dgs.render_metadata import actor_metadata
                        actor_record = actor_metadata(controller, frame_index, ego, base_camera,
                                                      camera_infos['intrinsics'], outputs['depth'])
                    rgb = (outputs['rgb'].detach().clamp(0, 1).cpu().numpy()*255).round().astype(np.uint8)
                    if save_geometry:
                        geometry_dir=args.output/'geometry';geometry_dir.mkdir(exist_ok=True)
                        np.savez_compressed(geometry_dir/f'{record["frame_idx"]:03d}_{camera_id}.npz',
                            depth=outputs.get('Static_depth',outputs['Background_depth']).detach().cpu().numpy()[...,0].astype(np.float32),
                            opacity=outputs.get('Static_opacity',outputs['Background_opacity']).detach().cpu().numpy()[...,0].astype(np.float16),
                            dynamic_opacity=outputs['Dynamic_opacity'].detach().cpu().numpy()[...,0].astype(np.float16))
                    if record['frame_idx'] in args.diagnostic_frames:
                        diagnostic_dir = args.output/'diagnostic_layers'
                        diagnostic_dir.mkdir(exist_ok=True)
                        raw_layers, layer_stats = {}, {}
                        for key in ('Background_rgb', 'RigidNodes_rgb', 'Dynamic_opacity', 'rgb_sky', 'rgb_sky_blend', 'rgb_gaussians', 'opacity'):
                            if key in outputs:
                                raw = outputs[key].detach().cpu().numpy()
                                raw_layers[key] = raw
                                layer_stats[key] = {'min':float(raw.min()), 'max':float(raw.max()),
                                    'below_zero_fraction':float((raw < 0).mean()), 'above_one_fraction':float((raw > 1).mean())}
                                layer = (np.clip(raw, 0, 1)*255).round().astype(np.uint8)
                                if layer.ndim == 3 and layer.shape[-1] == 1:
                                    layer = layer[..., 0]
                                imageio.imwrite(diagnostic_dir/f'{record["frame_idx"]:03d}_{camera_id}_{key}.png', layer)
                        np.savez_compressed(diagnostic_dir/f'{record["frame_idx"]:03d}_{camera_id}_raw.npz', **raw_layers)
                        write_json(diagnostic_dir/f'{record["frame_idx"]:03d}_{camera_id}_stats.json', layer_stats)
                filename = f'{record["frame_idx"]:03d}_{camera_id}.png'
                imageio.imwrite(args.output/filename, rgb)
                if not args.images_only and camera_id not in writers:
                    writers[camera_id] = imageio.get_writer(str(args.output/f'camera_{camera_id}.mp4'),
                        fps=controller.scene['identity']['fps'], macro_block_size=2,
                        ffmpeg_params=['-threads', '4'])
                if not args.images_only: writers[camera_id].append_data(rgb)
                image_record = {'camera_id': camera_id, 'timestamp_us': timestamp, 'file': filename}
                if actor_record is not None:
                    image_record['scene_metadata'] = actor_record
                frame_files.append(image_record)
            if frame_files:
                run['frames'].append({'frame_idx': record['frame_idx'], 'images': frame_files})
                print(f'NEXUS_3DGS_RENDER frame={record["frame_idx"]} cameras={len(frame_files)}', flush=True)
        if not run['frames']:
            raise ValueError('No camera timestamps overlap the exported Nexus clip')
        run['status'] = 'complete'
    except Exception as exc:
        run.update(status='failed', error=str(exc))
        raise
    finally:
        for writer in writers.values():
            writer.close()
        write_json(args.output/'run.json', run)
    print(f'NEXUS_3DGS_RENDER_OK output={args.output}', flush=True)


if __name__ == '__main__':
    main()
