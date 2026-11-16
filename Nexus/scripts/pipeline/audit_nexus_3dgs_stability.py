"""Separate trajectory, annotation-height, reconstruction and camera motion."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from nexus_3dgs.geometry import quaternion_matrix
from nexus_3dgs.runtime import TrajectoryController, apply_planar_motion


def summary(values):
    values = np.asarray(values)
    return {'min': float(values.min()), 'max': float(values.max()), 'range': float(np.ptp(values)),
        'max_frame_step': float(np.max(np.abs(np.diff(values)))) if len(values) > 1 else 0.,
        'second_difference_rms': float(np.sqrt(np.mean(np.diff(values, n=2)**2))) if len(values) > 2 else 0.}


def angles(matrix):
    return np.rad2deg([np.arctan2(matrix[2, 1], matrix[2, 2]),
        np.arctan2(-matrix[2, 0], np.hypot(matrix[2, 1], matrix[2, 2])),
        np.arctan2(matrix[1, 0], matrix[0, 0])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    rigid_state = checkpoint['models']['RigidNodes']
    rigid = SimpleNamespace(instance_ids=rigid_state['dataset_instance_ids'],
        instances_trans=rigid_state['instances_trans'], instances_quats=rigid_state['instances_quats'],
        instances_fv=rigid_state['instances_fv'])
    rigid.num_frames, rigid.num_instances = rigid.instances_fv.shape
    controller = TrajectoryController(args.bundle, args.scene, rigid)
    frames = [f for f in controller.scene['frames'] if f['timestamp_us'] >=
        int(controller.bundle['timestamps_us'][controller.manifest['current_index']])]
    frames = [f for f in frames if f['images'][0]['timestamp_us'] <= controller.bundle['timestamps_us'][-1]]
    matrices = {'generated': [], 'reference': [], 'logged_ego': []}
    for frame in frames:
        camera = next(c for c in frame['images'] if c['camera_id'] == 0)
        matrices['generated'].append(controller.sample(camera['timestamp_us'])[0])
        matrices['reference'].append(controller.sample(camera['timestamp_us'], True)[0])
        matrices['logged_ego'].append(camera['ego_to_world'])
    matrices = {key: np.asarray(value) for key, value in matrices.items()}
    rows = []
    current = controller.manifest['current_index']
    for agent, token in enumerate(controller.bundle['track_tokens']):
        valid = controller.bundle['valid'][current:, agent]
        pred = controller.bundle['predicted_poses_world'][current:, agent, :2, 3]
        ref = controller.bundle['reference_poses_world'][current:, agent, :2, 3]
        both = valid[1:] & valid[:-1]
        rows.append({'slot': int(controller.bundle['nexus_slots'][agent]), 'token': str(token),
            'valid_future_intervals': int(both.sum()),
            'predicted_path_length_m': float(np.linalg.norm(np.diff(pred, axis=0), axis=-1)[both].sum()),
            'logged_path_length_m': float(np.linalg.norm(np.diff(ref, axis=0), axis=-1)[both].sum())})
    bus_agent = 1
    bus_model = controller.bindings[bus_agent]
    world_from_scene = np.linalg.inv(controller.scene_from_world)
    trained_world = []
    for frame in frames:
        index = frame['frame_idx']
        p = np.eye(4)
        p[:3, :3] = quaternion_matrix(rigid.instances_quats[index, bus_model].numpy())
        p[:3, 3] = rigid.instances_trans[index, bus_model].numpy()
        trained_world.append(world_from_scene @ p)
    trained_world = np.asarray(trained_world)
    extrinsic = np.loadtxt(args.scene/'extrinsics/0.txt')
    intrinsics = np.loadtxt(args.scene/'intrinsics/0.txt')
    from PIL import Image
    old_image = REPO/'artifacts/nexus_3dgs/render_generated_integration5000/020_0.png'
    render_width, render_height = Image.open(old_image).size
    source_width, source_height = Image.open(args.scene/'images/020_0.jpg').size
    K = np.array([[intrinsics[0]*render_width/source_width, 0, intrinsics[2]*render_width/source_width],
        [0, intrinsics[1]*render_height/source_height, intrinsics[3]*render_height/source_height], [0, 0, 1.]])
    cam_state = checkpoint['models']['CamPose']
    projections = {key: [] for key in ('old_generated', 'registered_generated', 'old_reference', 'registered_reference')}
    ground = None
    ground_path = REPO/'artifacts/nexus_3dgs/stability_audit/ground_plane.json'
    def camera_correction(index):
        embedding = cam_state['embeds.weight'][index].numpy()
        d6 = embedding[3:]+cam_state['identity'].numpy()
        b1 = d6[:3]/np.linalg.norm(d6[:3])
        b2 = d6[3:]-b1*(b1 @ d6[3:]); b2 /= np.linalg.norm(b2)
        correction = np.eye(4)
        correction[:3, :3] = np.stack((b1, b2, np.cross(b1, b2)))
        correction[:3, 3] = embedding[:3]
        return correction
    if ground_path.is_file():
        from nexus_3dgs.grounding import GroundedRegistration
        ground = GroundedRegistration(controller, ground_path)
        camera_poses = []
        for frame in controller.scene['frames']:
            record = next(c for c in frame['images'] if c['camera_id'] == 0)
            camera_poses.append(controller.scene_from_world @ np.asarray(record['ego_to_world']) @ extrinsic @ camera_correction(frame['frame_idx']))
        ground.register_camera(0, camera_poses)
        projections['grounded_generated'] = []
    for j, frame in enumerate(frames):
        logged_camera = matrices['logged_ego'][j] @ extrinsic
        correction = camera_correction(frame['frame_idx'])
        registered_camera = logged_camera @ correction
        pairs = {
            'old_generated': (matrices['generated'][j, 0] @ extrinsic, matrices['generated'][j, bus_agent]),
            'old_reference': (logged_camera, matrices['reference'][j, bus_agent]),
            'registered_reference': (registered_camera, trained_world[j]),
            'registered_generated': (
                apply_planar_motion(registered_camera, matrices['generated'][j, 0], matrices['reference'][j, 0], np.eye(4)),
                apply_planar_motion(trained_world[j], matrices['generated'][j, bus_agent], matrices['reference'][j, bus_agent], np.eye(4)))
        }
        if ground is not None:
            pairs['grounded_generated'] = (world_from_scene @ ground.camera_scene_pose(0, matrices['generated'][j, 0]),
                world_from_scene @ ground.object_scene_pose(bus_model, matrices['generated'][j, bus_agent]))
        for name, (camera_pose, bus_pose) in pairs.items():
            point = (np.linalg.inv(camera_pose) @ bus_pose)[:3, 3]
            pixel = K @ point
            projections[name].append((pixel[:2]/pixel[2]).tolist())
    motion = {}
    for name, values in (('nexus_ego', matrices['generated'][:, 0]),
            ('reference_ego', matrices['reference'][:, 0]), ('logged_ego', matrices['logged_ego']),
            ('nexus_bus', matrices['generated'][:, bus_agent]),
            ('reference_bus', matrices['reference'][:, bus_agent]), ('trained_bus', trained_world)):
        rpy = np.array([angles(p[:3, :3]) for p in values])
        motion[name] = {'world_z_m': summary(values[:, 2, 3]), 'roll_deg': summary(rpy[:, 0]),
            'pitch_deg': summary(rpy[:, 1]), 'yaw_deg': summary(rpy[:, 2])}
    output = {'checkpoint_step': int(checkpoint['step']), 'actors': rows, 'motion': motion,
        'bus_center_projection_pixels': {key: {'u': summary(np.array(value)[:, 0]),
            'v': summary(np.array(value)[:, 1]), 'positions': value} for key, value in projections.items()},
        'projection_metric_scope': 'projected rigid object center, not measured optical flow or image quality',
        'camera_state_keys': list(checkpoint['models']['CamPose']),
        'bus_track_token': str(controller.bundle['track_tokens'][bus_agent]),
        'frame_count': len(frames)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
