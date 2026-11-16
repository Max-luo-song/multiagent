"""Apply exported trajectories to existing Omniscenegraph appearance assets."""
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

from .bundle import load_bundle
from .geometry import matrix_quaternion, resample_poses


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def retarget_camera(image_infos, camera_infos, camera_to_world):
    """Move camera and its world-space rays together, preserving intrinsics.

    Gaussian rasterization reads camera_infos, while the sky reads viewdirs.
    Updating just the former would leave the sky in the logged orientation.
    """
    old_rotation = camera_infos['camera_to_world'][:3, :3]
    new_rotation = camera_to_world[:3, :3]
    image_infos['viewdirs'] = image_infos['viewdirs'] @ old_rotation @ new_rotation.T
    image_infos['origins'] = camera_to_world[:3, 3].expand_as(image_infos['origins'])
    camera_infos['camera_to_world'] = camera_to_world


def apply_planar_motion(base_scene_pose, target_world_pose, reference_world_pose, scene_from_world):
    """Apply Nexus XY/yaw change while retaining reconstructed height/tilt.

    Nexus supplies planar states. Its exported z/roll/pitch are annotations,
    not generated motion, and must not overwrite the trained 3D registration.
    A target identical to the reference leaves reconstruction untouched.
    """
    from .geometry import pose
    def planar(value):
        return pose(value[0, 3], value[1, 3], yaw=np.arctan2(value[1, 0], value[0, 0]))
    delta_world = planar(target_world_pose) @ np.linalg.inv(planar(reference_world_pose))
    return scene_from_world @ delta_world @ np.linalg.inv(scene_from_world) @ base_scene_pose


def bind_instances(tokens, instances_info, model_ids, num_instances, explicit_map=None, strict=True):
    """Join Nexus track token -> original dataset ID -> model ID.

    Checkpoints made before dataset_instance_ids was added require an explicit
    track-token -> model-index JSON. Inferring correspondence from row order is
    deliberately unsupported because training filters static/invisible objects.
    """
    known = {str(value['id']): int(key) for key, value in instances_info.items()}
    if len(known) != len(instances_info):
        raise ValueError('Duplicate dataset track identities')
    if explicit_map is not None:
        mapping = {str(token): int(index) for token, index in explicit_map.items()}
        if any(token not in known for token in mapping):
            raise ValueError('Explicit identity map contains tracks outside the trained scene')
        if len(set(mapping.values())) != len(mapping) or any(not 0 <= v < num_instances for v in mapping.values()):
            raise ValueError('Explicit model indices must be unique and in range')
    else:
        if model_ids is None:
            raise ValueError('Legacy 3DGS checkpoint has no dataset_instance_ids; supply --instance-map or retrain with this adapter')
        model_ids = numpy(model_ids).tolist()
        if len(model_ids) != num_instances or len(set(model_ids)) != len(model_ids):
            raise ValueError('Invalid dataset_instance_ids in checkpoint')
        reverse = {int(value): index for index, value in enumerate(model_ids)}
        if not set(reverse).issubset({int(key) for key in instances_info}):
            raise ValueError('Checkpoint instance IDs do not belong to this scene')
        mapping = {token: reverse[index] for token, index in known.items() if index in reverse}
    bindings = {agent: mapping[str(token)] for agent, token in enumerate(tokens) if agent and str(token) in mapping}
    missing = [str(token) for agent, token in enumerate(tokens) if agent and agent not in bindings]
    if strict and missing:
        raise ValueError(f'{len(missing)} Nexus tracks have no trained rigid Gaussian asset: {missing[:8]}')
    if len(tokens) > 1 and not bindings:
        raise ValueError('No Nexus tracks bind to trained Gaussian assets')
    return bindings, missing


class TrajectoryController:
    def __init__(self, bundle_dir, scene_dir, rigid_model, start_frame=0, explicit_map=None, strict=True,
                 pose_policy='planar_delta'):
        self.bundle, self.manifest = load_bundle(bundle_dir)
        self.scene_dir = Path(scene_dir)
        self.scene = json.loads((self.scene_dir/'nexus_scene.json').read_text())
        if self.scene['status'] != 'prepared':
            raise ValueError('Prepared scene is incomplete')
        identity = self.scene['identity']
        if Path(identity['database']).resolve() != Path(self.manifest['database']).resolve() or identity['scenario_token'] != self.manifest['scenario_token']:
            raise ValueError('3DGS scene and Nexus prediction refer to different logs/scenarios')
        if rigid_model is None:
            raise ValueError('Checkpoint has no RigidNodes model for Nexus traffic control')
        self.rigid = rigid_model
        if pose_policy not in ('planar_delta', 'absolute'):
            raise ValueError('Unknown rigid pose policy')
        self.pose_policy = pose_policy
        self.grounding = None
        self.start_frame = int(start_frame)
        self.scene_from_world = np.linalg.inv(np.asarray(self.scene['frames'][start_frame]['ego_to_world']))
        self.bindings, self.missing = bind_instances(self.bundle['track_tokens'],
            json.loads((self.scene_dir/'instances/instances_info.json').read_text()),
            getattr(rigid_model, 'instance_ids', None), rigid_model.num_instances, explicit_map, strict)
        if rigid_model.num_frames+start_frame > len(self.scene['frames']):
            raise ValueError('Checkpoint duration exceeds prepared scene duration')

    def sample(self, timestamp_us, reference=False):
        key = 'reference_poses_world' if reference else 'predicted_poses_world'
        poses, valid = resample_poses(self.bundle['timestamps_us'], self.bundle[key], self.bundle['valid'], [timestamp_us])
        return poses[0], valid[0]

    @contextmanager
    def edit_frame(self, frame_index, timestamp_us, reference=False):
        """Temporarily update rigid poses/visibility; always restore after render."""
        import torch
        if not 0 <= frame_index < self.rigid.num_frames:
            raise ValueError('Frame outside trained 3DGS duration')
        poses, valid = self.sample(timestamp_us, reference)
        reference_poses, _ = self.sample(timestamp_us, True)
        if not valid[0]:
            raise ValueError('Requested camera timestamp is outside the Nexus clip')
        stable_all = self.grounding is not None and getattr(self.grounding, 'stabilize_all', False)
        ids = list(range(self.rigid.num_instances)) if stable_all else list(self.bindings.values())
        positions = {model_index:index for index,model_index in enumerate(ids)}
        with torch.no_grad():
            translation = self.rigid.instances_trans[frame_index, ids].clone()
            quaternion = self.rigid.instances_quats[frame_index, ids].clone()
            visibility = self.rigid.instances_fv[frame_index, ids].clone()
            try:
                if stable_all:
                    for model_index in set(ids)-set(self.bindings.values()):
                        logged = self.grounding.reference_poses[model_index].get(frame_index+self.start_frame)
                        if logged is None or not self.rigid.instances_fv[frame_index,model_index]:
                            continue
                        local = self.grounding.object_scene_pose(model_index, logged)
                        self.rigid.instances_trans[frame_index,model_index].copy_(torch.as_tensor(
                            local[:3,3],dtype=self.rigid.instances_trans.dtype,device=self.rigid.instances_trans.device))
                        self.rigid.instances_quats[frame_index,model_index].copy_(torch.as_tensor(
                            matrix_quaternion(local[:3,:3]),dtype=self.rigid.instances_quats.dtype,device=self.rigid.instances_quats.device))
                for agent, model_index in self.bindings.items():
                    index = positions[model_index]
                    grounded = self.grounding is not None and (stable_all or self.grounding.supports(poses[agent]))
                    if reference and self.pose_policy == 'planar_delta' and not grounded:
                        continue
                    self.rigid.instances_fv[frame_index, model_index] = bool(valid[agent])
                    if not valid[agent]:
                        continue
                    if grounded:
                        local = self.grounding.object_scene_pose(model_index, poses[agent])
                    elif self.pose_policy == 'absolute':
                        local = self.scene_from_world @ poses[agent]
                    else:
                        from .geometry import quaternion_matrix
                        base = np.eye(4)
                        base[:3, 3] = numpy(translation[index])
                        base[:3, :3] = quaternion_matrix(numpy(quaternion[index]))
                        local = apply_planar_motion(base, poses[agent], reference_poses[agent], self.scene_from_world)
                    self.rigid.instances_trans[frame_index, model_index].copy_(
                        torch.as_tensor(local[:3, 3], dtype=self.rigid.instances_trans.dtype, device=self.rigid.instances_trans.device))
                    self.rigid.instances_quats[frame_index, model_index].copy_(
                        torch.as_tensor(matrix_quaternion(local[:3, :3]), dtype=self.rigid.instances_quats.dtype, device=self.rigid.instances_quats.device))
                yield poses[0]
            finally:
                self.rigid.instances_trans[frame_index, ids] = translation
                self.rigid.instances_quats[frame_index, ids] = quaternion
                self.rigid.instances_fv[frame_index, ids] = visibility
