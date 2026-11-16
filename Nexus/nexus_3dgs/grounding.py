"""Rigid appearance registration on a validated local LiDAR ground plane.

This is an explicit planar-road rendering assumption. Nexus XY/yaw samples
are retained; unobserved height/tilt and camera mounting are calibrated once.
"""
import json
from pathlib import Path

import numpy as np

from .geometry import matrix_quaternion, quaternion_matrix, pose
from .runtime import numpy


def mean_pose(values):
    values = np.asarray(values)
    result = np.eye(4)
    result[:3, 3] = np.median(values[:, :3, 3], axis=0)
    quaternions = np.array([matrix_quaternion(p[:3, :3]) for p in values])
    _, vectors = np.linalg.eigh(quaternions.T @ quaternions)
    result[:3, :3] = quaternion_matrix(vectors[:, -1])
    return result


class GroundedRegistration:
    def __init__(self, controller, model_path, stabilize_all=False):
        self.controller = controller
        self.stabilize_all = stabilize_all
        self.model = json.loads(Path(model_path).read_text())
        if self.model['status'] != 'validated':
            raise ValueError('Ground plane has not passed validation')
        if Path(self.model['scene']).resolve() != controller.scene_dir.resolve():
            raise ValueError('Ground plane belongs to a different scene')
        if self.model['identity'] != controller.scene['identity']:
            raise ValueError('Ground plane source identity differs')
        scene_plane = np.r_[-np.array(self.model['coefficients_z_ax_by_c'][:2]), 1., -self.model['coefficients_z_ax_by_c'][2]]
        # Plane covectors transform by inverse transpose.
        self.ground_scene_from_world = np.linalg.inv(np.asarray(
            self.model.get('world_from_scene', controller.scene['frames'][0]['ego_to_world'])))
        self.world_plane = self.ground_scene_from_world.T @ scene_plane
        self.world_plane /= np.linalg.norm(self.world_plane[:3])
        if self.world_plane[2] < .9:
            raise ValueError('Ground plane is too steep for this local adapter')
        self.world_from_scene = np.linalg.inv(controller.scene_from_world)
        self.object_mounts, self.camera_mounts = {}, {}
        self.flat_mounts, self.reference_poses = {}, {}
        info = json.loads((controller.scene_dir/'instances/instances_info.json').read_text())
        translations = numpy(controller.rigid.instances_trans)
        quaternions = numpy(controller.rigid.instances_quats)
        visibility = numpy(controller.rigid.instances_fv)
        ids = numpy(controller.rigid.instance_ids)
        model_indices = range(len(ids)) if stabilize_all else controller.bindings.values()
        for model_index in model_indices:
            record = info[str(int(ids[model_index]))]['frame_annotations']
            reference = dict(zip(record['frame_idx'], record['obj_to_world']))
            self.reference_poses[model_index] = {int(k):np.asarray(v) for k,v in reference.items()}
            mounts, flat_mounts = [], []
            all_supported = True
            for local_frame in np.flatnonzero(visibility[:, model_index]):
                frame = int(local_frame)+controller.start_frame
                if not self.in_prediction_window(frame):
                    continue
                if frame not in reference:
                    continue
                trained = np.eye(4)
                trained[:3, :3] = quaternion_matrix(quaternions[local_frame, model_index])
                trained[:3, 3] = translations[local_frame, model_index]
                reference_pose = np.asarray(reference[frame])
                mounts.append(np.linalg.inv(self.road_pose(reference_pose)) @ self.world_from_scene @ trained)
                flat_mounts.append(np.linalg.inv(self.planar_pose(reference_pose)) @ self.world_from_scene @ trained)
                all_supported = all_supported and self.supports(reference_pose)
            if not mounts:
                if not np.any(visibility[:min(len(controller.scene['frames']),len(visibility)), model_index]):
                    continue
                # Extended reconstruction may contain actors seen only after
                # the exported prediction ends. They are invisible during
                # this render and require no mounting calibration.
                visible_frames=np.flatnonzero(visibility[:,model_index])+controller.start_frame
                if not any(self.in_prediction_window(int(f)) for f in visible_frames):
                    continue
                raise ValueError('No valid reconstruction poses for grounded instance')
            self.object_mounts[model_index] = mean_pose(mounts)
            if stabilize_all:
                # Decide once per track: no switching at the plane boundary.
                agent = next((a for a,m in controller.bindings.items() if m == model_index), None)
                if agent is not None:
                    targets = controller.bundle['predicted_poses_world'][:,agent]
                    valid = controller.bundle['valid'][:,agent]
                    all_supported = all_supported and all(self.supports(p) for p in targets[valid])
                if not all_supported:
                    self.flat_mounts[model_index] = mean_pose(flat_mounts)

    @staticmethod
    def planar_pose(target_world):
        return pose(target_world[0,3], target_world[1,3],
                    yaw=np.arctan2(target_world[1,0], target_world[0,0]))

    def in_prediction_window(self, frame):
        if not self.controller.scene.get('identity',{}).get('reconstruction_tail_seconds',0):
            return True
        timestamps=getattr(self.controller,'bundle',{}).get('timestamps_us')
        timestamp=self.controller.scene['frames'][frame].get('timestamp_us')
        return timestamps is None or timestamp is None or int(timestamps[0])<=timestamp<=int(timestamps[-1])

    def road_pose(self, target_world):
        x, y = target_world[:2, 3]
        nx, ny, nz, d = self.world_plane
        yaw = np.arctan2(target_world[1, 0], target_world[0, 0])
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.])
        normal = self.world_plane[:3]
        forward -= normal*(normal @ forward)
        forward /= np.linalg.norm(forward)
        result = np.eye(4)
        result[:3, :3] = np.column_stack((forward, np.cross(normal, forward), normal))
        result[:3, 3] = [x, y, -(nx*x+ny*y+d)/nz]
        return result

    def supports(self, target_world):
        position = self.ground_scene_from_world @ target_world[:, 3]
        low, high = np.asarray(self.model['supported_xy_bounds_scene'])
        return bool(np.all(position[:2] >= low) and np.all(position[:2] <= high))

    def object_scene_pose(self, model_index, target_world):
        if model_index in self.flat_mounts:
            return self.controller.scene_from_world @ self.planar_pose(target_world) @ self.flat_mounts[model_index]
        return self.controller.scene_from_world @ self.road_pose(target_world) @ self.object_mounts[model_index]

    def register_camera(self, camera_id, camera_scene_poses):
        mounts = []
        for index, camera_scene in enumerate(camera_scene_poses):
            record = self.controller.scene['frames'][self.controller.start_frame+index]
            if not self.in_prediction_window(self.controller.start_frame+index):
                continue
            logged = next(c for c in record['images'] if c['camera_id'] == camera_id)
            mounts.append(np.linalg.inv(self.road_pose(np.asarray(logged['ego_to_world']))) @ self.world_from_scene @ camera_scene)
        self.camera_mounts[camera_id] = mean_pose(mounts)

    def camera_scene_pose(self, camera_id, ego_world):
        return self.controller.scene_from_world @ self.road_pose(ego_world) @ self.camera_mounts[camera_id]

    def metadata(self):
        return {'policy': 'constant_rigid_registration_on_lidar_ground_plane',
            'ground_model': self.model, 'object_mounts': {str(k):v.tolist() for k,v in self.object_mounts.items()},
            'camera_mounts': {str(k):v.tolist() for k,v in self.camera_mounts.items()},
            'nexus_motion': 'unchanged XY/yaw; no generated trajectory smoothing',
            'stabilize_all_rigid_instances': self.stabilize_all,
            'flat_track_mounts': {str(k):v.tolist() for k,v in self.flat_mounts.items()},
            'outside_ground_coverage_policy': ('fixed per-track world height/tilt; planar motion preserved; local flat-track assumption'
                if self.stabilize_all else 'preserve per-frame reconstruction and apply planar delta'),
            'assumption': 'locally planar road and fixed camera/rigid-body mounting'}
