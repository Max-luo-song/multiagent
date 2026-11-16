"""CPU regression checks for coordinate/time/identity boundaries of the bridge."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import typing
import unittest

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))
from nexus_3dgs.bundle import FEATURE_MEANS, FEATURE_STD, load_bundle, save_bundle
from nexus_3dgs.geometry import interpolate_rotation, matrix_quaternion, pose, quaternion_matrix, resample_poses
from nexus_3dgs.runtime import TrajectoryController, bind_instances, retarget_camera, apply_planar_motion
from nexus_3dgs.nuplan import SensorLocator, write_json


class BridgeTests(unittest.TestCase):
    def test_normalization_agrees_with_nexus(self):
        tree = ast.parse((REPO/'nuplan_extent/planning/training/preprocessing/features/scene_tensor.py').read_text())
        constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('FEATURE_MEANS', 'FEATURE_STD')}
        np.testing.assert_array_equal(FEATURE_MEANS, constants['FEATURE_MEANS'])
        np.testing.assert_array_equal(FEATURE_STD, constants['FEATURE_STD'])

    def test_rotations_and_heading_wrap(self):
        for yaw in (0., .7, np.pi, -np.pi+.01):
            rotation = pose(0, 0, yaw=yaw)[:3, :3]
            np.testing.assert_allclose(quaternion_matrix(matrix_quaternion(rotation)), rotation, atol=1e-10)
        midway = interpolate_rotation(pose(0, 0, yaw=np.deg2rad(179))[:3, :3], pose(0, 0, yaw=np.deg2rad(-179))[:3, :3], .5)
        self.assertLess(midway[0, 0], -.9999)

    def test_no_gap_filling_or_extrapolation(self):
        source = np.array([[pose(0, 0)], [pose(10, 0)], [pose(20, 0)]])
        result, valid = resample_poses([0, 500000, 1000000], source, [[1], [0], [1]], [-1, 0, 250000, 750000, 1000000, 1000001])
        np.testing.assert_array_equal(valid[:, 0], [0, 1, 0, 0, 1, 0])
        continuous, mask = resample_poses([0, 1000000], source[[0, 2]], [[1], [1]], [250000])
        self.assertTrue(mask[0, 0])
        self.assertAlmostEqual(continuous[0, 0, 0, 3], 5.)

    def test_identity_survives_filtering_and_reordering(self):
        instances = {'7': {'id': 'car-a'}, '20': {'id': 'car-b'}}
        bindings, missing = bind_instances(['__ego__', 'car-a', 'car-b'], instances, [20, 7], 2)
        self.assertEqual(bindings, {1: 1, 2: 0})
        self.assertFalse(missing)
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            bind_instances(['__ego__', 'car-a'], instances, None, 2)
        with self.assertRaisesRegex(ValueError, 'unique'):
            bind_instances(['__ego__', 'car-a'], instances, None, 2, {'car-a': 0, 'car-b': 0})
        with self.assertRaisesRegex(ValueError, 'no trained'):
            bind_instances(['__ego__', 'missing-car'], instances, [20, 7], 2)

    def test_camera_and_sky_rays_move_together(self):
        old = torch.tensor(pose(100, 200, yaw=np.pi/2), dtype=torch.float32)
        new = torch.tensor(pose(105, 201, yaw=np.pi), dtype=torch.float32)
        camera_ray = torch.tensor([[[1., 0., 0.]]])
        image = {'viewdirs': camera_ray @ old[:3, :3].T, 'origins': old[:3, 3].reshape(1, 1, 3)}
        camera = {'camera_to_world': old}
        retarget_camera(image, camera, new)
        torch.testing.assert_close(image['viewdirs'], camera_ray @ new[:3, :3].T)
        torch.testing.assert_close(image['origins'], new[:3, 3].reshape(1, 1, 3))
        self.assertIs(camera['camera_to_world'], new)

    def test_planar_motion_preserves_reconstruction_height_and_tilt(self):
        origin = pose(100., 200.)
        base = pose(6.2, 1.1, z=.4)
        angle = .12
        base[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
        reference = pose(106., 201., z=50.)
        target = pose(109., 205., z=999., yaw=np.pi/2)
        changed = apply_planar_motion(base, target, reference, np.linalg.inv(origin))
        np.testing.assert_allclose(changed[:3, 3], [8.9, 5.2, .4], atol=1e-10)
        np.testing.assert_allclose(changed[:3, :3], pose(0, 0, yaw=np.pi/2)[:3, :3] @ base[:3, :3], atol=1e-10)
        np.testing.assert_allclose(apply_planar_motion(base, reference, reference, np.linalg.inv(origin)), base, atol=1e-10)

    def test_ground_registration_keeps_height_on_sloped_road(self):
        from types import SimpleNamespace
        from nexus_3dgs.grounding import GroundedRegistration
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'instances').mkdir()
            identity = {'database': '/tmp/test.db', 'scenario_token': 'test'}
            normal = np.array([-.01, 0., 1.]); normal /= np.linalg.norm(normal)
            forward = np.array([1., 0., 0.])-normal*normal[0]; forward /= np.linalg.norm(forward)
            rotation = np.column_stack((forward, np.cross(normal, forward), normal))
            references, trained, frames = [], [], []
            mount = pose(0, 0, z=1.5, yaw=.02)
            camera_mount = pose(.2, 0, z=2.)
            cameras = []
            for index, noise in enumerate((-.08, 0., .08)):
                reference = pose(float(index), 0.)
                road = reference.copy(); road[2, 3] = .01*index; road[:3, :3] = rotation
                value = road @ mount; value[2, 3] += noise
                references.append(reference)
                trained.append(value)
                cameras.append(road @ camera_mount)
                frames.append({'ego_to_world': np.eye(4).tolist(), 'images': [{'camera_id': 0, 'ego_to_world': reference.tolist()}]})
            trained = np.array(trained)
            write_json(root/'instances/instances_info.json', {'7': {'id': 'bus', 'frame_annotations': {
                'frame_idx': [0, 1, 2], 'obj_to_world': np.array(references).tolist()}}})
            ground_path = root/'ground.json'
            write_json(ground_path, {'status': 'validated', 'scene': str(root), 'identity': identity,
                'coefficients_z_ax_by_c': [.01, 0., 0.], 'world_from_scene': np.eye(4).tolist(),
                'supported_xy_bounds_scene': [[-1., -5.], [10., 5.]]})
            controller = SimpleNamespace(scene_dir=root, scene={'identity': identity, 'frames': frames},
                scene_from_world=np.eye(4), start_frame=0, bindings={1: 0},
                rigid=SimpleNamespace(instance_ids=np.array([7]), instances_fv=np.ones((3, 1), dtype=bool),
                    instances_trans=trained[:, None, :3, 3],
                    instances_quats=np.array([[matrix_quaternion(p[:3, :3])] for p in trained])))
            ground = GroundedRegistration(controller, ground_path)
            target = pose(5., 0., z=999.)
            road = target.copy(); road[2, 3] = .05; road[:3, :3] = rotation
            np.testing.assert_allclose(ground.object_scene_pose(0, target), road @ mount, atol=1e-10)
            ground.register_camera(0, cameras)
            np.testing.assert_allclose(ground.camera_scene_pose(0, target), road @ camera_mount, atol=1e-10)
            self.assertTrue(ground.supports(target))
            self.assertFalse(ground.supports(pose(11., 0.)))

    def test_edit_restores_checkpoint_even_after_render_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, scene = root/'bundle', root/'scene'
            (scene/'instances').mkdir(parents=True)
            times = np.array([0, 500000, 1000000], dtype=np.int64)
            poses = np.array([[pose(100, 200), pose(105+t, 201), pose(110+t, 202)] for t in range(3)])
            valid = np.ones((3, 3), dtype=bool)
            valid[1, 2] = False
            save_bundle(bundle, {'timestamps_us': times, 'predicted_poses_world': poses, 'reference_poses_world': poses,
                'valid': valid, 'track_tokens': np.array(['__ego__', 'car-a', 'car-b'])},
                {'schema_version': 1, 'dataset': 'nuplan', 'database': '/tmp/log.db', 'scenario_token': 'abc'})
            write_json(scene/'nexus_scene.json', {'status': 'prepared', 'identity': {'database': '/tmp/log.db', 'scenario_token': 'abc'},
                'frames': [{'ego_to_world': pose(100, 200).tolist()} for _ in times]})
            write_json(scene/'instances/instances_info.json', {'7': {'id': 'car-a'}, '20': {'id': 'car-b'}})
            from types import SimpleNamespace
            rigid = SimpleNamespace(num_frames=3, num_instances=2, instance_ids=torch.tensor([20, 7]),
                instances_trans=torch.nn.Parameter(torch.randn(3, 2, 3)),
                instances_quats=torch.nn.Parameter(torch.randn(3, 2, 4)), instances_fv=torch.ones(3, 2, dtype=torch.bool))
            original = [v.clone() for v in (rigid.instances_trans, rigid.instances_quats, rigid.instances_fv)]
            controller = TrajectoryController(bundle, scene, rigid, pose_policy='absolute')
            trans_parameter = rigid.instances_trans
            with self.assertRaisesRegex(RuntimeError, 'render failure'):
                with controller.edit_frame(1, 500000):
                    np.testing.assert_allclose(rigid.instances_trans[1, 1].detach().numpy(), [6., 1., 0.])
                    self.assertFalse(rigid.instances_fv[1, 0].item())
                    raise RuntimeError('render failure')
            self.assertIs(rigid.instances_trans, trans_parameter)
            for value, expected in zip((rigid.instances_trans, rigid.instances_quats, rigid.instances_fv), original):
                self.assertTrue(torch.equal(value, expected))
            calibrated = TrajectoryController(bundle, scene, rigid)
            with calibrated.edit_frame(1, 500000, reference=True):
                for value, expected in zip((rigid.instances_trans, rigid.instances_quats, rigid.instances_fv), original):
                    self.assertTrue(torch.equal(value, expected))
            with calibrated.edit_frame(1, 500000):
                # In this fixture predicted=reference, so a planar edit must
                # preserve the trained translation/height instead of snapping
                # the instance back to the database pose.
                torch.testing.assert_close(rigid.instances_trans[1, 1], original[0][1, 1])
                np.testing.assert_allclose(quaternion_matrix(rigid.instances_quats[1, 1].detach().numpy()),
                    quaternion_matrix(original[1][1, 1].detach().numpy()), atol=1e-6)
            for value, expected in zip((rigid.instances_trans, rigid.instances_quats, rigid.instances_fv), original):
                self.assertTrue(torch.equal(value, expected))
            broken = json.loads((scene/'nexus_scene.json').read_text())
            broken['identity']['scenario_token'] = 'other-scene'
            write_json(scene/'nexus_scene.json', broken)
            with self.assertRaisesRegex(ValueError, 'different logs'):
                TrajectoryController(bundle, scene, rigid)

    def test_actual_rigid_checkpoint_identity_roundtrip(self):
        # Execute the real checkpoint methods with a minimal tensor base. CUDA
        # rasterization/appearance imports are irrelevant to serialization.
        tree = ast.parse((REPO/'Omniscenegraph/models/nodes/rigid.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RigidNodes')
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('state_dict', 'load_state_dict', 'num_frames', 'num_instances')]
        class TensorBase:
            def state_dict(self):
                return {'instances_trans': self.instances_trans.clone(), 'instances_quats': self.instances_quats.clone()}
            def load_state_dict(self, value, **kwargs):
                with torch.no_grad():
                    self.instances_trans.copy_(value['instances_trans'])
                    self.instances_quats.copy_(value['instances_quats'])
        namespace = {'VanillaGaussians': TensorBase, 'Dict': typing.Dict, 'torch': torch, 'Parameter': torch.nn.Parameter}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<rigid checkpoint methods>', 'exec'), namespace)
        Rigid = namespace['RigidNodes']
        original = Rigid()
        original.device = torch.device('cpu')
        original.instance_ids = torch.tensor([20, 7])
        original.instances_fv = torch.ones(3, 2, dtype=torch.bool)
        original.instances_size = torch.ones(2, 3)
        original.point_ids = torch.tensor([[0], [1]])
        original.instances_trans = torch.randn(3, 2, 3)
        original.instances_quats = torch.randn(3, 2, 4)
        state = original.state_dict()
        loaded = Rigid()
        loaded.device = torch.device('cpu')
        loaded.load_state_dict(copy.deepcopy(state))
        self.assertTrue(torch.equal(loaded.instance_ids, original.instance_ids))
        self.assertTrue(torch.equal(loaded.instances_trans, original.instances_trans))
        state.pop('dataset_instance_ids')
        loaded.load_state_dict(state)
        self.assertIsNone(loaded.instance_ids)

    def test_camera_uses_image_time_pose_and_legacy_fallback(self):
        tree = ast.parse((REPO/'Omniscenegraph/datasets/nuplan/nuplan_sourceloader.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NuPlanCameraData')
        ns = {'CameraData': object, 'np': np, 'torch': torch, 'os': os, 'OPENCV2DATASET': np.eye(4)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<nuplan camera>', 'exec'), ns)
        Camera = ns['NuPlanCameraData']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('ego_pose', 'camera_ego_pose', 'intrinsics', 'extrinsics'):
                (root/name).mkdir()
            np.savetxt(root/'intrinsics/0.txt', [100., 100., 50., 50., 0., 0., 0., 0., 0.])
            np.savetxt(root/'extrinsics/0.txt', pose(1., 0.))
            for t in range(2):
                np.savetxt(root/'ego_pose'/f'{t:03d}.txt', pose(10.+t, 20.))
            np.savetxt(root/'camera_ego_pose/001_0.txt', pose(11.5, 20.))
            camera = Camera()
            camera.data_path, camera.cam_id = str(root), 0
            camera.load_size = camera.original_size = (100, 100)
            camera.start_timestep, camera.end_timestep = 0, 2
            camera.load_calibrations()
            np.testing.assert_allclose(camera.cam_to_worlds[:, 0, 3], [1., 2.5])
            np.testing.assert_allclose(Camera.get_camera2worlds(str(root), '0', 0, 2), camera.cam_to_worlds)

    def test_actual_camera_constructor_without_optional_masks(self):
        # Real constructor + device transfer, with I/O methods supplying a
        # tiny RGB/calibration fixture. No semantic supervision is fabricated.
        tree = ast.parse((REPO/'Omniscenegraph/datasets/base/pixel_source.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CameraData')
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('__init__', 'to')]
        namespace = {'torch': torch, 'Optional': typing.Optional, 'Tensor': torch.Tensor,
            'DATASETS_CONFIG': {'nuplan': {0: {'camera_name': 'CAM_F0', 'original_size': (2, 2)}}}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<actual camera constructor>', 'exec'), namespace)
        class FixtureCamera(namespace['CameraData']):
            def create_all_filelist(self):
                pass
            def load_calibrations(self):
                self.cam_to_worlds = torch.eye(4)[None]
                self.intrinsics, self.distortions = torch.eye(3)[None], torch.zeros(1, 5)
            def load_images(self):
                self.images = torch.zeros(1, 2, 2, 3)
            def load_egocar_mask(self):
                self.egocar_mask = torch.zeros(2, 2)
            def load_ego_template(self):
                self.ego_template = None
        camera = FixtureCamera(dataset_name='nuplan', data_path='unused', cam_id=0,
            load_sky_mask=False, load_dynamic_mask=False, load_road_mask=False)
        for name in ('dynamic_masks', 'human_masks', 'vehicle_masks', 'road_masks', 'sky_masks'):
            self.assertIsNone(getattr(camera, name))

    def test_sensor_package_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'nuplan-v1.1_val_camera_9/log/CAM_F0/image.jpg'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'sensor-file')
            locator = SensorLocator([root])
            self.assertEqual(locator.resolve('log/CAM_F0/image.jpg'), image)
            self.assertIsNone(locator.resolve('log/CAM_F0/missing.jpg'))
            with self.assertRaises(ValueError):
                locator.resolve('../outside/file.jpg')

    def test_actual_background_filter_uses_3d_boxes_without_2d_masks(self):
        from types import MethodType, SimpleNamespace
        tree = ast.parse((REPO/'Omniscenegraph/datasets/driving_dataset.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DrivingDataset')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
            ('project_aggregated_lidar_pts_wo_box', '_get_static_mask_for_frame')]
        def unexpected_new_intrinsics(*args, **kwargs):
            raise AssertionError('nuPlan undistortion must retain the camera intrinsic matrix')
        namespace = {'torch': torch, 'tqdm': lambda iterable, **kwargs: iterable,
            'cv2': SimpleNamespace(getOptimalNewCameraMatrix=unexpected_new_intrinsics)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), '<actual background filter>', 'exec'), namespace)
        class FixtureCamera(SimpleNamespace):
            def __len__(self):
                return 1
        camera = FixtureCamera(cam_name='CAM_F0', undistort=True, dynamic_masks=None, WIDTH=4, HEIGHT=4,
            intrinsics=torch.tensor([[[2., 0., 2.], [0., 2., 2.], [0., 0., 1.]]]),
            cam_to_worlds=torch.eye(4)[None], images=torch.ones(1, 4, 4, 3))
        points = torch.tensor([[0., 0., 5.], [3., 0., 5.]])
        ranges = points.norm(dim=-1, keepdim=True)
        lidar = SimpleNamespace(find_closest_timestep=lambda timestamp: 0,
            get_lidar_rays=lambda index: {'lidar_origins': torch.zeros_like(points),
                'lidar_viewdirs': points/ranges, 'lidar_ranges': ranges})
        box = torch.eye(4)[None, None]
        box[0, 0, 2, 3] = 5.
        pixel = SimpleNamespace(camera_data={0: camera}, normalized_time=torch.tensor([0.]),
            per_frame_instance_mask=torch.ones(1, 1, dtype=torch.bool),
            instances_pose=box, instances_size=torch.full((1, 3), 2.))
        dataset = SimpleNamespace(type='nuplan', device=torch.device('cpu'), pixel_source=pixel, lidar_source=lidar)
        dataset._get_static_mask_for_frame = MethodType(namespace['_get_static_mask_for_frame'], dataset)
        result = namespace['project_aggregated_lidar_pts_wo_box'](dataset)
        torch.testing.assert_close(result['pts'], points[1:])
        torch.testing.assert_close(result['colors'], torch.ones(1, 3))

    def test_nearest_depth_is_order_independent_and_excludes_hidden_points(self):
        ns = {}
        exec(compile((REPO/'Omniscenegraph/datasets/utils/lidar_projection.py').read_text(), '<actual zbuffer>', 'exec'), ns)
        raster = ns['nearest_depth_and_visibility']
        pixels = torch.tensor([[.2, .2], [.3, .1], [1.1, .2], [.4, .4]])
        depths = torch.tensor([4., 12., 7., 4.1])
        depth, visible = raster(pixels, depths, 2, 2, .2)
        torch.testing.assert_close(depth, torch.tensor([[4., 7.], [0., 0.]]))
        self.assertEqual(visible.tolist(), [True, False, True, True])
        order = torch.tensor([3, 2, 1, 0])
        reordered, reordered_visible = raster(pixels[order], depths[order], 2, 2, .2)
        torch.testing.assert_close(reordered, depth)
        self.assertEqual(reordered_visible.tolist(), visible[order].tolist())
        empty, mask = raster(torch.empty(0, 2), torch.empty(0), 2, 2)
        self.assertEqual(empty.sum().item(), 0.)
        self.assertEqual(len(mask), 0)
        with self.assertRaisesRegex(ValueError, 'positive depth'):
            raster(pixels[:1], torch.tensor([0.]), 2, 2)

    def test_actual_dataset_does_not_color_occluded_lidar(self):
        from types import SimpleNamespace
        ns = {'torch': torch, 'np': np, 'tqdm': lambda iterable, **kwargs: iterable,
            'logger': SimpleNamespace(info=lambda *args: None),
            'DepthMode': SimpleNamespace(SINGLE_FRAME_RAW='raw', SINGLE_FRAME_STATIC='static', MULTI_FRAME_STATIC='multi')}
        exec(compile((REPO/'Omniscenegraph/datasets/utils/lidar_projection.py').read_text(), '<actual zbuffer>', 'exec'), ns)
        exec(compile((REPO/'Omniscenegraph/datasets/utils/base_utils.py').read_text(), '<actual projection>', 'exec'), ns)
        tree = ast.parse((REPO/'Omniscenegraph/datasets/driving_dataset.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DrivingDataset')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'project_lidar_pts_on_images')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<actual dataset projection>', 'exec'), ns)
        class FixtureCamera(SimpleNamespace):
            def __len__(self):
                return 1
            def load_depth(self, depth):
                self.depth = depth
        camera = FixtureCamera(cam_name='CAM_F0', undistort=False, WIDTH=2, HEIGHT=1,
            intrinsics=torch.tensor([[[1., 0., .2], [0., 1., .2], [0., 0., 1.]]]),
            cam_to_worlds=torch.eye(4)[None], images=torch.tensor([[[[1., 0., 0.], [0., 1., 0.]]]]))
        points = torch.tensor([[0., 0., 5.], [0., 0., 10.], [10., 0., 10.]])
        ranges = points.norm(dim=-1, keepdim=True)
        lidar = SimpleNamespace(data_cfg={'projection_policy':'zbuffer'}, num_points=3,
            visible_masks=torch.zeros(3, dtype=torch.bool), colors=torch.full((3, 3), .5),
            find_closest_timestep=lambda timestamp: 0,
            get_lidar_rays=lambda index: {'lidar_origins':torch.zeros_like(points), 'lidar_viewdirs':points/ranges,
                'lidar_ranges':ranges, 'lidar_mask':torch.ones(3, dtype=torch.bool)})
        dataset = SimpleNamespace(type='nuplan', depth_mode='raw', device=torch.device('cpu'), lidar_source=lidar,
            pixel_source=SimpleNamespace(camera_data={0:camera}, normalized_time=torch.tensor([0.])))
        ns['project_lidar_pts_on_images'](dataset, delete_out_of_view_points=False)
        self.assertEqual(lidar.visible_masks.tolist(), [True, False, True])
        torch.testing.assert_close(lidar.colors, torch.tensor([[1., 0., 0.], [.5, .5, .5], [0., 1., 0.]]))
        torch.testing.assert_close(camera.depth, torch.tensor([[[5., 10.]]]))

    def test_binary_lidar_conversion(self):
        from scripts.pipeline.prepare_nuplan_3dgs import convert_lidar
        import struct
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/'cloud.pcd', Path(tmp)/'cloud.bin'
            header = ('VERSION .7\nFIELDS x y z intensity ring lidar_info\nSIZE 4 4 4 4 4 4\n'
                'TYPE F F F F I I\nCOUNT 1 1 1 1 1 1\nWIDTH 1\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1\nDATA binary\n')
            source.write_bytes(header.encode()+struct.pack('<ffffii', 1., 2., 3., .5, 4, 5))
            self.assertEqual(convert_lidar((source, target)), 1)
            np.testing.assert_allclose(np.fromfile(target, dtype=np.float32), [1., 2., 3., .5, 4., 5.])

    def test_rgb_training_without_fabricated_semantic_masks(self):
        from omegaconf import OmegaConf
        from types import SimpleNamespace
        tree = ast.parse((REPO/'Omniscenegraph/models/trainers/base.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BasicTrainer')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_losses')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_init_losses')
        namespace = {'Dict': typing.Dict, 'torch': torch, 'np': np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[init, method], type_ignores=[])), '<actual RGB loss>', 'exec'), namespace)
        trainer = SimpleNamespace(losses_dict=OmegaConf.create({'rgb': {'w': 1.}, 'ssim': {'w': 0.}}),
            ssim=lambda a, b: torch.tensor(1.), sky_opacity_loss_fn=None, depth_loss_fn=None,
            road_loss_fn=None, gaussian_classes={}, models={'Sky': None}, step=0)
        namespace['_init_losses'](trainer)
        self.assertIsNone(trainer.sky_opacity_loss_fn)
        pixels = torch.ones(2, 2, 3)
        output = {'rgb': torch.zeros(2, 2, 3)}
        loss = namespace['compute_losses'](trainer, output, {'pixels': pixels}, {}, None)
        self.assertAlmostEqual(loss['rgb_loss'].item(), 1.)
        loss = namespace['compute_losses'](trainer, output, {'pixels': pixels, 'road_masks': torch.ones(2, 2)}, {}, None)
        self.assertAlmostEqual(loss['rgb_loss'].item(), 0.)
        trainer.sky_opacity_loss_fn = object()
        with self.assertRaisesRegex(ValueError, 'real sky masks'):
            namespace['compute_losses'](trainer, output, {'pixels': pixels}, {}, None)

    def test_actual_render_forward_accepts_rgb_without_road_labels(self):
        # Exercise the real scene-graph dispatch, replacing only the unavailable
        # CUDA rasterizer/appearance models with deterministic CPU outputs.
        from types import SimpleNamespace
        tree = ast.parse((REPO/'Omniscenegraph/models/trainers/scene_graph.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiTrainer')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        ns = {'Dict': typing.Dict, 'torch': torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<actual render dispatch>', 'exec'), ns)
        gaussian = SimpleNamespace(**{k: torch.zeros(2, size) for k, size in
            (('means', 3), ('quats', 4), ('scales', 3), ('opacities', 1))})
        freeze = torch.zeros(2, dtype=torch.bool)
        rgb = torch.full((2, 2, 3), .25)
        render = lambda **kwargs: ({'rgb_gaussians': rgb, 'opacity': torch.ones(2, 2, 1)}, None)
        trainer = SimpleNamespace(normalized_timestamps=torch.tensor([0., 1.]), models={'Sky': lambda info: torch.zeros_like(rgb)},
            gaussian_classes={}, in_test_set=False, process_camera=lambda **kwargs: None,
            collect_gaussians=lambda **kwargs: (gaussian, freeze), render_gaussians=render,
            render_cfg=SimpleNamespace(near_plane=.1, far_plane=100., get=lambda *args: 0.),
            sky_image_infos=lambda infos,cam: infos,
            affine_transformation=lambda value, info: value, training=True, render_dynamic_mask=False)
        infos = {'normed_time': torch.tensor([0.]), 'img_idx': torch.tensor([0])}
        outputs, _ = ns['forward'](trainer, infos, {}, novel_view=True)
        torch.testing.assert_close(outputs['rgb'], rgb)
        self.assertNotIn('road_rgb', outputs)
        infos['road_masks'] = torch.ones(2, 2)
        outputs, _ = ns['forward'](trainer, infos, {}, novel_view=True)
        torch.testing.assert_close(outputs['road_rgb'], rgb)


def inspect_real_data():
    from PIL import Image
    summary = []
    for path in sorted((REPO/'artifacts/nexus_3dgs/processed').glob('*/nexus_scene.json')):
        scene = json.loads(path.read_text())
        if scene['status'] != 'prepared':
            continue
        frames, root = scene['frames'], path.parent
        for frame, count in zip(frames, scene['point_counts']):
            index = frame['frame_idx']
            if (root/'lidar'/f'{index:03d}.bin').stat().st_size != count*6*4:
                raise AssertionError('LiDAR dimensions disagree with preparation manifest')
            for camera in frame['images']:
                with Image.open(root/'images'/f'{index:03d}_{camera["camera_id"]}.jpg') as image:
                    image.verify()
                if abs(camera['timestamp_us']-frame['timestamp_us']) > 50000:
                    raise AssertionError('Camera/LiDAR synchronization exceeds tolerance')
        summary.append({'scene': str(root), 'frames': len(frames), 'images': scene['num_images'],
            'instances': scene['num_instances'], 'points': sum(scene['point_counts'])})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO/'artifacts/nexus_3dgs/verification.json')
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BridgeTests))
    real = inspect_real_data() if result.wasSuccessful() else []
    bundle_checks = []
    for path in sorted((REPO/'artifacts/nexus_3dgs').glob('*/manifest.json')):
        arrays, metadata = load_bundle(path.parent)
        bundle_checks.append({'bundle': str(path.parent), 'agents': len(arrays['track_tokens']),
            'bundle_sha256': metadata['bundle_sha256'],
            'max_reference_position_error_m': metadata['max_reference_position_error_m']})
    render_path = REPO/'artifacts/nexus_3dgs/comparison_integration5000/verification.json'
    render_verified = False
    if render_path.is_file():
        rendered = json.loads(render_path.read_text())
        checkpoint = Path(rendered['checkpoint'])
        render_verified = (rendered['status'] == 'passed' and rendered['actual_3dgs_render_verified']
            and rendered['bundle_sha256'] in {item['bundle_sha256'] for item in bundle_checks}
            and checkpoint.is_file()
            and hashlib.sha256(checkpoint.read_bytes()).hexdigest() == rendered['checkpoint_sha256'])
    report = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'prepared_scenes': real,
        'real_prediction_bundles': bundle_checks, 'actual_3dgs_render_verified': render_verified,
        'photorealistic_render_verified': render_verified,
        'render_verification': str(render_path) if render_path.is_file() else None}
    stability_path = REPO/'artifacts/nexus_3dgs/stability_comparison/verification.json'
    if stability_path.is_file():
        stability = json.loads(stability_path.read_text())
        report['stability_verification'] = {'report': str(stability_path), 'status': stability['status'],
            'same_prediction_before_after': stability['inputs_unchanged'],
            'old_vertical_projection_range_px': stability['old_bus_projected_vertical_range_px'],
            'grounded_vertical_projection_range_px': stability['grounded_bus_projected_vertical_range_px']}
    image_motion_path = REPO/'artifacts/nexus_3dgs/stability_audit/image_motion.json'
    if image_motion_path.is_file():
        motion = json.loads(image_motion_path.read_text())
        report['bus_texture_motion_check'] = {'report': str(image_motion_path),
            'old_vertical_range_px': motion['old']['median_texture_vertical_range_px'],
            'grounded_vertical_range_px': motion['grounded']['median_texture_vertical_range_px']}
    write_json(args.output, report)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print('NEXUS_3DGS_CHECKS_OK '+str(args.output))


if __name__ == '__main__':
    main()
