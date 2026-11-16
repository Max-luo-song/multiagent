"""Describe actual edited renderer poses for an interpretable traffic diagram."""
import numpy as np

from .geometry import quaternion_matrix


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def matched_camera_overrides(source, bundle_sha256):
    """Read unique exposures from a completed render of this same prediction."""
    if (source['status'] != 'complete' or source['reference']
            or source['camera_mode'] != 'generated_ego'
            or source['bundle_sha256'] != bundle_sha256):
        raise ValueError('Camera A/B source must be a completed generated-ego render of the identical prediction bundle')
    result={}
    for frame in source['frames']:
        for image in frame['images']:
            key=(frame['frame_idx'],image['camera_id'],image['timestamp_us'])
            if key in result:raise ValueError('Duplicate camera A/B exposure')
            metadata=image['scene_metadata'];C=np.asarray(metadata['camera_world'])
            if C.shape!=(4,4) or not np.isfinite(C).all():raise ValueError('Invalid source camera')
            np.testing.assert_allclose(C[3],[0,0,0,1],atol=1e-8,rtol=0)
            np.testing.assert_allclose(C[:3,:3].T@C[:3,:3],np.eye(3),atol=1e-5,rtol=0)
            if np.linalg.det(C[:3,:3])<.999:raise ValueError('Source camera rotation is not proper')
            result[key]=metadata
    if not result:raise ValueError('No camera A/B exposures')
    return result


def actor_metadata(controller, frame_index, ego_world, camera_scene, intrinsics, depth):
    """Call inside edit_frame: use the same poses and camera as the RGB render.

    Pixel visibility is a centre-depth heuristic for labelling, not segmentation.
    The diagram keeps every actor with a valid pose, including occluded actors.
    """
    rigid = controller.rigid
    world_from_scene = np.linalg.inv(controller.scene_from_world)
    camera_world = world_from_scene @ numpy(camera_scene)
    camera_from_scene = np.linalg.inv(numpy(camera_scene))
    K = numpy(intrinsics)
    depth = numpy(depth).squeeze()
    height, width = depth.shape
    inverse_bindings = {model: agent for agent, model in controller.bindings.items()}
    ego = np.asarray(ego_world).copy()
    ego[:3, 3] += 1.461 * ego[:3, 0]
    actors = [{'model_index': -1, 'agent_index': 0, 'nexus_slot': 0, 'track_token': '__ego__',
               'valid': True, 'center_world': ego[:3, 3].tolist(),
               'yaw_world_rad': float(np.arctan2(ego[1, 0], ego[0, 0])),
               'size_lwh': [5.176, 2.297, 1.8], 'center_visible': False}]
    ids = numpy(rigid.instance_ids)
    import json
    info = json.loads((controller.scene_dir / 'instances/instances_info.json').read_text())
    for model in range(rigid.num_instances):
        agent = inverse_bindings.get(model)
        pose = np.eye(4)
        pose[:3, 3] = numpy(rigid.instances_trans[frame_index, model])
        pose[:3, :3] = quaternion_matrix(numpy(rigid.instances_quats[frame_index, model]))
        world = world_from_scene @ pose
        size = numpy(rigid.instances_size[model])
        center = camera_from_scene @ pose[:, 3]
        uv = None
        visible = False
        if center[2] > .1:
            pixel = K @ center[:3]
            uv = pixel[:2] / pixel[2]
            x, y = np.rint(uv).astype(int)
            if 0 <= x < width and 0 <= y < height:
                tolerance = float(np.linalg.norm(size) / 2 + .5)
                visible = bool(abs(float(depth[y, x]) - center[2]) <= tolerance)
        actors.append({'model_index': model, 'agent_index': agent,
                       'nexus_slot': None if agent is None else int(controller.bundle['nexus_slots'][agent]),
                       'track_token': info[str(int(ids[model]))]['id'],
                       'valid': bool(rigid.instances_fv[frame_index, model]),
                       'center_world': world[:3, 3].tolist(),
                       'yaw_world_rad': float(np.arctan2(world[1, 0], world[0, 0])),
                       'size_lwh': size.tolist(), 'center_uv': None if uv is None else uv.tolist(),
                       'center_visible': visible})
    return {'actors': actors, 'camera_world': camera_world.tolist(), 'intrinsics': K.tolist(),
            'width': width, 'height': height, 'label_visibility': 'centre-depth heuristic only'}
