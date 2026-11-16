"""Export saved Nexus predictions without loading a Nexus/3DGS model."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import pose, yaw_matrix
from . import nuplan

# Kept independent of the SDK-heavy SceneTensor module; regression tests compare
# this contract with that module's constants.
FEATURE_MEANS = np.array([0., 0., 0., 0., 0., 0., 4.5, 2.])
FEATURE_STD = np.array([52., 52., .5, .5, 2., 2., 2.5, .8])
SCHEMA_VERSION = 1


def decode(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] != 8:
        raise ValueError('nuPlan Nexus tensors must have exactly eight state channels')
    return value*(2*FEATURE_STD)+FEATURE_MEANS


def read_run(run_dir, prediction=None):
    """Resolve both single-scene demos and the existing multi-city run format."""
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir/'run.json').read_text())
    prediction = Path(prediction).resolve() if prediction else run_dir/'prediction.pt'
    if not prediction.is_file():
        raise FileNotFoundError(f'{prediction}; use --prediction for a multi-scene run')
    if 'database' in manifest and 'scenario_token' in manifest:
        context = {'database': manifest['database'], 'scenario_token': manifest['scenario_token']}
    else:
        records = manifest.get('records', {}).values()
        record = next((r for r in records if r['prediction_file'] == prediction.name), None)
        if record is None:
            raise ValueError('Prediction is not registered in run.json')
        if hashlib.sha256(prediction.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Prediction checksum does not match run.json')
        plan = json.loads((run_dir/'frozen_plan.json').read_text())
        scenario = next(s for s in plan['scenarios'] if s['token'] == record['token'])
        context = {'database': scenario['database'], 'scenario_token': scenario['token']}
    context['run_dir'] = str(run_dir)
    context['prediction'] = str(prediction)
    return context


def _sample(array, batch_index):
    if hasattr(array, 'detach'):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    return array[batch_index] if array.ndim == 4 else array


def export_arrays(payload, context, task='sampled_tensor', current_index=4, dt=.5, batch_index=0):
    if task not in payload:
        raise ValueError(f'Prediction has no {task!r}; available keys: {sorted(payload)}')
    gt = decode(_sample(payload['scene_tensor_gt'], batch_index))
    pred = decode(_sample(payload[task], batch_index))
    validity = payload.get('validity', payload.get('valid_mask'))
    if validity is None:
        raise ValueError('Missing validity/valid_mask')
    if hasattr(validity, 'detach'):
        validity = validity.detach().cpu().numpy()
    validity = np.asarray(validity)
    if validity.ndim == 3:
        validity = validity[batch_index]
    validity = validity.astype(bool)
    if gt.shape != pred.shape or validity.shape != gt.shape[:2]:
        raise ValueError('Predicted, reference and validity shapes disagree')
    if not 0 <= current_index < gt.shape[1] or dt <= 0:
        raise ValueError('Invalid time sampling')
    if not validity[0].all():
        raise ValueError('Ego reference must be valid for the entire clip')
    active = np.flatnonzero(validity.any(axis=1))
    gt, pred, validity = gt[active], pred[active], validity[active]
    if active[0] != 0 or not np.isfinite(gt[validity]).all() or not np.isfinite(pred[validity]).all():
        raise ValueError('Missing ego or non-finite valid states')
    if np.any(pred[validity, 6:8] <= 0):
        raise ValueError('Valid predicted vehicles have nonpositive length/width')
    if np.any(np.linalg.norm(pred[validity, 2:4], axis=-1) < 1e-6):
        raise ValueError('Valid predicted states have undefined heading')

    with nuplan.connect(context['database']) as db:
        anchor = nuplan.anchor_frame(db, context['scenario_token'])
        ideal = anchor['lidar_timestamp'] + np.rint((np.arange(gt.shape[1])-current_index)*dt*1e6).astype(np.int64)
        frames = nuplan.nearest_frames(db, ideal)
        annotations = [nuplan.boxes(db, f['lidar_token']) for f in frames]
        # Derive the frame from the reference ego state, so nonzero raster
        # offsets are supported. Database ego x/y are rear-axle coordinates.
        local_ego = gt[0, current_index]
        local_pose = pose(local_ego[0], local_ego[1], yaw=np.arctan2(local_ego[3], local_ego[2]))
        world_from_local = pose(anchor['x'], anchor['y'], yaw=nuplan.ego_yaw(anchor)) @ np.linalg.inv(local_pose)
        world_yaw = np.arctan2(world_from_local[1, 0], world_from_local[0, 0])

        def world_states(states):
            points = states[..., :2] @ world_from_local[:2, :2].T + world_from_local[:2, 3]
            heading = np.arctan2(states[..., 3], states[..., 2])+world_yaw
            return points, heading

        gt_xy, gt_heading = world_states(gt)
        pred_xy, pred_heading = world_states(pred)
        tokens, bindings = ['__ego__'], []
        assigned = set()
        for agent in range(1, len(active)):
            # Recover legacy identities from exact reference annotations, never
            # from generated positions or an assumed array/index ordering.
            choices = np.flatnonzero(validity[agent])
            t = min(choices, key=lambda k: abs(int(k)-current_index))
            matches = []
            for obj in annotations[t]:
                if obj['category_name'] not in ('vehicle', 'pedestrian', 'bicycle'):
                    continue
                xy_error = float(np.linalg.norm(gt_xy[agent, t]-[obj['x'], obj['y']]))
                yaw_error = abs(float(np.arctan2(np.sin(gt_heading[agent, t]-obj['yaw']), np.cos(gt_heading[agent, t]-obj['yaw']))))
                size_error = float(np.max(np.abs(gt[agent, t, 6:8]-[obj['length'], obj['width']])))
                if xy_error < .05 and yaw_error < .02 and size_error < .05:
                    matches.append((obj, xy_error))
            if len(matches) != 1 or matches[0][0]['track_token'] in assigned:
                raise ValueError(f'Cannot uniquely recover track token for Nexus slot {active[agent]} (matches={len(matches)})')
            obj, error = matches[0]
            tokens.append(obj['track_token'])
            assigned.add(tokens[-1])
            bindings.append({'nexus_slot': int(active[agent]), 'track_token': tokens[-1],
                'category': obj['category_name'], 'reference_frame': int(t), 'match_error_m': error})

        T, A = gt.shape[1], len(active)
        predicted = np.broadcast_to(np.eye(4), (T, A, 4, 4)).copy()
        reference = predicted.copy()
        max_reference_error = 0.
        for t, (frame, objects) in enumerate(zip(frames, annotations)):
            logged_ego = nuplan.ego_pose(frame)
            for agent, token in enumerate(tokens):
                if not validity[agent, t]:
                    continue
                if agent == 0:
                    z = frame['z']
                    residual = logged_ego[:3, :3] @ yaw_matrix(-nuplan.ego_yaw(frame))
                    ref_xy = [frame['x'], frame['y']]
                else:
                    obj = next((o for o in objects if o['track_token'] == token), None)
                    if obj is None:
                        raise ValueError(f'Reference track {token} missing at valid frame {t}')
                    z, residual, ref_xy = obj['z'], np.eye(3), [obj['x'], obj['y']]
                error = float(np.linalg.norm(gt_xy[agent, t]-ref_xy))
                max_reference_error = max(max_reference_error, error)
                if error > .1:
                    raise ValueError(f'Reference coordinate/time mismatch: agent={token} frame={t} error={error:.3f}m')
                for target, xy, heading in ((predicted, pred_xy, pred_heading), (reference, gt_xy, gt_heading)):
                    target[t, agent, :3, :3] = residual @ yaw_matrix(heading[agent, t])
                    target[t, agent, :3, 3] = [*xy[agent, t], z]

    metadata = {**context, 'schema_version': SCHEMA_VERSION, 'dataset': 'nuplan',
        'task': task, 'current_index': current_index, 'sample_interval_s': dt,
        'coordinate_frame': 'dataset_world', 'pose_convention': 'object_to_world_column_vectors',
        'ego_reference': 'rear_axle', 'units': {'length': 'metres', 'time': 'microseconds'},
        'agent_bindings': bindings, 'world_from_nexus': world_from_local.tolist(),
        'max_reference_position_error_m': max_reference_error,
        'height_policy': 'retain_logged_z_and_ego_roll_pitch; no terrain extrapolation',
        'status': 'trajectories_exported', 'photorealistic_render_verified': False}
    arrays = {'timestamps_us': ideal, 'source_lidar_timestamps_us': np.array([f['lidar_timestamp'] for f in frames], dtype=np.int64),
        'track_tokens': np.array(tokens), 'nexus_slots': active,
        'predicted_poses_world': predicted, 'reference_poses_world': reference,
        'valid': validity.T, 'predicted_size_lw': pred[..., 6:8].transpose(1, 0, 2)}
    return arrays, metadata


def save_bundle(output_dir, arrays, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir/'trajectories.npz'
    np.savez_compressed(path, **arrays)
    metadata = {**metadata, 'bundle_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    nuplan.write_json(output_dir/'manifest.json', metadata)
    return metadata


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['schema_version'] != SCHEMA_VERSION or manifest['dataset'] != 'nuplan':
        raise ValueError('Unsupported trajectory bundle')
    path = directory/'trajectories.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['bundle_sha256']:
        raise ValueError('Trajectory bundle checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, manifest
