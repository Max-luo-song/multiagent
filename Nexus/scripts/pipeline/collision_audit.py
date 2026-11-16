"""CPU rectangle-overlap counters with explicit vehicle and pair denominators."""

import ast
from pathlib import Path

import numpy as np


FEATURE_SOURCE = Path('nuplan_extent/planning/training/preprocessing/features/scene_tensor.py')


def scene_scaling(path=FEATURE_SOURCE):
    """Read the upstream literal scales without importing TensorFlow/model code."""
    constants = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('FEATURE_MEANS','FEATURE_STD'):
                    constants[target.id] = ast.literal_eval(node.value)
    assert all(len(constants[k]) == 8 for k in ('FEATURE_MEANS','FEATURE_STD'))
    return (np.asarray(constants['FEATURE_MEANS'],dtype=np.float32),
            np.asarray(constants['FEATURE_STD'],dtype=np.float32))


def decode_scene(values, scaling):
    # Preserve the float32 operation order used by the saved-output evaluator.
    means, deviations = scaling
    values = np.asarray(values,dtype=np.float32)
    return values*np.float32(2)*deviations + means


def rectangle_overlaps(states, valid):
    """Symmetric SAT adjacency for [x,y,cos,sin,vx,vy,length,width] states.

    Positions are interpreted as box centers, matching the released collision
    implementation. Boundary contact counts as overlap. No 10 m culling is used.
    """
    states, valid = np.asarray(states), np.asarray(valid,dtype=bool)
    assert states.ndim == 2 and states.shape[1] == 8 and valid.shape == states.shape[:1]
    indices = np.flatnonzero(valid)
    result = np.zeros((len(valid),len(valid)),dtype=bool)
    if not len(indices):
        return result
    selected = states[indices]
    if not np.isfinite(selected[:,[0,1,2,3,6,7]]).all():
        raise ValueError('Non-finite valid bounding-box state')
    if np.any(selected[:,6:8] <= 0):
        raise ValueError('Non-positive valid bounding-box dimensions')
    if len(indices) < 2:
        return result
    # arctan2 follows the upstream interpretation even if cos/sin are not unit length.
    heading = np.arctan2(selected[:,3],selected[:,2])
    longitudinal = np.stack((np.cos(heading),np.sin(heading)),axis=-1).astype(np.float64)
    # Re-normalize for orthogonal projections; headings otherwise retain upstream precision.
    longitudinal /= np.linalg.norm(longitudinal,axis=-1,keepdims=True)
    lateral = np.stack((-longitudinal[:,1],longitudinal[:,0]),axis=-1)
    xy = selected[:,:2].astype(np.float64)
    delta = xy[None,:,:]-xy[:,None,:]
    half_l, half_w = selected[:,6].astype(np.float64)/2, selected[:,7].astype(np.float64)/2
    overlap = np.ones((len(indices),len(indices)),dtype=bool)
    for axis, self_radius in ((longitudinal,half_l),(lateral,half_w)):
        projection = np.abs(np.einsum('ijd,id->ij',delta,axis))
        other_radius = (np.abs(axis@longitudinal.T)*half_l[None,:]
                        +np.abs(axis@lateral.T)*half_w[None,:])
        overlap &= projection <= self_radius[:,None]+other_radius
    overlap &= overlap.T  # axes of both boxes must overlap
    np.fill_diagonal(overlap,False)
    result[np.ix_(indices,indices)] = overlap
    return result


def collision_counts(states, validity, goal_conditioned=False):
    """Return additive counts for all 16 future frames in one 21-frame scene.

    Both all-agent and non-ego-only geometry are exposed. Generated-frame counts
    additionally omit each supplied final valid state for the conditioned task;
    those supplied states still act as obstacles for other generated vehicles.
    """
    states = np.asarray(states)
    valid = np.asarray(validity,dtype=bool)
    assert states.shape[1:] == (21,8) and valid.shape == states.shape[:2]
    generated = valid.copy()
    generated[:,:5] = False
    if goal_conditioned:
        last = np.where(valid,np.arange(21)[None,:],-1).max(axis=-1)
        indices = np.flatnonzero(last >= 5)
        generated[indices,last[indices]] = False
    adjacency, adjacency10 = [], []
    for t in range(5,21):
        overlaps = rectangle_overlaps(states[:,t],valid[:,t])
        delta = states[:,t,None,:2]-states[None,:,t,:2]
        within10 = (delta*delta).sum(axis=-1) <= 100
        adjacency.append(overlaps)
        adjacency10.append(overlaps & within10)
    adjacency = np.stack(adjacency)
    adjacency10 = np.stack(adjacency10)
    result = {}
    for label, start in (('all_agents',0),('non_ego_only',1)):
        adj = adjacency[:,start:,start:]
        adj10 = adjacency10[:,start:,start:]
        frame_valid = valid[start:,5:].T
        frame_generated = generated[start:,5:].T
        collided = adj.any(axis=-1)
        pair_counts = np.triu(adj,k=1).sum(axis=(1,2))
        pair10_counts = np.triu(adj10,k=1).sum(axis=(1,2))
        result[label] = dict(
            valid_vehicle_frames=int(frame_valid.sum()),
            colliding_vehicle_frames=int(collided.sum()),
            valid_vehicles=int(frame_valid.any(axis=0).sum()),
            colliding_vehicles_any_future=int(collided.any(axis=0).sum()),
            generated_vehicle_frames=int(frame_generated.sum()),
            colliding_generated_vehicle_frames=int((collided & frame_generated).sum()),
            pair_events=int(pair_counts.sum()),
            pair_events_missed_by_10m=int((pair_counts-pair10_counts).sum()),
            valid_by_future_step=frame_valid.sum(axis=1).astype(int).tolist(),
            collided_by_future_step=collided.sum(axis=1).astype(int).tolist(),
            pair10_by_future_step=pair10_counts.astype(int).tolist())
    result['ego'] = dict(valid_future_frames=int(valid[0,5:].sum()),
                         colliding_future_frames=int(adjacency[:,0].any(axis=-1).sum()))
    return result


def add_counts(destination, value):
    for key, number in value.items():
        if isinstance(number,dict):
            add_counts(destination.setdefault(key,{}),number)
        elif isinstance(number,list):
            destination[key] = [int(a+b) for a,b in zip(destination.get(key,[0]*len(number)),number)]
        else:
            destination[key] = destination.get(key,0)+number


def finalize_counts(value):
    result = {scope:dict(counts) for scope,counts in value.items()}
    for scope in ('all_agents','non_ego_only'):
        counts = result[scope]
        for label, numerator, denominator in (
            ('vehicle_frame_collision_fraction','colliding_vehicle_frames','valid_vehicle_frames'),
            ('vehicle_any_future_collision_fraction','colliding_vehicles_any_future','valid_vehicles'),
            ('generated_vehicle_frame_collision_fraction','colliding_generated_vehicle_frames','generated_vehicle_frames')):
            counts[label] = counts[numerator]/counts[denominator] if counts[denominator] else None
        indices = list(range(1,16,2))
        ratios = [counts['pair10_by_future_step'][t]/counts['valid_by_future_step'][t]
                  for t in indices if counts['valid_by_future_step'][t]]
        counts['upstream_integer_horizon_pair_ratio'] = sum(ratios)/len(ratios) if ratios else None
        denominators = counts['valid_by_future_step']
        counts['collided_vehicle_fraction_by_future_step'] = [n/d if d else None
            for n,d in zip(counts['collided_by_future_step'],denominators)]
    counts = result['ego']
    counts['collision_frame_fraction'] = (counts['colliding_future_frames']/counts['valid_future_frames']
                                         if counts['valid_future_frames'] else None)
    return result
