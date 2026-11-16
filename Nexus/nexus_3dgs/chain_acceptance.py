"""Chain physical audit with explicit A/B/C slots and labeled B-fixed diagnostics.

All physical thresholds and checks are unchanged.
"""
import json
from pathlib import Path


def evaluate(payload, *, reference_root, lane_root, variant, actor_slot, third_slot, reactor_slot, fixed_reactor=False):
    import numpy as np
    from shapely import wkb
    from shapely.geometry import Polygon
    from .bundle import decode
    from .scene_projection import pair_separations
    reference_root = Path(reference_root)
    lane_root = Path(lane_root)
    protocol = json.loads((reference_root/'protocol.json').read_text())
    W = np.array(protocol['world_from_nexus'])
    pred = payload['sampled_tensor'][0].numpy()
    cond = payload['conditioned_scene_tensor'][0].numpy()
    keep = payload['task_mask'][0].numpy().astype(bool)
    allvalid = payload['valid_mask'][0].numpy().astype(bool)
    active = np.flatnonzero(allvalid.any(-1))
    states = decode(pred)[active]
    original = decode(payload['scene_tensor_gt'][0].numpy())[active]
    valid = allvalid[active]
    sizes = original[:, 4, 6:8]
    assert (pred[keep] == cond[keep]).all()
    assert (cond[~keep] == 0).all()
    assert not keep[third_slot, 5:].any()
    assert keep[reactor_slot, 5:].all() if fixed_reactor else not keep[reactor_slot, 5:].any()
    assert keep[actor_slot].all()
    times = np.arange(1, 81)*.1
    grid = times/.5+4
    lo = np.minimum(np.floor(grid).astype(int), 19)
    f = grid-lo
    positions = (states[:, lo, :2]*(1-f[None, :, None])
                 +states[:, lo+1, :2]*f[None, :, None])
    angles = np.arctan2(states[..., 3], states[..., 2])
    delta = angles[:, lo+1]-angles[:, lo]
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    yaw = angles[:, lo]+delta*f
    seen = valid[:, lo]&valid[:, lo+1]
    seen = np.where(f[None]==0, valid[:, lo], seen)
    seen = np.where(f[None]==1, valid[:, lo+1], seen)
    boxes = np.concatenate([positions.transpose(1,0,2), yaw.T[..., None]], -1)
    boxes[:, 0, :2] += 1.461*np.c_[np.cos(boxes[:, 0, 2]), np.sin(boxes[:, 0, 2])]
    with np.load(reference_root/'inputs/background_replay.npz') as data:
        bg = {k:data[k] for k in data.files}
    # The frozen replay already has exact 0.1s samples including time zero.
    np.testing.assert_allclose(bg['times_s'][1:], times)
    bg_yaw = np.arctan2(bg['heading'][1:,:,1], bg['heading'][1:,:,0])
    bg_boxes = np.concatenate([bg['centers'][1:], bg_yaw[...,None]], -1)
    combined = np.concatenate([boxes,bg_boxes],1)
    visibility = np.concatenate([seen.T,bg['valid'][1:].astype(bool)],1)
    n = len(active)
    separation, validpairs = pair_separations(combined, visibility, np.concatenate([sizes,bg['sizes']]))
    pairmask = np.ones(separation.shape[-2:],bool);pairmask[n:,n:] = False
    overlap = (separation<0)&validpairs&pairmask[None]
    roles = {'B':int(np.flatnonzero(active==reactor_slot)[0]),'A':int(np.flatnonzero(active==actor_slot)[0]),'C':int(np.flatnonzero(active==third_slot)[0])}
    pair_stats = {}
    for name,i,j in [('BA',roles['B'],roles['A']),('BC',roles['B'],roles['C']),('AC',roles['A'],roles['C'])]:
        visible = seen[i]&seen[j]
        hits = overlap[:,i,j]&visible
        pair_stats[name] = {'overlap_samples':int(hits.sum()),
                            'minimum_signed_gap_m':float(separation[visible,i,j].min()),
                            'overlap_times_s':times[hits].tolist()}
    road = wkb.loads((reference_root/'inputs/drivable.wkb').read_bytes())
    road_tolerant = road.buffer(.02)

    def body(box,size):
        x,y,angle = box;c,s = np.cos(angle),np.sin(angle)
        rot = np.array([[c,s],[-s,c]])
        corners = np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*size/2@rot+[x,y]
        return Polygon(corners@W[:2,:2].T+W[:2,3])

    eligible = []
    for i in range(n):
        center = states[i,4,:2].copy()
        if i==0:center += 1.461*np.array([np.cos(angles[i,4]),np.sin(angles[i,4])])
        eligible.append(road_tolerant.covers(body([*center,angles[i,4]],sizes[i])))
    eligible_offroad = 0
    role_offroad = {r:0 for r in roles}
    lane_stats = {}
    for r,i in roles.items():
        lane_path = lane_root/f'{int(active[i])}.wkb'
        if lane_path.exists():
            lane = wkb.loads(lane_path.read_bytes()).buffer(.02)
            hit = np.array([not lane.covers(body(boxes[t,i],sizes[i])) and seen[i,t] for t in range(80)])
            lane_stats[r] = {'box_outside_original_lane_samples':int(hit.sum()),
                             'first_outside_s':float(times[hit][0]) if hit.any() else None}
    for t,i in np.argwhere(seen.T):
        off = not road_tolerant.covers(body(boxes[t,i],sizes[i]))
        eligible_offroad += bool(off and eligible[i])
        for r,j in roles.items():
            if j==i:role_offroad[r] += int(off)
    velocity = np.diff(states[...,:2],axis=1)/.5
    acceleration = np.diff(velocity,axis=1)/.5
    accnorm = np.linalg.norm(acceleration,axis=-1)
    acceleration_mask = valid[:,:-2]&valid[:,1:-1]&valid[:,2:]
    acceleration_mask[:,:3] = False
    future_mask = valid[:,4:-1]&valid[:,5:]
    h = states[:,5:,2:4]
    h = h/np.maximum(np.linalg.norm(h,axis=-1,keepdims=True),1e-8)
    vv = velocity[:,4:]
    along = (vv*h).sum(-1)
    lateral = vv[...,0]*h[...,1]-vv[...,1]*h[...,0]
    heading_error = np.degrees(np.arctan2(abs(lateral),along))
    moving = future_mask&(np.linalg.norm(vv,axis=-1)>.5)
    motion = {}
    for role,index in roles.items():
        if role=='A':continue  # Immutable input is prechecked independently.
        a = float(accnorm[index,acceleration_mask[index]].max())
        slip = float(heading_error[index,moving[index]].max()) if moving[index].any() else 0.
        reverse = float((np.maximum(-along[index],0)*future_mask[index]).sum()*.5)
        head_speed = np.linalg.norm(states[index,5:,4:6],axis=-1)
        xy_speed = np.linalg.norm(vv[index],axis=-1)
        motion[role] = {'max_xy_acceleration_mps2':a,'max_heading_error_deg':slip,
                        'reverse_distance_m':reverse,
                        'max_xy_vs_speed_channel_difference_mps':float(abs(head_speed-xy_speed)[future_mask[index]].max()),
                        'minimum_first2p5s_xy_speed_mps':float(xy_speed[:5].min()),
                        'xy_interval_speeds_mps':xy_speed.tolist(),
                        'accepted':a<=5 and slip<=10 and reverse<=.2
                        and float(abs(head_speed-xy_speed)[future_mask[index]].max())<=1.6}
    role_overlap = {r:int((overlap[:,i,:].any(-1)|overlap[:,:,i].any(-1)).sum()) for r,i in roles.items()}
    normal_lane_ok = variant!='normal' or all(lane_stats.get(r,{}).get('box_outside_original_lane_samples',80)==0 for r in ('B','C'))
    primary_ok = (not any(role_overlap.values()) and not any(role_offroad.values())
                  and all(x['accepted'] for x in motion.values()) and normal_lane_ok)
    free_actor = ~keep[active,5:].all((-1,-2))
    acceleration_violations = int(((accnorm>5)&acceleration_mask&free_actor[:,None]).sum())
    heading_violations = int(((heading_error>10)&moving&free_actor[:,None]).sum())
    full_ok = (primary_ok and not overlap.any() and eligible_offroad==0
               and acceleration_violations==0 and heading_violations==0)
    return {'scope':'80 future 0.1s checks; original sizes; rear-axle offset; lane identity and XY-derived motion; not continuous-time proof',
            'variant':variant,'conditions_exact':True,'B_C_future_free':not fixed_reactor,'C_future_free':True,'fixed_B_diagnostic':fixed_reactor,
            'finite_valid_states':bool(np.isfinite(states[valid]).all()),
            'roles':pair_stats,'role_any_overlap_samples':role_overlap,
            'all_predicted_involved_overlap_samples':int(overlap.sum()),
            'predicted_vs_background_overlap_samples':int(overlap[:,:n,n:].sum()),
            'role_box_offroad_samples':role_offroad,'eligible_box_offroad_samples':int(eligible_offroad),
            'original_lane':lane_stats,'normal_lane_keeping_accepted':bool(normal_lane_ok),
            'motion':motion,'all_free_acceleration_over5_samples':acceleration_violations,
            'all_free_heading_over10deg_samples':heading_violations,
            'primary_three_accepted':bool(primary_ok),'full_scene_accepted':bool(full_ok),
            'approved_for_downstream':bool(full_ok)}
