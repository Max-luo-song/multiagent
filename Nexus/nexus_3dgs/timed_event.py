"""Compile explicit actor/time behavior requests into local Nexus conditions.

Only history is read when constructing a condition. Future existence masks
remain dataset supplied, and are reported separately by the runner.
"""
import numpy as np

from .bundle import FEATURE_MEANS, FEATURE_STD, decode


def compile_event(original, validity, slot, request, variant='event', current=4, dt=.5):
    if variant not in ('normal', 'event'):
        raise ValueError('Unknown variant')
    if slot <= 0 or slot >= original.shape[0]:
        raise ValueError('Event actor must be a non-ego Nexus slot')
    start, end = float(request['start_s']), float(request['end_s'])
    horizon = (original.shape[1] - current - 1) * dt
    if not np.isfinite([start, end]).all() or not 0 < start < end <= horizon:
        raise ValueError('Expected 0 < start_s < end_s <= horizon')
    if any(not np.isclose(t/dt, round(t/dt), atol=1e-8) for t in (start, end)):
        raise ValueError('Event times must lie on the native 0.5-second grid')
    indices = np.arange(current + round(start/dt), current + round(end/dt) + 1)
    if not validity[slot, indices].all() or not validity[slot, current-1:current+1].all():
        raise ValueError('Event actor must be present in history and throughout the event')
    # Deliberately decode only past/current state values.
    history = decode(original[slot, :current+1])
    anchor = history[-1].copy()
    velocity = (history[-1, :2] - history[-2, :2])/dt
    if np.linalg.norm(velocity) < .1:
        velocity[:] = 0  # explicit stationary-history noise tolerance
    heading = anchor[2:4]/np.linalg.norm(anchor[2:4])
    normal = np.array([-heading[1], heading[0]])
    times = (np.arange(original.shape[1])-current)*dt
    xy = anchor[:2] + times[:, None]*velocity
    behavior = request['behavior']
    parameters = request.get('parameters', {})
    duration = end-start
    u = np.clip((times-start)/duration, 0, 1)
    smooth = 10*u**3 - 15*u**4 + 6*u**5
    if behavior in ('reverse_approach', 'lateral_shift'):
        amount = float(parameters['distance_m'])
        if not np.isfinite(amount) or not 0 < amount <= 10:
            raise ValueError('distance_m must be finite and in (0,10]')
        if behavior == 'reverse_approach':
            if np.linalg.norm(velocity) > .5:
                raise ValueError('Reverse approach currently requires an initially stopped actor')
            direction = -heading
        else:
            side = parameters['side']
            if side not in ('left', 'right') or np.linalg.norm(velocity) < 1:
                raise ValueError('Lateral shift requires side left/right and moving history >=1m/s')
            direction = normal*(1 if side == 'left' else -1)
        if variant == 'event':
            xy += amount*smooth[:, None]*direction
    elif behavior == 'brake':
        speed = float(np.linalg.norm(velocity))
        decel = float(parameters['deceleration_mps2'])
        if not np.isfinite(decel) or not 0 < decel <= 10 or speed < 1:
            raise ValueError('Brake requires moving history >=1m/s and deceleration in (0,10]')
        if variant == 'event':
            tau = np.clip(times-start, 0, speed/decel)
            distance = speed*tau-.5*decel*tau**2
            xy = anchor[:2] + np.minimum(times,start)[:,None]*velocity + distance[:,None]*(velocity/speed)
    else:
        raise ValueError('Supported behaviors: reverse_approach, lateral_shift, brake')
    states = np.broadcast_to(anchor, (len(times), 8)).copy()
    states[:, :2] = xy
    # Match the native finite-difference convention for provided velocities.
    states[1:, 4:6] = np.diff(xy, axis=0)/dt
    if behavior == 'lateral_shift' and variant == 'event':
        angle = np.arctan2(states[:,5], states[:,4])
        states[:,2:4] = np.stack([np.cos(angle),np.sin(angle)],axis=-1)
    tensor = np.zeros_like(original)
    tensor[:, :current+1] = original[:, :current+1]
    keep = np.zeros_like(original, dtype=bool)
    keep[:, :current+1] = True
    tensor[slot, indices] = ((states[indices]-FEATURE_MEANS)/(2*FEATURE_STD)).astype(original.dtype)
    keep[slot, indices] = True
    audit = {'actor_slot':slot,'variant':variant,'behavior':behavior,
             'event_indices':indices.tolist(),'event_times_s':times[indices].tolist(),
             'history_derived_velocity_mps':velocity.tolist(),
             'condition_states_physical':states[indices].tolist(),
             'condition_displacement_m':(xy[indices[-1]]-xy[indices[0]]).tolist(),
             'ego_future_conditioned':bool(keep[0,current+1:].any()),
             'unconditioned_values_zero':bool((tensor[~keep]==0).all()),
             'condition_source':'history plus explicit event parameters; no logged future state values',
             'actor_outside_event_future_free':True,'time_interval_inclusive':True}
    return tensor, keep, audit
