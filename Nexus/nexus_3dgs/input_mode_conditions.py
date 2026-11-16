"""Explicit conditions for matched historical-event and future-event comparisons."""
import numpy as np
from .bundle import decode,FEATURE_MEANS,FEATURE_STD


MODES=('bp_history','history_full_target','future_full_target','pure_sg_free_ego','official_sg_fixed_ego','pure_sg_bad_schedule')


def compile_mode(original,valid,actor,mode,variant,distance=3.,current=4,dt=.5,longitudinal_offset=2.):
    if mode not in MODES or variant not in ('normal','event'):raise ValueError('Unknown mode/variant')
    if actor<=0 or not valid[actor].all():raise ValueError('Target must be valid throughout the clip')
    history=decode(original[actor,:current+1])
    velocity=np.diff(history[:,:2],axis=0)/dt
    velocity=np.concatenate([velocity[:1],velocity],axis=0)
    forward=history[-1,2:4]/np.linalg.norm(history[-1,2:4]);right=np.array([forward[1],-forward[0]])
    speed=float(np.linalg.norm(velocity[-1]))
    if speed<1:raise ValueError('Cut-in comparison requires a moving target')
    times=(np.arange(original.shape[1])-current)*dt
    history_event=mode in ('bp_history','history_full_target')
    start,end=(-2.,0.) if history_event else (.5,2.5)
    u=np.clip((times-start)/(end-start),0,1)
    smooth=10*u**3-15*u**4+6*u**5
    derivative=(30*u**2-60*u**3+30*u**4)/(end-start)
    states=np.broadcast_to(history[-1],(len(times),8)).copy()
    states[:,:2]=history[-1,:2]+times[:,None]*velocity[-1]
    states[:current+1,:2]=history[:,:2]
    # Shared input edit, declared before sampling: avoid a historical cut-in
    # whose completed current state already intersects the ego physical box.
    states[:,:2]+=longitudinal_offset*forward
    states[:,4:6]=velocity[-1]
    states[:current+1,4:6]=velocity
    if variant=='event':
        states[:,:2]+=distance*smooth[:,None]*right
        states[:,4:6]+=distance*derivative[:,None]*right
    angle=np.arctan2(states[:,5],states[:,4])
    states[:,2:4]=np.stack([np.cos(angle),np.sin(angle)],axis=-1)
    encoded=((states-FEATURE_MEANS)/(2*FEATURE_STD)).astype(original.dtype)
    keep=np.zeros_like(original,dtype=bool)
    values=np.zeros_like(original)
    if mode in ('bp_history','history_full_target','future_full_target'):
        keep[:,:current+1]=True
        values[:,:current+1]=original[:,:current+1]
    keep[actor,:current+1]=True;values[actor,:current+1]=encoded[:current+1]
    if mode!='bp_history':
        keep[actor,current+1:]=True;values[actor,current+1:]=encoded[current+1:]
    if mode=='official_sg_fixed_ego':
        keep[0]=True;values[0]=original[0]
    condition={'mode':mode,'variant':variant,'target_slot':actor,'event_time_window_s':[start,end],
               'target_history_states_physical':states[:current+1].tolist(),
               'target_future_states_physical':states[current+1:].tolist() if mode!='bp_history' else None,
               'ego_future_free':bool(not keep[0,current+1:].any()),
               'all_actor_histories_fixed':bool(keep[:,:current+1].all()),
               'unknown_values_zero':bool((values[~keep]==0).all()),
               'target_velocity_source':'finite-difference history plus analytic derivative of the counterfactual lateral displacement',
               'history_event':history_event,'runtime_denoise_unknown_history':mode!='pure_sg_bad_schedule'}
    condition['shared_target_longitudinal_input_offset_m']=longitudinal_offset
    return values,keep,condition
