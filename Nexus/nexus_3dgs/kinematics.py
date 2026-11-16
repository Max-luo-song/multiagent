"""Conservative, explicitly recorded repairs of sampled vehicle pose outliers."""
import numpy as np
from .geometry import yaw_matrix


def angle_difference(a, b):
    return np.arctan2(np.sin(a-b), np.cos(a-b))


def repair_isolated_half_turns(poses, valid, fixed, max_step_m=.25):
    """Resolve an isolated ~pi heading branch error on an almost still car.

    Both neighbors must agree, each translation step must be <= max_step_m,
    and all three samples must be valid. Given conditions are immutable.
    This is not general trajectory smoothing or a learned-model correction.
    """
    original=np.asarray(poses);valid=np.asarray(valid,dtype=bool);fixed=np.asarray(fixed,dtype=bool)
    if original.shape!=(*valid.shape,4,4) or fixed.shape!=valid.shape:
        raise ValueError('Pose, validity and condition masks disagree')
    result=original.copy();changes=[]
    yaw=np.arctan2(original[...,1,0],original[...,0,0])
    for t in range(1,len(original)-1):
        for a in np.flatnonzero(valid[t-1]&valid[t]&valid[t+1]&~fixed[t]):
            before,center,after=yaw[t-1,a],yaw[t,a],yaw[t+1,a]
            steps=np.linalg.norm(np.diff(original[t-1:t+2,a,:2,3],axis=0),axis=-1)
            if (steps.max()>max_step_m or abs(angle_difference(before,after))>np.deg2rad(20)
                or abs(angle_difference(center,before))<np.deg2rad(150)
                or abs(angle_difference(center,after))<np.deg2rad(150)):
                continue
            corrected=angle_difference(center+np.pi,0.)
            if max(abs(angle_difference(corrected,before)),abs(angle_difference(corrected,after)))>np.deg2rad(20):
                continue
            result[t,a,:3,:3]=yaw_matrix(np.pi)@original[t,a,:3,:3]
            changes.append({'sample_index':t,'agent_index':int(a),'yaw_before_deg':float(np.rad2deg(center)),
                            'yaw_after_deg':float(np.rad2deg(corrected)),'max_neighbor_translation_m':float(steps.max())})
    return result,changes


def repair_low_speed_tail_slips(poses, valid, fixed, timestamps, current_index,
                                wheelbase_m=2.7, max_steer_deg=35.):
    """Project impossible low-speed turns/lateral jumps without using future GT.

    Only an outlier and the remaining contiguous free tail are changed. Validity,
    past/given states and ordinary motion remain untouched. Body yaw is bounded
    by travelled longitudinal distance plus 3 degrees of annotation tolerance.
    """
    original=np.asarray(poses);valid=np.asarray(valid,bool);fixed=np.asarray(fixed,bool)
    ts=np.asarray(timestamps,dtype=np.int64);result=original.copy();changes=[]
    if original.shape!=(*valid.shape,4,4) or fixed.shape!=valid.shape or len(ts)!=len(original):
        raise ValueError('Pose/time/condition dimensions disagree')
    yaw=np.arctan2(original[...,1,0],original[...,0,0])
    for a in range(valid.shape[1]):
        repairing=False
        for t in range(current_index+1,len(ts)):
            if not(valid[t-1,a] and valid[t,a]) or fixed[t,a]:
                repairing=False;continue
            dt=(ts[t]-ts[t-1])/1e6
            if dt<=0:raise ValueError('Non-increasing time')
            previous=np.arctan2(result[t-1,a,1,0],result[t-1,a,0,0])
            raw_step=original[t,a,:2,3]-original[t-1,a,:2,3]
            forward=np.array([np.cos(previous),np.sin(previous)]);side=np.array([-forward[1],forward[0]])
            along=float(raw_step@forward);lateral=float(raw_step@side)
            speed=np.linalg.norm(raw_step)/dt;jump=abs(angle_difference(yaw[t,a],previous))
            allowed=abs(along)/wheelbase_m*np.tan(np.deg2rad(max_steer_deg))+np.deg2rad(3.)
            outlier=(speed<2.5 and (jump>max(np.deg2rad(30.),2*allowed) or
                      (abs(lateral)>.25 and jump>np.deg2rad(10.))))
            if not (outlier or repairing):continue
            repairing=True
            # Keep small normal annotation/position noise; reject a sideways
            # teleport while the vehicle is moving very little longitudinally.
            lateral_limit=.03+.1*abs(along)
            limited_lateral=np.clip(lateral,-lateral_limit,lateral_limit)
            limited_jump=np.clip(angle_difference(yaw[t,a],previous),-allowed,allowed)
            target_yaw=previous+limited_jump
            result[t,a,:2,3]=result[t-1,a,:2,3]+along*forward+limited_lateral*side
            result[t,a,:3,:3]=yaw_matrix(angle_difference(target_yaw,yaw[t,a]))@original[t,a,:3,:3]
            changes.append({'sample_index':t,'agent_index':a,'triggered_by_outlier':bool(outlier),
                'yaw_before_deg':float(np.rad2deg(yaw[t,a])), 'yaw_after_deg':float(np.rad2deg(target_yaw)),
                'raw_lateral_step_m':lateral,'accepted_lateral_step_m':float(limited_lateral),
                'position_change_m':float(np.linalg.norm(result[t,a,:2,3]-original[t,a,:2,3]))})
    return result,changes
