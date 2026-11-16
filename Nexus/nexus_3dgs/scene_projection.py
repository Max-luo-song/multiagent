"""Offline longitudinal projection of free actors against all scene vehicles."""
import numpy as np
from scipy.optimize import minimize
from .clearance import box_separation


def pair_separations(boxes,valid,sizes):
    sep=box_separation(boxes[:,:,None,:],sizes[:,None,:],boxes[:,None,:,:],sizes[None,:,:])
    pairs=valid[:,:,None]&valid[:,None,:]&np.triu(np.ones((len(sizes),len(sizes)),bool),1)
    return sep,pairs


def project_forward_offsets(times,knots,boxes,valid,sizes,movable,directions,fixed_knots,
                            previous_velocity=None,clearance=.15,max_offset=8.):
    """Keep history/conditions and all logged actors fixed; advance free actors.

    The original sampled path is translated along each actor's current heading
    with smooth, nonnegative offsets. No actor is removed or made invisible.
    """
    times,knots=np.asarray(times),np.asarray(knots)
    movable=np.asarray(movable,dtype=int);directions=np.asarray(directions)
    N,K=len(movable),len(knots)
    if not N:raise ValueError('No free actors available for scene projection')
    weight=np.stack([np.interp(times,knots,np.eye(K)[i]) for i in range(K)],axis=-1)
    knot_indices=np.array([np.argmin(np.abs(times-t)) for t in knots]);dt=np.diff(knots)
    initial_sep,pairs=pair_separations(boxes,valid,sizes)
    touched=np.zeros(len(sizes),bool);touched[movable]=True
    changeable=touched[:,None]|touched[None,:]
    if ((initial_sep<=0)&pairs&~changeable).any():raise ValueError('An overlapping pair has no free actor')
    mask=pairs&changeable
    required=np.full((len(sizes),len(sizes)),clearance)
    initial_pairs=pairs[0]
    if ((initial_sep[0]<=0)&initial_pairs).any():raise ValueError('Initial scene already overlaps; immutable history cannot be repaired')
    required[initial_pairs]=np.minimum(clearance,initial_sep[0][initial_pairs]*.5)
    previous_velocity=np.zeros((N,2)) if previous_velocity is None else previous_velocity
    def unpack(x):return np.c_[np.zeros(N),x.reshape(N,K-1)]
    def shift(x):
        result=boxes.copy();offsets=weight@unpack(x).T
        result[:,movable,:2]+=offsets[:,:,None]*directions[None,:,:]
        return result
    def constraints(x):
        shifted=shift(x)
        sep,_=pair_separations(shifted,valid,sizes)
        xy=shifted[knot_indices][:,movable,:2]
        velocity=np.diff(xy,axis=0)/dt[:,None,None]
        acceleration=np.diff(np.concatenate((previous_velocity[None],velocity)),axis=0)/dt[:,None,None]
        return np.r_[(sep-required)[mask],(8.-np.linalg.norm(velocity,axis=-1)).ravel(),
                     (4.-np.linalg.norm(acceleration,axis=-1)).ravel()]
    def objective(x):
        s=unpack(x)
        return float(np.mean(s*s)+.3*np.mean(np.diff(s,n=2,axis=-1)**2))
    bounds=[(0,0) if fixed_knots[i,k] else (0,max_offset) for i in range(N) for k in range(1,K)]
    attempts=[]
    for end_offset in (4.,6.,0.):
        seed=np.tile(np.linspace(0,end_offset,K),(N,1))[:,1:]
        seed[np.asarray(fixed_knots)[:,1:]]=0
        result=minimize(objective,seed.ravel(),bounds=bounds,method='SLSQP',
                        constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':350,'ftol':1e-9})
        minimum=float(constraints(result.x).min());attempts.append({'success':bool(result.success),'message':result.message,'minimum_constraint':minimum})
        if result.success and minimum>=-1e-5:
            return unpack(result.x),{'attempts':attempts,'objective':float(result.fun),'clearance_m':clearance,
                                     'max_offset_m':float(unpack(result.x).max())}
    raise ValueError('No feasible full-scene projection: '+str(attempts))
