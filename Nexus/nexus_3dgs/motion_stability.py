"""Bounded 3D motion refinement; never filters rendered pixels."""
import numpy as np
from scipy.optimize import lsq_linear


def bounded_curve_correction(values, strength=20., maximum=.05, fixed=None,
                             step_targets=None, step_weight=0.):
    """Fit bounded additive corrections, keeping locked values exact.

    A second-difference penalty suppresses isolated fitting noise. An optional
    first-difference target can enforce lateral motion consistent with heading.
    """
    y=np.asarray(values,dtype=float)
    if y.ndim!=1 or not np.isfinite(y).all() or maximum<=0 or strength<0:raise ValueError('Invalid curve or refinement bounds')
    n=len(y);fixed=np.zeros(n,bool) if fixed is None else np.asarray(fixed,bool)
    if fixed.shape!=(n,):raise ValueError('Lock shape mismatch')
    free=np.flatnonzero(~fixed)
    if n<3 or not len(free):return y.copy()
    D=np.diff(np.eye(n),n=2,axis=0)
    matrices=[np.eye(n),np.sqrt(strength)*D];targets=[np.zeros(n),-np.sqrt(strength)*(D@y)]
    if step_targets is not None:
        target=np.asarray(step_targets,float)
        if target.shape!=(n-1,) or step_weight<0:raise ValueError('Invalid step targets')
        L=np.diff(np.eye(n),axis=0);matrices.append(np.sqrt(step_weight)*L);targets.append(np.sqrt(step_weight)*(target-L@y))
    A=np.concatenate(matrices)[:,free];b=np.concatenate(targets)
    fit=lsq_linear(A,b,bounds=(-maximum,maximum),tol=1e-12,max_iter=200)
    if not fit.success:raise RuntimeError('Bounded curve refinement failed: '+fit.message)
    result=y.copy();result[free]+=fit.x
    np.testing.assert_array_equal(result[fixed],y[fixed])
    if np.max(np.abs(result-y))>maximum+1e-9:raise AssertionError('Correction exceeds declared bound')
    return result


def valid_segments(valid):
    ids=np.flatnonzero(np.asarray(valid,bool))
    return np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []


def regularize_observed_bodies(poses,valid,max_lateral_m=.05,max_rotation_deg=.75):
    """Refine each continuous observed track in SE(3), without freezing motion.

    Longitudinal motion, height, visibility and all invalid poses are unchanged.
    The two segment endpoints are locked. This handles scene fitting jitter,
    not Nexus prediction outliers.
    """
    from scipy.spatial.transform import Rotation
    original=np.asarray(poses,float);valid=np.asarray(valid,bool)
    if original.shape!=(*valid.shape,4,4):raise ValueError('Pose/visibility shape mismatch')
    result=original.copy();records=[]
    for m in range(valid.shape[1]):
        for ids in valid_segments(valid[:,m]):
            if len(ids)<7:continue
            P=original[ids,m];yaw=np.unwrap(np.arctan2(P[:,1,0],P[:,0,0]));reference=float(np.median(yaw));forward=np.array([np.cos(reference),np.sin(reference)]);side=np.array([-forward[1],forward[0]])
            local=P[:,:2,3]-P[0,:2,3];lateral=local@side;locked=np.zeros(len(ids),bool);locked[[0,-1]]=True
            fitted=bounded_curve_correction(lateral,10.,max_lateral_m,locked)
            result[ids,m,:2,3]+=((fitted-lateral)[:,None]*side)
            R0=P[0,:3,:3];v=Rotation.from_matrix(P[:,:3,:3]@R0.T).as_rotvec();fitted_rotation=np.stack([bounded_curve_correction(v[:,k],20.,np.deg2rad(max_rotation_deg),locked) for k in range(3)],-1)
            delta=fitted_rotation-v;angle=np.linalg.norm(delta,axis=-1);delta*=np.minimum(1,np.deg2rad(max_rotation_deg)/np.maximum(angle,1e-12))[:,None]
            result[ids,m,:3,:3]=Rotation.from_rotvec(v+delta).as_matrix()@R0
            result[ids[[0,-1]],m]=P[[0,-1]]
            np.testing.assert_allclose((result[ids,m,:2,3]-P[:,:2,3])@forward,0,atol=1e-9)
            records.append({'model_index':m,'first_frame':int(ids[0]),'last_frame':int(ids[-1]),'maximum_lateral_change_m':float(np.max(abs(fitted-lateral))),'maximum_rotation_change_deg':float(np.rad2deg(np.max(np.linalg.norm(delta,axis=-1)))),'lateral_second_difference_rms_before':float(np.sqrt(np.mean(np.diff(lateral,2)**2))),'lateral_second_difference_rms_after':float(np.sqrt(np.mean(np.diff(fitted,2)**2))),'rotation_second_difference_rms_deg_before':float(np.rad2deg(np.sqrt(np.mean(np.diff(v,n=2,axis=0)**2)))),'rotation_second_difference_rms_deg_after':float(np.rad2deg(np.sqrt(np.mean(np.diff(v+delta,n=2,axis=0)**2))))})
    np.testing.assert_array_equal(result[~valid],original[~valid]);np.testing.assert_array_equal(result[:,:,2,3],original[:,:,2,3])
    return result,records
