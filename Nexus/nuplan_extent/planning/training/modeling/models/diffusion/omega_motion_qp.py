"""Late-denoising convex feasibility restoration of XY trajectories.

The solve takes place on the clean anchor before constructing the reverse
kernel. Heading stays fixed; free velocity updates can shrink toward the
network anchor to allocate the same KL budget. This module makes no claim
that the full eight-channel state satisfies a bicycle model.
"""
import numpy as np
import torch
from scipy.optimize import Bounds,LinearConstraint,minimize


def restore_motion(refiner,best,base,coefficient,variance):
    c=refiner.config
    state=(best*refiner.scale+refiner.mean).detach().cpu().numpy()[0]
    original=state.copy()
    valid=refiner.valid[0].cpu().numpy()
    keep=refiner.keep[0].cpu().numpy()
    sizes=refiner.sizes[0].cpu().numpy()
    A,T=valid.shape
    dt=.5
    D2=np.diff(np.eye(T),n=2,axis=0)/dt**2
    D3=np.diff(np.eye(T),n=3,axis=0)/dt**3
    D1=np.diff(np.eye(T),axis=0)/dt
    horizon=(T-5)*dt
    times=np.arange(1,round(horizon/.1)+1)*.1
    bg=refiner.background
    if bg is not None:
        bt=bg['times_s'].cpu().numpy()
        times=np.unique(np.r_[times,bt[(bt>0)&(bt<=horizon)]])
        bc=bg['centers'].cpu().numpy();bh=bg['heading'].cpu().numpy()
        bs=bg['sizes'].cpu().numpy();bv=bg['valid'].cpu().numpy().astype(bool)
    L=np.zeros((len(times),T))
    for k,t in enumerate(times):
        u=t/dt+4;lo=min(int(np.floor(u)),T-2);f=u-lo
        L[k,lo]=1-f;L[k,lo+1]=f
    angle=np.arctan2(state[...,3],state[...,2])
    heading=[];seen=[]
    for agent in range(A):
        values=[];mask=[]
        for k,t in enumerate(times):
            u=t/dt+4;lo=min(int(np.floor(u)),T-2);f=u-lo
            difference=angle[agent,lo+1]-angle[agent,lo]
            values.append(angle[agent,lo]+f*np.arctan2(np.sin(difference),np.cos(difference)))
            mask.append(valid[agent,lo] and valid[agent,lo+1])
        heading.append(np.c_[np.cos(values),np.sin(values)]);seen.append(mask)
    heading=np.asarray(heading);seen=np.asarray(seen)
    side=np.stack([-heading[...,1],heading[...,0]],-1)
    offset=np.zeros_like(heading);offset[0]=1.461*heading[0]
    logs=[]
    for agent in range(A):
        if not valid[agent,5:].any():continue
        start=state[agent,:,:2].reshape(-1).copy()
        lower=np.full(2*T,-np.inf);upper=np.full(2*T,np.inf)
        frozen=keep[agent,:,:2]|~valid[agent,:,None]
        lower[frozen.reshape(-1)]=start[frozen.reshape(-1)]
        upper[frozen.reshape(-1)]=start[frozen.reshape(-1)]
        matrix=[];lo_bounds=[];hi_bounds=[]
        av=valid[agent,:-2]&valid[agent,1:-1]&valid[agent,2:]
        av[:3]=False
        jv=valid[agent,:-3]&valid[agent,1:-2]&valid[agent,2:-1]&valid[agent,3:]
        jv[:2]=False
        for derivative,mask,limit in ((D2,av,c.qp_acceleration_mps2),(D3,jv,c.qp_jerk_mps3)):
            block=np.kron(derivative[mask],np.eye(2))
            matrix.extend(block);lo_bounds.extend([-limit/np.sqrt(2)]*len(block));hi_bounds.extend([limit/np.sqrt(2)]*len(block))
        # Preserve specified velocities where an adjacent position is free.
        for t in range(5,T):
            if not valid[agent,t-1:t+1].all():continue
            for axis in (0,1):
                if keep[agent,t,4+axis] and not frozen[t-1:t+1,axis].all():
                    row=np.zeros((T,2));row[:,axis]=D1[t-1]
                    matrix.append(row.ravel());lo_bounds.append(state[agent,t,4+axis]);hi_bounds.append(state[agent,t,4+axis])
        pos=np.einsum('kt,atd->akd',L,state[...,:2])+offset
        def add_separation(other_pos,other_head,other_size,mask):
            other_side=np.stack([-other_head[...,1],other_head[...,0]],-1)
            axes=np.stack([heading[agent],side[agent],other_head,other_side],1)
            delta=other_pos-pos[agent]
            ri=(np.abs(np.einsum('kd,knd->kn',heading[agent],axes))*sizes[agent,0]+
                np.abs(np.einsum('kd,knd->kn',side[agent],axes))*sizes[agent,1])/2
            rj=(np.abs(np.einsum('kd,knd->kn',other_head,axes))*other_size[0]+
                np.abs(np.einsum('kd,knd->kn',other_side,axes))*other_size[1])/2
            projected=np.einsum('kd,knd->kn',delta,axes)
            separation=np.abs(projected)-ri-rj
            choice=separation.argmax(-1)
            for k in np.flatnonzero(mask):
                axis=axes[k,choice[k]]*np.sign(projected[k,choice[k]] or 1.)
                row=(-L[k,:,None]*axis).ravel()
                # A positive buffer cannot be imposed instantly on an
                # observed close pair. Ramp it during the first second.
                margin=c.qp_clearance_m*min(times[k],1.)
                rhs=(ri[k,choice[k]]+rj[k,choice[k]]+margin
                     -np.dot(other_pos[k]-offset[agent,k],axis))
                matrix.append(row);lo_bounds.append(rhs);hi_bounds.append(np.inf)
        for other in range(A):
            if other!=agent:
                add_separation(pos[other],heading[other],sizes[other],seen[agent]&seen[other])
        if bg is not None:
            for other in range(len(bs)):
                points=np.c_[np.interp(times,bt,bc[:,other,0]),np.interp(times,bt,bc[:,other,1])]
                yaw=np.unwrap(np.arctan2(bh[:,other,1],bh[:,other,0]))
                angles=np.interp(times,bt,yaw);heads=np.c_[np.cos(angles),np.sin(angles)]
                visible=np.zeros(len(times),bool)
                for k,t in enumerate(times):
                    if bt[0]<=t<=bt[-1]:
                        right=min(max(np.searchsorted(bt,t),1),len(bt)-1)
                        visible[k]=bv[right-1,other] and bv[right,other]
                add_separation(points,heads,bs[other],seen[agent]&visible)
        if refiner.drivable is not None:
            xy=state[agent,:,:2]
            query=pos[agent]
            corners=query[:,None]+heading[agent,:,None]*sizes[agent,0]/2*np.array([1,1,-1,-1])[None,:,None]
            corners+=side[agent,:,None]*sizes[agent,1]/2*np.array([1,-1,-1,1])[None,:,None]
            queries=np.concatenate([query[:,None],corners],1)
            with torch.enable_grad():
                points=torch.as_tensor(queries,device=best.device,dtype=best.dtype).requires_grad_()
                distances=refiner.road_distance(points)
                gradient=torch.autograd.grad(distances.sum(),points)[0]
            distances=distances.detach().cpu().numpy();gradient=gradient.detach().cpu().numpy()
            # Retain the established eligibility policy for recorded parked boxes.
            current=state[agent,4,:2]+(1.461*state[agent,4,2:4] if agent==0 else 0)
            direction=state[agent,4,2:4]
            direction=direction/max(np.linalg.norm(direction),.1)
            normal=np.array([-direction[1],direction[0]])
            signs=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])
            corners_now=current+signs[:,0,None]*sizes[agent,0]/2*direction+signs[:,1,None]*sizes[agent,1]/2*normal
            current_points=np.vstack([current,corners_now])
            with torch.no_grad():current_sdf=refiner.road_distance(best.new_tensor(current_points)).cpu().numpy()
            eligible=[current_sdf[0]>=0]+[current_sdf[1:].min()>=0]*4
            for k in np.flatnonzero(seen[agent]):
                for corner in range(5):
                    if not eligible[corner]:continue
                    g=gradient[k,corner]
                    if np.linalg.norm(g)<1e-7 and distances[k,corner]>=0:continue
                    row=(L[k,:,None]*g).ravel()
                    # The restoration target is the actual road boundary;
                    # the smooth guidance still uses its separate buffer.
                    rhs=-distances[k,corner]+np.dot(g,L[k]@xy)
                    matrix.append(row);lo_bounds.append(rhs);hi_bounds.append(np.inf)
        if not matrix:continue
        M=np.asarray(matrix);lb=np.asarray(lo_bounds);ub=np.asarray(hi_bounds)
        residual=M@start
        before=max(float(np.maximum(lb-residual,0).max(initial=0)),float(np.maximum(residual-ub,0).max(initial=0)))
        if before<=1e-5:
            logs.append({'agent':int(refiner.indices[agent]),'status':'already_feasible'});continue
        # Analytic quadratic objective and constraint Jacobians avoid finite
        # difference calls through the diffusion model or renderer.
        smooth=np.kron(D2[av],np.eye(2))
        H=np.eye(2*T)+.001*smooth.T@smooth
        fun=lambda v:.5*np.dot(v-start,v-start)+.0005*np.dot(smooth@v,smooth@v)
        jac=lambda v:H@v-start
        equality=np.isfinite(lb)&np.isfinite(ub)&(lb==ub)
        constraints=[]
        for mask in (equality,~equality):
            if mask.any():constraints.append(LinearConstraint(M[mask],lb[mask],ub[mask]))
        result=minimize(fun,start,jac=jac,bounds=Bounds(lower,upper),
            constraints=constraints,method='SLSQP',
            options={'maxiter':150,'ftol':1e-9,'disp':False})
        candidate=result.x.reshape(T,2)
        after=max(float(np.maximum(lb-M@result.x,0).max(initial=0)),float(np.maximum(M@result.x-ub,0).max(initial=0)))
        trial=best.detach().clone()
        proposed=best.new_tensor((candidate-refiner.mean[:2].cpu().numpy())/refiner.scale[:2].cpu().numpy())
        trial[0,agent,:,:2]=torch.where(torch.as_tensor(frozen,device=best.device),best[0,agent,:,:2],proposed)
        delta=trial-base
        precision=torch.where(variance>1e-12,coefficient.square()/variance.clamp_min(1e-12),0.)
        kl=.5*(precision*delta.square()).sum((-1,-2))
        relief=False
        if float(kl[0,agent])>c.kl_budget:
            velocity=.5*(precision[0,agent,:,4:6]*delta[0,agent,:,4:6].square()).sum()
            fixed=float(kl[0,agent]-velocity)
            if fixed<c.kl_budget and float(velocity)>0:
                factor=max(0.,(c.kl_budget-fixed)/float(velocity))**.5
                trial[0,agent,:,4:6]=base[0,agent,:,4:6]+delta[0,agent,:,4:6]*factor
                delta=trial-base
                kl=.5*(precision*delta.square()).sum((-1,-2));relief=True
        accept=after<1e-4 and float(kl[0,agent])<=c.kl_budget*(1+1e-4)
        if accept:
            best=trial;state[agent,:,:2]=candidate
        logs.append({'agent':int(refiner.indices[agent]),'status':'accepted' if accept else 'rejected',
            'solver_message':str(result.message),'constraint_violation_before':before,'constraint_violation_after':after,
            'kl_agent':float(kl[0,agent]),'velocity_budget_relief':relief,'success':bool(result.success)})
    return best,logs
