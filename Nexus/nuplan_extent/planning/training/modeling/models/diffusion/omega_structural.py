"""Per-agent KL budgets and dense constraints for the second-stage pilot.

Independent OMEGA-inspired engineering variant. Given actor states remain
exact; no trajectory editing occurs after sampling. Optional recorded replay
obstacles are explicit extra scene information, not history-only prediction.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .omega_guidance import AnchorRefiner,GuidanceConfig


@dataclass
class StructuralConfig(GuidanceConfig):
    dense_substeps:int=5
    jerk_weight:float=.2
    max_jerk_mps3:float=8.
    boundary_weight:float=4.
    optimize_heading:bool=True
    ego_heading_only:bool=False
    road_clearance_m:float=.3
    following_weight:float=0.
    motion_qp:bool=False
    qp_acceleration_mps2:float=4.
    qp_jerk_mps3:float=6.
    qp_clearance_m:float=.3


def sat_separation(ci,hi,si,cj,hj,sj):
    ci,hi,si,cj,hj,sj=torch.broadcast_tensors(ci,hi,si,cj,hj,sj)
    pi=torch.stack([-hi[...,1],hi[...,0]],-1)
    pj=torch.stack([-hj[...,1],hj[...,0]],-1)
    bi,bj=torch.stack([hi,pi],-2),torch.stack([hj,pj],-2)
    axes=torch.cat([bi,bj],-2)
    ri=((bi.unsqueeze(-3)*axes.unsqueeze(-2)).sum(-1).abs()*si.unsqueeze(-2)/2).sum(-1)
    rj=((bj.unsqueeze(-3)*axes.unsqueeze(-2)).sum(-1).abs()*sj.unsqueeze(-2)/2).sum(-1)
    return (((cj-ci).unsqueeze(-2)*axes).sum(-1).abs()-ri-rj).amax(-1)


class StructuralRefiner(AnchorRefiner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.config.solver!='adam_physical':
            raise ValueError('Structural pilot supports physical Adam only')
        if self.config.dense_substeps<1:
            raise ValueError('Positive substeps required')
        self.background=None
        self.drivable=None
        # Historical lane-compatible lead identities select a consistent
        # separation branch; otherwise SAT can reward crossing through a lead
        # vehicle to reach the opposite side of a local collision minimum.
        state=self.physical_history[:,:,4]
        direction=state[...,2:4]/state[...,2:4].norm(dim=-1,keepdim=True).clamp_min(.1)
        center=state[...,:2].clone();center[:,0]+=1.461*direction[:,0]
        delta=center[:,None]-center[:,:,None]
        longitudinal=(delta*direction[:,:,None]).sum(-1)
        normal=torch.stack([-direction[...,1],direction[...,0]],-1)
        lateral=(delta*normal[:,:,None]).sum(-1).abs()
        compatible=(direction[:,:,None]*direction[:,None]).sum(-1)>.7
        compatible&=(longitudinal>0)&(longitudinal<40)
        compatible&=lateral<(self.sizes[:,:,None,1]+self.sizes[:,None,:,1])/2+.5
        compatible&=self.valid[:,:,4,None]&self.valid[:,None,:,4]
        score=torch.where(compatible,longitudinal,torch.full_like(longitudinal,float('inf')))
        nearest=score.argmin(-1)
        self.follow_i=torch.nonzero(torch.isfinite(score.amin(-1))[0]).flatten()
        self.follow_j=nearest[0,self.follow_i]
        self.follow_direction=direction[:,self.follow_i]

    def set_background_replay(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.background={k:torch.as_tensor(data[k],device=self.scene.device,dtype=self.scene.dtype)
                             for k in ('times_s','centers','heading','sizes','valid')}

    def set_drivable_sdf(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.drivable={k:torch.as_tensor(data[k],device=self.scene.device,dtype=self.scene.dtype)
                           for k in ('sdf','origin','resolution')}

    def road_distance(self,points):
        data=self.drivable
        H,W=data['sdf'].shape
        grid=(points-data['origin'])/(data['resolution']*points.new_tensor([W-1,H-1]))*2-1
        result=torch.nn.functional.grid_sample(data['sdf'][None,None],grid.reshape(1,1,-1,2),
            mode='bilinear',padding_mode='border',align_corners=True)
        return result.reshape(points.shape[:-1])

    def interpolate(self,state,valid,steps=None,times=None):
        if times is None:
            steps=steps or self.config.dense_substeps
            grid=torch.arange(4*steps,(state.shape[-2]-1)*steps+1,device=state.device)/steps
        else:
            grid=times/.5+4
        lo=grid.floor().long().clamp(0,state.shape[-2]-2)
        u=(grid-lo).clamp(0,1)
        hi=lo+1
        xy=state[:,:,lo,:2]*(1-u[:,None])+state[:,:,hi,:2]*u[:,None]
        angle=torch.atan2(state[...,3],state[...,2])
        difference=angle[:,:,hi]-angle[:,:,lo]
        difference=torch.atan2(difference.sin(),difference.cos())
        yaw=angle[:,:,lo]+u*difference
        heading=torch.stack([yaw.cos(),yaw.sin()],-1)
        active=valid[:,:,lo]&valid[:,:,hi]
        # Exact endpoint validity does not depend on a neighboring interval.
        active=torch.where((u==0)[None,None],valid[:,:,lo],active)
        active=torch.where((u==1)[None,None],valid[:,:,hi],active)
        return xy,heading,active

    def objectives(self,x,interaction=True):
        c=self.config
        state=x*self.scale+self.mean
        xy,h,v=state[...,:2],state[...,2:4],state[...,4:6]
        norm=h.norm(dim=-1,keepdim=True)
        fallback=torch.zeros_like(h);fallback[...,0]=1
        hn=torch.where(norm>.1,h/norm.clamp_min(.1),fallback)
        state=torch.cat([xy,hn,state[...,4:]],-1)
        valid=self.valid
        future=valid.clone();future[:,:,:5]=False
        edge=valid[:,:,1:]&valid[:,:,:-1];edge[:,:,:4]=False
        triple=valid[:,:,:-2]&valid[:,:,1:-1]&valid[:,:,2:];triple[:,:,:3]=False
        quartet=valid[:,:,:-3]&valid[:,:,1:-2]&valid[:,:,2:-1]&valid[:,:,3:];quartet[:,:,:2]=False

        def agent_mean_sum(value,mask):
            while mask.ndim<value.ndim:mask=mask.unsqueeze(-1)
            mask=mask.expand_as(value)
            return ((value*mask).flatten(2).sum(-1)/mask.flatten(2).sum(-1).clamp_min(1)).sum()/x.shape[0]

        velocity=(xy[:,:,1:]-xy[:,:,:-1])/.5
        acceleration=(velocity[:,:,1:]-velocity[:,:,:-1])/.5
        jerk=(acceleration[:,:,1:]-acceleration[:,:,:-1])/.5
        losses={}
        residual=(velocity-v[:,:,1:]).square().sum(-1)
        losses['kinematic']=agent_mean_sum(residual,edge)
        known=self.keep.all(-1)
        boundary=(known[:,:,1:]!=known[:,:,:-1])&edge
        losses['boundary']=agent_mean_sum(residual,boundary)
        losses['acceleration']=agent_mean_sum(torch.relu(acceleration.norm(dim=-1)-c.max_acceleration_mps2).square(),triple)
        losses['jerk']=agent_mean_sum(torch.relu(jerk.norm(dim=-1)-c.max_jerk_mps3).square(),quartet)
        losses['smoothness']=agent_mean_sum(acceleration.square().sum(-1),triple)
        slip=v[...,0]*hn[...,1]-v[...,1]*hn[...,0]
        heading_error=slip.square()+(h.square().sum(-1)-1).square()
        losses['heading']=agent_mean_sum(heading_error,future)
        yaw_step=torch.atan2(hn[:,:,:-1,0]*hn[:,:,1:,1]-hn[:,:,:-1,1]*hn[:,:,1:,0],
                             (hn[:,:,:-1]*hn[:,:,1:]).sum(-1))/.5
        losses['heading']+=agent_mean_sum(torch.relu(yaw_step.abs()-c.max_yaw_rate_rps).square(),edge)
        if self.drivable is not None:
            centers=xy.clone();centers[:,0]+=1.461*hn[:,0]
            lateral=torch.stack([-hn[...,1],hn[...,0]],-1)
            corners=centers[:,:,:,None]+self.sizes[:,:,None,None,0:1]/2*hn[:,:,:,None]*x.new_tensor([1,1,-1,-1])[None,None,None,:,None]
            corners=corners+self.sizes[:,:,None,None,1:2]/2*lateral[:,:,:,None]*x.new_tensor([1,-1,-1,1])[None,None,None,:,None]
            center_sdf=self.road_distance(centers)
            corner_sdf=self.road_distance(corners)
            # Do not demand that a historically partly off-road parked box
            # suddenly becomes fully on-road. Its center remains constrained.
            eligible=center_sdf[:,:,4].detach()>=0
            box_eligible=corner_sdf[:,:,4].detach().amin(-1)>=0
            center_cost=torch.relu(c.road_clearance_m-center_sdf).square()
            corner_cost=torch.relu(c.road_clearance_m-corner_sdf.amin(-1)).square()
            losses['road']=agent_mean_sum(center_cost,future&eligible[:,:,None])
            losses['road']+=agent_mean_sum(corner_cost,future&box_eligible[:,:,None])
        elif self.segments is not None and len(self.segments):
            a,d=self.segments[:,0],self.segments[:,1]-self.segments[:,0]
            q=xy.unsqueeze(-2)-a
            u=((q*d).sum(-1)/d.square().sum(-1).clamp_min(1e-8)).clamp(0,1)
            distance=(q-u.unsqueeze(-1)*d).norm(dim=-1).amin(-1)
            eligible=distance[:,:,4].detach()<c.road_tolerance_m
            losses['road']=agent_mean_sum(torch.relu(distance-c.road_tolerance_m).square(),future&eligible[:,:,None])
        else:losses['road']=x.sum()*0
        losses['collision']=x.sum()*0
        losses['following']=x.sum()*0
        if interaction:
            centers,heading,active=self.interpolate(state,valid)
            centers=centers.clone();centers[:,0]+=1.461*heading[:,0]
            ii,jj=torch.triu_indices(x.shape[1],x.shape[1],1,device=x.device)
            sizes=self.sizes[:,:,None,:]
            sep=sat_separation(centers[:,ii],heading[:,ii],sizes[:,ii],centers[:,jj],heading[:,jj],sizes[:,jj])
            mask=active[:,ii]&active[:,jj]
            losses['collision']=(torch.relu(c.clearance_m-sep).square()*mask).sum()/centers.shape[-2]
            if c.following_weight and len(self.follow_i):
                fi,fj=self.follow_i,self.follow_j
                axis=self.follow_direction[:,:,None]
                normal=torch.stack([-axis[...,1],axis[...,0]],-1)
                delta=centers[:,fj]-centers[:,fi]
                gap=(delta*axis).sum(-1)
                lateral=(delta*normal).sum(-1).abs()
                def radius(index):
                    head=heading[:,index]
                    side=torch.stack([-head[...,1],head[...,0]],-1)
                    size=self.sizes[:,index,None]
                    return ((head*axis).sum(-1).abs()*size[...,0]+(side*axis).sum(-1).abs()*size[...,1])/2
                required=radius(fi)+radius(fj)+c.clearance_m
                width=(self.sizes[:,fi,None,1]+self.sizes[:,fj,None,1])/2+c.clearance_m
                gate=torch.sigmoid((width-lateral)/.15)
                pairs=active[:,fi]&active[:,fj]
                losses['following']=(torch.relu(required-gap).square()*gate*pairs).sum()/centers.shape[-2]
            if self.background is not None:
                bg=self.background
                pos,head,seen=self.interpolate(state,valid,times=bg['times_s'])
                pos=pos.clone();pos[:,0]+=1.461*head[:,0]
                # Shape B,A,T,M,2; all M recorded actors are immutable.
                sep_bg=sat_separation(pos[:,:,:,None],head[:,:,:,None],self.sizes[:,:,None,None],
                    bg['centers'][None,None],bg['heading'][None,None],bg['sizes'][None,None,None])
                mask_bg=seen[:,:,:,None]&bg['valid'][None,None].bool()
                losses['collision']+=(torch.relu(c.clearance_m-sep_bg).square()*mask_bg).sum()/len(bg['times_s'])
        total=sum(getattr(c,key+'_weight')*value for key,value in losses.items())
        return total,losses

    def __call__(self,anchor,coefficient,variance,constrain):
        c=self.config
        full=anchor
        anchor,coef,var=[z[:,self.indices] for z in (anchor,coefficient,variance)]
        active=(var>1e-12)&(coef.abs()>1e-12)
        free=(~self.keep)&self.valid.unsqueeze(-1)&active
        free[...,6:]=False
        if not c.optimize_heading:free[...,2:4]=False
        elif c.ego_heading_only:free[:,self.indices!=0,:,2:4]=False
        if not free.any():return full
        with torch.inference_mode(False),torch.enable_grad():
            base=torch.where(self.keep,self.scene,anchor).detach().clone()
            coef,var,free=coef.clone(),var.clone(),free.clone()
            precision=torch.where(active,coef.square()/var.clamp_min(1e-12),0.)
            kl=lambda delta:.5*(precision*delta.square()).sum((-1,-2))
            offset=torch.zeros_like(base,requires_grad=True)
            opt=torch.optim.Adam([offset],lr=c.learning_rate)
            interaction=self.phase!='warmup'
            best=base
            before=float(self.objectives(base,interaction)[0])
            best_value=before
            for _ in range(c.iterations):
                opt.zero_grad(set_to_none=True)
                candidate=base+offset/self.scale*free
                objective,_=self.objectives(candidate,interaction)
                loss=objective+c.kl_weight*kl(candidate-base).mean()
                loss.backward()
                if not torch.isfinite(offset.grad).all():raise FloatingPointError('Nonfinite structural gradient')
                opt.step()
                with torch.no_grad():
                    offset.mul_(free)
                    budgets=kl(offset/self.scale)
                    factor=(c.kl_budget/budgets.clamp_min(1e-20)).sqrt().clamp(max=1.)
                    offset.mul_(factor[:,:,None,None])
                    candidate=base+offset/self.scale
                    value=float(self.objectives(candidate,interaction)[0]+c.kl_weight*kl(candidate-base).mean())
                    if value<best_value:best_value,best=value,candidate.clone()
            restoration=[]
            if c.motion_qp and float(var.max())<1e-4:
                from .omega_motion_qp import restore_motion
                best,restoration=restore_motion(self,best,base,coef,var)
                best_value=float(self.objectives(best,interaction)[0]+c.kl_weight*kl(best-base).mean())
            budgets=kl(best-base)
            assert float(budgets.max())<=c.kl_budget*(1+1e-4)
            assert torch.equal(best[self.keep],self.scene[self.keep])
            _,parts=self.objectives(best,interaction)
            self.records.append({'phase':self.phase,'objective_before':before,'objective_after_with_kl':best_value,
                'kl_max':float(budgets.sum(-1).max()),'kl_max_per_agent':float(budgets.max()),
                'kl_budget_scope':'per agent; total budget is agent_count times configured budget',
                'max_xy_anchor_change_m':float(((best-base)[...,:2]*104).norm(dim=-1).max()),
                'terms_after':{k:float(v) for k,v in parts.items()},'motion_restoration':restoration})
        result=full.clone();result[:,self.indices]=best.detach()
        return result
