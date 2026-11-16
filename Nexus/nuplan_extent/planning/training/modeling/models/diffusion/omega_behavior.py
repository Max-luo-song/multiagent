"""Optional ego-only forward/nonholonomic preference inside anchor refinement.

Uses generated XY differences and heading, never logged future values.
Prescribed actor reversal and all conditions remain handled by StructuralRefiner.
This soft preference is not a bicycle-model feasibility guarantee.
"""
from dataclasses import dataclass
import torch
from .omega_structural import StructuralConfig,StructuralRefiner


@dataclass
class BehaviorConfig(StructuralConfig):
    ego_motion_weight:float=0.
    ego_curvature_weight:float=0.
    ego_max_curvature_per_m:float=.2


class BehaviorRefiner(StructuralRefiner):
    def objectives(self,x,interaction=True):
        total,losses=super().objectives(x,interaction)
        weight=self.config.ego_motion_weight
        curvature_weight=self.config.ego_curvature_weight
        if weight==0 and curvature_weight==0:return total,losses
        if min(weight,curvature_weight)<0:raise ValueError('Nonnegative ego preference weights required')
        # indices maps the active compact tensor to original Nexus slots.
        ego=torch.nonzero(self.indices==0).flatten()
        if len(ego)!=1:raise ValueError('Exactly one ego slot required')
        k=int(ego[0])
        state=x[:,k]*self.scale+self.mean
        velocity=(state[:,1:,:2]-state[:,:-1,:2])/.5
        raw_heading=state[:,1:,2:4]
        norm=raw_heading.norm(dim=-1,keepdim=True)
        fallback=torch.zeros_like(raw_heading);fallback[...,0]=1
        heading=torch.where(norm>.1,raw_heading/norm.clamp_min(.1),fallback)
        along=(velocity*heading).sum(-1)
        lateral=velocity[...,0]*heading[...,1]-velocity[...,1]*heading[...,0]
        mask=self.valid[:,k,1:]&self.valid[:,k,:-1]
        mask=mask.clone();mask[:,:4]=False
        # Do not penalize any explicitly specified adjacent XY states.
        fixed=self.keep[:,k,:,:2].all(-1)
        mask&=~(fixed[:,1:]&fixed[:,:-1])
        state_residual=(velocity-state[:,1:,4:6]).square().sum(-1)
        cost=lateral.square()+10*torch.relu(-along).square()+.2*state_residual
        loss=(cost*mask).sum()/mask.sum().clamp_min(1)
        losses['ego_motion']=loss
        total=total+weight*loss
        if curvature_weight:
            previous=state[:,:-1,2:4]
            previous_norm=previous.norm(dim=-1,keepdim=True)
            previous=torch.where(previous_norm>.1,previous/previous_norm.clamp_min(.1),fallback)
            yaw_rate=torch.atan2(previous[...,0]*heading[...,1]-previous[...,1]*heading[...,0],
                (previous*heading).sum(-1))/.5
            # Necessary curvature bound with radius >=5 m; not a complete
            # discrete bicycle equation or a steering actuator model.
            bound=self.config.ego_max_curvature_per_m*velocity.norm(dim=-1)
            curvature=(torch.relu(yaw_rate.abs()-bound).square()*mask).sum()/mask.sum().clamp_min(1)
            losses['ego_curvature']=curvature
            total=total+curvature_weight*curvature
        return total,losses
