"""Whole-body lane preference on top of observed-speed cruise guidance."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from .omega_cruise import CruiseConfig,CruiseRefiner


@dataclass
class CorridorCruiseConfig(CruiseConfig):
    lane_body_weight: float = 10000.
    lane_body_clearance_m: float = .25
    incoming_lane_relaxation: float = .1
    control_learning_rate: float = .03


class CorridorCruiseRefiner(CruiseRefiner):
    def __init__(self,*args,route_path,**kwargs):
        super().__init__(*args,route_path=route_path,**kwargs)
        with np.load(Path(route_path).parent/'lane_corridors_sdf.npz') as data:
            self.lane_sdf=self.scene.new_tensor(data['sdf'])
            self.lane_origin=self.scene.new_tensor(data['origin'])
            self.lane_resolution=float(data['resolution'])
            slots=data['slots'].tolist()
        self.lane_actors=[];self.lane_risk_relaxation=[]
        given=self.physical_history[0,:,5:,:2]
        for slot in slots:
            found=torch.nonzero(self.indices==slot).flatten()
            if len(found)!=1:raise ValueError('Lane-corridor actor unavailable')
            actor=int(found[0]);incoming=False
            if actor in self.routes:
                start,delta=self.routes[actor]
                for other in torch.nonzero(self.given_future).flatten().tolist():
                    if not bool(self.initial_ahead[actor,other]):continue
                    q=given[other,:,None]-start
                    u=((q*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-8)).clamp(0,1)
                    distance=(q-u[...,None]*delta).square().sum(-1).amin(-1).sqrt()
                    width=(self.sizes[0,actor,1]+self.sizes[0,other,1])/2+.3
                    incoming |= bool((distance<width).any())
            self.lane_actors.append(actor)
            self.lane_risk_relaxation.append(self.config.incoming_lane_relaxation if incoming else 1.)

    def project_controls(self,controls):
        super().project_controls(controls)
        # Keep the optimization VARIABLE inside the speed-dependent lateral-
        # acceleration bound too. Otherwise the rollout clips a hidden large
        # curvature and its zero gradient can prevent correction of lane drift.
        with torch.no_grad():
            c=self.config
            velocity=[self.initial_speed]
            acceleration=(controls[...,0]*3.).clamp(-c.braking_limit_mps2,c.propulsion_limit_mps2)
            for t in range(acceleration.shape[1]):
                velocity.append((velocity[-1]+acceleration[:,t]*.5).clamp(0,40))
            velocity=torch.stack(velocity,1)
            maximum=torch.maximum(velocity[:,:-1],velocity[:,1:])
            bound=(c.lateral_acceleration_limit_mps2/maximum.square().clamp_min(1.)).clamp_max(c.curvature_limit_per_m)
            # Stay a numerical epsilon inside the rollout's bound, retaining
            # a steering gradient instead of landing beyond it by roundoff.
            bound*=1.-1e-4
            step=c.curvature_slew_limit_per_ms*.5
            # Backward tightening leaves a nonempty slew-feasible interval
            # even when the next interval's speed makes its curvature smaller.
            for t in range(bound.shape[1]-2,-1,-1):
                bound[:,t]=torch.minimum(bound[:,t],bound[:,t+1]+step)
            curvature=controls[...,1]*.02
            previous=torch.zeros_like(curvature[:,0])
            for t in range(curvature.shape[1]):
                curvature[:,t]=torch.maximum(torch.minimum(curvature[:,t],bound[:,t]),-bound[:,t])
                curvature[:,t]=torch.maximum(torch.minimum(curvature[:,t],previous+step),previous-step)
                previous=curvature[:,t]
            controls[...,1].copy_(curvature/.02)

    def control_objective(self,controls,base):
        total,state,parts=super().control_objective(controls,base)
        physical=state*self.scale+self.mean
        penalty=state.sum()*0
        H,W=self.lane_sdf.shape[-2:]
        for layer,actor in enumerate(self.lane_actors):
            xy=physical[:,actor,5:,:2];head=physical[:,actor,5:,2:4]
            center=xy+(1.461*head if int(self.indices[actor])==0 else 0)
            side=torch.stack([-head[...,1],head[...,0]],-1)
            signs=state.new_tensor([[1,1],[1,-1],[-1,-1],[-1,1]])
            size=self.sizes[:,actor]
            corner=(center[:,:,None]+head[:,:,None]*size[:,None,None,0:1]/2*signs[None,None,:,0:1]
                    +side[:,:,None]*size[:,None,None,1:2]/2*signs[None,None,:,1:2])
            grid=(corner-self.lane_origin)/(self.lane_resolution*state.new_tensor([W-1,H-1]))*2-1
            distance=torch.nn.functional.grid_sample(self.lane_sdf[layer][None,None],grid.reshape(1,1,-1,2),
                 mode='bilinear',padding_mode='border',align_corners=True).reshape(corner.shape[:-1])
            mask=self.valid[:,actor,5:]&self.controlled[actor]
            cost=torch.relu(self.config.lane_body_clearance_m-distance.amin(-1)).square()
            penalty+=(cost*mask).sum()/mask.sum().clamp_min(1)*self.lane_risk_relaxation[layer]
        parts['whole_body_lane']=penalty
        return total+self.config.lane_body_weight*penalty,state,parts
