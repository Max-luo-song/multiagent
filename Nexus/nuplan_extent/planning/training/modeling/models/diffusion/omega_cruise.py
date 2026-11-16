"""Observed-speed soft preferences for control-space Nexus generation.

Targets come only from the current observed speed.  They are costs, not future
conditions: braking, stopping, and avoidance remain free when required by the
traffic constraints.  A stable target also supplies a recovery incentive after
the risk recedes, instead of tracking an increasingly slow diffusion proposal.
"""
from dataclasses import dataclass
import torch

from .omega_kinematic import KinematicConfig, KinematicRefiner


@dataclass
class CruiseConfig(KinematicConfig):
    cruise_speed_weight: float = 8.
    cruise_terminal_speed_weight: float = 4.
    cruise_moving_threshold_mps: float = 1.
    anchor_longitudinal_weight: float = .03


class CruiseRefiner(KinematicRefiner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cruise_reference = self.initial_speed.detach().clone()
        self.cruise_eligible = (self.controlled
                                & (self.cruise_reference > self.config.cruise_moving_threshold_mps))

    def rollout(self, controls, base):
        """Same physical forward rollout with an escapable zero-speed bound.

        The projected forward speed is nonnegative. An identity surrogate
        derivative through the bound lets the optimizer release an excessively
        negative acceleration after stopping; an ordinary saturated clamp would
        otherwise remove every speed-recovery gradient in that region.
        """
        c=self.config
        a=(controls[...,0]*3.).clamp(-c.braking_limit_mps2,c.propulsion_limit_mps2)
        speed=[self.initial_speed]
        for t in range(a.shape[1]):
            proposal=speed[-1]+a[:,t]*.5
            projected=proposal.clamp(0,40)
            speed.append(proposal+(projected-proposal).detach())
        speed=torch.stack(speed,1)
        fraction=controls.new_tensor([.1,.3,.5,.7,.9])
        mid=speed[:,:-1,None]+(speed[:,1:]-speed[:,:-1])[:,:,None]*fraction
        maximum=torch.maximum(speed[:,:-1],speed[:,1:])
        limit=(c.lateral_acceleration_limit_mps2/maximum.square().clamp_min(1.)).clamp_max(c.curvature_limit_per_m)
        curvature=torch.maximum(torch.minimum(controls[...,1]*.02,limit),-limit)
        step_yaw=(mid*curvature[:,:,None]*.1).flatten(1)
        yaw=self.initial_yaw[:,None]+step_yaw.cumsum(1)
        mid_yaw=yaw-step_yaw/2
        distance=mid.flatten(1)*.1
        displacement=distance[...,None]*torch.stack([mid_yaw.cos(),mid_yaw.sin()],-1)
        xy=self.initial[:,None,:2]+displacement.cumsum(1)
        native_yaw=yaw[:,4::5]
        heading=torch.stack([native_yaw.cos(),native_yaw.sin()],-1)
        velocity=speed[:,1:,None]*heading
        sizes=self.initial[:,None,6:8].expand(-1,heading.shape[1],-1)
        future=torch.cat([xy[:,4::5],heading,velocity,sizes],-1)
        original=base*self.scale+self.mean
        future=torch.where(self.controlled[:,None,None],future,original[0,:,5:])
        state=torch.cat([self.physical_history[:,:,:5],future[None]],2)
        normalized=torch.where(self.keep,self.scene,(state-self.mean)/self.scale)
        return normalized,speed,curvature

    def control_objective(self, controls, base):
        total, state, parts = super().control_objective(controls, base)
        physical = state*self.scale+self.mean
        # For our rollout the heading is unit and speed is nonnegative. This
        # equals its magnitude but retains a useful derivative at zero speed.
        speed = (physical[:, :, 5:, 4:6]*physical[:, :, 5:, 2:4]).sum(-1)
        target = self.cruise_reference[None, :, None]
        mask = self.valid[:, :, 5:] & self.cruise_eligible[None, :, None]
        # Absolute future XY tracking also fixes an implicit progress schedule.
        # Keep the original lateral proposal preference, but relax its along-
        # road component so a slow proposal does not defeat observed cruising.
        anchor = base*self.scale+self.mean
        displacement = physical[:, :, 5:, :2]-anchor[:, :, 5:, :2]
        longitudinal = (displacement*self.initial_heading[None, :, None]).sum(-1)
        free = self.valid[:, :, 5:] & self.controlled[None, :, None]
        longitudinal_prior = ((longitudinal.square()*free).sum(-1)
                              / free.sum(-1).clamp_min(1)).sum()
        total += (self.config.anchor_longitudinal_weight-self.config.anchor_xy_weight)*longitudinal_prior
        error = (speed-target).square()
        cruise = ((error*mask).sum(-1)/mask.sum(-1).clamp_min(1)).sum()
        terminal_mask = mask[:, :, -4:]
        recovery = ((error[:, :, -4:]*terminal_mask).sum(-1)
                    / terminal_mask.sum(-1).clamp_min(1)).sum()
        parts.update(observed_speed_preference=cruise, terminal_speed_recovery=recovery,
                     anchor_longitudinal=longitudinal_prior)
        return (total+self.config.cruise_speed_weight*cruise
                +self.config.cruise_terminal_speed_weight*recovery), state, parts
