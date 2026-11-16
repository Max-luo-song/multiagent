"""Opt-in control-space guidance for Nexus clean anchors.

Future XY, heading and velocity share a bounded nonholonomic rollout.  Only
observed states initialize the rollout; no future ego/C positions are supplied.
The neural clean anchor is a soft prior, not a set of future conditions.

This is a new inference algorithm, not an exact Gaussian-KL OMEGA variant.
Collision and lane preferences remain optimization objectives; callers MUST
independently reject infeasible final samples.  Nothing is repaired after the
sampler returns, and this module never changes the released/default sampler.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .omega_structural import StructuralConfig, StructuralRefiner


@dataclass
class KinematicConfig(StructuralConfig):
    start_call: int = 8
    control_iterations: int = 65
    final_control_iterations: int = 180
    control_learning_rate: float = .07
    braking_limit_mps2: float = 4.5
    propulsion_limit_mps2: float = 2.5
    lateral_acceleration_limit_mps2: float = 2.
    curvature_limit_per_m: float = .12
    control_jerk_limit_mps3: float = 6.
    curvature_slew_limit_per_ms: float = .025
    anchor_xy_weight: float = .15
    anchor_speed_weight: float = .1
    lane_preference_weight: float = 30.
    dynamic_following_weight: float = 150.
    entering_actor_yield_weight: float = 300.
    following_headway_s: float = .65
    control_effort_weight: float = .2


class KinematicRefiner(StructuralRefiner):
    """Joint acceleration/curvature optimization inside reverse diffusion."""

    def __init__(self, *args, route_path, **kwargs):
        super().__init__(*args, **kwargs)
        c = self.config
        if c.following_weight or c.motion_qp:
            raise ValueError('Use dynamic following and control rollouts, not legacy following/QP')
        self.calls = 0
        self.previous_controls = None
        self.last_clean_anchor = None
        self.initial = self.physical_history[0, :, 4].detach()
        heading = self.initial[:, 2:4]
        heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(.1)
        self.initial_heading = heading
        self.initial_yaw = torch.atan2(heading[:, 1], heading[:, 0])
        signed_speed = (self.initial[:, 4:6] * heading).sum(-1)
        self.initial_speed = signed_speed.clamp_min(0)
        self.controlled = self.valid[0, :, 4] & ~self.keep[0, :, 5:].any((-1, -2))
        self.controlled &= signed_speed > -.5
        if not self.controlled[0]:
            raise ValueError('Observed forward-moving ego with free future required')
        # Speeds before the current observation are used only for jerk boundary.
        prior = self.physical_history[0, :, 3]
        prior_head = prior[:, 2:4] / prior[:, 2:4].norm(dim=-1, keepdim=True).clamp_min(.1)
        prior_speed = (prior[:, 4:6] * prior_head).sum(-1)
        self.initial_acceleration = ((signed_speed-prior_speed)/.5).clamp(
            -c.braking_limit_mps2, c.propulsion_limit_mps2)
        self.routes = {}
        with np.load(route_path, allow_pickle=False) as data:
            for compact, slot in enumerate(self.indices.tolist()):
                key = f'route_{slot}'
                if key not in data or not self.controlled[compact]:
                    continue
                points = self.scene.new_tensor(data[key])
                self.routes[compact] = (points[:-1], points[1:]-points[:-1])
        self.controlled_slots = self.indices[self.controlled].tolist()
        self.given_future = self.keep[0, :, 5:].all((-1, -2))
        current_center = self.initial[:, :2].clone()
        current_center[0] += 1.461*self.initial_heading[0]
        relative = current_center[None]-current_center[:, None]
        initial_along = (relative*self.initial_heading[:, None]).sum(-1)
        alignment = (self.initial_heading[:, None]*self.initial_heading[None]).sum(-1)
        self.initial_ahead = (initial_along>0)&(initial_along<40)&(alignment>.7)

    def project_controls(self, controls):
        """Project actuator limits, including changes across native intervals."""
        c = self.config
        with torch.no_grad():
            a = controls[..., 0] * 3.
            a.clamp_(-c.braking_limit_mps2, c.propulsion_limit_mps2)
            previous = self.initial_acceleration
            for t in range(a.shape[1]):
                a[:, t].copy_(torch.maximum(torch.minimum(a[:, t], previous+c.control_jerk_limit_mps3*.5),
                                           previous-c.control_jerk_limit_mps3*.5))
                previous = a[:, t]
            controls[..., 0].copy_(a/3.)
            k = controls[..., 1] * .02
            k.clamp_(-c.curvature_limit_per_m, c.curvature_limit_per_m)
            previous = torch.zeros_like(k[:, 0])
            for t in range(k.shape[1]):
                change = c.curvature_slew_limit_per_ms*.5
                k[:, t].copy_(torch.maximum(torch.minimum(k[:, t], previous+change), previous-change))
                previous = k[:, t]
            controls[..., 1].copy_(k/.02)
            controls[~self.controlled] = 0

    def rollout(self, controls, base):
        """0.1s midpoint integration; samples returned at native 0.5s times."""
        c = self.config
        a = (controls[..., 0]*3.).clamp(-c.braking_limit_mps2, c.propulsion_limit_mps2)
        speed = [self.initial_speed]
        for t in range(a.shape[1]):
            speed.append((speed[-1]+a[:, t]*.5).clamp(0, 40))
        speed = torch.stack(speed, 1)
        fraction = controls.new_tensor([.1, .3, .5, .7, .9])
        v_mid = speed[:, :-1, None] + (speed[:, 1:]-speed[:, :-1])[:, :, None]*fraction
        maximum_speed = torch.maximum(speed[:, :-1], speed[:, 1:])
        max_curvature = (c.lateral_acceleration_limit_mps2 / maximum_speed.square().clamp_min(1.))
        max_curvature = max_curvature.clamp_max(c.curvature_limit_per_m)
        curvature = controls[..., 1]*.02
        curvature = torch.maximum(torch.minimum(curvature, max_curvature), -max_curvature)
        d_yaw = (v_mid*curvature[:, :, None]*.1).flatten(1)
        yaw_end = self.initial_yaw[:, None] + d_yaw.cumsum(1)
        yaw_mid = yaw_end-d_yaw/2
        distance = v_mid.flatten(1)*.1
        displacement = distance[..., None]*torch.stack([yaw_mid.cos(), yaw_mid.sin()], -1)
        xy_dense = self.initial[:, None, :2] + displacement.cumsum(1)
        xy = xy_dense[:, 4::5]
        yaw = yaw_end[:, 4::5]
        heading = torch.stack([yaw.cos(), yaw.sin()], -1)
        velocity = speed[:, 1:, None]*heading
        sizes = self.initial[:, None, 6:8].expand(-1, xy.shape[1], -1)
        future = torch.cat([xy, heading, velocity, sizes], -1)
        state = base*self.scale+self.mean
        future = torch.where(self.controlled[:, None, None], future, state[0, :, 5:])
        state = torch.cat([self.physical_history[:, :, :5], future[None]], 2)
        # Preserve every supplied channel exactly, including fixed A's future.
        normalized = (state-self.mean)/self.scale
        normalized = torch.where(self.keep, self.scene, normalized)
        return normalized, speed, curvature

    def control_objective(self, controls, base):
        c = self.config
        state, speeds, curvature = self.rollout(controls, base)
        total, parts = super().objectives(state, interaction=True)
        physical = state*self.scale+self.mean
        anchor = base*self.scale+self.mean
        mask = self.valid[:, :, 5:] & self.controlled[None, :, None]
        def actor_mean_sum(value):
            return ((value*mask).sum(-1)/mask.sum(-1).clamp_min(1)).sum()
        anchor_xy = actor_mean_sum((physical[:, :, 5:, :2]-anchor[:, :, 5:, :2]).square().sum(-1))
        target_speed = anchor[:, :, 5:, 4:6].norm(dim=-1).clamp(0, 40)
        anchor_speed = actor_mean_sum((speeds[None, :, 1:]-target_speed).square())
        lane = state.sum()*0
        for actor, (start, delta) in self.routes.items():
            xy = physical[:, actor, 5:, :2]
            q = xy[:, :, None]-start
            u = ((q*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-8)).clamp(0, 1)
            distance2 = (q-u[..., None]*delta).square().sum(-1).amin(-1)
            m = mask[:, actor]
            lane += (distance2*m).sum()/m.sum().clamp_min(1)
        # Determine front/lateral relations at every generated time, including
        # a newly entering actor.  No historical lead identity is frozen.
        xy, heading, seen = self.interpolate(physical, self.valid)
        center = xy.clone()
        center[:, 0] += 1.461*heading[:, 0]
        delta = center[:, None]-center[:, :, None]
        axis = heading[:, :, None]
        normal = torch.stack([-axis[..., 1], axis[..., 0]], -1)
        along = (delta*axis).sum(-1)
        lateral = (delta*normal).sum(-1).abs()
        compatible = (heading[:, :, None]*heading[:, None]).sum(-1)>.7
        pairmask = seen[:, :, None]&seen[:, None]
        pairmask &= self.controlled[None, :, None, None]
        eye = torch.eye(len(self.indices), device=state.device, dtype=torch.bool)
        pairmask &= ~eye[None, :, :, None]
        width = (self.sizes[:, :, None, None, 1]+self.sizes[:, None, :, None, 1])/2+.3
        length = (self.sizes[:, :, None, None, 0]+self.sizes[:, None, :, None, 0])/2
        future_velocity = (physical[:, :, 1:, :2]-physical[:, :, :-1, :2])/.5
        current_speed = future_velocity[:, :, 4:].norm(dim=-1)
        native_speed = torch.cat([self.initial_speed[None, :, None], current_speed], -1)
        dense_speed = torch.nn.functional.interpolate(
            native_speed, size=along.shape[-1], mode='linear', align_corners=True)
        required = length+.5+self.config.following_headway_s*dense_speed[:, :, None]
        # Compact support: vehicles clearly in adjacent lanes must not create
        # a persistent artificial braking incentive through sigmoid tails.
        gate = ((width+.4-lateral)/.4).clamp(0, 1)*torch.sigmoid(along/.3)
        following = (torch.relu(required-along).square()*gate*compatible*pairmask).sum()/along.shape[-1]
        # Preserve yielding order for a prescribed actor entering this actor's
        # map lane from ahead. The activation depends on the GIVEN actor and
        # observed lane topology, not on free ego/C future waypoints. Therefore
        # a candidate cannot turn this safety term off just by swerving toward
        # a neighboring lane or accelerating past the entering actor.
        yielding = state.sum()*0
        for actor, (start, segment) in self.routes.items():
            for other in torch.nonzero(self.given_future).flatten().tolist():
                if not bool(self.initial_ahead[actor, other]):
                    continue
                points = center[:, other]
                q = points[:, :, None]-start
                u = ((q*segment).sum(-1)/segment.square().sum(-1).clamp_min(1e-8)).clamp(0, 1)
                distance2 = (q-u[..., None]*segment).square().sum(-1)
                distance, choice = distance2.min(-1)
                direction = segment[choice]
                direction = direction/direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                order_gap = ((center[:, other]-center[:, actor])*direction).sum(-1)
                extent = (self.sizes[:, actor, 0]+self.sizes[:, other, 0])/2
                desired = extent[:, None]+.5+c.following_headway_s*dense_speed[:, actor]
                lateral_extent = (self.sizes[:, actor, 1]+self.sizes[:, other, 1])/2+.3
                activation = ((lateral_extent[:, None]+.4-distance.clamp_min(1e-8).sqrt())/.4).clamp(0, 1)
                active = seen[:, actor]&seen[:, other]
                yielding += (torch.relu(desired-order_gap).square()*activation*active).sum()/along.shape[-1]
        effort = (controls.square()*self.controlled[:, None, None]).sum()/mask.sum().clamp_min(1)
        parts.update(anchor_xy=anchor_xy, anchor_speed=anchor_speed,
                     lane_preference=lane, dynamic_following=following,
                     entering_actor_yield=yielding, control_effort=effort)
        c = self.config
        total = (total+c.anchor_xy_weight*anchor_xy+c.anchor_speed_weight*anchor_speed
                 +c.lane_preference_weight*lane+c.dynamic_following_weight*following
                 +c.entering_actor_yield_weight*yielding
                 +c.control_effort_weight*effort)
        return total, state, parts

    def __call__(self, anchor, coefficient, variance, constrain):
        self.calls += 1
        if self.calls < self.config.start_call:
            return anchor
        full = anchor
        base = torch.where(self.keep, self.scene, anchor[:, self.indices]).detach()
        with torch.inference_mode(False), torch.enable_grad():
            base = base.clone()
            if self.previous_controls is None:
                controls = base.new_zeros((len(self.indices), base.shape[2]-5, 2))
            else:
                controls = self.previous_controls.clone()
            controls.requires_grad_()
            self.project_controls(controls)
            optimizer = torch.optim.Adam([controls], lr=self.config.control_learning_rate)
            iterations = (self.config.final_control_iterations if float(variance.max())<1e-4
                          else self.config.control_iterations)
            before = None
            best_cost = float('inf')
            best_controls = controls.detach().clone()
            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = self.control_objective(controls, base)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite control objective; reject this sample')
                value = float(loss.detach())
                if before is None:
                    before = value
                if value < best_cost:
                    best_cost = value
                    best_controls = controls.detach().clone()
                loss.backward()
                if not torch.isfinite(controls.grad).all():
                    raise FloatingPointError('Nonfinite control gradient; reject this sample')
                optimizer.step()
                self.project_controls(controls)
            loss, _, _ = self.control_objective(controls, base)
            if float(loss) < best_cost:
                best_controls = controls.detach().clone()
            final_loss, best, parts = self.control_objective(best_controls, base)
            self.previous_controls = best_controls
            assert torch.equal(best[self.keep], self.scene[self.keep])
            self.records.append({'call':self.calls, 'iterations':iterations,
                                 'controlled_slots':self.controlled_slots,
                                 'objective_before':before,
                                 'objective_after':float(final_loss),
                                 'terms_after':{k:float(v) for k,v in parts.items()},
                                 'trust_region':'soft physical anchor preference; no exact KL guarantee',
                                 'safety_acceptance':'requires independent final gate'})
        result = full.clone()
        result[:, self.indices] = best.detach()
        self.last_clean_anchor = result.detach()
        return result
