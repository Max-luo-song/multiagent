"""Experimental OMEGA-inspired clean-anchor guidance for released Nexus.

Paper: https://arxiv.org/html/2512.07661v4 (sections 3.2/3.3).
This is an independent, deliberately scoped implementation, not official
OMEGA code. It uses finite-difference kinematics and lane-segment proxies,
not the paper's auxiliary bicycle controls, route DFS, or SE-IBR game.
"""
from dataclasses import dataclass

import numpy as np
import torch


SCALE = (104., 104., 1., 1., 4., 4., 5., 1.6)
MEAN = (0., 0., 0., 0., 0., 0., 4.5, 2.)


@dataclass
class GuidanceConfig:
    solver: str = 'adam_physical'
    iterations: int = 40
    learning_rate: float = .10
    kl_budget: float = 20.
    kl_weight: float = .02
    collision_weight: float = 30.
    kinematic_weight: float = 4.
    acceleration_weight: float = 2.
    smoothness_weight: float = .1
    heading_weight: float = 2.
    road_weight: float = 2.
    max_acceleration_mps2: float = 5.
    max_yaw_rate_rps: float = .6
    clearance_m: float = .15
    road_tolerance_m: float = 2.5
    # Adam supports physical coordinates or a normalized-coordinate ablation.
    # The exact trust region always uses normalized diffusion coordinates.
    warmup_steps: int = 24
    residual_time: float = .15


def gaussian_kl(delta, coefficient, variance):
    """Exact equal-covariance KL per batch, with deterministic axes excluded.

    KL = sum(A^2 * delta^2 / (2 * variance)). A is squared as required by
    Gaussian KL (also derived in OMEGA appendix A.1).
    """
    movable = (variance > 1e-12) & (coefficient.abs() > 1e-12)
    precision = torch.where(movable, coefficient.square() / variance.clamp_min(1e-12), 0.)
    return .5 * (precision * delta.square()).flatten(1).sum(-1)


class AnchorRefiner:
    def __init__(self, config, scene, valid, keep, segments=None):
        self.config = config
        if config.iterations < 1 or config.kl_budget <= 0 or config.learning_rate <= 0:
            raise ValueError("Positive iterations, KL budget and learning rate required")
        if config.solver not in ('adam', 'adam_physical', 'lbfgs'):
            raise ValueError('Unknown optimizer')
        if scene.shape[0] != 1:
            raise ValueError('Experimental refiner currently supports batch size 1')
        self.indices = valid[0].bool().any(-1).nonzero().flatten()
        if not len(self.indices) or int(self.indices[0]) != 0:
            raise ValueError('Ego must be valid')
        scene = scene[:, self.indices]
        self.scene = scene.detach()
        self.valid = valid[:, self.indices].bool()
        self.keep = keep[:, self.indices].bool()
        self.segments = segments
        self.scale = scene.new_tensor(SCALE)
        self.mean = scene.new_tensor(MEAN)
        self.physical_history = scene * self.scale + self.mean
        # Dimensions come only from current observations, never generated
        # future sizes; shrinkage cannot make the collision objective easier.
        self.sizes = self.physical_history[:, :, 4, 6:8].clamp_min(.1)
        self.phase = "joint"
        self.records = []

    def objectives(self, x, interaction=True):
        c = self.config
        state = x * self.scale + self.mean
        xy, h, v = state[..., :2], state[..., 2:4], state[..., 4:6]
        hnorm = h.norm(dim=-1, keepdim=True)
        # Absent tokens can carry zero cos/sin. atan2(0,0) has undefined
        # derivatives even when its later loss is masked to zero.
        fallback = torch.zeros_like(h)
        fallback[..., 0] = 1
        hn = torch.where(hnorm > .1, h/hnorm.clamp_min(.1), fallback)
        valid = self.valid
        future = valid.clone()
        future[:, :, :5] = False
        pair_valid = valid[:, :, 1:] & valid[:, :, :-1]
        pair_valid[:, :, :4] = False
        triple = pair_valid[:, :, 1:] & pair_valid[:, :, :-1]
        # The history-to-first-future acceleration must be included.
        triple[:, :, 3] = valid[:, :, 3:6].all(-1)
        velocity = (xy[:, :, 1:] - xy[:, :, :-1]) / .5
        acceleration = (velocity[:, :, 1:] - velocity[:, :, :-1]) / .5

        def avg(value, mask):
            while mask.ndim < value.ndim:
                mask = mask.unsqueeze(-1)
            return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1)

        losses = {}
        losses['kinematic'] = avg((velocity-v[:, :, 1:]).square(), pair_valid)
        # Penalize lateral slipping; signed forward speed permits reversing.
        slip = v[..., 0]*hn[..., 1] - v[..., 1]*hn[..., 0]
        losses['heading'] = avg(slip.square() + (h.square().sum(-1)-1).square(), future)
        angle_diff = torch.atan2(
            hn[:, :, :-1, 0]*hn[:, :, 1:, 1]-hn[:, :, :-1, 1]*hn[:, :, 1:, 0],
            (hn[:, :, :-1]*hn[:, :, 1:]).sum(-1)) / .5
        losses['heading'] += avg(torch.relu(angle_diff.abs()-c.max_yaw_rate_rps).square(), pair_valid)
        losses['acceleration'] = avg(torch.relu(acceleration.norm(dim=-1)-c.max_acceleration_mps2).square(), triple)
        losses['smoothness'] = avg(acceleration.square(), triple)
        if self.segments is not None and len(self.segments):
            a, d = self.segments[:, 0], self.segments[:, 1]-self.segments[:, 0]
            q = xy.unsqueeze(-2)-a
            u = ((q*d).sum(-1)/d.square().sum(-1).clamp_min(1e-8)).clamp(0, 1)
            distance = (q-u.unsqueeze(-1)*d).norm(dim=-1).amin(-1)
            # Apply only to agents whose current position is near this lane
            # representation. Parked/off-map vehicles are not pulled to roads.
            current_xy = self.physical_history[:, :, 4, :2]
            cq = current_xy.unsqueeze(-2)-a
            cu = ((cq*d).sum(-1)/d.square().sum(-1).clamp_min(1e-8)).clamp(0, 1)
            eligible = (cq-cu.unsqueeze(-1)*d).norm(dim=-1).amin(-1) < c.road_tolerance_m
            losses['road'] = avg(torch.relu(distance-c.road_tolerance_m).square(), future & eligible.unsqueeze(-1))
        else:
            losses['road'] = x.sum()*0
        if interaction:
            # Differentiable separating-axis penetration for physical boxes.
            # Unlike a circle approximation this accommodates long buses.
            centers = xy.clone()
            centers[:, 0] = centers[:, 0] + 1.461*hn[:, 0]
            perp = torch.stack([-hn[..., 1], hn[..., 0]], -1)
            basis = torch.stack([hn, perp], -2)  # B,A,T,2,2
            ii, jj = torch.triu_indices(x.shape[1], x.shape[1], 1, device=x.device)
            bi, bj = basis[:, ii], basis[:, jj]
            axes = torch.cat([bi, bj], -2)
            delta = centers[:, jj]-centers[:, ii]
            radius_i = ((bi.unsqueeze(-3)*axes.unsqueeze(-2)).sum(-1).abs()*self.sizes[:, ii, None, None, :]/2).sum(-1)
            radius_j = ((bj.unsqueeze(-3)*axes.unsqueeze(-2)).sum(-1).abs()*self.sizes[:, jj, None, None, :]/2).sum(-1)
            sep = ((delta.unsqueeze(-2)*axes).sum(-1).abs()-radius_i-radius_j).amax(-1)
            collision_mask = future[:, ii] & future[:, jj]
            # Sum pairs per timestamp to avoid dilution in larger scenes.
            losses['collision'] = (torch.relu(c.clearance_m-sep).square()*collision_mask).sum()/future.shape[-1]
        else:
            losses['collision'] = x.sum()*0
        total = sum(getattr(c, key+'_weight') * value for key, value in losses.items())
        return total, losses

    def _lbfgs(self, base, coefficient, variance, free, interaction):
        """Optimize in whitened reverse-kernel coordinates with line search.

        The Euclidean ball in these coordinates is exactly the Gaussian KL
        ball. Strong-Wolfe line search resolves physical unit/curvature
        differences without hand-setting separate XY/heading/velocity steps.
        """
        c = self.config
        precision = coefficient.square()/variance.clamp_min(1e-12)
        inverse_root = precision.clamp_min(1e-12).rsqrt()
        offset = torch.zeros_like(base, requires_grad=True)
        optimizer = torch.optim.LBFGS([offset], lr=1., max_iter=c.iterations,
            max_eval=2*c.iterations, line_search_fn='strong_wolfe',
            tolerance_grad=1e-6, tolerance_change=1e-9)
        best = base
        best_value = float(self.objectives(base, interaction)[0])

        def closure():
            nonlocal best,best_value
            optimizer.zero_grad(set_to_none=True)
            white = offset*free
            norm = white.square().flatten(1).sum(-1).clamp_min(1e-20).sqrt()
            factor = ((2*c.kl_budget)**.5/norm).clamp(max=1.)
            delta = white*factor.view(-1,1,1,1)*inverse_root
            candidate = base+delta
            objective,_ = self.objectives(candidate, interaction)
            loss = objective+c.kl_weight*gaussian_kl(delta,coefficient,variance).mean()
            loss.backward()
            if not torch.isfinite(offset.grad).all():
                raise FloatingPointError('Nonfinite L-BFGS guidance gradient')
            if float(loss) < best_value:
                best_value = float(loss)
                best = candidate.detach().clone()
            return loss

        optimizer.step(closure)
        return best,best_value

    def __call__(self, anchor, coefficient, variance, constrain):
        c = self.config
        full_anchor = anchor
        anchor = anchor[:, self.indices]
        coefficient = coefficient[:, self.indices]
        variance = variance[:, self.indices]
        # Finite positive variance is necessary for a bounded Gaussian KL.
        # Frozen timesteps and deterministic final axes receive zero updates.
        active = (variance > 1e-12) & (coefficient.abs() > 1e-12)
        free = (~self.keep) & self.valid.unsqueeze(-1) & active
        free[..., 6:] = False
        base = torch.where(self.keep, self.scene, anchor).detach()
        if not free.any():
            return full_anchor
        interaction = self.phase != 'warmup'
        # Clone outside inference_mode so autograd can safely save tensors.
        with torch.inference_mode(False), torch.enable_grad():
            base = base.clone()
            coeff, var = coefficient.detach().clone(), variance.detach().clone()
            free = free.clone()
            # Both parameterizations are retained for reproducible ablation.
            # The physical-coordinate Adam candidate performed better on the
            # current development case. Neither is a universal solver claim.
            offset = torch.zeros_like(base, requires_grad=True)
            optimization_scale = self.scale if c.solver=='adam_physical' else 104.
            optimizer = torch.optim.Adam([offset], lr=c.learning_rate)
            best = base
            with torch.no_grad():
                best_value = float(self.objectives(base, interaction)[0])
            start_value = best_value
            if c.solver == 'lbfgs':
                best,best_value = self._lbfgs(base,coeff,var,free,interaction)
            for _ in range(c.iterations if c.solver in ('adam','adam_physical') else 0):
                optimizer.zero_grad(set_to_none=True)
                candidate = base + (offset/optimization_scale)*free
                objective, _ = self.objectives(candidate, interaction)
                kl = gaussian_kl(candidate-base, coeff, var)
                loss = objective + c.kl_weight*kl.mean()
                loss.backward()
                if not torch.isfinite(offset.grad).all():
                    raise FloatingPointError('Nonfinite guidance gradient')
                optimizer.step()
                with torch.no_grad():
                    offset.mul_(free)
                    delta = offset/optimization_scale
                    kl = gaussian_kl(delta, coeff, var)
                    factor = (c.kl_budget/kl.clamp_min(1e-20)).sqrt().clamp(max=1)
                    offset.mul_(factor.view(-1, 1, 1, 1))
                    candidate = base + offset/optimization_scale
                    value = float(self.objectives(candidate, interaction)[0] + c.kl_weight*gaussian_kl(candidate-base, coeff, var).mean())
                    if value < best_value:
                        best_value, best = value, candidate.clone()
            kl = gaussian_kl(best-base, coeff, var)
            if float(kl.max()) > c.kl_budget*(1+1e-4):
                raise AssertionError('KL trust region exceeded')
            if not torch.equal(best[self.keep], self.scene[self.keep]):
                raise AssertionError('Condition changed')
            self.records.append({'phase':self.phase, 'objective_before':start_value,
                                 'objective_after_with_kl':best_value, 'kl_max':float(kl.max()),
                                 'max_xy_anchor_change_m':float(((best-base)[..., :2]*104).norm(dim=-1).max())})
        result = full_anchor.clone()
        result[:, self.indices] = best.detach()
        return result


def sample_two_phase(diffuser, scene, valid, keep, global_context, z_t=None,
                     raw_map=None, return_intermediates=False):
    """Warm up whole future, then reduce one frame's residual noise at a time.

    Fixed history and earlier refined frames never get denoised again. The
    numerical residual at t=0 uses Nexus's original finite log-SNR schedule.
    This is offline conditional sampling, not an external feedback rollout.
    """
    refiner = getattr(diffuser, 'anchor_refiner', None)
    config = refiner.config if refiner else GuidanceConfig()
    low = config.residual_time
    if not 0 < low < 1 or config.warmup_steps < 2:
        raise ValueError('Invalid two-phase schedule')
    if z_t is None:
        z_t = torch.randn_like(scene)
    keep = keep.bool()
    z_t = torch.where(keep, scene, z_t)
    context = torch.cat([scene*keep, keep], -1)
    schedule = diffuser.logsnr_schedule_cosine if diffuser.schedule == 'cosine' else diffuser.logsnr_schedule_cosine_shifted
    constrain = dict(mode=diffuser.constrain_mode, gamma=diffuser.constraint_gamma,
                     raw_map=raw_map, valid_mask=valid, keep_mask=keep, scene_tensor=scene)
    shape = tuple(scene.shape[:3])
    fixed_tokens = keep.all(-1).cpu().numpy()
    levels = np.full(shape, 1., dtype=np.float32)
    levels[fixed_tokens] = 0
    intermediate = []

    def step(next_levels):
        nonlocal z_t, levels
        previous = z_t
        result, mu = diffuser.sampler_step(scene, z_t, keep, levels, next_levels,
            context, global_context, valid, schedule, False, True, scene, constrain)
        changed = torch.as_tensor(next_levels < levels, device=scene.device).unsqueeze(-1)
        # Even finite t=0 model estimates cannot perturb frozen timestamps.
        z_t = torch.where(changed, result, previous)
        z_t = torch.where(keep, scene, z_t)
        levels = next_levels
        if return_intermediates:
            intermediate.append(z_t.clone())

    if refiner:
        refiner.phase = 'warmup'
    for level in np.linspace(1., low, config.warmup_steps+1)[1:]:
        following = np.full(shape, level, dtype=np.float32)
        following[fixed_tokens] = 0
        step(following)
    if refiner:
        refiner.phase = 'rolling'
    for tau in range(5, scene.shape[2]):
        following = levels.copy()
        following[:, :, tau] = 0
        step(following)
    return z_t, intermediate
