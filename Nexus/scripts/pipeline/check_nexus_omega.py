"""Numerical checks for trust bounds, immutable conditions and rolling freeze."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('omega_under_test', ROOT/'nuplan_extent/planning/training/modeling/models/diffusion/omega_guidance.py')
omega = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = omega
spec.loader.exec_module(omega)


class GuidanceChecks(unittest.TestCase):
    def test_kl_matches_gaussian_distributions(self):
        delta = torch.tensor([[[[.2, -.3]]]], dtype=torch.float64)
        coefficient = torch.full_like(delta, .7)
        variance = torch.full_like(delta, .2)
        p = torch.distributions.Normal(coefficient*delta, variance.sqrt())
        q = torch.distributions.Normal(torch.zeros_like(delta), variance.sqrt())
        expected = torch.distributions.kl_divergence(p, q).sum()
        torch.testing.assert_close(omega.gaussian_kl(delta, coefficient, variance)[0], expected)
        self.assertEqual(float(omega.gaussian_kl(delta, coefficient, variance*0)), 0.)

    def test_refinement_preserves_conditions_invalid_states_and_kl(self):
        shape = (1, 3, 9, 8)
        physical = torch.zeros(shape)
        physical[..., 2] = 1
        physical[..., 6:] = torch.tensor([5., 2.])
        physical[:, 1, :, 1] = 20
        scale, mean = torch.tensor(omega.SCALE), torch.tensor(omega.MEAN)
        scene = (physical-mean)/scale
        valid = torch.ones(shape[:3], dtype=torch.bool)
        valid[:, 2] = False
        valid[:, 1, 7:] = False
        scene[:, 1, 7:, 2:4] = 0
        keep = torch.zeros(shape, dtype=torch.bool)
        keep[:, :, :5] = True
        keep[:, 1, 6] = True
        anchor = scene.clone()
        anchor[:, 0, 5:, 0] += .03  # implausible 3.12m first-step jump
        refiner = omega.AnchorRefiner(omega.GuidanceConfig(iterations=12, kl_budget=.15), scene, valid, keep)
        A, variance = torch.ones_like(anchor), torch.ones_like(anchor)*.01
        before = refiner.objectives(anchor[:, refiner.indices])[0]
        out = refiner(anchor, A, variance, {})
        self.assertTrue(torch.equal(out[keep], scene[keep]))
        self.assertTrue(torch.equal(out[:, 2], anchor[:, 2]))
        self.assertTrue(torch.equal(out[~valid], anchor[~valid]))
        self.assertTrue(torch.equal(out[..., 6:], anchor[..., 6:]))
        self.assertLessEqual(float(omega.gaussian_kl(out-anchor, A, variance).max()), .15002)
        self.assertLess(float(refiner.objectives(out[:, refiner.indices])[0]), float(before))

    def test_rolling_zero_freezes_previous_frames(self):
        class Sampler:
            schedule = 'cosine'
            constrain_mode = []
            constraint_gamma = []
            anchor_refiner = None
            logsnr_schedule_cosine = None
            seen = []

            def sampler_step(self, scene, x, keep, u_t, u_s, *args):
                self.seen.append((u_t.copy(), u_s.copy()))
                # Deliberately corrupt all outputs; scheduler must discard
                # changes on fixed tokens and non-active timesteps.
                return x+1, x+1

        sampler = Sampler()
        scene = torch.zeros(1, 2, 9, 8)
        keep = torch.zeros_like(scene, dtype=torch.bool)
        keep[:, :, :5] = True
        keep[:, 1, 7] = True
        out, _ = omega.sample_two_phase(sampler, scene, torch.ones(1, 2, 9), keep,
                                       None, z_t=torch.zeros_like(scene))
        self.assertTrue(torch.equal(out[keep], scene[keep]))
        torch.testing.assert_close(out[0, 0, 5:, 0], torch.full((4,), 25.))
        for u_t, u_s in sampler.seen:
            self.assertTrue((u_s <= u_t).all())
            self.assertTrue((u_t[:, :, :5] == 0).all())
        for u_t, u_s in sampler.seen[-4:]:
            self.assertTrue(((u_t-u_s)>0).sum(axis=-1).max() <= 1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
