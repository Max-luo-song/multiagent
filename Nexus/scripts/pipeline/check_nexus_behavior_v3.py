"""Check behavior auditing against analytic forward/reverse trajectories."""
import unittest
import numpy as np
import torch
from audit_nexus_behavior_v3 import measure
from check_nexus_omega_v2 import base,module,PACKAGE
from importlib import import_module
behavior=import_module(PACKAGE+'.omega_behavior')


class BehaviorChecks(unittest.TestCase):
    def case(self,angle=0.):
        state=np.zeros((2,21,8),dtype=np.float64)
        direction=np.array([np.cos(angle),np.sin(angle)])
        state[...,2:4]=direction;state[...,6:]=[5.,2.]
        times=np.arange(-4,17)*.5
        state[0,:,:2]=times[:,None]*direction
        state[0,:,4:6]=direction
        state[1,:,:2]=-times[:,None]*direction
        state[1,:,4:6]=-direction
        normalized=(state-np.array([0,0,0,0,0,0,4.5,2]))/np.array([104,104,1,1,4,4,5,1.6])
        return {'sampled_tensor':torch.from_numpy(normalized[None]),
                'valid_mask':torch.ones(1,2,21,dtype=torch.bool)}

    def test_rigid_rotation_invariance_and_bus_reverse_exclusion(self):
        for angle in [0.,1.1,-2.8]:
            metrics=measure(self.case(angle),np)['ego']
            self.assertAlmostEqual(metrics['final_forward_progress_m'],8.)
            self.assertAlmostEqual(metrics['final_left_displacement_m'],0.)
            self.assertAlmostEqual(metrics['xy_velocity_head_rms_mps'],0.)
            self.assertAlmostEqual(metrics['xy_lateral_velocity_rms_mps'],0.)
            self.assertFalse(metrics['unrequested_reverse_over_0p2m'])

    def test_reverse_and_sliding_detected(self):
        payload=self.case()
        payload['sampled_tensor'][:,0,:,0]*=-1
        ego=measure(payload,np)['ego']
        self.assertAlmostEqual(ego['reverse_distance_body_axis_m'],8.)
        self.assertAlmostEqual(ego['moving_heading_error_max_deg'],180.)
        self.assertTrue(ego['unrequested_reverse_over_0p2m'])
        payload=self.case();payload['sampled_tensor'][:,0,:,2:4]=torch.tensor([0.,1.])
        ego=measure(payload,np)['ego']
        self.assertAlmostEqual(ego['xy_lateral_velocity_rms_mps'],1.)
        self.assertAlmostEqual(ego['moving_heading_error_max_deg'],90.)

    def test_invalid_values_do_not_affect_metrics(self):
        payload=self.case();payload['valid_mask'][:,0,15:]=False
        before=measure(payload,np)['ego']
        payload['sampled_tensor'][:,0,15:]=12345
        after=measure(payload,np)['ego']
        self.assertEqual(before,after)


class EgoPreferenceChecks(unittest.TestCase):
    def case(self,weight):
        state=torch.zeros(1,2,9,8);state[...,2]=1;state[...,6:]=torch.tensor([5.,2.])
        state[:,1,:,1]=20
        scene=(state-torch.tensor(base.MEAN))/torch.tensor(base.SCALE)
        valid=torch.ones(1,2,9,dtype=torch.bool)
        keep=torch.zeros_like(scene,dtype=torch.bool);keep[:,:,:5]=True;keep[:,1,6:]=True
        config=behavior.BehaviorConfig(iterations=3,kl_budget=1.,ego_motion_weight=weight)
        return behavior.BehaviorRefiner(config,scene,valid,keep),scene,valid,keep

    def test_zero_weight_reproduces_original_refiner(self):
        ref,scene,valid,keep=self.case(0.)
        original=module.StructuralRefiner(ref.config,scene,valid,keep)
        anchor=scene.clone();anchor[:,0,5:,0]=.01
        args=(anchor,torch.ones_like(anchor),torch.ones_like(anchor)*.01,{})
        torch.testing.assert_close(ref(*args),original(*args),atol=0,rtol=0)

    def test_preference_uses_ego_xy_and_excludes_bus_reversal(self):
        ref,scene,_,_=self.case(100.)
        def cost(value):return float(ref.objectives(value)[1]['ego_motion'])
        bus_reverse=scene.clone()
        bus_reverse[:,1,5:,0]=-torch.arange(1,5)*.5/104
        bus_reverse[:,1,5:,4]=-.25
        self.assertEqual(cost(bus_reverse),0.)
        reverse=scene.clone();reverse[:,0,5:,0]=-torch.arange(1,5)*.5/104
        reverse[:,0,5:,4]=-.25
        self.assertAlmostEqual(cost(reverse),10.,places=5)
        slide=scene.clone();slide[:,0,5:,1]=torch.arange(1,5)*.5/104
        slide[:,0,5:,5]=.25
        self.assertAlmostEqual(cost(slide),1.,places=5)

    def test_finite_heading_gradient_and_hard_conditions(self):
        ref,scene,_,keep=self.case(100.)
        anchor=scene.clone();anchor[:,0,5:,0]=torch.arange(1,5)*.5/104
        anchor[:,0,5:,1]=torch.arange(1,5)*.2/104
        anchor.requires_grad_()
        ref.objectives(anchor)[1]['ego_motion'].backward()
        self.assertTrue(torch.isfinite(anchor.grad).all())
        self.assertGreater(float(anchor.grad[:,0,5:,2:4].abs().sum()),0.)
        out=ref(anchor.detach(),torch.ones_like(anchor),torch.ones_like(anchor)*.01,{})
        self.assertTrue(torch.equal(out[keep],scene[keep]))
        kl=.5*((out-anchor.detach()).square()/.01).sum((-1,-2))
        self.assertLessEqual(float(kl.max()),1.0001)

    def test_heading_scope_keeps_other_actor_heading_exact(self):
        ref,scene,valid,keep=self.case(100.)
        ref.config.ego_heading_only=True
        anchor=scene.clone();anchor[:,1,5,2]=.6;anchor[:,1,5,3]=.3
        anchor[:,1,5,4]=2.
        out=ref(anchor,torch.ones_like(anchor),torch.ones_like(anchor)*.01,{})
        torch.testing.assert_close(out[:,1,5:,2:4],anchor[:,1,5:,2:4],atol=0,rtol=0)

    def test_curvature_rejects_stationary_rotation(self):
        ref,scene,_,_=self.case(0.)
        ref.config.ego_curvature_weight=1000.
        turning=scene.clone()
        angle=torch.arange(1,5)*.1
        turning[:,0,5:,2]=angle.cos();turning[:,0,5:,3]=angle.sin()
        turning.requires_grad_()
        losses=ref.objectives(turning)[1]
        self.assertGreater(float(losses['ego_curvature']),.03)
        losses['ego_curvature'].backward()
        self.assertTrue(torch.isfinite(turning.grad).all())
        self.assertGreater(float(turning.grad[...,2:4].abs().sum()),0.)
        # Moving on a radius 10 m arc satisfies the necessary curvature bound.
        turning=turning.detach()
        turning[:,0,5:,0]=10*angle.sin()/104
        turning[:,0,5:,1]=10*(1-angle.cos())/104
        self.assertLess(float(ref.objectives(turning)[1]['ego_curvature']),1e-6)


if __name__=='__main__':unittest.main()
