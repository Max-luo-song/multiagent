"""Numerical checks for dense geometry, per-agent KL and hard conditions."""
import importlib.util
from pathlib import Path
import sys
import unittest
import types
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
PACKAGE='omega_test_package'
package=types.ModuleType(PACKAGE)
package.__path__=[str(ROOT/'nuplan_extent/planning/training/modeling/models/diffusion')]
sys.modules[PACKAGE]=package
from importlib import import_module
base=import_module(PACKAGE+'.omega_guidance')
module=import_module(PACKAGE+'.omega_structural')


class StructuralChecks(unittest.TestCase):
    def setup_case(self):
        state=torch.zeros(1,3,9,8)
        state[...,2]=1;state[...,6:]=torch.tensor([5.,2.])
        state[:,1,:,1]=12
        scene=(state-torch.tensor(base.MEAN))/torch.tensor(base.SCALE)
        valid=torch.ones(1,3,9,dtype=torch.bool);valid[:,2]=False;valid[:,1,8]=False
        keep=torch.zeros_like(scene,dtype=torch.bool);keep[:,:,:5]=True;keep[:,1,6]=True
        config=module.StructuralConfig(iterations=8,kl_budget=.2)
        return module.StructuralRefiner(config,scene,valid,keep),scene,valid,keep

    def test_subframe_collision_detected(self):
        ref,scene,valid,keep=self.setup_case()
        states=scene[:,:2]*ref.scale+ref.mean
        states[:,0,:,0]=-10;states[:,1,:,0]=10;states[:,1,:,1]=0
        states[:,0,5:,0]=10;states[:,1,5:,0]=-10
        xy,h,v=ref.interpolate(states,ref.valid[:,:2])
        sep=module.sat_separation(xy[:,0],h[:,0],ref.sizes[:,0,None],xy[:,1],h[:,1],ref.sizes[:,1,None])
        self.assertLess(float(sep.min()),0.)
        self.assertGreater(float(sep[:,0]),0.)
        self.assertGreater(float(sep[:,5]),0.)

    def test_broadcast_geometry_and_finite_gradient(self):
        xy=torch.zeros(1,2,3,1,2,requires_grad=True)
        h=torch.zeros_like(xy);h[...,0]=1
        other=torch.ones(1,1,3,4,2)*10
        sep=module.sat_separation(xy,h,torch.ones(1,2,1,1,2),other,h[:,:1],torch.ones(1,1,1,4,2))
        self.assertEqual(tuple(sep.shape),(1,2,3,4))
        sep.sum().backward();self.assertTrue(torch.isfinite(xy.grad).all())

    def test_exact_conditions_per_agent_budget_and_invalid_states(self):
        ref,scene,valid,keep=self.setup_case()
        anchor=scene.clone();anchor[:,0,5:,0]+=.03
        A=torch.ones_like(anchor);variance=torch.ones_like(anchor)*.01
        out=ref(anchor,A,variance,{})
        self.assertTrue(torch.equal(out[keep],scene[keep]))
        self.assertTrue(torch.equal(out[~valid],anchor[~valid]))
        self.assertTrue(torch.equal(out[...,6:],anchor[...,6:]))
        kl=.5*((out-anchor).square()/.01).sum((-1,-2))
        self.assertLessEqual(float(kl.max()),.20002)
        self.assertLessEqual(ref.records[-1]['objective_after_with_kl'],ref.records[-1]['objective_before'])

    def test_drivable_field_gradient(self):
        ref,scene,valid,keep=self.setup_case()
        # Signed distance to a vertical boundary at local x=-2.
        sdf=torch.arange(21,dtype=torch.float32)[None].repeat(21,1)-8
        ref.drivable={'sdf':sdf,'origin':torch.tensor([-10.,-10.]),'resolution':torch.tensor(1.)}
        points=torch.tensor([[[0.,0.],[1.,0.]]],requires_grad=True)
        distances=ref.road_distance(points)
        torch.testing.assert_close(distances,torch.tensor([[2.,3.]]))
        distances.sum().backward()
        torch.testing.assert_close(points.grad,torch.tensor([[[1.,0.],[1.,0.]]]))
        total,_=ref.objectives(scene[:,:2])
        self.assertTrue(torch.isfinite(total))

    def test_history_velocity_uses_no_future_values(self):
        spec=importlib.util.spec_from_file_location('v2_runner_for_test',ROOT/'scripts/pipeline/run_nexus_omega_v2.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        raw=np.zeros((2,21,8),dtype=np.float32)
        raw[...,:2]=np.arange(21)[None,:,None]*.03
        valid=np.ones((2,21),dtype=bool)
        original=raw.copy()
        first=runner.restore_history_velocity(original,raw,valid)
        disturbed=raw.copy();disturbed[:,5:]=12345
        second=runner.restore_history_velocity(disturbed,disturbed,valid)
        np.testing.assert_array_equal(first[:,:5],second[:,:5])
        np.testing.assert_array_equal(first[...,:4],original[...,:4])
        np.testing.assert_array_equal(first[...,6:],original[...,6:])

    def test_history_order_rejects_crossing_to_opposite_side(self):
        ref,scene,valid,keep=self.setup_case()
        scene[:,1,:,0]=10/104.;scene[:,1,:,1]=0
        config=module.StructuralConfig(iterations=2,following_weight=1.)
        ref=module.StructuralRefiner(config,scene,valid,keep)
        crossing=scene[:,:2].clone();crossing[:,0,5:,0]=20/104.
        _,losses=ref.objectives(crossing)
        self.assertGreater(float(losses['following']),1.)
        passing=crossing.clone();passing[:,0,5:,1]=20/104.
        _,away=ref.objectives(passing)
        self.assertLess(float(away['following']),float(losses['following']))

    def test_motion_restoration_enforces_initial_continuity(self):
        ref,scene,valid,keep=self.setup_case()
        scene[:,0,:5,0]=torch.arange(5)*2.5/104
        scene[:,0,:5,4]=5/4
        config=module.StructuralConfig(iterations=2,kl_budget=20000,motion_qp=True)
        ref=module.StructuralRefiner(config,scene,valid,keep)
        anchor=scene[:,:2].clone()
        qp=import_module(PACKAGE+'.omega_motion_qp')
        out,logs=qp.restore_motion(ref,anchor,anchor,torch.ones_like(anchor),torch.ones_like(anchor)*.01)
        self.assertTrue(any(x['status']=='accepted' for x in logs))
        self.assertTrue(torch.equal(out[ref.keep],anchor[ref.keep]))
        xy=out[0,0,:,:2]*104
        acceleration=torch.diff(xy,n=2,dim=0)/.25
        self.assertLessEqual(float(acceleration[3:].norm(dim=-1).max()),4.001)

    def test_motion_budget_allocation_keeps_geometry_and_conditions(self):
        ref,scene,valid,keep=self.setup_case()
        scene[:,0,:5,0]=torch.arange(5)*2.5/104
        config=module.StructuralConfig(iterations=2,kl_budget=100,motion_qp=True)
        ref=module.StructuralRefiner(config,scene,valid,keep)
        anchor=scene[:,:2].clone();best=anchor.clone();best[:,0,5:,4]=.2
        qp=import_module(PACKAGE+'.omega_motion_qp')
        out,logs=qp.restore_motion(ref,best,anchor,torch.ones_like(anchor),torch.ones_like(anchor)*.001)
        self.assertTrue(any(x.get('velocity_budget_relief') for x in logs))
        self.assertTrue(torch.equal(out[ref.keep],anchor[ref.keep]))
        kl=.5*((out-anchor).square()/.001).sum((-1,-2))
        self.assertLessEqual(float(kl.max()),100.01)


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
