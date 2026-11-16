"""Regression checks for all-actor registration, physical boxes and support."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from nexus_3dgs.bundle import save_bundle
from nexus_3dgs.geometry import pose
from nexus_3dgs.runtime import TrajectoryController
from nexus_3dgs.grounding import GroundedRegistration
from nexus_3dgs.clearance import box_separation, rear_axle_to_box
from nexus_3dgs.kinematics import repair_isolated_half_turns,repair_low_speed_tail_slips
from nexus_3dgs.scene_projection import project_forward_offsets,pair_separations


class FollowupChecks(unittest.TestCase):
    def test_auxiliary_render_gradients_reach_refinement(self):
        spec=importlib.util.spec_from_file_location('foreground_supervision',REPO/'Omniscenegraph/models/foreground_supervision.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        main=torch.zeros(1,3,2,requires_grad=True)
        aux=torch.zeros_like(main,requires_grad=True)
        main.absgrad=torch.ones_like(main)
        aux.absgrad=torch.full_like(aux,3.)
        result=module.refinement_gradients({'means2d':main},[{'means2d':aux}])
        torch.testing.assert_close(result,torch.full_like(main,4.))
        (main.sum()+aux.sum()*2).backward()
        result=module.refinement_gradients({'means2d':main},[{'means2d':aux}],absgrad=False)
        torch.testing.assert_close(result,torch.full_like(main,3.))
        torch.testing.assert_close(module.refinement_gradients({'means2d':main}),main.absgrad)

    def test_terminal_low_speed_turn_and_side_slip(self):
        xy=np.array([[0,0],[.02,.05],[.04,.1],[-.73,.55],[-.31,.94]])
        values=np.array([[pose(*p,yaw=np.deg2rad(y))] for p,y in zip(xy,[68,68,68,195,191])])
        valid=np.ones((5,1),bool);fixed=np.zeros_like(valid);fixed[:3]=True
        result,changes=repair_low_speed_tail_slips(values,valid,fixed,np.arange(5)*500000,2)
        self.assertEqual([c['sample_index'] for c in changes],[3,4])
        np.testing.assert_array_equal(result[:3],values[:3])
        yaw=np.unwrap(np.arctan2(result[:,0,1,0],result[:,0,0,0]))
        self.assertLess(np.rad2deg(np.abs(np.diff(yaw))).max(),20)
        self.assertTrue(all(abs(c['accepted_lateral_step_m'])<.1 for c in changes))

    def test_tail_repair_preserves_given_state_and_real_moving_turn(self):
        values=np.array([[pose(3*t,0,yaw=np.deg2rad(y))] for t,y in enumerate([0,10,70,150])])
        fixed=np.zeros((4,1),bool);fixed[:2]=True
        result,changes=repair_low_speed_tail_slips(values,np.ones((4,1),bool),fixed,np.arange(4)*500000,1)
        self.assertFalse(changes);np.testing.assert_array_equal(result,values)
        fixed[:]=True
        values[3,0,:3,:3]=pose(0,0,yaw=np.pi)[:3,:3]
        result,changes=repair_low_speed_tail_slips(values,np.ones((4,1),bool),fixed,np.arange(4)*500000,1)
        self.assertFalse(changes);np.testing.assert_array_equal(result,values)

    def test_vehicle_owned_rgb_blocks_background_color_gradient(self):
        spec=importlib.util.spec_from_file_location('foreground_supervision',REPO/'Omniscenegraph/models/foreground_supervision.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        bg=torch.full((1,2,3),.2,requires_grad=True);fg=torch.full((1,2,3),.5,requires_grad=True)
        owned=module.foreground_owned_rgb(fg+.2*bg,fg,torch.tensor([[True,False]]))
        owned[0,0].sum().backward(retain_graph=True)
        self.assertEqual(float(bg.grad.abs().sum()),0.)
        self.assertGreater(float(fg.grad.abs().sum()),0.)
        bg.grad.zero_();fg.grad.zero_();owned[0,1].sum().backward()
        self.assertGreater(float(bg.grad[0,1].abs().sum()),0.)

    def test_scene_projection_anticipates_follower_without_hiding_it(self):
        times=np.linspace(0,4,41);knots=np.linspace(0,4,9)
        boxes=np.zeros((len(times),3,3));boxes[:,0,:2]=[-20,10]
        boxes[:,2,0]=-8+1.5*times
        valid=np.ones((len(times),3),bool);sizes=np.tile([4.,2.],(3,1))
        source=boxes.copy()
        offsets,_=project_forward_offsets(times,knots,boxes,valid,sizes,[1],np.array([[1.,0.]]),np.array([[True]+[False]*8]))
        repaired=boxes.copy();repaired[:,1,0]+=np.interp(times,knots,offsets[0])
        sep,pairs=pair_separations(repaired,valid,sizes)
        self.assertGreater(float(sep[pairs].min()),0)
        self.assertEqual(float(offsets[0,0]),0.)
        np.testing.assert_array_equal(boxes,source)
        np.testing.assert_array_equal(repaired[:,[0,2]],source[:,[0,2]])

    def test_scene_projection_rejects_fixed_overlapping_pair(self):
        times=np.linspace(0,2,21);knots=np.linspace(0,2,5)
        boxes=np.zeros((21,3,3));boxes[:,0,:2]=[20,20]
        with self.assertRaisesRegex(ValueError,'no free actor'):
            project_forward_offsets(times,knots,boxes,np.ones((21,3),bool),np.tile([4.,2.],(3,1)),[0],np.array([[1.,0.]]),np.array([[True]+[False]*4]))

    def test_isolated_static_half_turn_preserves_position_and_other_states(self):
        values=np.array([[pose(.1*t,0,yaw=np.deg2rad(y))] for t,y in enumerate([45,46,-134,46,45])])
        valid=np.ones((5,1),bool);fixed=np.zeros_like(valid);fixed[[0,-1]]=True
        original=values.copy();corrected,changes=repair_isolated_half_turns(values,valid,fixed)
        self.assertEqual(len(changes),1)
        self.assertEqual(changes[0]['sample_index'],2)
        self.assertAlmostEqual(np.rad2deg(np.arctan2(corrected[2,0,1,0],corrected[2,0,0,0])),46)
        np.testing.assert_array_equal(corrected[...,:3,3],original[...,:3,3])
        np.testing.assert_array_equal(values,original)
        np.testing.assert_array_equal(corrected[[0,1,3,4]],original[[0,1,3,4]])

    def test_given_heading_or_invalid_neighbor_blocks_repair(self):
        values=np.array([[pose(0,0,yaw=y)] for y in [0,np.pi,0]])
        valid=np.ones((3,1),bool);fixed=np.zeros_like(valid);fixed[1]=True
        corrected,changes=repair_isolated_half_turns(values,valid,fixed)
        self.assertFalse(changes);np.testing.assert_array_equal(corrected,values)
        fixed[:]=False;valid[0]=False
        corrected,changes=repair_isolated_half_turns(values,valid,fixed)
        self.assertFalse(changes);np.testing.assert_array_equal(corrected,values)

    def test_moving_turn_and_angle_wrap_are_unchanged(self):
        for values in [np.array([[pose(2*t,0,yaw=y)] for t,y in enumerate([0,np.pi,0])]),
                       np.array([[pose(0,0,yaw=np.deg2rad(y))] for y in [179,-180,-179]])]:
            corrected,changes=repair_isolated_half_turns(values,np.ones((3,1),bool),np.zeros((3,1),bool))
            self.assertFalse(changes);np.testing.assert_array_equal(corrected,values)

    def test_rear_axle_offset_finds_missed_front_overlap(self):
        size=np.array([5.176,2.297]);bus=np.array([10.,0.,0.]);bus_size=[12.,2.4]
        axle=np.array([0.,0.,0.])
        self.assertGreater(box_separation(axle,size,bus,bus_size),0)
        center=rear_axle_to_box(axle,1.461)
        self.assertLess(box_separation(center,size,bus,bus_size),0)

    def test_clearance_matches_independent_polygon_intersections(self):
        from shapely.geometry import Polygon
        rng=np.random.default_rng(34)
        def polygon(p,s):
            c,t=np.cos(p[2]),np.sin(p[2]);R=np.array([[c,-t],[t,c]])
            return Polygon((np.array([[1,1],[-1,1],[-1,-1],[1,-1]])*s/2)@R.T+p[:2])
        for _ in range(300):
            a,b=rng.normal(size=(2,3))*[5,5,1]
            sa,sb=rng.uniform([2,1],[14,3],size=(2,2))
            self.assertEqual(bool(box_separation(a,sa,b,sb)<=0),polygon(a,sa).intersects(polygon(b,sb)))

    def test_rotated_gaussian_support_and_gradient(self):
        spec=importlib.util.spec_from_file_location('support',REPO/'Omniscenegraph/models/gaussians/support.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        angle=.7;c,t=np.cos(angle),np.sin(angle)
        R=torch.tensor([[[c,-t,0],[t,c,0],[0,0,1]]],dtype=torch.float32)
        means=torch.tensor([[.8,0,0],[4.,0,0]])
        scales=torch.tensor([[2.,.8,.5],[1.,1.,1.]],requires_grad=True)
        sizes=torch.tensor([[2.,2.,2.],[2.,2.,2.]])
        capped,valid=module.confine_gaussian_support(means,scales,R.expand(2,-1,-1),sizes,sigma=3,margin=.08)
        extent=3*torch.sqrt((R.square()*capped[0,None,None,:].square()).sum(-1))
        self.assertTrue((extent[0]+means[0].abs()<=sizes[0]/2+.080001).all())
        self.assertEqual(valid.tolist(),[True,False])
        capped.sum().backward();self.assertTrue(torch.isfinite(scales.grad).all())

    def test_unbound_outside_track_is_stable_and_restored_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scene=root/'scene';(scene/'instances').mkdir(parents=True)
            times=np.array([0,500000,1000000]);values=np.array([[pose(0,0),pose(5,0)] for _ in times])
            save_bundle(root/'bundle',{'timestamps_us':times,'predicted_poses_world':values,
                        'reference_poses_world':values,'valid':np.ones((3,2),bool),
                        'track_tokens':np.array(['__ego__','controlled'])},
                        {'schema_version':1,'dataset':'nuplan','database':'/tmp/example.db','scenario_token':'x'})
            identity={'database':'/tmp/example.db','scenario_token':'x'}
            (scene/'nexus_scene.json').write_text(json.dumps({'status':'prepared','identity':identity,
                'frames':[{'ego_to_world':np.eye(4).tolist()} for _ in times]}))
            info={str(i):{'id':token,'frame_annotations':{'frame_idx':[0,1,2],
                  'obj_to_world':[pose(x,0).tolist() for _ in times]}}
                  for i,token,x in [(7,'controlled',5),(8,'unbound',20)]}
            (scene/'instances/instances_info.json').write_text(json.dumps(info))
            rigid=SimpleNamespace(num_frames=3,num_instances=2,instance_ids=torch.tensor([7,8]),
                instances_trans=torch.nn.Parameter(torch.tensor([[[5.,0,.2],[20.,0,0]],[[5.,0,.3],[20.,0,.5]],[[5.,0,.4],[20.,0,1.]]])),
                instances_quats=torch.nn.Parameter(torch.tensor([[[1.,0,0,0]]*2]*3)),
                instances_fv=torch.ones((3,2),dtype=torch.bool))
            controller=TrajectoryController(root/'bundle',scene,rigid)
            model=root/'ground.json';model.write_text(json.dumps({'status':'validated','scene':str(scene),
                'identity':identity,'coefficients_z_ax_by_c':[0,0,0],
                'supported_xy_bounds_scene':[[-10,-10],[10,10]]}))
            controller.grounding=GroundedRegistration(controller,model,stabilize_all=True)
            self.assertIn(1,controller.grounding.flat_mounts)
            original=[v.clone() for v in [rigid.instances_trans,rigid.instances_quats,rigid.instances_fv]]
            with self.assertRaisesRegex(RuntimeError,'simulate render failure'):
                with controller.edit_frame(0,0):
                    self.assertAlmostEqual(rigid.instances_trans[0,1,2].item(),.5)
                    raise RuntimeError('simulate render failure')
            for actual,expected in zip([rigid.instances_trans,rigid.instances_quats,rigid.instances_fv],original):
                self.assertTrue(torch.equal(actual,expected))


if __name__=='__main__':unittest.main(verbosity=2)
