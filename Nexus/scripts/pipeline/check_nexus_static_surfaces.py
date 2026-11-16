"""Analytic geometry and gradient checks for optional static-surface changes."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]
from nexus_3dgs.surface_lifting import lift_pixels,rectify_gaussians


class StaticSurfaces(unittest.TestCase):
    def test_depth_rectification_keeps_original_projected_ellipses(self):
        C=np.eye(4);C[:3,:3]=Rotation.from_euler('x',110,degrees=True).as_matrix();C[2,3]=2
        K=np.array([[700.,0,480],[0,700,270],[0,0,1.]])
        uv=np.array([[280,420],[480,490],[680,400]],float)
        rays=np.column_stack((uv,np.ones(3)))@np.linalg.inv(K).T
        means=(rays*np.array([5.,12.,20.])[:,None])@C[:3,:3].T+C[:3,3]
        target=-2/(rays@C[:3,:3].T)[:,2]
        logs=np.array([[-2.,-3.,-4.],[-3.,-2.,-3.],[-2.,-4.,-3.]])
        quat=Rotation.from_euler('xyz',[[10,20,30],[0,40,-10],[60,0,10]],degrees=True).as_quat()[:,[3,0,1,2]]
        points,new_logs,new_q=rectify_gaussians(means,logs,quat,K,C,target,np.tile([0.,0.,1.],(3,1)))
        np.testing.assert_allclose(points[:,2],0,atol=1e-9)
        def pixels_covariance(points,logs,quat):
            xyz=(points-C[:3,3])@C[:3,:3]
            projection=xyz[:,:2]/xyz[:,2,None]*700+[480,270]
            R=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix();cov=(R*np.exp(2*logs)[:,None,:])@R.transpose(0,2,1)
            result=[]
            for p,S in zip(xyz,cov):
                J=np.array([[700/p[2],0,-700*p[0]/p[2]**2],[0,700/p[2],-700*p[1]/p[2]**2]])@C[:3,:3].T
                result.append(J@S@J.T)
            return projection,np.array(result)
        p0,c0=pixels_covariance(means,logs,quat);p1,c1=pixels_covariance(points,new_logs,new_q)
        np.testing.assert_allclose(p0,p1,atol=1e-9);np.testing.assert_allclose(c0,c1,rtol=1e-7,atol=1e-7)

    def test_camera_comparison_rejects_wrong_predictions_and_recorded_camera(self):
        import copy
        from nexus_3dgs.render_metadata import matched_camera_overrides
        image={'camera_id':0,'timestamp_us':100,'scene_metadata':{'camera_world':np.eye(4).tolist()}}
        run={'status':'complete','reference':False,'camera_mode':'generated_ego','bundle_sha256':'same',
             'frames':[{'frame_idx':20,'images':[image]}]}
        self.assertIn((20,0,100),matched_camera_overrides(run,'same'))
        with self.assertRaises(ValueError):matched_camera_overrides(run,'different')
        changed=copy.deepcopy(run);changed['reference']=True
        with self.assertRaises(ValueError):matched_camera_overrides(changed,'same')
        changed=copy.deepcopy(run);changed['frames'][0]['images'].append(copy.deepcopy(image))
        with self.assertRaisesRegex(ValueError,'Duplicate'):matched_camera_overrides(changed,'same')

    def test_reconstruction_tail_cannot_change_camera_mount(self):
        from nexus_3dgs.grounding import GroundedRegistration
        from nexus_3dgs.geometry import pose
        registration=object.__new__(GroundedRegistration)
        registration.controller=SimpleNamespace(start_frame=0,bundle={'timestamps_us':np.array([0,100])},
            scene={'identity':{'reconstruction_tail_seconds':7},'frames':[{'timestamp_us':t,'images':[{'camera_id':0,'ego_to_world':pose(x,0).tolist()}]} for t,x in [(0,0),(100,1),(200,2)]]})
        registration.world_plane=np.array([0.,0.,1.,0.]);registration.world_from_scene=np.eye(4);registration.camera_mounts={}
        registration.register_camera(0,np.array([pose(0,0,z=2),pose(1,0,z=2),pose(2,0,z=20)]))
        self.assertAlmostEqual(registration.camera_mounts[0][2,3],2.)

    def test_inverse_depth_fit_rejects_gross_triangulation_outliers(self):
        spec=importlib.util.spec_from_file_location('far_geometry',ROOT/'scripts/pipeline/correct_nexus_far_geometry.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rng=np.random.default_rng(18);uv=rng.uniform([20,10],[240,100],(100,2))
        X=np.column_stack(((uv-[480,270])/500,np.ones(100)))
        truth=np.array([.001,.0002,.006]);depth=1/(X@truth)
        depth[:30]*=rng.uniform(.2,.5,30)
        coef,inlier,error=module.fit_inverse_depth(uv,depth)
        np.testing.assert_allclose(coef,truth,atol=1e-10)
        self.assertGreaterEqual(inlier.sum(),70)
        self.assertLess(np.median(error),1e-8)

    def test_oblique_surface_preserves_camera_pixels_and_footprints(self):
        C=np.eye(4);C[:3,:3]=Rotation.from_euler('x',110,degrees=True).as_matrix();C[2,3]=2
        K=np.array([[700.,0,480],[0,700,270],[0,0,1.]])
        uv=np.array([[280,420],[480,490],[680,400]],float)
        xyz,logs,q,valid=lift_pixels(uv,K,C,[.01,.02,1,0])
        self.assertTrue(valid.all())
        np.testing.assert_allclose(xyz@np.array([.01,.02,1]),0,atol=1e-10)
        camera=(xyz-C[:3,3])@C[:3,:3]
        projected=camera[:,:2]/camera[:,2,None]*700+[480,270]
        np.testing.assert_allclose(projected,uv,atol=1e-9)
        R=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        cov=(R*np.exp(2*logs)[:,None,:])@R.transpose(0,2,1)
        for i,p in enumerate(camera):
            J=np.array([[700/p[2],0,-700*p[0]/p[2]**2],[0,700/p[2],-700*p[1]/p[2]**2]])@C[:3,:3].T
            np.testing.assert_allclose(J@cov[i]@J.T,np.eye(2)*.7**2,atol=.02)

    def test_parallel_and_rear_facing_rays_are_invalid(self):
        C=np.eye(4);C[2,3]=2
        _,_,_,valid=lift_pixels([[0,0]],np.eye(3),C,[0,0,1,0])
        self.assertFalse(valid.any())
        _,_,_,valid=lift_pixels([[0,0]],np.eye(3),C,[0,1,0,-1])
        self.assertFalse(valid.any())

    def test_lidar_gradient_moves_only_supported_point_toward_surface(self):
        from models.gaussians.lidar_surface_prior import LidarSurfacePrior
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'anchors.npz'
            np.savez(path,anchors=np.zeros((2,3),np.float32),normals=np.tile([0,0,1],(2,1)),valid=[True,False])
            model=SimpleNamespace(_means=torch.tensor([[0.,0.,.2],[3.,0.,1.]],requires_grad=True),
                                  _scales=torch.full((2,3),-2.,requires_grad=True),
                                  _quats=torch.tensor([[1.,0,0,0],[1.,0,0,0]],requires_grad=True),quat_act=lambda q:torch.nn.functional.normalize(q,dim=-1))
            prior=LidarSurfacePrior(path,'cpu');loss=sum(prior.losses(model).values());loss.backward()
            self.assertGreater(model._means.grad[0,2].item(),0)
            self.assertEqual(model._means.grad[1].abs().sum().item(),0)
            self.assertGreater(model._scales.grad[0,2].item(),0)
            self.assertTrue(torch.isfinite(model._quats.grad).all())


if __name__=='__main__':unittest.main()
