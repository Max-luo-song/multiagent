"""Behavioral checks for identity-preserving, trainable camera appearance."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]
import torch
from models.modules import AffineTransform


class AppearanceChecks(unittest.TestCase):
    def test_optimized_camera_and_sky_rays_point_to_the_same_world_pixels(self):
        from models.trainers.base import BasicTrainer
        from types import SimpleNamespace
        from datasets.base.pixel_source import get_rays
        angle=torch.tensor(.3);R=torch.tensor([[angle.cos(),0,angle.sin()],[0,1,0],[-angle.sin(),0,angle.cos()]])
        raw=torch.eye(4);corrected=raw.clone();corrected[:3,:3]=R;corrected[:3,3]=torch.tensor([2.,1.,.5]);K=torch.tensor([[100.,0,40.],[0,100.,30.],[0,0,1.]])
        u=torch.tensor([0,30,70]);v=torch.tensor([0,20,40]);origins,directions,_=get_rays(u,v,raw,K)
        image={'origins':origins,'viewdirs':directions};cam=SimpleNamespace(camtoworlds_gt=raw,camtoworlds=corrected)
        expected_origins,expected_directions,_=get_rays(u,v,corrected,K)
        actual=BasicTrainer.sky_image_infos(SimpleNamespace(render_cfg={'synchronize_sky_camera':True}),image,cam)
        torch.testing.assert_close(actual['origins'],expected_origins);torch.testing.assert_close(actual['viewdirs'],expected_directions)
        self.assertIs(BasicTrainer.sky_image_infos(SimpleNamespace(render_cfg={}),image,cam),image)

    def test_actor_color_preserved_under_different_background_exposure(self):
        from nexus_3dgs.replay import change_color_basis
        torch.manual_seed(7);rgb=torch.rand(300,3)
        source=torch.cat((torch.eye(3)+.03*torch.randn(3,3),.03*torch.randn(3,1)),1)
        target=torch.cat((torch.eye(3)+.04*torch.randn(3,3),.02*torch.randn(3,1)),1)
        converted=change_color_basis(rgb,source,target)
        torch.testing.assert_close(converted@target[:,:3].T+target[:,3],rgb@source[:,:3].T+source[:,3],atol=2e-7,rtol=1e-6)

    def test_initial_transform_is_exact_identity_for_every_image(self):
        m=AffineTransform('Affine',4,device=torch.device('cpu'))
        expected=torch.cat((torch.eye(3),torch.zeros(3,1)),1).expand(4,-1,-1)
        torch.testing.assert_close(m({'img_idx':torch.arange(4)}),expected,atol=0,rtol=0)

    def test_can_learn_different_exposure_for_identical_rgb_in_two_images(self):
        torch.manual_seed(24);m=AffineTransform('Affine',2,device=torch.device('cpu'));optimizer=torch.optim.Adam(m.parameters(),lr=.01);rgb=torch.full((2,3),.4);target=torch.tensor([[.2,.2,.2],[.6,.6,.6]])
        for _ in range(250):
            A=m({'img_idx':torch.arange(2)});prediction=(A[:,:,:3]@rgb[...,None])[...,0]+A[:,:,3];loss=(prediction-target).square().mean();optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertLess(float(loss),1e-5)
        self.assertGreater(float((m.embedding.weight[0]-m.embedding.weight[1]).norm()),.01)

    def test_legacy_load_and_opt_in_revival_preserve_rendered_transforms(self):
        m=AffineTransform('Affine',3,device=torch.device('cpu'));old={k:torch.zeros_like(v) for k,v in m.state_dict().items()};old['decoder.2.bias']=torch.linspace(-.03,.03,12);m.load_state_dict(old)
        before=m({'img_idx':torch.arange(3)}).detach().clone();self.assertTrue(m.revive_constant_features());after=m({'img_idx':torch.arange(3)}).detach();self.assertTrue(torch.equal(before,after));self.assertFalse(m.revive_constant_features())

if __name__=='__main__':unittest.main()
