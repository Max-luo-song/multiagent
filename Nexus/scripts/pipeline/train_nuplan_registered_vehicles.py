"""Refit vehicle radiance/opacity against real RGB with all geometry frozen."""
import argparse, json, random, runpy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'Omniscenegraph')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['config', 'checkpoint', 'output-root']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--background-only', action='store_true', help='Train background/sky with fixed rendered vehicles and genuine interior masks')
    p.add_argument('--poses-only',action='store_true',help='Fit only measured vehicle body poses; all Gaussian assets and camera parameters stay fixed')
    p.add_argument('--refit-assets',action='store_true',help='Refit Gaussian geometry/radiance with registered measured body poses fixed')
    a = p.parse_args()
    import torch, yaml
    from models.trainers.base import BasicTrainer
    from datasets.base.pixel_source import ScenePixelSource
    cfg = yaml.safe_load(a.config.read_text())
    scene = Path(cfg['data']['data_root']) / cfg['data']['scene_idx']
    mask = json.loads((scene / 'vehicle_interior_masks/manifest.json').read_text())
    assert mask['status'] == 'complete' and 'LiDAR' in mask['kind']
    assert cfg['trainer']['losses']['vehicle_ownership']['foreground_only_rgb']
    original_optimizer = BasicTrainer.initialize_optimizer
    original_sample = ScenePixelSource.propose_training_image
    original_step=BasicTrainer.optimizer_step
    assert sum([a.background_only,a.poses_only,a.refit_assets])<=1
    appearance = ('_features_dc', '_features_rest', '_opacities')

    def initialize(self):
        original_optimizer(self)
        # The source checkpoint's step is useful for SH activation but a new
        # appearance-only optimizer must run its own requested iteration count.
        self.step = 0
        for name, model in self.models.items():
            for param in model.parameters():
                param.requires_grad_(False)
        if a.poses_only:
            rigid=self.models['RigidNodes'];self._original_body_poses={key:getattr(rigid,key).detach().clone() for key in ['instances_trans','instances_quats']}
            keep=torch.ones(rigid.num_frames,dtype=torch.bool,device=self.device);keep[10::10]=False;self._pose_train_frames=keep
            for key in ['instances_trans','instances_quats']:
                param=getattr(rigid,key);param.requires_grad_(True);mask=keep[:,None,None].to(param.dtype);param.register_hook(lambda grad,mask=mask:grad*mask)
        elif a.background_only:
            for name in ['Background', 'Sky']:
                for param in self.models[name].parameters():param.requires_grad_(True)
        else:
            for key in (('_means','_scales','_quats')+appearance if a.refit_assets else appearance):
                getattr(self.models['RigidNodes'], key).requires_grad_(True)
        for name in self.gaussian_classes:
            assert self.models[name].ctrl_cfg.refine_interval > self.num_iters
            assert self.models[name].ctrl_cfg.reset_alpha_interval > self.num_iters
        print('REGISTERED_COMPOSITE_OPTIMIZER background_only='+str(a.background_only), flush=True)

    def optimizer_step(self):
        original_step(self)
        if not a.poses_only:return
        with torch.no_grad():
            rigid=self.models['RigidNodes'];keep=self._pose_train_frames;initial=self._original_body_poses
            delta=rigid.instances_trans[keep]-initial['instances_trans'][keep]
            delta*=torch.clamp(.25/torch.linalg.vector_norm(delta,dim=-1,keepdim=True).clamp_min(1e-12),max=1)
            rigid.instances_trans[keep]=initial['instances_trans'][keep]+delta
            q=torch.nn.functional.normalize(rigid.instances_quats[keep],dim=-1);q0=torch.nn.functional.normalize(initial['instances_quats'][keep],dim=-1)
            q*=torch.where((q*q0).sum(-1,keepdim=True)>=0,1.,-1.)
            angle=torch.acos((q*q0).sum(-1,keepdim=True).clamp(-1,1));fraction=torch.clamp(.0261799388/angle.clamp_min(1e-12),max=1)
            rigid.instances_quats[keep]=torch.nn.functional.normalize(q0+(q-q0)*fraction,dim=-1)

    def sample(self, candidate_indices=None):
        if not hasattr(self, '_registered_vehicle_candidates'):
            cameras = len(cfg['data']['pixel_source']['cameras'])
            self._registered_vehicle_candidates = {
                c: [int(i) for i in candidate_indices if int(i) % cameras == c]
                for c in range(cameras)}
        c = 0 if random.random() < .7 else random.choice([1, 2, 3])
        return random.choice(self._registered_vehicle_candidates[c])

    BasicTrainer.initialize_optimizer = initialize
    BasicTrainer.optimizer_step=optimizer_step
    ScenePixelSource.propose_training_image = sample
    sys.argv = ['train.py', '--config_file', str(a.config), '--resume_from', str(a.checkpoint),
                '--output_root', str(a.output_root), '--project', 'nuplan', '--run_name', a.run_name]
    try:
        runpy.run_path(str(ROOT / 'Omniscenegraph/tools/train.py'), run_name='__main__')
    finally:
        BasicTrainer.initialize_optimizer = original_optimizer
        BasicTrainer.optimizer_step=original_step
        ScenePixelSource.propose_training_image = original_sample
    directory = a.output_root / 'nuplan' / a.run_name
    before = torch.load(a.checkpoint, map_location='cpu', weights_only=False)['models']
    final_state=torch.load(directory / 'checkpoint_final.pth', map_location='cpu', weights_only=False);after=final_state['models']
    if a.poses_only:
        import numpy as np
        from scipy.spatial.transform import Rotation
        from nexus_3dgs.geometry import quaternion_matrix,matrix_quaternion
        oldrigid,newrigid=before['RigidNodes'],after['RigidNodes'];nframes=len(newrigid['instances_trans']);held=np.arange(10,nframes,10);train=np.setdiff1d(np.arange(nframes),held)
        for key in ['instances_trans','instances_quats']:assert torch.equal(newrigid[key][held],oldrigid[key][held])
        delta=(newrigid['instances_trans']-oldrigid['instances_trans']).numpy()
        for m in range(delta.shape[1]):
            for d in range(3):delta[held,m,d]=np.interp(held,train,delta[train,m,d])
            rot=np.array([quaternion_matrix(q.numpy()) for q in newrigid['instances_quats'][:,m]])
            rot0=np.array([quaternion_matrix(q.numpy()) for q in oldrigid['instances_quats'][:,m]])
            residual=Rotation.from_matrix(rot@rot0.transpose(0,2,1)).as_rotvec()
            values=np.stack([np.interp(held,train,residual[train,d]) for d in range(3)],-1)
            for f,R in zip(held,Rotation.from_rotvec(values).as_matrix()):newrigid['instances_quats'][f,m]=torch.tensor(matrix_quaternion(R@rot0[f]))
        newrigid['instances_trans'][held]=oldrigid['instances_trans'][held]+torch.tensor(delta[held])
        torch.save(final_state,directory/'checkpoint_final.pth')
    changed = []
    for name in before:
        for key, value in before[name].items():
            allowed=(name in ['Background','Sky']) if a.background_only else (name=='RigidNodes' and key in (['instances_trans','instances_quats'] if a.poses_only else (('_means','_scales','_quats')+appearance if a.refit_assets else appearance)))
            if allowed:
                if not torch.equal(value, after[name][key]):
                    changed.append(key)
            else:
                assert torch.equal(value, after[name][key]), (name, key)
    assert changed
    report = {'status': 'passed', 'parameters_changed': changed, 'background_only': a.background_only,
              'all_parameters_outside_allowed_training_models_bit_exact': True,
              'allowed_models': ['Background', 'Sky'] if a.background_only else (['measured body poses only; held-out pose residuals interpolated from neighboring training frames'] if a.poses_only else (['RigidNodes Gaussian geometry and appearance; measured body poses fixed'] if a.refit_assets else ['RigidNodes appearance and opacity'])),
              'real_lidar_derived_vehicle_masks': str(scene / 'vehicle_interior_masks/manifest.json'),
              'heldout_images_not_used_in_gradient_training': True,
              'validation_scope': 'RGB holdout; poses use all LiDAR; reused vehicle assets were fitted to the original full scene',
              'nexus_trained_or_trajectories_changed': False}
    (directory / 'vehicle_refit_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print('REGISTERED_VEHICLES_TRAIN_COMPLETE ' + str(directory), flush=True)


if __name__ == '__main__':
    main()
