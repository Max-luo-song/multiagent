"""Fine-tune static geometry and sky, protecting all actors and cameras."""
import argparse
import gc
import json
from pathlib import Path
import random
import runpy
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--resume-from',type=Path,required=True)
    p.add_argument('--anchors',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--run-name',required=True)
    p.add_argument('--train-camera-appearance',action='store_true',help='Revive and train the legacy constant exposure branch; keep camera poses frozen')
    p.add_argument('--appearance-only',action='store_true',help='Preserve the reconstructed background geometry and opacity, fit only radiance')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root),str(root/'Omniscenegraph')]
    import torch
    import yaml
    from models.trainers.base import BasicTrainer
    from models.gaussians.lidar_surface_prior import LidarSurfacePrior
    from datasets.base.pixel_source import ScenePixelSource
    cfg=yaml.safe_load(a.config.read_text());scene=Path(cfg['data']['data_root'])/cfg['data']['scene_idx']
    frames=json.loads((scene/'nexus_scene.json').read_text())['frames']
    import numpy as np
    xy=np.array([f['ego_to_world'] for f in frames])[:,:2,3];moving=set(np.flatnonzero(np.linalg.norm(xy-xy[20],axis=-1)>.1).tolist())
    heldout={80,85,90,95,99};camera_count=len(cfg['data']['pixel_source']['cameras'])
    original_init=BasicTrainer.initialize_optimizer;original_sample=ScenePixelSource.propose_training_image
    def initialize(self):
        if a.train_camera_appearance:
            revived=self.models['Affine'].revive_constant_features()
            print('AFFINE_FEATURES_REVIVED '+str(revived),flush=True)
        original_init(self)
        for name,model in self.models.items():
            for key,param in model.named_parameters():
                allowed=name in ('Background','Sky') or (a.train_camera_appearance and name=='Affine')
                if a.appearance_only and name=='Background':allowed=key in ('_features_dc','_features_rest')
                param.requires_grad_(allowed)
        for name in self.gaussian_classes:assert self.models[name].ctrl_cfg.refine_interval>self.num_iters
        self.background_surface_prior=LidarSurfacePrior(a.anchors,torch.device('cuda'))
        print('STATIC_ONLY_OPTIMIZER actors_camera_frozen=True affine_trainable='+str(a.train_camera_appearance),flush=True)
    def sample(self,candidate_indices=None):
        if not hasattr(self,'_static_motion_candidates'):
            allowed=[int(i) for i in candidate_indices if not(int(i)%camera_count==0 and int(i)//camera_count in heldout)]
            self._static_motion_candidates={}
            for c in range(camera_count):
                all_views=[i for i in allowed if i%camera_count==c]
                moving_views=[i for i in all_views if i//camera_count in moving]
                self._static_motion_candidates[c]=(all_views,moving_views)
        c=0 if random.random()<.65 else random.choice(list(range(1,camera_count)))
        all_views,moving_views=self._static_motion_candidates[c]
        return random.choice(moving_views if moving_views and random.random()<.65 else all_views)
    BasicTrainer.initialize_optimizer=initialize;ScenePixelSource.propose_training_image=sample
    sys.argv=['train.py','--config_file',str(a.config),'--resume_from',str(a.resume_from),
              '--output_root',str(a.output_root),'--project','nuplan','--run_name',a.run_name]
    try:runpy.run_path(str(root/'Omniscenegraph/tools/train.py'),run_name='__main__')
    finally:
        BasicTrainer.initialize_optimizer=original_init;ScenePixelSource.propose_training_image=original_sample
    checkpoint=a.output_root/'nuplan'/a.run_name/'checkpoint_final.pth'
    before=torch.load(a.resume_from,map_location='cpu',weights_only=False)['models']
    after=torch.load(checkpoint,map_location='cpu',weights_only=False)['models']
    from scripts.pipeline.merge_nexus_vehicle_asset import equal
    for name,value in before.items():
        if name not in ('Background','Sky') and not(a.train_camera_appearance and name=='Affine'):equal(value,after[name])
    if a.appearance_only:
        for key,value in before['Background'].items():
            if key not in ('_features_dc','_features_rest'):equal(value,after['Background'][key])
    report={'status':'passed','actors_cameras_bitwise_preserved':True,'affine_bitwise_preserved':not a.train_camera_appearance,
            'background_geometry_opacity_bitwise_preserved':a.appearance_only,
            'moving_view_sampling_probability':.65,'front_view_sampling_probability':.65,
            'fine_tuning_held_out_front_frames':sorted(heldout),
            'holdout_scope':'excluded from this fine-tune; base checkpoint previously saw all frames, not an independent unseen-scene benchmark'}
    (checkpoint.parent/'protected_parameters.json').write_text(json.dumps(report,indent=2)+'\n')
    print('STATIC_TRAIN_OK '+str(checkpoint),flush=True);del before,after;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
