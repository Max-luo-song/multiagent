"""Fit one vehicle while freezing every other asset, with exact reload checks."""
import argparse
import gc
import json
from pathlib import Path
import runpy
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--resume-from',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--run-name',required=True)
    p.add_argument('--track-token',required=True);p.add_argument('--jobs',type=Path)
    p.add_argument('--finalize',type=Path)
    p.add_argument('--appearance-indices',type=Path,help='Fit only colors/opacity at these Gaussian indices; all geometry and poses stay frozen')
    p.add_argument('--canonical-pose',type=Path,help='Compensate high-degree color at this reference pose, then fit degree 1')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(root),str(root/'Omniscenegraph'),str(root/'Omniscenegraph/sim_render/cam')]
    import torch
    import yaml
    from models.trainers.base import BasicTrainer
    from datasets.base.pixel_source import ScenePixelSource
    cfg=yaml.safe_load(a.config.read_text());scene=Path(cfg['data']['data_root'])/cfg['data']['scene_idx']
    info=json.loads((scene/'instances/instances_info.json').read_text())
    dataset_id=next(int(i) for i,r in info.items() if r['id']==a.track_token)
    original_initialize=BasicTrainer.initialize_optimizer;original_sample=ScenePixelSource.propose_training_image
    policy={}
    def initialize_frozen_optimizer(self):
        original_initialize(self)
        for model in self.models.values():
            for param in model.parameters():param.requires_grad_(False)
        rigid=self.models['RigidNodes']
        index=int(torch.nonzero(rigid.instance_ids==dataset_id,as_tuple=True)[0].item())
        selected=rigid.point_ids[:,0]==index
        if a.appearance_indices:
            import numpy as np
            chosen=torch.from_numpy(np.load(a.appearance_indices)).long().to(rigid._means.device)
            assert selected[chosen].all();selected=torch.zeros_like(selected);selected[chosen]=True
            if a.canonical_pose:
                from models.gaussians.basics import spherical_harmonics
                pose=json.loads(a.canonical_pose.read_text());B=torch.tensor(pose['body_to_scene'],dtype=torch.float32,device=rigid._means.device);C=torch.tensor(pose['camera_to_scene'],dtype=torch.float32,device=rigid._means.device)
                with torch.no_grad():
                    world=rigid._means[chosen]@B[:3,:3].T+B[:3,3];direction=torch.nn.functional.normalize(world-C[:3,3],dim=-1)
                    sh=torch.cat((rigid._features_dc[chosen,None,:],rigid._features_rest[chosen]),dim=1)
                    rigid._features_dc[chosen]+=(spherical_harmonics(3,direction,sh)-spherical_harmonics(1,direction,sh))/.28209479177387814
                    rigid._features_rest[chosen,3:]=0
        names=('_opacities','_features_dc','_features_rest') if a.appearance_indices else ('_means','_scales','_quats','_opacities','_features_dc','_features_rest')
        for name in names:
            param=getattr(rigid,name);param.requires_grad_(True)
            mask=selected.reshape((-1,)+(1,)*(param.ndim-1)).to(param.dtype)
            if a.canonical_pose and name=='_features_rest':
                mask=mask.expand_as(param).clone();mask[:,3:]=0
            param.register_hook(lambda grad,mask=mask: grad*mask)
        for name in (() if a.appearance_indices else ('instances_trans','instances_quats')):
            param=getattr(rigid,name);param.requires_grad_(True)
            mask=torch.zeros_like(param);mask[:,index]=1
            param.register_hook(lambda grad,mask=mask: grad*mask)
        for name in self.gaussian_classes:
            assert self.models[name].ctrl_cfg.refine_interval>self.num_iters
        policy.update(model_index=index,selected_gaussians=int(selected.sum()),track_token=a.track_token,
                      other_parameters_frozen=True,gaussian_add_remove_and_opacity_reset_disabled=True,appearance_only=bool(a.appearance_indices))
        print('BUS_ONLY_OPTIMIZER '+json.dumps(policy),flush=True)
    BasicTrainer.initialize_optimizer=initialize_frozen_optimizer
    if a.appearance_indices:
        def sample(self,candidate_indices=None):
            import random
            allowed=[int(i) for i in candidate_indices if int(i)%len(cfg['data']['pixel_source']['cameras'])==0 and int(i)//len(cfg['data']['pixel_source']['cameras']) not in {80,85,90,95,99}]
            return random.choice(allowed)
        ScenePixelSource.propose_training_image=sample
    sys.argv=['train.py','--config_file',str(a.config),'--resume_from',str(a.resume_from),
              '--output_root',str(a.output_root),'--project','nuplan','--run_name',a.run_name]
    try:
        runpy.run_path(str(root/'Omniscenegraph/tools/train.py'),run_name='__main__')
    finally:
        BasicTrainer.initialize_optimizer=original_initialize
        ScenePixelSource.propose_training_image=original_sample
    checkpoint=a.output_root/'nuplan'/a.run_name/'checkpoint_final.pth'
    before=torch.load(a.resume_from,map_location='cpu',weights_only=False)['models']
    after=torch.load(checkpoint,map_location='cpu',weights_only=False)['models']
    from scripts.pipeline.merge_nexus_vehicle_asset import equal
    for name,model in before.items():
        if name!='RigidNodes':equal(model,after[name])
    old,new=before['RigidNodes'],after['RigidNodes'];index=policy['model_index']
    equal(old['points_ids'],new['points_ids'])
    if a.appearance_indices:
        import numpy as np
        chosen=torch.from_numpy(np.load(a.appearance_indices)).long();untouched=torch.ones(len(old['_means']),dtype=torch.bool);untouched[chosen]=False
        for key,value in old.items():
            if key in ('_features_dc','_features_rest','_opacities'):equal(value[untouched],new[key][untouched])
            else:equal(value,new[key])
        if a.canonical_pose:assert torch.count_nonzero(new['_features_rest'][chosen,3:])==0
    other=old['points_ids'].reshape(-1)!=index
    for key,value in old.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0]==len(other):
            equal(value[other],new[key][other])
        elif key in ('instances_trans','instances_quats'):
            mask=torch.arange(value.shape[1])!=index;equal(value[:,mask],new[key][:,mask])
        else:equal(value,new[key])
    policy.update(status='passed',reloaded_checkpoint_verified=True,checkpoint=str(checkpoint))
    (checkpoint.parent/'frozen_parameter_verification.json').write_text(json.dumps(policy,indent=2)+'\n')
    print('BUS_ONLY_FREEZE_VERIFIED',flush=True)
    del before,after,old,new
    gc.collect();torch.cuda.empty_cache()
    if a.jobs:
        sys.argv=['render_nexus_jobs.py','--jobs',str(a.jobs)]
        runpy.run_path(str(root/'Omniscenegraph/sim_render/cam/render_nexus_jobs.py'),run_name='__main__')
    if a.finalize:runpy.run_path(str(a.finalize),run_name='__main__')


if __name__=='__main__':main()
