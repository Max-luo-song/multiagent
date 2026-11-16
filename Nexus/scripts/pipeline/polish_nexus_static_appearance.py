"""Fit static appearance with geometry, vehicles, cameras and exposure frozen."""
import argparse,json,random,runpy,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','config','reference','output-root'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--run-name',required=True);a=p.parse_args()
    ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]
    import torch,numpy as np,yaml
    from models.trainers.base import BasicTrainer
    from datasets.base.pixel_source import ScenePixelSource
    from models.gaussians.basics import spherical_harmonics
    cfg=yaml.safe_load(a.config.read_text());scene=Path(cfg['data']['data_root'])/cfg['data']['scene_idx']
    first=json.loads((scene/'nexus_scene.json').read_text())['frames'][0]['ego_to_world']
    record=json.loads((a.reference/'run.json').read_text())['frames'][0]['images'][0]['scene_metadata'];C=np.linalg.inv(first)@record['camera_world']
    original_init=BasicTrainer.initialize_optimizer;original_sample=ScenePixelSource.propose_training_image
    def initialize(self):
        bg=self.models['Background'];center=torch.tensor(C[:3,3],device=bg._means.device,dtype=bg._means.dtype)
        with torch.no_grad():
            for start in range(0,bg.num_points,100000):
                end=start+100000;direction=bg._means[start:end]-center;direction/=direction.norm(dim=-1,keepdim=True)
                sh=torch.cat((bg._features_dc[start:end,None,:],bg._features_rest[start:end]),dim=1)
                delta=spherical_harmonics(3,direction,sh)-spherical_harmonics(1,direction,sh)
                bg._features_dc[start:end]+=delta/.28209479177387814
            bg._features_rest[:,3:]=0
        original_init(self)
        for name,model in self.models.items():
            for key,param in model.named_parameters():
                param.requires_grad_(name=='Sky' or (name=='Background' and key in ['_features_dc','_features_rest','_opacities']))
        def first_order_only(gradient):
            gradient=gradient.clone();gradient[:,3:]=0;return gradient
        bg._features_rest.register_hook(first_order_only)
        print('STATIC_APPEARANCE_ONLY geometry_and_all_traffic_camera_affine_frozen=True SH_degree=1',flush=True)
    def sample(self,candidate_indices=None):
        if not hasattr(self,'_appearance_candidates'):
            allowed=[int(i) for i in candidate_indices if not(int(i)%3==0 and int(i)//3 in {80,85,90,95,99})]
            self._appearance_candidates={c:([i for i in allowed if i%3==c],[i for i in allowed if i%3==c and i//3>=75]) for c in range(3)}
        c=0 if random.random()<.8 else random.choice([1,2]);all_views,moving=self._appearance_candidates[c]
        return random.choice(moving if moving and random.random()<.5 else all_views)
    BasicTrainer.initialize_optimizer=initialize;ScenePixelSource.propose_training_image=sample
    sys.argv=['train.py','--config_file',str(a.config),'--resume_from',str(a.checkpoint),'--output_root',str(a.output_root),'--project','nuplan','--run_name',a.run_name]
    try:runpy.run_path(str(ROOT/'Omniscenegraph/tools/train.py'),run_name='__main__')
    finally:BasicTrainer.initialize_optimizer=original_init;ScenePixelSource.propose_training_image=original_sample
    output=a.output_root/'nuplan'/a.run_name/'checkpoint_final.pth';before=torch.load(a.checkpoint,map_location='cpu',weights_only=False);after=torch.load(output,map_location='cpu',weights_only=False)
    from scripts.pipeline.merge_nexus_vehicle_asset import equal
    for name,model in before['models'].items():
        if name not in ['Background','Sky']:equal(model,after['models'][name])
    for key in ['_means','_scales','_quats']:equal(before['models']['Background'][key],after['models']['Background'][key])
    assert torch.count_nonzero(after['models']['Background']['_features_rest'][:,3:])==0
    report={'status':'passed','geometry_vehicles_cameras_affine_bitwise_preserved':True,'background_rendered_SH_degree':1,'heldout_front_frames':[80,85,90,95,99],
            'scope':'static color/opacity and sky only, original real observations, no generated-view RGB targets'}
    (output.parent/'protected_parameters.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
