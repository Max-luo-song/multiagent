"""Train one new scene, saving its own AABB and explicit validation scope."""
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Omniscenegraph'),str(ROOT)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--run-name',required=True)
    a=p.parse_args()
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(a.config);scene=Path(cfg.data.data_root)/cfg.data.scene_idx
    coverage=a.config.parent/'coverage.json'
    if not coverage.exists():
        saved_argv=sys.argv
        sys.argv=['audit_nexus_pilot_coverage.py','--scene',str(scene),'--bundle',str(a.config.parent/'bundle'),'--output',str(coverage)]
        try:runpy.run_path(str(ROOT/'scripts/pipeline/audit_nexus_pilot_coverage.py'),run_name='__main__')
        finally:sys.argv=saved_argv
    import datasets.driving_dataset as module
    import torch
    from models.trainers.base import BasicTrainer
    torch.set_num_threads(8)
    directory=a.output_root/'nuplan'/a.run_name;directory.mkdir(parents=True,exist_ok=False)
    original=module.DrivingDataset
    original_optimizer=BasicTrainer.initialize_optimizer
    protected={}
    def initialize_optimizer(self):
        original_optimizer(self)
        for model_name,model in self.models.items():
            for key,param in model.named_parameters():
                if model_name=='CamPose' or (model_name=='RigidNodes' and key in ('instances_trans','instances_quats')):
                    param.requires_grad_(False);protected[(model_name,key)]=param.detach().cpu().clone()
        print('PILOT_POSES_FROZEN_BY_REQUIRES_GRAD',list(protected),flush=True)
    BasicTrainer.initialize_optimizer=initialize_optimizer
    def construct(*args,**kwargs):
        data=original(*args,**kwargs)
        (directory/'scene_geometry.json').write_text(json.dumps({'scene_aabb':data.get_aabb().reshape(2,3).cpu().tolist(),
            'test_timesteps':list(map(int,data.test_timesteps)),
            'scope':'RGB gradient holdout only; initialization and LiDAR use this scene. Engineering reconstruction check, not unseen-scene benchmark.'},indent=2)+'\n')
        return data
    module.DrivingDataset=construct
    sys.argv=['train.py','--config_file',str(a.config),'--output_root',str(a.output_root),'--project','nuplan','--run_name',a.run_name]
    try:runpy.run_path(str(ROOT/'Omniscenegraph/tools/train.py'),run_name='__main__')
    finally:
        module.DrivingDataset=original
        BasicTrainer.initialize_optimizer=original_optimizer
    state=torch.load(directory/'checkpoint_final.pth',map_location='cpu',weights_only=False)['models']
    for (name,key),value in protected.items():assert torch.equal(value,state[name][key]),(name,key)
    (directory/'pose_protection.json').write_text(json.dumps({'status':'passed','camera_and_recorded_vehicle_poses_bitwise_unchanged':True,'freeze_method':'requires_grad=False; positive configured learning rates retained to avoid log(0) schedules'},indent=2)+'\n')
    print('PILOT_TRAIN_COMPLETE',directory,flush=True)


if __name__=='__main__':main()
