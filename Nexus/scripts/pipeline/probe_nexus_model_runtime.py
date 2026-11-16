"""Load verified Nexus weights and execute a real cached-scene CUDA sample."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts/pipeline'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report={'status':'started','python':sys.executable,'checks':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
    save();os.environ.update(MPLBACKEND='Agg',WANDB_MODE='disabled',NUPLAN_DATA_STORE='local')
    try:
        print('NEXUS_PROBE imports',flush=True)
        import numpy as np
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
        from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
        from run_repro_demo import CHECKPOINT_SHA256,file_sha256,model_state_dict
        torch.set_num_threads(2)
        report['versions']={'torch':torch.__version__,'numpy':np.__version__}
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        checkpoint=REPO.parent/'checkpoints/nuplan.ckpt'
        if file_sha256(checkpoint)!=CHECKPOINT_SHA256:raise ValueError('Official checkpoint checksum changed')
        model=instantiate(OmegaConf.load(a.source_run/'config.yaml').model)
        model.load_state_dict(model_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)),strict=True)
        model=model.cuda().eval().requires_grad_(False);versions=[x._version for x in model.parameters()]
        report['checks'].append('strict_official_checkpoint_load')
        path=a.source_run/'raw_features/096_a89c1931e7b25976.pt'
        raw=torch.load(path,map_location='cpu',weights_only=False)
        scene=SceneTensor(**{k:v.numpy().copy() for k,v in raw.items()})
        scene=ResetVAugmentor().augment({'scene_tensor':scene},{})[0]['scene_tensor']
        features=SceneTensor(**{k:torch.from_numpy(getattr(scene,k).copy()).unsqueeze(0).cuda()
                              for k in ('tensor','validity','road_graph','road_graph_validity')})
        print('NEXUS_PROBE actual CUDA inference',flush=True);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        keep=torch.zeros_like(features.tensor[...,:8]);keep[:,:,:5]=1
        with torch.inference_mode():result=model.forward_inference({'scene_tensor':features,'task_mask':keep},return_intermidates=False)
        output=result['sampled_tensor'];valid=features.validity.bool()
        if not torch.isfinite(output[valid]).all():raise RuntimeError('Non-finite sampled states')
        if versions!=[x._version for x in model.parameters()]:raise RuntimeError('Weights changed during runtime probe')
        report['checks'].append('cached_nuplan_feature_cuda_inference_finite')
        report.update(status='passed',gpu=torch.cuda.get_device_name(),output_shape=list(output.shape),
                      valid_state_count=int(valid.sum()),model_weights_unchanged=True)
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:save()
    print('NEXUS_MODEL_RUNTIME_OK',flush=True)


if __name__=='__main__':main()
