"""Prepare, reconstruct and validate each frozen pilot scene sequentially."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',type=Path,required=True);a=p.parse_args()
    plan_path=a.pilot/'frozen_scenes.json';plan=json.loads(plan_path.read_text());assert plan['status']=='frozen'
    state_path=a.pilot/'run.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'status':'started','frozen_plan_sha256':sha(plan_path),'scenes':{}}
    assert state['frozen_plan_sha256']==sha(plan_path)
    source=ROOT/'artifacts/nuplan_evaluation/independent256_v1';manifest=json.loads((source/'run.json').read_text())
    def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
    def launch(script,args,log):
        command=['bash',str(ROOT/'scripts/pipeline/nexus_3dgs.sh'),'script',str(ROOT/'scripts/pipeline'/script),*map(str,args)]
        state['active_command']=command;save();print('PILOT_COMMAND',script,str(log),flush=True)
        with log.open('w') as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    state['status']='running';save()
    try:
        for row in plan['scenarios']:
            key=str(row['index']);entry=state['scenes'].setdefault(key,{'token':row['token'],'category':row['pilot_category'],'status':'started'})
            if entry['status']=='complete':continue
            work=a.pilot/f'{row["index"]:03d}_{row["pilot_category"]}';work.mkdir(exist_ok=True)
            bundle=work/'bundle';entry['work']=str(work)
            if not (bundle/'manifest.json').exists():
                import torch
                from nexus_3dgs.bundle import export_arrays,save_bundle
                record=manifest['records'][f'released_test_recipe:{row["index"]}:0'];prediction=source/record['prediction_file'];assert sha(prediction)==record['sha256']
                payload=torch.load(prediction,map_location='cpu',weights_only=False)
                arrays,meta=export_arrays(payload,{'database':row['database'],'scenario_token':row['token'],'prediction':str(prediction),'prediction_sha256':record['sha256'],'seed':0},task='intent_conditioned_behavior_pred')
                save_bundle(bundle,arrays,meta);entry['source_prediction_sha256']=record['sha256'];save()
            scene=work/'processed'/(Path(row['database']).stem+'__'+row['token'])
            prepared=scene/'nexus_scene.json'
            if not prepared.exists() or json.loads(prepared.read_text())['status']!='prepared':
                entry['status']='preparing';save()
                launch('prepare_nuplan_3dgs.py',['--bundle',bundle,'--output-root',work/'processed','--cameras',0,1,2,4,'--fps',10,'--workers',2],work/'prepare.log')
            entry['scene']=str(scene)
            config=work/'pilot_config.yaml'
            if not config.exists():
                from omegaconf import OmegaConf
                cfg=OmegaConf.load(scene/'train_config.yaml');cfg.trainer.optim.num_iters=5000;cfg.trainer.render.synchronize_sky_camera=True
                cfg.data.pixel_source.test_image_stride=10
                cfg.logging.vis_freq=100000000;cfg.logging.print_freq=250;cfg.logging.saveckpt_freq=2500
                cfg.render.render_full=False;cfg.render.render_test=False
                OmegaConf.save(cfg,config)
            from omegaconf import OmegaConf
            cfg=OmegaConf.load(config)
            if cfg.model.RigidNodes.optim.ins_rotation.lr==0:
                # Preserve the failed first configuration as evidence.
                config=work/'pilot_config_fixed_poses.yaml'
                if not config.exists():
                    cfg.model.CamPose.optim.all.lr=1e-5
                    cfg.model.RigidNodes.optim.ins_rotation.lr=1e-5;cfg.model.RigidNodes.optim.ins_rotation.lr_final=5e-6
                    cfg.model.RigidNodes.optim.ins_translation.lr=.0005;cfg.model.RigidNodes.optim.ins_translation.lr_final=.0001
                    cfg.logging.vis_freq=100000000
                    OmegaConf.save(cfg,config)
            entry['config_sha256']=sha(config)
            trained=work/'reconstruction'/'nuplan'/'initial_5000';checkpoint=trained/'checkpoint_final.pth'
            previous=Path(entry.get('checkpoint',checkpoint))
            finished=sorted((work/'reconstruction'/'nuplan').glob('initial_5000*/checkpoint_final.pth')) if (work/'reconstruction'/'nuplan').exists() else []
            if previous.is_file():checkpoint=previous;trained=checkpoint.parent
            elif finished:checkpoint=finished[-1];trained=checkpoint.parent
            if not checkpoint.exists():
                entry['status']='reconstructing';save()
                if trained.exists():
                    n=2
                    while (trained.parent/f'initial_5000_attempt{n}').exists():n+=1
                    trained=trained.parent/f'initial_5000_attempt{n}';checkpoint=trained/'checkpoint_final.pth'
                launch('train_nexus_3dgs_pilot.py',['--config',config,'--output-root',work/'reconstruction','--run-name',trained.name],work/(trained.name+'.log'))
            assert checkpoint.exists();entry['checkpoint']=str(checkpoint);entry['checkpoint_sha256']=sha(checkpoint)
            evaluation=work/'evaluation'
            if entry.get('evaluation') and (Path(entry['evaluation'])/'validation.json').exists():evaluation=Path(entry['evaluation'])
            if not (evaluation/'validation.json').exists():
                entry['status']='evaluating';save()
                if evaluation.exists():
                    n=2
                    while (work/f'evaluation_attempt{n}').exists():n+=1
                    evaluation=work/f'evaluation_attempt{n}'
                launch('evaluate_nexus_3dgs_pilot.py',['--checkpoint',checkpoint,'--scene',scene,'--bundle',bundle,'--output',evaluation],work/(evaluation.name+'.log'))
            validation=json.loads((evaluation/'validation.json').read_text());assert validation['status']=='complete'
            entry.update(status='complete',evaluation=str(evaluation),metrics=validation['metrics'],unbound_prediction_tokens=validation['unbound_prediction_tokens'])
            save();print('PILOT_SCENE_COMPLETE',key,row['pilot_category'],json.dumps(entry['metrics']),flush=True)
        state['status']='complete';state.pop('active_command',None);state.pop('error',None);save();print('THREE_SCENE_PILOT_COMPLETE',flush=True)
    except Exception as exc:
        state.update(status='failed_resumable',error=repr(exc));save();raise


if __name__=='__main__':main()
