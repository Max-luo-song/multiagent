"""Resumable cached-input comparison of published and causal training recipes.

No optimization, database rebuilding, report generation or archive creation.
The reference's saved predictions supply the common-agent baseline.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from run_nuplan_metrics import json_write
from run_nuplan_multicity import audit_displacement
from run_repro_demo import CHECKPOINT_SHA256, file_sha256, model_state_dict
from training_recipe_common import BRANCHES, check_prepared


def add_audit(total, seed, city, branch, value):
    dest = total.setdefault(str(seed),{}).setdefault(city,{}).setdefault(branch,
            {'scenes':0, 'bp':{'error_sum_m':0.,'valid_points':0}, 'goal':{'error_sum_m':0.,'valid_points':0}})
    dest['scenes'] += 1
    for task in ('bp','goal'):
        for key in ('error_sum_m','valid_points'):
            dest[task][key] += value[task][key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--prepared-features',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,default=os.environ.get('NEXUS_WORKDIR'))
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    reference, prepared_dir = args.reference_run.resolve(), args.prepared_features.resolve()
    prepared, baseline = check_prepared(prepared_dir,reference)
    scenes, seeds, batch_size = baseline['scenarios'], baseline['seeds'], baseline['batch_size']
    assert len(scenes)==128 and seeds==[0,1,2] and batch_size==4
    assert baseline['evaluated_scene_seeds']==384
    if args.check_only:
        print('RECIPE_CONTROL_CHECK_OK: 128 cached scenes, 2 recipes, 3 seeds, common 64 agent slots',flush=True)
        return
    if args.workdir is None:
        parser.error('Source cache_env.sh first')
    work = args.workdir.resolve()
    repo = work/'Nexus'
    checkpoint = work/'checkpoints/nuplan.ckpt'
    print('Checking official checkpoint SHA256...',flush=True)
    assert file_sha256(checkpoint)==baseline['checkpoint_sha256']==CHECKPOINT_SHA256
    run = args.run_dir.resolve()
    assert run not in (reference,prepared_dir)
    run.mkdir(parents=True,exist_ok=True)
    fingerprint = dict(plan_sha256=baseline['plan_sha256'], checkpoint_sha256=CHECKPOINT_SHA256,
                       prepared_manifest_sha256=file_sha256(prepared_dir/'run.json'),
                       reference_manifest_sha256=file_sha256(reference/'run.json'),
                       runner_sha256=file_sha256(Path(__file__)), seeds=seeds,batch_size=batch_size,
                       audit_source_sha256=file_sha256(Path(__file__).with_name('run_nuplan_multicity.py')),
                       helper_sha256=file_sha256(Path(__file__).with_name('training_recipe_common.py')))
    state_path = run/'run.json'
    if state_path.exists():
        manifest = json.loads(state_path.read_text())
        assert all(manifest[k]==v for k,v in fingerprint.items()), 'Resume input or source mismatch'
        if manifest['status']=='complete':
            print('ALREADY_COMPLETE\nRUN_DIR='+str(run),flush=True)
            return
    else:
        manifest = dict(fingerprint,status='started',scope='train_test_preprocessing_and_causal_boundary_control',
                        reference_run=str(reference),prepared_features=str(prepared_dir),records={},training_updates=0,
                        branches=list(BRANCHES),common_agent_slots=64,scenarios=scenes,
                        input_comparison='Published training recipe jointly changes maps, velocities and agent cap. '
                                         'Causal branch changes only current velocity at the history boundary.',
                        rng_note='Reuse reference batch seed labels; different agent counts imply different noise '
                                 'layouts, so reference and training-recipe inference are not identical-noise pairs. '
                                 'The two 64-agent branches do reuse identical RNG seeds and tensor shapes.',
                        official_metric_note='Official metrics here use 64-agent training inputs and native map '
                                             'validity; use common_agent_comparison.json for matched position comparison.')
    manifest['status']='running'
    json_write(state_path,manifest)
    print('RUN_DIR='+str(run),flush=True)
    print('TARGET: 128 scenes x 3 seeds x 2 recipes x 2 tasks; cached inputs, no DB reads',flush=True)
    os.environ.update(MPLBACKEND='Agg',WANDB_MODE='disabled',NUPLAN_DATA_STORE='local')
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'third_party/nuplan-devkit'))
    try:
        print('[1/3] Loading dependencies and model',flush=True)
        import numpy as np
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor, decode_scene_tensor
        assert torch.cuda.is_available(), 'CUDA unavailable'
        torch.set_num_threads(2)
        cfg = OmegaConf.load(reference/'config.yaml')
        assert list(cfg.model.diffuser.constrain_mode)==['map','sma','clip']
        assert list(cfg.model.diffuser.constrain_gamma)==[1.,1.,1.]
        assert cfg.model.diffuser.steps==32 and cfg.model.diffuser.scheduling_matrix=='full_sequence'
        OmegaConf.save(cfg,run/'config.yaml')
        model = instantiate(cfg.model)
        state = torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(model_state_dict(state),strict=True)
        del state
        model = model.cuda().eval().requires_grad_(False)
        versions = [p._version for p in model.parameters()]
        manifest.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda_runtime=torch.version.cuda)
        plan = json.loads((reference/'frozen_plan.json').read_text())
        metric_cfg = json.loads(Path(plan['reference_configuration']).with_name('run.json').read_text())['metric_configurations']
        metrics = {branch:{str(seed):{name:instantiate(OmegaConf.create(config)).cpu()
                   for group in metric_cfg.values() for name,config in group.items()} for seed in seeds} for branch in BRANCHES}
        totals = {}
        print('CHECKPOINT_LOAD_OK\n[2/3] Paired 64-agent recipe inference and common-agent comparisons',flush=True)
        for offset in range(0,128,batch_size):
            batch = scenes[offset:offset+batch_size]
            features = {}
            for branch in BRANCHES:
                if any(f'{branch}:{s["index"]}:{seed}' not in manifest['records'] for s in batch for seed in seeds):
                    features[branch]=[]
                    for s in batch:
                        info=prepared['records'][str(s['index'])]['features'][branch]
                        features[branch].append(torch.load(prepared_dir/info['file'],map_location='cpu',weights_only=False))
            for seed in seeds:
                base_outputs={}
                for s in batch:
                    record=baseline['records'][f'{s["index"]}:{seed}']
                    path=reference/record['prediction_file']
                    assert file_sha256(path)==record['sha256']
                    cpu=torch.load(path,map_location='cpu',weights_only=False)
                    for key in ('behavior_pred','intent_conditioned_behavior_pred','scene_tensor_gt','valid_mask','task_mask'):
                        cpu[key]=cpu[key][:,:64].clone()
                    base_outputs[s['index']]=cpu
                    value=audit_displacement(cpu,decode_scene_tensor)
                    for city in ('all',s['city']):
                        add_audit(totals,seed,city,'saved_test_recipe_common64',value)
                for branch in BRANCHES:
                    keys=[f'{branch}:{s["index"]}:{seed}' for s in batch]
                    if any(key not in manifest['records'] for key in keys):
                        batch_seed=baseline['records'][f'{batch[0]["index"]}:{seed}']['batch_rng_seed']
                        assert all(baseline['records'][f'{s["index"]}:{seed}']['batch_rng_seed']==batch_seed for s in batch)
                        random.seed(batch_seed);np.random.seed(batch_seed)
                        torch.manual_seed(batch_seed);torch.cuda.manual_seed_all(batch_seed)
                        scene=SceneTensor(**{k:torch.stack([f[k] for f in features[branch]]).cuda()
                               for k in ('tensor','validity','road_graph','road_graph_validity')})
                        torch.cuda.synchronize();started=time.perf_counter()
                        with torch.inference_mode():
                            result=model.forward_validation({'scene_tensor':scene})['behavior_prediction_out']
                        torch.cuda.synchronize();seconds=time.perf_counter()-started
                        for j,(s,key) in enumerate(zip(batch,keys)):
                            if key in manifest['records']:
                                continue
                            cpu={k:v[j:j+1].detach().cpu().clone() if torch.is_tensor(v) else v for k,v in result.items()}
                            assert all(torch.isfinite(cpu[k]).all() for k in ('behavior_pred','intent_conditioned_behavior_pred','scene_tensor_gt'))
                            target=run/f'{branch}_{s["index"]:03d}_seed{seed}.pt'
                            temporary=target.with_suffix('.tmp');torch.save(cpu,temporary);temporary.replace(target)
                            manifest['records'][key]=dict(index=s['index'],token=s['token'],city=s['city'],seed=seed,branch=branch,
                                prediction_file=target.name,sha256=file_sha256(target),batch_rng_seed=batch_seed,
                                batch_two_task_seconds=seconds)
                        json_write(state_path,manifest)
                    for s,key in zip(batch,keys):
                        record=manifest['records'][key];path=run/record['prediction_file']
                        assert file_sha256(path)==record['sha256']
                        cpu=torch.load(path,map_location='cpu',weights_only=False)
                        old=base_outputs[s['index']]
                        assert torch.equal(cpu['valid_mask'],old['valid_mask'])
                        assert torch.allclose(cpu['scene_tensor_gt'][...,:4],old['scene_tensor_gt'][...,:4],atol=1e-6,rtol=1e-6)
                        value=audit_displacement(cpu,decode_scene_tensor)
                        old_value=audit_displacement(old,decode_scene_tensor)
                        assert all(value[t]['valid_points']==old_value[t]['valid_points'] for t in ('bp','goal'))
                        for city in ('all',s['city']):
                            add_audit(totals,seed,city,branch,value)
                        for metric in metrics[branch][str(seed)].values():
                            metric.update({'behavior_prediction_out':copy.deepcopy(cpu)}, {})
            print(f'RECIPE_PROGRESS scenes={offset+len(batch)}/128 saved_predictions={len(manifest["records"])}/768',flush=True)
        print('[3/3] Saving native official metrics and matched-agent displacement',flush=True)
        official={branch:{seed:{name:{k:float(v.detach().cpu()) for k,v in metric.compute().items()}
                  for name,metric in group.items()} for seed,group in groups.items()} for branch,groups in metrics.items()}
        assert all(np.isfinite(v) for groups in official.values() for group in groups.values() for metric in group.values() for v in metric.values())
        for cities in totals.values():
            for branches in cities.values():
                for value in branches.values():
                    for task in ('bp','goal'):
                        assert value[task]['valid_points']>0
                        value[task]['mean_error_m']=value[task]['error_sum_m']/value[task]['valid_points']
        assert versions==[p._version for p in model.parameters()]
        assert len(manifest['records'])==768
        json_write(run/'official_metrics_by_recipe_seed.json',official)
        json_write(run/'common_agent_comparison.json',totals)
        json_write(run/'prepared_input_manifest.json',prepared)
        manifest.update(status='complete',evaluated_scenarios=128,evaluated_scene_seed_recipes=768,
                        task_outputs=1536,model_parameter_versions_unchanged=True)
        json_write(state_path,manifest)
        print('NUPLAN_RECIPE_CONTROL_OK\nRUN_DIR='+str(run),flush=True)
    except Exception as exc:
        manifest.update(status='failed_resumable',error=repr(exc));json_write(state_path,manifest)
        raise


if __name__=='__main__':
    main()
