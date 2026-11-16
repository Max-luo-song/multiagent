"""Evaluate frozen independent scenes with released-test and causal recipes.

Published weights and sampling settings remain fixed. Native metrics retain
their own input populations; position comparisons additionally use common slots.
Every scene/recipe/seed prediction is checkpointed for resumable inference.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time

from nuplan_plan_queries import validate_plan_metadata,QUERY_SOURCE
from prepare_nuplan_training_recipe import causal_history
from run_nuplan_metrics import json_write
from run_nuplan_multicity import audit_displacement
from run_repro_demo import CHECKPOINT_SHA256,file_sha256,model_state_dict,prepare_maps


def add_displacement(total,seed,city,recipe,values):
    dest=total.setdefault(str(seed),{}).setdefault(city,{}).setdefault(recipe,
        {'scenes':0,'bp':{'error_sum_m':0.,'valid_points':0},'goal':{'error_sum_m':0.,'valid_points':0}})
    dest['scenes']+=1
    for task in ('bp','goal'):
        for key in ('error_sum_m','valid_points'):
            dest[task][key]+=values[task][key]


def check_plan(plan_path):
    plan=json.loads(plan_path.read_text())
    samples=plan['scenarios']
    assert len(samples)==len({s['token'] for s in samples})==256
    assert len({s['trip'] for s in samples})==32
    assert not ({s['trip'] for s in samples}&set(plan['excluded_previous_trips']))
    reference=Path(plan['reference_run'])/'run.json'
    assert file_sha256(reference)==plan['reference_manifest_sha256']
    previous=json.loads(reference.read_text())
    assert previous['status']=='complete'
    assert {s['trip'] for s in previous['scenarios']}<=set(plan['excluded_previous_trips'])
    assert not ({s['token'] for s in samples}&{s['token'] for s in previous['scenarios']})
    assert [s['index'] for s in samples]==list(range(256))
    assert plan['recipes']==['released_test_recipe','causal_training_recipe']
    assert plan['seeds']==[0,1,2] and plan['batch_size']==4
    assert plan['sampling']=={'steps':32,'schedule':'full_sequence'}
    assert plan['constraints']=={'mode':['map','sma','clip'],'gamma':[1.,1.,1.]}
    assert file_sha256(Path(plan['configuration']))==plan['configuration_sha256']
    assert file_sha256(Path(plan['metric_reference_manifest']))==plan['metric_reference_sha256']
    assert file_sha256(Path(plan['scenario_query_source']))==plan['scenario_query_sha256']
    for path,digest in plan['preprocessing_source_sha256'].items():
        assert file_sha256(Path(path))==digest,'Preprocessing changed: '+path
    assert file_sha256(Path(__file__).with_name('prepare_nuplan_training_recipe.py'))==plan['causal_helper_sha256']
    assert validate_plan_metadata(plan)==256
    report_path=plan_path.with_name(plan_path.stem+'_builder_check.json')
    report=json.loads(report_path.read_text())
    assert report['status']=='complete' and report['plan_sha256']==file_sha256(plan_path)
    assert report['checked_scenarios']==256
    assert {item['index'] for item in report.get('feature_checks',[])}>={0,1,2,3}
    for item in report['feature_checks']:
        assert file_sha256(Path(item['raw_feature']))==item['sha256']
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,default=os.environ.get('NEXUS_WORKDIR'))
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    print('Checking independent scene plan, source hashes and official builder validation...',flush=True)
    plan=check_plan(args.plan)
    if args.check_only:
        print('INDEPENDENT_HOLDOUT_CHECK_OK: 256 scenes, 4 cities, 32 new trips; 2 recipes, 3 seeds',flush=True)
        return
    if args.workdir is None:
        parser.error('Source cache_env.sh first')
    work=args.workdir.resolve();repo=work/'Nexus';run=args.run_dir.resolve()
    checkpoint=work/'checkpoints/nuplan.ckpt'
    print('Checking official checkpoint SHA256...',flush=True)
    assert file_sha256(checkpoint)==CHECKPOINT_SHA256
    run.mkdir(parents=True,exist_ok=True)
    state_path=run/'run.json'
    fingerprint=dict(plan_sha256=file_sha256(args.plan),checkpoint_sha256=CHECKPOINT_SHA256,
                     runner_sha256=file_sha256(Path(__file__)),
                     audit_helper_sha256=file_sha256(Path(__file__).with_name('run_nuplan_multicity.py')),
                     causal_helper_sha256=plan['causal_helper_sha256'])
    if state_path.exists():
        state=json.loads(state_path.read_text())
        assert all(state[k]==v for k,v in fingerprint.items()),'Resume source or plan mismatch'
        if state['status']=='complete':
            print('ALREADY_COMPLETE\nRUN_DIR='+str(run),flush=True);return
    else:
        state=dict(fingerprint,status='started',scope=plan['scope'],scenarios=plan['scenarios'],records={},
            recipes=plan['recipes'],seeds=plan['seeds'],batch_size=plan['batch_size'],training_updates=0,
            primary_recipe=plan['primary_recipe'],excluded_previous_trips=plan['excluded_previous_trips'],
            timing_note='Native time values are batched and are not paper per-scene latency.',
            native_metric_note='Native metrics use each recipe input and agent cap; compare position errors '
                               'with common64_displacement_by_recipe_city_seed.json. No lowest-score recipe selection.',
            rng_note='sha256(plan hash + seed + fixed batch tokens). Same labels across recipes do not imply '
                     'identical noise arrays because agent dimensions differ.',
            recipe_note=plan['recipe_note'])
    state['status']='running';json_write(state_path,state)
    print('RUN_DIR='+str(run),flush=True)
    print('TARGET: 256 independent scenes x 3 seeds x 2 recipes x 2 tasks; batch_size=4',flush=True)
    os.environ.update(MPLBACKEND='Agg',WANDB_MODE='disabled',NUPLAN_DATA_STORE='local')
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'third_party/nuplan-devkit'))
    try:
        print('[1/3] Loading dependencies, official model and fixed input recipes',flush=True)
        import numpy as np
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor,decode_scene_tensor,encode_scene_tensor
        from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
        from nuplan_extent.planning.training.data_augmentation.only_centerline_reset_v import CenterlineVAugmentor
        assert torch.cuda.is_available(),'CUDA unavailable'
        torch.set_num_threads(2)
        pipeline=runpy.run_path(str(repo/'scripts/pipeline/pipeline.py'),run_name='independent_holdout_helpers')
        cfg=OmegaConf.load(plan['configuration'])
        cfg.scenario_filter.limit_total_scenarios=None;cfg.scenario_filter.timestamp_threshold_s=None
        cfg.scenario_builder.map_root=str(prepare_maps(Path('/inspire/hdd/global_public/public_datas/nuPlan/maps'),work/'data/nuplan/maps'))
        assert list(cfg.model.diffuser.constrain_mode)==plan['constraints']['mode']
        assert list(cfg.model.diffuser.constrain_gamma)==plan['constraints']['gamma']
        assert cfg.model.diffuser.steps==32 and cfg.model.diffuser.scheduling_matrix=='full_sequence'
        OmegaConf.save(cfg,run/'config.yaml');json_write(run/'frozen_plan.json',plan)
        model=instantiate(cfg.model)
        checkpoint_data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(model_state_dict(checkpoint_data),strict=True);del checkpoint_data
        model=model.cuda().eval().requires_grad_(False);versions=[p._version for p in model.parameters()]
        state.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda_runtime=torch.version.cuda)
        metric_cfg=json.loads(Path(plan['metric_reference_manifest']).read_text())['metric_configurations']
        metrics={recipe:{str(seed):{name:instantiate(OmegaConf.create(config)).cpu()
                    for group in metric_cfg.values() for name,config in group.items()}
                    for seed in plan['seeds']} for recipe in plan['recipes']}
        worker=pipeline['build_worker'](cfg);feature_builder=model.get_list_of_required_feature()[0]
        reset,centerline=ResetVAugmentor(),CenterlineVAugmentor()
        raw_cache=run/'raw_features';raw_cache.mkdir(exist_ok=True)
        totals={};omitted={}
        print('CHECKPOINT_LOAD_OK\n[2/3] Independent-scene features, inference and metrics',flush=True)
        for offset in range(0,256,4):
            batch=plan['scenarios'][offset:offset+4]
            features={recipe:[] for recipe in plan['recipes']};lookup=None
            if any(f'{recipe}:{s["index"]}:{seed}' not in state['records']
                   for s in batch for seed in plan['seeds'] for recipe in plan['recipes']):
                print(f'FEATURE_PROGRESS scenes={offset+1}-{offset+len(batch)}/256',flush=True)
                for sample in batch:
                    path=raw_cache/f'{sample["index"]:03d}_{sample["token"]}.pt'
                    if path.exists():
                        raw=SceneTensor(**{k:v.numpy().copy() for k,v in torch.load(path,map_location='cpu',weights_only=False).items()})
                    else:
                        if lookup is None:
                            cfg.scenario_builder.db_files=sorted({s['database'] for s in batch})
                            cfg.scenario_filter.log_names=sorted({s['log_name'] for s in batch})
                            cfg.scenario_filter.scenario_tokens=[s['token'] for s in batch]
                            builder=pipeline['build_scenario_builder'](cfg)
                            built=pipeline['build_scenarios_from_config'](cfg,builder,worker)
                            lookup={str(s.token):s for s in built}
                            missing=[s['token'] for s in batch if s['token'] not in lookup]
                            assert not missing,f'Official builder omitted frozen tokens: {missing}'
                        scenario=lookup[sample['token']]
                        assert scenario.log_name==sample['log_name'] and scenario._initial_lidar_timestamp==sample['timestamp_us']
                        raw=feature_builder.get_features_from_scenario(scenario)
                        assert raw.tensor.shape==(129,21,8)
                        temporary=path.with_suffix('.tmp')
                        torch.save({k:torch.from_numpy(v.copy()) for k,v in vars(raw).items()},temporary);temporary.replace(path)
                    released=reset.augment({'scene_tensor':copy.deepcopy(raw)}, {})[0]['scene_tensor']
                    training=centerline.augment({'scene_tensor':copy.deepcopy(raw)}, {})[0]['scene_tensor']
                    causal=causal_history(training,raw,decode_scene_tensor,encode_scene_tensor)
                    assert causal.tensor.shape==(64,21,8)
                    assert np.array_equal(released.validity[:64],causal.validity)
                    assert np.allclose(released.tensor[:64,:,:4],causal.tensor[:,:,:4],atol=1e-6,rtol=1e-6)
                    features['released_test_recipe'].append(released);features['causal_training_recipe'].append(causal)
            for seed in plan['seeds']:
                common_valid=None;common_gt=None
                for recipe in plan['recipes']:
                    keys=[f'{recipe}:{s["index"]}:{seed}' for s in batch]
                    if any(k not in state['records'] for k in keys):
                        material=fingerprint['plan_sha256']+':'+str(seed)+':'+','.join(s['token'] for s in batch)
                        batch_seed=int(hashlib.sha256(material.encode()).hexdigest()[:8],16)
                        random.seed(batch_seed);np.random.seed(batch_seed)
                        torch.manual_seed(batch_seed);torch.cuda.manual_seed_all(batch_seed)
                        scene=SceneTensor(**{k:torch.stack([torch.from_numpy(getattr(f,k).copy()) for f in features[recipe]]).cuda()
                            for k in ('tensor','validity','road_graph','road_graph_validity')})
                        torch.cuda.synchronize();started=time.perf_counter()
                        with torch.inference_mode():
                            output=model.forward_validation({'scene_tensor':scene})['behavior_prediction_out']
                        torch.cuda.synchronize();seconds=time.perf_counter()-started
                        for j,(sample,key) in enumerate(zip(batch,keys)):
                            if key in state['records']:
                                continue
                            cpu={k:v[j:j+1].detach().cpu().clone() if torch.is_tensor(v) else v for k,v in output.items()}
                            assert all(torch.isfinite(cpu[k]).all() for k in ('behavior_pred','intent_conditioned_behavior_pred','scene_tensor_gt'))
                            path=run/f'{recipe}_{sample["index"]:03d}_seed{seed}.pt'
                            temporary=path.with_suffix('.tmp');torch.save(cpu,temporary);temporary.replace(path)
                            state['records'][key]=dict(index=sample['index'],token=sample['token'],city=sample['city'],recipe=recipe,
                                seed=seed,batch_rng_seed=batch_seed,batch_two_task_seconds=seconds,prediction_file=path.name,sha256=file_sha256(path))
                        json_write(state_path,state)
                    valid_batch=[];gt_batch=[]
                    for sample,key in zip(batch,keys):
                        record=state['records'][key];path=run/record['prediction_file']
                        assert file_sha256(path)==record['sha256']
                        cpu=torch.load(path,map_location='cpu',weights_only=False)
                        for metric in metrics[recipe][str(seed)].values():
                            metric.update({'behavior_prediction_out':copy.deepcopy(cpu)}, {})
                        if recipe=='released_test_recipe':
                            omitted[key]=int(cpu['valid_mask'][:,64:,5:].bool().sum())
                        # Only the position audit is explicitly matched to common agent slots.
                        matched={k:(v[:,:64].clone() if k in ('behavior_pred','intent_conditioned_behavior_pred','scene_tensor_gt','valid_mask','task_mask') else v)
                                 for k,v in cpu.items()}
                        valid_batch.append(matched['valid_mask']);gt_batch.append(matched['scene_tensor_gt'][...,:4])
                        value=audit_displacement(matched,decode_scene_tensor)
                        for city in ('all',sample['city']):
                            add_displacement(totals,seed,city,recipe,value)
                    if common_valid is None:
                        common_valid,common_gt=valid_batch,gt_batch
                    else:
                        assert all(torch.equal(a,b) for a,b in zip(common_valid,valid_batch))
                        assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-6) for a,b in zip(common_gt,gt_batch))
            print(f'HOLDOUT_PROGRESS scenes={offset+len(batch)}/256 saved_predictions={len(state["records"])}/1536',flush=True)
        print('[3/3] Saving native metrics and common-agent displacement',flush=True)
        official={recipe:{seed:{name:{k:float(v.detach().cpu()) for k,v in metric.compute().items()}
            for name,metric in group.items()} for seed,group in groups.items()} for recipe,groups in metrics.items()}
        assert all(np.isfinite(v) for groups in official.values() for group in groups.values() for metric in group.values() for v in metric.values())
        for cities in totals.values():
            for recipes in cities.values():
                for task in ('bp','goal'):
                    assert len({value[task]['valid_points'] for value in recipes.values()})==1
                for value in recipes.values():
                    for task in ('bp','goal'):
                        assert value[task]['valid_points']>0
                        value[task]['mean_error_m']=value[task]['error_sum_m']/value[task]['valid_points']
        assert len(state['records'])==1536 and versions==[p._version for p in model.parameters()]
        json_write(run/'official_metrics_by_recipe_seed.json',official)
        json_write(run/'common64_displacement_by_recipe_city_seed.json',totals)
        json_write(run/'reference_future_points_outside_common64.json',omitted)
        state.update(status='complete',evaluated_scenarios=256,evaluated_scene_seed_recipes=1536,task_outputs=3072,
                     model_parameter_versions_unchanged=True)
        state.pop('error',None);json_write(state_path,state)
        print('NUPLAN_INDEPENDENT_HOLDOUT_OK\nRUN_DIR='+str(run),flush=True)
    except Exception as exc:
        state.update(status='failed_resumable',error=repr(exc));json_write(state_path,state);raise


if __name__=='__main__':
    main()
