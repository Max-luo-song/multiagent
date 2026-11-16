"""Run a frozen multi-city validation plan; automatically resume saved samples.

Official preprocessing/constraints remain fixed. Batched GPU inference is split
back into single-scene inputs for official metric updates. Saved tensor files
allow resuming aggregation without repeating completed GPU batches.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time

from run_nuplan_metrics import json_write
from run_repro_demo import CHECKPOINT_SHA256, file_sha256, model_state_dict, prepare_maps
from nuplan_plan_queries import validate_plan_metadata, QUERY_SOURCE


def audit_displacement(output, decode):
    """All future half-second points; explicitly exclude every provided goal."""
    import torch
    gt = decode(output['scene_tensor_gt']).double()
    valid = output['valid_mask'].bool().clone()
    valid[:, 0] = False
    future = valid.clone()
    future[:, :, :5] = False
    last = (valid.long() * torch.arange(valid.shape[-1])).max(-1).values
    goals = torch.zeros_like(valid)
    goals.scatter_(2, last.unsqueeze(-1), True)
    result = {}
    for label, key in [('bp', 'behavior_pred'), ('goal', 'intent_conditioned_behavior_pred')]:
        mask = future if label == 'bp' else future & ~goals
        errors = (decode(output[key]).double()[..., :2] - gt[..., :2]).norm(dim=-1)
        result[label] = {'error_sum_m': float(errors[mask].sum()), 'valid_points': int(mask.sum())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, default=os.environ.get('NEXUS_WORKDIR'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    parser.add_argument('--batch-size', type=int, choices=[1,2,4], default=4)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    scenarios = plan['scenarios']
    assert len(scenarios) == len({s['token'] for s in scenarios}) == 128
    assert len({s['trip'] for s in scenarios}) == 32
    assert set(Counter(s['city'] for s in scenarios).values()) == {32}
    assert len(Counter(s['city'] for s in scenarios)) == 4
    assert not (set(plan['excluded_development_trips']) & {s['trip'] for s in scenarios})
    assert len(args.seeds) == len(set(args.seeds))
    for s in scenarios:
        assert Path(s['database']).is_file()
    print('Checking all frozen tokens against nuPlan scene-boundary filtering...', flush=True)
    assert validate_plan_metadata(plan) == 128
    assert file_sha256(QUERY_SOURCE) == plan['scenario_query_sha256'], 'Official scenario query changed'
    if args.check_only:
        print('PLAN_CHECK_OK: 128 queryable unique scenes, 4 cities, 32 trips, development trips excluded')
        return
    if args.workdir is None:
        parser.error('Source cache_env.sh first')
    work = args.workdir.resolve()
    repo = work / 'Nexus'
    assert file_sha256(repo / plan['model_configuration_source']) == plan['model_configuration_sha256']
    checkpoint = work / 'checkpoints/nuplan.ckpt'
    print('Checking checkpoint SHA256...', flush=True)
    assert file_sha256(checkpoint) == CHECKPOINT_SHA256
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    fingerprint = {'plan_sha256': file_sha256(args.plan), 'checkpoint_sha256': CHECKPOINT_SHA256,
                   'seeds': args.seeds, 'batch_size': args.batch_size}
    state_path = run / 'run.json'
    if state_path.exists():
        manifest = json.loads(state_path.read_text())
        assert all(manifest[k] == v for k,v in fingerprint.items()), 'Resume configuration mismatch'
        assert manifest['runner_sha256'] == file_sha256(Path(__file__)), 'Runner changed since this run began'
        if manifest['status'] == 'complete':
            print('ALREADY_COMPLETE\nRUN_DIR=' + str(run))
            return
    else:
        manifest = dict(fingerprint, status='started', scope=plan['scope'], records={}, training_updates=0,
                        runner_sha256=file_sha256(Path(__file__)), scenarios=scenarios,
                        rng_rule='sha256(plan_hash + seed + fixed_batch_tokens); rerun whole pending batch',
                        timing_note='official timing is batched, not per-scene paper latency')
    manifest['status'] = 'running'
    json_write(state_path, manifest)
    print('RUN_DIR=' + str(run), flush=True)
    print('TARGET: 128 scenes x '+str(len(args.seeds))+' seeds x 2 tasks; batch_size='+str(args.batch_size), flush=True)
    print('[1/3] Loading dependencies; this phase can have low GPU utilization', flush=True)
    os.environ.update(MPLBACKEND='Agg', WANDB_MODE='disabled', NUPLAN_DATA_STORE='local')
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch CUDA unavailable')
    torch.set_num_threads(2)
    pipeline = runpy.run_path(str(repo / 'scripts/pipeline/pipeline.py'), run_name='multicity_helpers')
    from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
    from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor, decode_scene_tensor
    augment = ResetVAugmentor()
    cfg = OmegaConf.load(plan['reference_configuration'])
    assert list(cfg.model.diffuser.constrain_mode) == plan['constraints']['mode']
    assert list(cfg.model.diffuser.constrain_gamma) == plan['constraints']['gamma']
    assert cfg.model.diffuser.steps == plan['sampling']['steps']
    assert cfg.model.diffuser.scheduling_matrix == plan['sampling']['schedule']
    cfg.scenario_filter.limit_total_scenarios = None
    cfg.scenario_filter.timestamp_threshold_s = None
    cfg.scenario_builder.map_root = str(prepare_maps(Path('/inspire/hdd/global_public/public_datas/nuPlan/maps'), work/'data/nuplan/maps'))
    OmegaConf.save(cfg, run/'config.yaml')
    print('Instantiating model and loading official checkpoint...', flush=True)
    model = instantiate(cfg.model)
    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(model_state_dict(checkpoint_data), strict=True)
    del checkpoint_data
    model = model.cuda().eval().requires_grad_(False)
    versions = [p._version for p in model.parameters()]
    manifest.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda_runtime=torch.version.cuda)
    print('CHECKPOINT_LOAD_OK', flush=True)
    metric_cfg = json.loads((Path(plan['reference_configuration']).with_name('run.json')).read_text())['metric_configurations']
    metrics = {str(seed): {name: instantiate(OmegaConf.create(config)).cpu() for group in metric_cfg.values() for name,config in group.items()} for seed in args.seeds}
    audit_totals = {str(seed): {} for seed in args.seeds}
    worker = pipeline['build_worker'](cfg)
    feature_builder = model.get_list_of_required_feature()[0]
    cache = run/'raw_features'
    cache.mkdir(exist_ok=True)
    print('[2/3] Batched GPU inference, per-scene metrics, automatic checkpointing', flush=True)
    try:
        for offset in range(0,len(scenarios),args.batch_size):
            batch = scenarios[offset:offset+args.batch_size]
            # Log-aligned groups are at most four samples; builder reads one log.
            lookup = None
            features = []
            missing_any = any(f'{s["index"]}:{seed}' not in manifest['records'] for s in batch for seed in args.seeds)
            if missing_any:
                print(f'FEATURE_PROGRESS preparing scenes {offset+1}-{offset+len(batch)}', flush=True)
                for s in batch:
                    feature_path = cache/f'{s["index"]:03d}_{s["token"]}.pt'
                    if feature_path.exists():
                        raw = SceneTensor(**{k:v.numpy().copy() for k,v in torch.load(feature_path,map_location='cpu',weights_only=False).items()})
                    else:
                        if lookup is None:
                            cfg.scenario_builder.db_files = sorted({b['database'] for b in batch})
                            cfg.scenario_filter.log_names = sorted({b['log_name'] for b in batch})
                            cfg.scenario_filter.scenario_tokens = [b['token'] for b in batch]
                            builder = pipeline['build_scenario_builder'](cfg)
                            built = pipeline['build_scenarios_from_config'](cfg,builder,worker)
                            lookup = {str(v.token):v for v in built}
                            missing = [b['token'] for b in batch if b['token'] not in lookup]
                            if missing:
                                raise RuntimeError(f'Official scenario builder omitted planned tokens: {missing}; '
                                                   f'returned={sorted(lookup)}')
                        scenario = lookup[s['token']]
                        assert scenario.log_name == s['log_name'] and scenario._initial_lidar_timestamp == s['timestamp_us']
                        raw = feature_builder.get_features_from_scenario(scenario)
                        assert raw.tensor.shape == (129,21,8)
                        temporary = feature_path.with_suffix('.tmp')
                        torch.save({k:torch.from_numpy(v.copy()) for k,v in vars(raw).items()},temporary)
                        temporary.replace(feature_path)
                    modified,_ = augment.augment({'scene_tensor':copy.deepcopy(raw)}, {})
                    features.append(modified['scene_tensor'])
            for seed in args.seeds:
                keys = [f'{s["index"]}:{seed}' for s in batch]
                if any(k not in manifest['records'] for k in keys):
                    material = fingerprint['plan_sha256']+':'+str(seed)+':'+','.join(s['token'] for s in batch)
                    batch_seed = int(hashlib.sha256(material.encode()).hexdigest()[:8],16)
                    random.seed(batch_seed); np.random.seed(batch_seed)
                    torch.manual_seed(batch_seed); torch.cuda.manual_seed_all(batch_seed)
                    scene = SceneTensor(**{k:torch.stack([torch.from_numpy(getattr(f,k).copy()) for f in features]).cuda() for k in ('tensor','validity','road_graph','road_graph_validity')})
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        output = model.forward_validation({'scene_tensor':scene})['behavior_prediction_out']
                    torch.cuda.synchronize()
                    batch_seconds = time.perf_counter()-started
                    for j,(s,key) in enumerate(zip(batch,keys)):
                        if key in manifest['records']:
                            continue
                        cpu = {k:v[j:j+1].detach().cpu().clone() if torch.is_tensor(v) else v for k,v in output.items()}
                        for k in ('behavior_pred','intent_conditioned_behavior_pred','scene_tensor_gt'):
                            assert torch.isfinite(cpu[k]).all(), f'Non-finite output {key} {k}'
                        path = run/f'prediction_{s["index"]:03d}_seed{seed}.pt'
                        temporary = path.with_suffix('.tmp')
                        torch.save(cpu,temporary); temporary.replace(path)
                        manifest['records'][key] = {'index':s['index'],'token':s['token'],'city':s['city'],'seed':seed,
                                                    'batch_rng_seed':batch_seed,'batch_two_task_seconds':batch_seconds,
                                                    'prediction_file':path.name,'sha256':file_sha256(path)}
                    json_write(state_path,manifest)
                # Replaying saved single-scene payloads gives identical metric
                # aggregation for a new run and a resumed run.
                for s,key in zip(batch,keys):
                    record = manifest['records'][key]
                    path = run/record['prediction_file']
                    assert file_sha256(path) == record['sha256']
                    cpu = torch.load(path,map_location='cpu',weights_only=False)
                    for metric in metrics[str(seed)].values():
                        metric.update({'behavior_prediction_out':copy.deepcopy(cpu)}, {})
                    sample_audit = audit_displacement(cpu, decode_scene_tensor)
                    for city in ('all',s['city']):
                        totals = audit_totals[str(seed)].setdefault(city,{'bp':{'error_sum_m':0.,'valid_points':0},'goal':{'error_sum_m':0.,'valid_points':0},'scenes':0})
                        totals['scenes'] += 1
                        for task in ('bp','goal'):
                            for item in ('error_sum_m','valid_points'):
                                totals[task][item] += sample_audit[task][item]
            print(f'EVAL_PROGRESS scenes={min(offset+args.batch_size,128)}/128 saved_scene_seeds={len(manifest["records"])}/{128*len(args.seeds)}',flush=True)
        print('[3/3] Aggregating per-seed official and audited displacement metrics',flush=True)
        official = {seed:{name:{k:float(v.detach().cpu()) for k,v in metric.compute().items()} for name,metric in group.items()} for seed,group in metrics.items()}
        assert all(np.isfinite(v) for group in official.values() for m in group.values() for v in m.values())
        for cities in audit_totals.values():
            for values in cities.values():
                for task in ('bp','goal'):
                    assert values[task]['valid_points'] > 0, 'No valid points for displacement audit'
                    values[task]['mean_error_m'] = values[task]['error_sum_m']/values[task]['valid_points']
        assert versions == [p._version for p in model.parameters()]
        json_write(run/'official_metrics_by_seed.json',official)
        json_write(run/'audited_displacement_by_city_seed.json',audit_totals)
        json_write(run/'frozen_plan.json',plan)
        manifest.update(status='complete',evaluated_scenarios=128,evaluated_scene_seeds=len(manifest['records']),model_parameter_versions_unchanged=True)
        json_write(state_path,manifest)
        print('NUPLAN_MULTICITY_OK\nRUN_DIR='+str(run),flush=True)
    except Exception as exc:
        manifest.update(status='failed_resumable',error=repr(exc))
        json_write(state_path,manifest)
        raise


if __name__ == '__main__':
    main()
