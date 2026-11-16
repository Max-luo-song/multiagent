"""Evaluate a bounded official nuPlan validation subset without training.

Uses the repository's feature builder, ResetVAugmentor, forward_validation, and
aggregated metric classes. This pilot is not the full paper benchmark.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import runpy
import sqlite3
import sys
import time
from datetime import datetime

from run_repro_demo import CHECKPOINT_SHA256, file_sha256, model_state_dict, prepare_maps


def json_write(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def parse_log_splits(path):
    splits, section = {}, None
    for line in path.read_text().splitlines():
        if line.startswith('  ') and not line.startswith('    ') and line.strip().endswith(':'):
            section = line.strip()[:-1]
            splits[section] = []
        elif section and line.lstrip().startswith('- '):
            splits[section].append(line.strip()[2:])
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, default=os.environ.get('NEXUS_WORKDIR'))
    parser.add_argument('--num-scenarios', type=int, default=8)
    parser.add_argument('--num-dbs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--constraints', choices=('official', 'clip'), default='official')
    args = parser.parse_args()
    if args.workdir is None or min(args.num_scenarios, args.num_dbs) < 1:
        parser.error('Provide a workdir and positive scenario/database counts')
    work = args.workdir.resolve()
    repo = work / 'Nexus'
    shared = Path('/inspire/hdd/global_public/public_datas/nuPlan')
    split_path = repo / 'third_party/nuplan-devkit/nuplan/planning/script/config/common/splitter/nuplan.yaml'
    splits = parse_log_splits(split_path)
    val_names = set(splits['val'])
    train_names = set(splits['train'])
    available = sorted(p for p in (shared / 'plan_data/val').glob('*.db') if p.stem in val_names)
    dbs = available[:args.num_dbs]
    if len(dbs) != args.num_dbs:
        raise RuntimeError('Requested validation databases are unavailable')
    assert not any(p.stem in train_names for p in dbs)
    bounds = {}
    for db in dbs:
        with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as connection:
            bounds[db.stem] = connection.execute('SELECT min(timestamp), max(timestamp) FROM lidar_pc').fetchone()
    checkpoint = work / 'checkpoints/nuplan.ckpt'
    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError('Checkpoint SHA256 mismatch')
    run = repo / 'artifacts/nuplan_metrics' / datetime.now().strftime('val_pilot_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    print('RUN_DIR=' + str(run), flush=True)
    manifest = {
        'status': 'started', 'scope': 'bounded_validation_pilot_not_paper_benchmark',
        'seed': args.seed, 'requested_scenarios': args.num_scenarios,
        'database_selection': 'first N lexicographically sorted available official val logs',
        'databases': [str(p) for p in dbs], 'available_val_databases': len(available),
        'split_sha256': file_sha256(split_path), 'checkpoint_sha256': CHECKPOINT_SHA256,
        'runner_sha256': file_sha256(Path(__file__)), 'training_updates': 0,
        'constraints': args.constraints, 'metric_inputs_isolated': True, 'records': [],
        'paper_protocol_reference': 'https://arxiv.org/html/2504.10485v1#A4.SS1',
    }
    json_write(run / 'run.json', manifest)
    os.environ.update(MPLBACKEND='Agg', WANDB_MODE='disabled', NUPLAN_DATA_STORE='local')
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest.update(torch=torch.__version__, cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    print('[1/4] Importing official pipeline and metrics', flush=True)
    pipeline_path = repo / 'scripts/pipeline/pipeline.py'
    pipeline = runpy.run_path(str(pipeline_path), run_name='nexus_eval_helpers')
    from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
    augment = ResetVAugmentor()
    cfg = OmegaConf.load(repo / 'scripts/pipeline/config.yaml')
    OmegaConf.set_struct(cfg, False)
    sampling = copy.deepcopy(cfg.model.future_trajectory_sampling)
    cfg.model = OmegaConf.load(repo / 'nuplan_extent/planning/script/config/common/model/nexus.yaml')
    cfg.model.future_trajectory_sampling = sampling  # Target builder unused for these metrics.
    cfg.model.diffuser.scheduling_matrix = 'full_sequence'
    cfg.model.diffuser.constrain_mode = ['map', 'sma', 'clip'] if args.constraints == 'official' else ['clip']
    cfg.model.diffuser.constrain_gamma = [1.0] * len(cfg.model.diffuser.constrain_mode)
    cfg.scenario_builder.data_root = str(work / 'data/nuplan')
    cfg.scenario_builder.map_root = str(prepare_maps(shared / 'maps', work / 'data/nuplan/maps'))
    cfg.scenario_builder.db_files = [str(p) for p in dbs]
    cfg.scenario_builder.include_cameras = False
    cfg.scenario_builder.max_workers = 1
    cfg.scenario_filter.log_names = [p.stem for p in dbs]
    cfg.scenario_filter.num_scenarios_per_type = None
    cfg.scenario_filter.limit_total_scenarios = args.num_scenarios * 8
    cfg.scenario_filter.shuffle = False
    cfg.scenario_filter.timestamp_threshold_s = 15
    cfg.worker = OmegaConf.create({'_target_': 'nuplan.planning.utils.multithreading.worker_sequential.Sequential', '_convert_': 'all'})
    cfg.checkpoint = str(checkpoint)
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, run / 'config.yaml')
    manifest['pipeline_sha256'] = file_sha256(pipeline_path)
    metric_configs = {}
    metrics = {}
    for group in ('behavior_prediction_metrics', 'collision_rate', 'traj_metric', 'time_metric'):
        path = repo / f'nuplan_extent/planning/script/config/training/aggregated_metric/{group}.yaml'
        group_cfg = OmegaConf.load(path)
        metric_configs[group] = OmegaConf.to_container(group_cfg, resolve=True)
        for name, config in group_cfg.items():
            metrics[name] = instantiate(config).cpu()
    manifest['metric_configurations'] = metric_configs
    metric_files = repo / 'nuplan_extent/planning/training/modeling/aggregated_metrics'
    manifest['metric_source_sha256'] = {p.name: file_sha256(p) for p in metric_files.glob('*.py')}
    json_write(run / 'run.json', manifest)

    print('[2/4] Strict checkpoint loading and validation scenario selection', flush=True)
    model = pipeline['build_torch_module_wrapper'](cfg.model)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(model_state_dict(state), strict=True)
    del state
    model = model.cuda().eval().requires_grad_(False)
    versions = [p._version for p in model.parameters()]
    print('CHECKPOINT_LOAD_OK', flush=True)
    worker = pipeline['build_worker'](cfg)
    builder = pipeline['build_scenario_builder'](cfg)
    candidates = pipeline['build_scenarios_from_config'](cfg, builder, worker)
    unique = {}
    for scenario in candidates:
        if scenario.log_name not in bounds:
            raise RuntimeError('Unexpected non-validation log')
        timestamp = scenario._initial_lidar_timestamp
        first, last = bounds[scenario.log_name]
        if timestamp - first >= 2_000_000 and last - timestamp >= 8_000_000:
            unique[(scenario.log_name, str(scenario.token))] = scenario
    scenarios = sorted(unique.values(), key=lambda s: (s.log_name, s._initial_lidar_timestamp, str(s.token)))[:args.num_scenarios]
    if len(scenarios) != args.num_scenarios:
        raise RuntimeError(f'Only {len(scenarios)} eligible unique scenarios, expected {args.num_scenarios}')
    manifest['selected_scenarios'] = [{'token': str(s.token), 'log_name': s.log_name, 'timestamp_us': s._initial_lidar_timestamp, 'type': s.scenario_type} for s in scenarios]
    json_write(run / 'run.json', manifest)
    feature_builder = model.get_list_of_required_feature()[0]

    print('[3/4] GPU validation and isolated official metric updates', flush=True)
    for index, scenario in enumerate(scenarios):
        feature_start = time.perf_counter()
        raw = feature_builder.get_features_from_scenario(scenario)
        features, _ = augment.augment({'scene_tensor': raw}, {}, scenario)
        raw = features['scene_tensor']
        if raw.tensor.shape[1:] != (21, 8):
            raise RuntimeError(f'Unexpected feature shape: {raw.tensor.shape}')
        scene = pipeline['unsqueeze'](pipeline['to_tensor'](raw, torch.device('cuda')))
        feature_seconds = time.perf_counter() - feature_start
        torch.cuda.synchronize()
        inference_start = time.perf_counter()
        with torch.inference_mode():
            predictions = model.forward_validation({'scene_tensor': scene})
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start
        output = predictions['behavior_prediction_out']
        for key in ('behavior_pred', 'intent_conditioned_behavior_pred', 'scene_tensor_gt'):
            if not torch.isfinite(output[key]).all():
                raise RuntimeError(f'Non-finite values: {key}, token={scenario.token}')
        cpu = {key: value.detach().cpu().clone() if torch.is_tensor(value) else value for key, value in output.items()}
        torch.save(cpu, run / f'prediction_{index:03d}_{scenario.token}.pt')
        map_before, valid_before = cpu['road_graph'].clone(), cpu['valid_mask'].clone()
        for metric in metrics.values():
            # Official metrics mutate mask views and unnormalize maps in place.
            metric.update({'behavior_prediction_out': copy.deepcopy(cpu)}, {})
        assert torch.equal(cpu['road_graph'], map_before)
        assert torch.equal(cpu['valid_mask'], valid_before)
        record = {'index': index, 'token': str(scenario.token), 'log_name': scenario.log_name,
                  'scenario_type': scenario.scenario_type, 'feature_seconds': feature_seconds,
                  'two_task_synchronized_seconds': inference_seconds,
                  'official_unsynchronized_mean_task_seconds': float(cpu['time_consume']),
                  'valid_non_ego_future_tokens': int(cpu['valid_mask'][:, 1:, 5:].sum()),
                  'full_future_trajectories_including_ego': int(cpu['valid_mask'][:, :, 5:].bool().all(-1).sum())}
        manifest['records'].append(record)
        json_write(run / 'run.json', manifest)
        print(f'SCENARIO_OK {index+1}/{len(scenarios)} token={scenario.token} synchronized_two_task_seconds={inference_seconds:.3f}', flush=True)

    print('[4/4] Aggregating metrics and saving results', flush=True)
    result = {}
    states = {}
    for name, metric in metrics.items():
        result[name] = {key: float(value.detach().cpu()) for key, value in metric.compute().items()}
        states[name] = {key: value.detach().cpu().tolist() for key, value in metric.metric_state.items() if torch.is_tensor(value)}
    if not all(np.isfinite(v) for values in result.values() for v in values.values()):
        raise RuntimeError('Non-finite aggregate metric, inspect denominators')
    assert versions == [p._version for p in model.parameters()], 'Model parameters changed during evaluation'
    json_write(run / 'metrics.json', result)
    json_write(run / 'metric_states.json', states)
    manifest.update(status='complete', evaluated_scenarios=len(scenarios), model_parameter_versions_unchanged=True)
    json_write(run / 'run.json', manifest)
    print(json.dumps(result, indent=2))
    print('NUPLAN_METRICS_PILOT_OK\nRUN_DIR=' + str(run), flush=True)


if __name__ == '__main__':
    main()
