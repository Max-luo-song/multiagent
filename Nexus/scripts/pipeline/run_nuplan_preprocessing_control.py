"""Paired official-reset-v versus raw-velocity control on saved pilot scenes.

Keep the checkpoint, scenes, maps, constraints, and per-scene sampling RNG equal.
Do not alter the official augmentor or claim the raw branch is the paper setup.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time
from datetime import datetime

from run_repro_demo import CHECKPOINT_SHA256, file_sha256, model_state_dict
from run_nuplan_metrics import json_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, default=os.environ.get('NEXUS_WORKDIR'))
    args = parser.parse_args()
    work = args.workdir.resolve()
    repo = work / 'Nexus'
    reference_dir = args.reference_run.resolve()
    reference = json.loads((reference_dir / 'run.json').read_text())
    if reference['status'] != 'complete':
        raise ValueError('Reference pilot must be complete')
    checkpoint = work / 'checkpoints/nuplan.ckpt'
    assert file_sha256(checkpoint) == reference['checkpoint_sha256'] == CHECKPOINT_SHA256
    run = repo / 'artifacts/nuplan_metrics' / datetime.now().strftime('preprocessing_control_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'started', 'scope': 'paired_preprocessing_diagnostic_not_paper_benchmark',
                'reference_run': str(reference_dir), 'checkpoint_sha256': CHECKPOINT_SHA256,
                'runner_sha256': file_sha256(Path(__file__)), 'seed': reference['seed'],
                'training_updates': 0, 'records': [], 'sampling_rng_paired': True,
                'branches': ['official_reset_v', 'raw_velocity']}
    json_write(run / 'run.json', manifest)
    print('RUN_DIR=' + str(run), flush=True)
    os.environ.update(MPLBACKEND='Agg', WANDB_MODE='disabled', NUPLAN_DATA_STORE='local')
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    torch.set_num_threads(2)
    random.seed(reference['seed'])
    np.random.seed(reference['seed'])
    torch.manual_seed(reference['seed'])
    torch.cuda.manual_seed_all(reference['seed'])
    print('[1/3] Loading official pipeline and unchanged checkpoint', flush=True)
    pipeline = runpy.run_path(str(repo / 'scripts/pipeline/pipeline.py'), run_name='nexus_control_helpers')
    from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
    from nuplan_extent.planning.training.preprocessing.features.scene_tensor import decode_scene_tensor
    augment = ResetVAugmentor()
    cfg = OmegaConf.load(reference_dir / 'config.yaml')
    cfg.scenario_filter.scenario_tokens = [s['token'] for s in reference['selected_scenarios']]
    cfg.scenario_filter.limit_total_scenarios = None
    cfg.scenario_filter.timestamp_threshold_s = None
    OmegaConf.save(cfg, run / 'config.yaml')
    metrics = {}
    for branch in manifest['branches']:
        metrics[branch] = {}
        for group in reference['metric_configurations'].values():
            for name, config in group.items():
                metrics[branch][name] = instantiate(OmegaConf.create(config)).cpu()
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
    lookup = {(str(s.token), s.scenario_type): s for s in candidates}
    feature_builder = model.get_list_of_required_feature()[0]
    print('[2/3] Paired inference with identical RNG states', flush=True)
    for index, requested in enumerate(reference['selected_scenarios']):
        scenario = lookup.get((requested['token'], requested['type']))
        if scenario is None:
            scenario = next(s for s in candidates if str(s.token) == requested['token'])
        raw = feature_builder.get_features_from_scenario(scenario)
        augmented, _ = augment.augment({'scene_tensor': copy.deepcopy(raw)}, {}, scenario)
        baseline = augmented['scene_tensor']
        original_file = reference_dir / f'prediction_{index:03d}_{scenario.token}.pt'
        original = torch.load(original_file, map_location='cpu', weights_only=False)
        assert torch.equal(torch.from_numpy(baseline.tensor).unsqueeze(0), original['scene_tensor_gt'])
        assert torch.equal(torch.from_numpy(raw.road_graph).unsqueeze(0), original['road_graph'])
        # Diagnose whether velocity channels still agree with observed motion.
        raw_decoded = decode_scene_tensor(raw.tensor)
        baseline_decoded = decode_scene_tensor(baseline.tensor)
        valid_pairs = raw.validity[1:, :4].astype(bool) & raw.validity[1:, 1:5].astype(bool)
        position_velocity = np.diff(raw_decoded[1:, :5, :2], axis=1) / 0.5
        raw_velocity = raw_decoded[1:, 1:5, 4:6]
        reset_velocity = baseline_decoded[1:, 1:5, 4:6]
        velocity_record = {
            'raw_velocity_vs_position_difference_mean_l2': float(np.linalg.norm(raw_velocity - position_velocity, axis=-1)[valid_pairs].mean()),
            'reset_velocity_vs_position_difference_mean_l2': float(np.linalg.norm(reset_velocity - position_velocity, axis=-1)[valid_pairs].mean()),
            'raw_history_speed_mean': float(np.linalg.norm(raw_velocity, axis=-1)[valid_pairs].mean()),
            'reset_history_speed_mean': float(np.linalg.norm(reset_velocity, axis=-1)[valid_pairs].mean()),
            'valid_history_pairs': int(valid_pairs.sum()),
        }
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        after_rng = None
        branch_records = {}
        for branch, feature in [('official_reset_v', baseline), ('raw_velocity', raw)]:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            scene = pipeline['unsqueeze'](pipeline['to_tensor'](copy.deepcopy(feature), torch.device('cuda')))
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                prediction = model.forward_validation({'scene_tensor': scene})
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            if after_rng is None:
                after_rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
            else:
                assert torch.equal(torch.get_rng_state(), after_rng[0])
                assert torch.equal(torch.cuda.get_rng_state(), after_rng[1])
            output = prediction['behavior_prediction_out']
            cpu = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in output.items()}
            for key in ('behavior_pred', 'intent_conditioned_behavior_pred'):
                assert torch.isfinite(cpu[key]).all()
            assert torch.equal(cpu['scene_tensor_gt'][..., :4], original['scene_tensor_gt'][..., :4])
            assert torch.equal(cpu['valid_mask'], original['valid_mask'])
            torch.save(cpu, run / f'{branch}_{index:03d}_{scenario.token}.pt')
            for metric in metrics[branch].values():
                metric.update({'behavior_prediction_out': copy.deepcopy(cpu)}, {})
            branch_records[branch] = {'two_task_seconds': seconds}
        # Keep unaugmented features for future controls without database reads.
        torch.save({k: torch.from_numpy(v.copy()) for k, v in vars(raw).items() if isinstance(v, np.ndarray)}, run / f'raw_features_{index:03d}_{scenario.token}.pt')
        record = {'token': str(scenario.token), 'index': index, 'velocity_diagnostics': velocity_record, 'branches': branch_records}
        manifest['records'].append(record)
        json_write(run / 'run.json', manifest)
        print(f'PAIRED_SCENARIO_OK {index+1}/{len(reference["selected_scenarios"])} token={scenario.token}', flush=True)
    print('[3/3] Saving paired metric results', flush=True)
    result = {branch: {name: {k: float(v.detach().cpu()) for k, v in metric.compute().items()} for name, metric in group.items()} for branch, group in metrics.items()}
    assert all(np.isfinite(value) for group in result.values() for metric in group.values() for value in metric.values())
    assert versions == [p._version for p in model.parameters()]
    manifest.update(status='complete', evaluated_scenarios=len(manifest['records']), model_parameter_versions_unchanged=True,
                    raw_feature_and_map_equality_verified=True)
    json_write(run / 'metrics.json', result)
    json_write(run / 'run.json', manifest)
    print(json.dumps(result, indent=2))
    print('NUPLAN_PREPROCESSING_CONTROL_OK\nRUN_DIR=' + str(run), flush=True)


if __name__ == '__main__':
    main()
