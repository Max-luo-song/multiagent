"""Paired constraint sensitivity diagnostic using cached raw features."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime

from run_repro_demo import CHECKPOINT_SHA256, file_sha256, model_state_dict
from run_nuplan_metrics import json_write


VARIANTS = {
    'map_sma_strong': {'mode': ['map', 'sma', 'clip'], 'gamma': [1.0, 1.0, 1.0]},
    'map_sma_weak': {'mode': ['map', 'sma', 'clip'], 'gamma': [0.1, 0.1, 1.0]},
    'clip_only': {'mode': ['clip'], 'gamma': [1.0]},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, default=os.environ.get('NEXUS_WORKDIR'))
    args = parser.parse_args()
    repo = args.workdir.resolve() / 'Nexus'
    control = args.reference_run.resolve()
    previous = json.loads((control / 'run.json').read_text())
    assert previous['status'] == 'complete'
    pilot = json.loads((Path(previous['reference_run']) / 'run.json').read_text())
    checkpoint = repo.parent / 'checkpoints/nuplan.ckpt'
    assert file_sha256(checkpoint) == CHECKPOINT_SHA256
    run = repo / 'artifacts/nuplan_metrics' / datetime.now().strftime('constraint_control_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'started', 'scope': 'constraint_sensitivity_pilot_not_paper_benchmark',
                'reference_run': str(control), 'checkpoint_sha256': CHECKPOINT_SHA256,
                'runner_sha256': file_sha256(Path(__file__)), 'variants': VARIANTS,
                'preprocessing': 'raw_velocity', 'training_updates': 0,
                'sampling_rng_paired': True, 'records': []}
    json_write(run / 'run.json', manifest)
    print('RUN_DIR=' + str(run), flush=True)
    os.environ.update(MPLBACKEND='Agg', WANDB_MODE='disabled', NUPLAN_DATA_STORE='local')
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    torch.set_num_threads(2)
    random.seed(previous['seed'])
    np.random.seed(previous['seed'])
    torch.manual_seed(previous['seed'])
    torch.cuda.manual_seed_all(previous['seed'])
    print('[1/2] Importing model and strictly loading checkpoint', flush=True)
    from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
    cfg = OmegaConf.load(control / 'config.yaml')
    OmegaConf.save(cfg, run / 'config.yaml')
    metrics = {}
    for name in VARIANTS:
        metrics[name] = {}
        for group in pilot['metric_configurations'].values():
            for key, config in group.items():
                metrics[name][key] = instantiate(OmegaConf.create(config)).cpu()
    model = instantiate(cfg.model)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(model_state_dict(state), strict=True)
    del state
    model = model.cuda().eval().requires_grad_(False)
    versions = [p._version for p in model.parameters()]
    print('CHECKPOINT_LOAD_OK\n[2/2] Paired cached-feature inference', flush=True)
    for record in previous['records']:
        index, token = record['index'], record['token']
        raw = torch.load(control / f'raw_features_{index:03d}_{token}.pt', map_location='cpu', weights_only=False)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        after_rng = None
        for name, variant in VARIANTS.items():
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            model.diffuser.constrain_mode = variant['mode']
            model.diffuser.constraint_gamma = variant['gamma']
            scene = SceneTensor(**{k: v.cuda().unsqueeze(0).clone() for k, v in raw.items()})
            with torch.inference_mode():
                prediction = model.forward_validation({'scene_tensor': scene})
            if after_rng is None:
                after_rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
            else:
                assert torch.equal(torch.get_rng_state(), after_rng[0])
                assert torch.equal(torch.cuda.get_rng_state(), after_rng[1])
            cpu = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in prediction['behavior_prediction_out'].items()}
            for key in ('behavior_pred', 'intent_conditioned_behavior_pred'):
                assert torch.isfinite(cpu[key]).all()
            torch.save(cpu, run / f'{name}_{index:03d}_{token}.pt')
            for metric in metrics[name].values():
                metric.update({'behavior_prediction_out': copy.deepcopy(cpu)}, {})
        manifest['records'].append({'index': index, 'token': token})
        json_write(run / 'run.json', manifest)
        print(f'CONSTRAINT_SCENARIO_OK {index+1}/{len(previous["records"])} token={token}', flush=True)
    result = {branch: {name: {key: float(value.detach().cpu()) for key, value in metric.compute().items()} for name, metric in group.items()} for branch, group in metrics.items()}
    assert all(np.isfinite(value) for group in result.values() for metric in group.values() for value in metric.values())
    assert versions == [p._version for p in model.parameters()]
    json_write(run / 'metrics.json', result)
    manifest.update(status='complete', evaluated_scenarios=len(manifest['records']), model_parameter_versions_unchanged=True)
    json_write(run / 'run.json', manifest)
    print(json.dumps(result, indent=2))
    print('NUPLAN_CONSTRAINT_CONTROL_OK\nRUN_DIR=' + str(run), flush=True)


if __name__ == '__main__':
    main()
