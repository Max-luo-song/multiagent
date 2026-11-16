"""Paired constraint ablation on one fixed cached scene, with no weight updates.

All variants are reported. This diagnostic changes inference constraints and
must not be presented as the released paper configuration or interaction test.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'scripts/pipeline'))
from run_repro_demo import CHECKPOINT_SHA256, file_sha256, model_state_dict

VARIANTS = {
    'released_constraints': (['map', 'sma', 'clip'], [1., 1., 1.]),
    'without_sma': (['map', 'clip'], [1., 1.]),
    'clip_only': (['clip'], [1.]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--index', type=int, default=96)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.source_run/'frozen_plan.json').read_text())
    sample = next(s for s in plan['scenarios'] if s['index'] == args.index)
    raw_path = args.source_run/'raw_features'/f'{args.index:03d}_{sample["token"]}.pt'
    checkpoint = REPO.parent/'checkpoints/nuplan.ckpt'
    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Nexus checkpoint differs from verified official weights')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'started', 'scope': 'single_scene_paired_constraint_diagnostic',
        'database': sample['database'], 'scenario_token': sample['token'], 'seed': args.seed,
        'raw_feature': str(raw_path.resolve()), 'raw_feature_sha256': file_sha256(raw_path),
        'checkpoint_sha256': CHECKPOINT_SHA256, 'training_updates': 0,
        'rng_pairing': 'same single-scene input shapes, same RNG state before each sample',
        'designed_interaction_condition': False, 'variants': {}}
    def save():
        (args.output/'run.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    os.environ.update(MPLBACKEND='Agg', WANDB_MODE='disabled', NUPLAN_DATA_STORE='local')
    print('MOTION_DIAGNOSTIC loading model', flush=True)
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
    from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
    from nexus_3dgs.bundle import decode, export_arrays, save_bundle
    torch.set_num_threads(2)
    cfg = OmegaConf.load(args.source_run/'config.yaml')
    model = instantiate(cfg.model)
    model.load_state_dict(model_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)), strict=True)
    model = model.cuda().eval().requires_grad_(False)
    versions = [p._version for p in model.parameters()]
    raw = torch.load(raw_path, map_location='cpu', weights_only=False)
    scene = SceneTensor(**{k:v.numpy().copy() for k, v in raw.items()})
    scene = ResetVAugmentor().augment({'scene_tensor': scene}, {})[0]['scene_tensor']
    try:
        for name, (modes, weights) in VARIANTS.items():
            random.seed(args.seed); np.random.seed(args.seed)
            torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
            model.diffuser.constrain_mode, model.diffuser.constraint_gamma = modes, weights
            current = SceneTensor(**{key:torch.from_numpy(getattr(scene, key).copy()).unsqueeze(0).cuda()
                for key in ('tensor', 'validity', 'road_graph', 'road_graph_validity')})
            with torch.inference_mode():
                result = model.forward_validation({'scene_tensor': current})['behavior_prediction_out']
            cpu = {key:value.detach().cpu().clone() if torch.is_tensor(value) else value for key, value in result.items()}
            path = args.output/f'{name}.pt'
            torch.save(cpu, path)
            valid = cpu['valid_mask'][0, :, 4:].bool().numpy()
            means = decode(cpu['behavior_pred'][0].numpy())[:, 4:, :2]
            both = valid[:, 1:] & valid[:, :-1]
            lengths = (np.linalg.norm(np.diff(means, axis=1), axis=-1)*both).sum(axis=1)
            active = np.flatnonzero(valid.any(axis=1))
            item = {'modes': modes, 'weights': weights, 'prediction_file': path.name,
                'sha256': file_sha256(path), 'path_length_m_by_slot': {str(a):float(lengths[a]) for a in active}}
            context = {'database': sample['database'], 'scenario_token': sample['token'],
                'run_dir': str(args.output.resolve()), 'prediction': str(path.resolve()),
                'sampling_diagnostic': name, 'sampling_constraints': {'modes': modes, 'weights': weights},
                'seed': args.seed, 'checkpoint_sha256': CHECKPOINT_SHA256}
            try:
                arrays, metadata = export_arrays(cpu, context, task='behavior_pred')
                save_bundle(args.output/name, arrays, metadata)
                item['bundle'] = str((args.output/name).resolve())
            except ValueError as exc:
                item['export_error'] = str(exc)
            report['variants'][name] = item
            save()
            print('MOTION_VARIANT '+name+' '+json.dumps(item), flush=True)
        if versions != [p._version for p in model.parameters()]:
            raise RuntimeError('Model parameters changed during diagnostic')
        report.update(status='complete', model_parameters_unchanged=True)
        save()
    except Exception as exc:
        report.update(status='failed', error=str(exc)); save()
        raise


if __name__ == '__main__':
    main()
