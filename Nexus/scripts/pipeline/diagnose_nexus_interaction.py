"""Paired single-bus intervention with other vehicles' futures unconditioned.

Same history, map, validity and random numbers. This measures conditional
dependence in a single scene; it does not establish safe or realistic reactions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'scripts/pipeline'))
from run_repro_demo import CHECKPOINT_SHA256, file_sha256, model_state_dict


def audit_variant(states, valid, conditioned_slot):
    import numpy as np
    from collision_audit import rectangle_overlaps
    collisions = [rectangle_overlaps(states[:, t], valid[:, t]) for t in range(5, states.shape[1])]
    adjacency = np.asarray(collisions)
    colliding = adjacency.any(axis=-1)
    generated_valid = valid[:, 5:].T.copy()
    generated_valid[:, conditioned_slot] = False
    velocity = np.diff(states[:, :, :2], axis=1)/.5
    velocity_valid = valid[:, :-1] & valid[:, 1:]
    acceleration = np.diff(velocity, axis=1)/.5
    acceleration_valid = velocity_valid[:, :-1] & velocity_valid[:, 1:]
    return {'collision_scope': 'oriented 2D vehicle rectangles at 16 future samples, no continuous-time guarantee',
            'unconditioned_valid_vehicle_frames': int(generated_valid.sum()),
            'unconditioned_colliding_vehicle_frames': int((colliding & generated_valid).sum()),
            'ego_colliding_future_frames': int(colliding[:, 0].sum()),
            'ego_valid_future_frames': int(valid[0, 5:].sum()),
            'max_ego_position_derived_speed_mps': float(np.linalg.norm(velocity[0, 4:][velocity_valid[0, 4:]], axis=-1).max()),
            'max_ego_position_derived_acceleration_mps2': float(np.linalg.norm(acceleration[0, 3:][acceleration_valid[0, 3:]], axis=-1).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--index', type=int, default=96)
    parser.add_argument('--bus-slot', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.source_run/'frozen_plan.json').read_text())
    scene_record = next(s for s in plan['scenarios'] if s['index'] == args.index)
    raw_path = args.source_run/'raw_features'/f'{args.index:03d}_{scene_record["token"]}.pt'
    checkpoint = REPO.parent/'checkpoints/nuplan.ckpt'
    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Official Nexus checkpoint checksum changed')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(MPLBACKEND='Agg', WANDB_MODE='disabled', NUPLAN_DATA_STORE='local')
    print('INTERACTION_DIAGNOSTIC loading model', flush=True)
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
    from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
    from nexus_3dgs.bundle import FEATURE_MEANS, FEATURE_STD, decode, export_arrays, save_bundle
    torch.set_num_threads(2)
    cfg = OmegaConf.load(args.source_run/'config.yaml')
    model = instantiate(cfg.model)
    model.load_state_dict(model_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)), strict=True)
    model = model.cuda().eval().requires_grad_(False)
    versions = [p._version for p in model.parameters()]
    raw = torch.load(raw_path, map_location='cpu', weights_only=False)
    scene = SceneTensor(**{k:v.numpy().copy() for k, v in raw.items()})
    scene = ResetVAugmentor().augment({'scene_tensor': scene}, {})[0]['scene_tensor']
    original = torch.from_numpy(scene.tensor[..., :8].copy()).unsqueeze(0).cuda()
    validity = torch.from_numpy(scene.validity.copy()).unsqueeze(0).cuda()
    validity[~validity[:, :, :5].any(dim=-1)] = 0
    if args.bus_slot == 0 or not validity[0, args.bus_slot].all():
        raise ValueError('The conditioned bus must be present throughout the clip')
    keep = torch.zeros_like(original)
    keep[:, :, :5] = 1
    keep[:, args.bus_slot, 5:] = 1
    # Erase unconditioned future values even though the native sampler masks
    # them, so no constraint or initialization can read hidden future states.
    common = original.clone()
    common[:, :, 5:] = 0
    gt_cpu = original.cpu()
    valid_cpu = validity.cpu()
    stop = decode(gt_cpu[0, args.bus_slot, 4].numpy())
    stop[4:6] = 0
    stop = torch.as_tensor((stop-FEATURE_MEANS)/(2*FEATURE_STD), dtype=original.dtype, device='cuda')
    report = {'status': 'started', 'database': scene_record['database'], 'scenario_token': scene_record['token'],
              'checkpoint_sha256': CHECKPOINT_SHA256, 'raw_feature_sha256': file_sha256(raw_path),
              'seed': args.seed, 'conditioned_slot': args.bus_slot, 'training_updates': 0,
              'conditions': 'all agents past/current; one bus full future; all other future state values erased',
              'validity_policy': 'logged future presence retained identically in both variants',
              'rng_pairing': 'same explicit CUDA torch.randn_like latent and reset CPU/CUDA RNG at each sample; differs from released validation CPU latent generator',
              'constraints': {'modes': model.diffuser.constrain_mode, 'weights': model.diffuser.constraint_gamma},
              'scope': 'single-scene conditional response diagnostic; no safety or realism conclusion', 'variants': {}}
    def save():
        (args.output/'run.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    results = {}
    try:
        for name in ('bus_stop', 'bus_follow_logged'):
            random.seed(args.seed); np.random.seed(args.seed)
            torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
            tensor = common.clone()
            tensor[:, args.bus_slot, 5:] = stop if name == 'bus_stop' else original[:, args.bus_slot, 5:]
            features = SceneTensor(tensor=tensor, validity=validity.clone(),
                                   road_graph=torch.from_numpy(scene.road_graph.copy()).unsqueeze(0).cuda(),
                                   road_graph_validity=torch.from_numpy(scene.road_graph_validity.copy()).unsqueeze(0).cuda())
            noise = torch.randn_like(original)
            model.eval()
            with torch.inference_mode():
                result = model.forward_inference({'scene_tensor': features, 'task_mask': keep.clone()},
                                                 noise=noise, return_intermidates=False)
            prediction = result['sampled_tensor'].detach().cpu()
            torch.testing.assert_close(prediction[keep.cpu().bool()], tensor.cpu()[keep.cpu().bool()], rtol=0, atol=0)
            payload = {'sampled_tensor': prediction, 'scene_tensor_gt': gt_cpu, 'valid_mask': valid_cpu,
                       'task_mask': keep.cpu(), 'conditioned_scene_tensor': tensor.cpu()}
            path = args.output/f'{name}.pt'
            torch.save(payload, path)
            context = {'database': scene_record['database'], 'scenario_token': scene_record['token'],
                       'prediction': str(path.resolve()), 'prediction_sha256': file_sha256(path),
                       'intervention': name, 'conditioned_slot': args.bus_slot, 'seed': args.seed}
            item = {'prediction': path.name, 'sha256': file_sha256(path),
                    'initial_noise_sha256': hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                    'given_logged_future_for_bus': name == 'bus_follow_logged', 'given_other_agents_future_states': False}
            try:
                arrays, metadata = export_arrays(payload, context, task='sampled_tensor')
                save_bundle(args.output/name, arrays, metadata)
                item['bundle'] = str((args.output/name).resolve())
                item['conditioned_track_token'] = str(arrays['track_tokens'][list(arrays['nexus_slots']).index(args.bus_slot)])
            except ValueError as exc:
                item['export_error'] = str(exc)
            results[name] = decode(prediction[0].numpy())
            report['variants'][name] = item
            save()
            print('INTERACTION_VARIANT '+name, flush=True)
        rows = []
        valid = valid_cpu[0].numpy().astype(bool)
        left, right = results['bus_stop'], results['bus_follow_logged']
        for slot in np.flatnonzero(valid[:, 5:].any(axis=-1)):
            present = valid[slot, 5:]
            delta = np.linalg.norm(left[slot, 5:, :2]-right[slot, 5:, :2], axis=-1)[present]
            paths = {}
            for name, values in results.items():
                both = valid[slot, 4:-1] & valid[slot, 5:]
                paths[name] = float(np.linalg.norm(np.diff(values[slot, 4:, :2], axis=0), axis=-1)[both].sum())
            rows.append({'slot': int(slot), 'directly_conditioned': bool(slot == args.bus_slot),
                         'mean_response_distance_m': float(delta.mean()), 'max_response_distance_m': float(delta.max()),
                         'last_valid_response_distance_m': float(delta[-1]), 'path_lengths_m': paths})
        if versions != [p._version for p in model.parameters()]:
            raise RuntimeError('Nexus weights changed during inference')
        hashes = [v['initial_noise_sha256'] for v in report['variants'].values()]
        if len(set(hashes)) != 1:
            raise RuntimeError('Paired initial noise differs')
        report.update(status='complete', model_parameters_unchanged=True, actors=rows,
                      geometry_audit={name: audit_variant(values, valid, args.bus_slot) for name, values in results.items()})
        save()
        print(json.dumps(rows, indent=2), flush=True)
    except Exception as exc:
        report.update(status='failed', error=str(exc)); save()
        raise


if __name__ == '__main__':
    main()
