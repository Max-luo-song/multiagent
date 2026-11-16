"""Audit train/test preprocessing and cache two 64-agent diagnostic inputs on CPU."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

from run_nuplan_metrics import json_write
from run_repro_demo import file_sha256
from training_recipe_common import BRANCHES, SOURCE_FILES, check_prepared


def causal_history(exact, raw, decode, encode):
    """Replace only current velocity; use the last observed position interval.

    No future positions are read for the history boundary. If the last observed
    interval is invalid, retain the raw current measured velocity.
    """
    import numpy as np
    feature = copy.deepcopy(exact)
    observed = decode(raw.tensor[:64, :5]).copy()
    valid_pair = raw.validity[:64, 3].astype(bool) & raw.validity[:64, 4].astype(bool)
    velocity = np.where(valid_pair[:, None], (observed[:, 4, :2]-observed[:, 3, :2])*2,
                        observed[:, 4, 4:6])
    state = observed[:, 4].copy()
    state[:, 4:6] = velocity
    feature.tensor[:, 4, 4:6] = encode(state)[:, 4:6].astype(feature.tensor.dtype)
    return feature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference, output = args.reference_run.resolve(), args.output.resolve()
    baseline = json.loads((reference/'run.json').read_text())
    assert baseline['status'] == 'complete' and baseline['evaluated_scenarios'] == 128
    if (output/'run.json').is_file() and json.loads((output/'run.json').read_text())['status'] == 'complete':
        check_prepared(output, reference)
        print('PREPARED_FEATURES_ALREADY_COMPLETE\nFEATURE_DIR='+str(output), flush=True)
        return
    samples = baseline['scenarios']
    for s in samples:
        assert (reference/'raw_features'/f'{s["index"]:03d}_{s["token"]}.pt').is_file()
    output.mkdir(parents=True, exist_ok=True)
    state = dict(status='preparing', scope='published_training_recipe_and_history_boundary_diagnostic',
                 reference_run=str(reference), reference_manifest_sha256=file_sha256(reference/'run.json'),
                 plan_sha256=baseline['plan_sha256'], branches=list(BRANCHES), scenarios=samples,
                 preparation_runner_sha256=file_sha256(Path(__file__)),
                 source_sha256={p:file_sha256(Path(p)) for p in SOURCE_FILES}, records={},
                 training_updates=0, gpu_inference=False,
                 comparison_note='Joint recipe comparison; velocity, map validity and agent cap change together. '
                                 'Only common first 64 agent slots are used for paired displacement comparison.',
                 causal_note='Published recipe uses forward position difference at current frame. '
                             'Causal branch replaces only current velocity using last observed interval, '
                             'falling back to raw current velocity when that interval is invalid.')
    json_write(output/'run.json',state)
    print('CPU_ONLY: loading existing features; no model or GPU inference', flush=True)
    os.environ['MPLBACKEND'] = 'Agg'
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(repo)); sys.path.insert(0,str(repo/'third_party/nuplan-devkit'))
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor, decode_scene_tensor, encode_scene_tensor
    from nuplan_extent.planning.training.data_augmentation.only_centerline_reset_v import CenterlineVAugmentor
    from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
    centerline, reset = CenterlineVAugmentor(), ResetVAugmentor()
    pair_count = 0
    speed_sums = dict(raw=0., test_reset=0., published_training=0., causal_history=0.)
    future_dependent_scenes = 0
    try:
        for s in samples:
            path = reference/'raw_features'/f'{s["index"]:03d}_{s["token"]}.pt'
            stored = torch.load(path,map_location='cpu',weights_only=False)
            raw = SceneTensor(**{k:v.numpy().copy() for k,v in stored.items()})
            test = reset.augment({'scene_tensor':copy.deepcopy(raw)}, {})[0]['scene_tensor']
            exact = centerline.augment({'scene_tensor':copy.deepcopy(raw)}, {})[0]['scene_tensor']
            causal = causal_history(exact,raw,decode_scene_tensor,encode_scene_tensor)
            assert exact.tensor.shape == causal.tensor.shape == (64,21,8)
            assert np.allclose(exact.tensor[...,:4],test.tensor[:64,...,:4],atol=1e-7)
            assert np.array_equal(exact.validity,raw.validity[:64])
            assert np.array_equal(causal.tensor[:,:,:4],exact.tensor[:,:,:4])
            assert np.array_equal(causal.tensor[:,5:],exact.tensor[:,5:])
            # Perturb ONLY the unknown future, retaining all observed states.
            probe = copy.deepcopy(raw)
            probe.tensor[:,5:,0] += .01
            changed = centerline.augment({'scene_tensor':copy.deepcopy(probe)}, {})[0]['scene_tensor']
            fixed_probe = causal_history(changed,probe,decode_scene_tensor,encode_scene_tensor)
            assert np.array_equal(fixed_probe.tensor[:,:5],causal.tensor[:,:5]), 'Future leaked into causal history'
            boundary_valid = exact.validity[:,4].astype(bool) & exact.validity[:,5].astype(bool)
            delta = np.abs(changed.tensor[:,4,4:6]-exact.tensor[:,4,4:6])[boundary_valid]
            dependency = bool(delta.size and delta.max()>1e-6)
            future_dependent_scenes += int(dependency)
            valid_pairs = raw.validity[1:64,:4].astype(bool) & raw.validity[1:64,1:5].astype(bool)
            count = int(valid_pairs.sum()); pair_count += count
            for label,feature in [('raw',raw),('test_reset',test),('published_training',exact),('causal_history',causal)]:
                speeds = np.linalg.norm(decode_scene_tensor(feature.tensor)[1:64,1:5,4:6],axis=-1)
                speed_sums[label] += float(speeds[valid_pairs].sum())
            features = {}
            for branch,feature in zip(BRANCHES,(exact,causal)):
                name = f'{s["index"]:03d}_{branch}.pt'
                target = output/name
                temporary = target.with_suffix('.tmp')
                torch.save({k:torch.from_numpy(v.copy()) for k,v in vars(feature).items()},temporary)
                temporary.replace(target)
                features[branch] = dict(file=name,sha256=file_sha256(target))
            state['records'][str(s['index'])] = dict(token=s['token'],features=features,raw_feature_sha256=file_sha256(path),
                raw_valid_map_points=int(raw.road_graph_validity[...,0].sum()),
                training_valid_map_points=int(exact.road_graph_validity[...,0].sum()),
                published_history_changed_after_future_perturbation=dependency,
                causal_history_unchanged_after_future_perturbation=True)
            if (s['index']+1)%16==0:
                print(f'FEATURE_AUDIT_PROGRESS {s["index"]+1}/128',flush=True)
                json_write(output/'run.json',state)
        state.update(status='complete',prepared_scenarios=128,valid_history_pairs=pair_count,
                     observed_history_speed_mean_mps={k:v/pair_count for k,v in speed_sums.items()},
                     published_history_future_dependent_scenes=future_dependent_scenes,
                     causal_future_invariance_passed_scenes=128,
                     checkpoint_training_recipe_confirmed=False)
        json_write(output/'run.json',state)
        check_prepared(output,reference)
        print('TRAINING_RECIPE_FEATURES_OK\nFEATURE_DIR='+str(output),flush=True)
        print('PUBLISHED_CURRENT_VELOCITY_USES_FUTURE_SCENES='+str(future_dependent_scenes),flush=True)
        print('HISTORY_SPEEDS_MPS='+json.dumps(state['observed_history_speed_mean_mps']),flush=True)
    except Exception as exc:
        state.update(status='failed',error=repr(exc));json_write(output/'run.json',state)
        raise


if __name__ == '__main__':
    main()
