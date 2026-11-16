"""CPU integration check: build every frozen scene with the real nuPlan builder."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from run_nuplan_metrics import json_write
from run_repro_demo import file_sha256
from nuplan_plan_queries import validate_plan_metadata, QUERY_SOURCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-cache',type=Path)
    parser.add_argument('--feature-limit',type=int,default=4)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    expected_count=len(plan['scenarios'])
    assert validate_plan_metadata(plan) == expected_count
    print('METADATA_CHECK_OK; importing real scenario builder for CPU integration check', flush=True)
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(repo))
    sys.path.insert(0, str(repo/'third_party/nuplan-devkit'))
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(plan.get('reference_configuration') or plan['configuration'])
    cfg.scenario_filter.limit_total_scenarios = None
    cfg.scenario_filter.timestamp_threshold_s = None
    groups = defaultdict(list)
    for s in plan['scenarios']:
        groups[s['database']].append(s)
    worker = instantiate(cfg.worker)
    checked = []
    feature_checks=[]
    if args.feature_cache:
        import copy
        import numpy as np
        import torch
        torch.set_num_threads(2)
        from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
        from nuplan_extent.planning.training.data_augmentation.only_centerline_reset_v import CenterlineVAugmentor
        from nuplan_extent.planning.training.preprocessing.features.scene_tensor import decode_scene_tensor,encode_scene_tensor
        from prepare_nuplan_training_recipe import causal_history
        feature_builder=instantiate(cfg.model.feature_builders[0])
        reset,centerline=ResetVAugmentor(),CenterlineVAugmentor()
        args.feature_cache.mkdir(parents=True,exist_ok=True)
    for database, samples in groups.items():
        cfg.scenario_builder.db_files = [database]
        cfg.scenario_filter.log_names = [samples[0]['log_name']]
        cfg.scenario_filter.scenario_tokens = [s['token'] for s in samples]
        builder = instantiate(cfg.scenario_builder)
        built = builder.get_scenarios(instantiate(cfg.scenario_filter), worker)
        actual = {str(s.token): s for s in built}
        assert set(actual) == {s['token'] for s in samples}, (database, sorted(actual))
        for s in samples:
            found = actual[s['token']]
            assert found.log_name == s['log_name']
            assert found._initial_lidar_timestamp == s['timestamp_us']
            assert found.scenario_type == s['type']
            checked.append(dict(index=s['index'], token=s['token'], log_name=found.log_name))
            if args.feature_cache and s['index']<args.feature_limit:
                raw=feature_builder.get_features_from_scenario(found)
                assert raw.tensor.shape==(129,21,8) and np.isfinite(raw.tensor).all()
                released=reset.augment({'scene_tensor':copy.deepcopy(raw)}, {})[0]['scene_tensor']
                trained=centerline.augment({'scene_tensor':copy.deepcopy(raw)}, {})[0]['scene_tensor']
                causal=causal_history(trained,raw,decode_scene_tensor,encode_scene_tensor)
                assert causal.tensor.shape==(64,21,8)
                assert np.array_equal(released.validity[:64],causal.validity)
                assert np.allclose(released.tensor[:64,:,:4],causal.tensor[:,:,:4],atol=1e-6,rtol=1e-6)
                target=args.feature_cache/f'{s["index"]:03d}_{s["token"]}.pt'
                temporary=target.with_suffix('.tmp')
                torch.save({k:torch.from_numpy(v.copy()) for k,v in vars(raw).items()},temporary)
                temporary.replace(target)
                feature_checks.append(dict(index=s['index'],token=s['token'],raw_feature=str(target.resolve()),
                                           sha256=file_sha256(target),recipes_checked=2))
        print(f'BUILDER_CHECK_PROGRESS scenes={len(checked)}/{expected_count}', flush=True)
    assert len(checked) == expected_count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_write(args.output, dict(status='complete', plan_sha256=file_sha256(args.plan),
                                scenario_query_sha256=file_sha256(QUERY_SOURCE),
                                checked_scenarios=expected_count, checked_databases=len(groups), records=checked,
                                feature_checks=feature_checks))
    print(f'OFFICIAL_BUILDER_CHECK_OK: {expected_count}/{expected_count} scenarios; no model or CUDA inference', flush=True)


if __name__ == '__main__':
    main()
