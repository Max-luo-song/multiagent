"""Restore only a named bus's pre-refit Gaussian asset; keep every v8 pose."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['base', 'asset-source', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--track-token', required=True)
    a = p.parse_args()
    import torch
    import yaml
    from scripts.pipeline.merge_nexus_vehicle_asset import equal
    torch.set_num_threads(8)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source_hashes = {str(path): digest(path) for path in [a.base, a.asset_source]}
    cfg = yaml.safe_load((a.base.parent / 'config.yaml').read_text())
    scene = Path(cfg['data']['data_root']) / cfg['data']['scene_idx']
    info = json.loads((scene / 'instances/instances_info.json').read_text())
    dataset_id = next(int(i) for i, row in info.items() if row['id'] == a.track_token)
    base = torch.load(a.base, map_location='cpu', weights_only=False)
    source = torch.load(a.asset_source, map_location='cpu', weights_only=False)
    old = base['models']['RigidNodes']
    donor = source['models']['RigidNodes']
    equal(old['dataset_instance_ids'], donor['dataset_instance_ids'])
    equal(old['points_ids'], donor['points_ids'])
    equal(old['instances_size'], donor['instances_size'])
    index = int(torch.nonzero(old['dataset_instance_ids'] == dataset_id).item())
    selected = old['points_ids'].reshape(-1) == index
    keys = ['_means', '_scales', '_quats', '_features_dc', '_features_rest', '_opacities']
    changed = []
    for key in keys:
        if not torch.equal(old[key][selected], donor[key][selected]):
            changed.append(key)
        old[key][selected] = donor[key][selected]
    assert changed
    a.output.mkdir(parents=True, exist_ok=False)
    output = a.output / 'checkpoint_final.pth'
    torch.save(base, output)
    shutil.copyfile(a.base.parent / 'config.yaml', a.output / 'config.yaml')
    original = torch.load(a.base, map_location='cpu', weights_only=False)
    saved = torch.load(output, map_location='cpu', weights_only=False)
    for name, model in original['models'].items():
        if name != 'RigidNodes':
            equal(model, saved['models'][name])
            continue
        for key, value in model.items():
            if key in keys:
                equal(value[~selected], saved['models'][name][key][~selected])
                equal(donor[key][selected], saved['models'][name][key][selected])
            else:
                equal(value, saved['models'][name][key])
    for path, before in source_hashes.items():
        assert digest(Path(path)) == before
    report = {
        'status': 'passed', 'base': str(a.base.resolve()),
        'asset_source': str(a.asset_source.resolve()), 'source_sha256': source_hashes,
        'output_sha256': digest(output), 'track_token': a.track_token,
        'model_index': index, 'bus_gaussians': int(selected.sum()),
        'changed_bus_parameters': changed,
        'all_v8_body_poses_camera_background_and_other_assets_bit_exact': True,
        'no_training_or_trajectory_edits': True,
        'scope': 'diagnostic bus-only pre-refit asset restoration; visual quality pending evaluation',
    }
    (a.output / 'asset_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
