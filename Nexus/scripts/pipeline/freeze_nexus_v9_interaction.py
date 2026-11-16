"""Freeze the accepted visual baseline and the inputs of a paired experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/nexus_3dgs/interaction_v9_20261025'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    base = ROOT / 'artifacts/nexus_3dgs'
    inputs = {
        'visual_checkpoint': base/'optimization_v9/bus_asset_restored/checkpoint_final.pth',
        'visual_config': base/'optimization_v9/bus_asset_restored/config.yaml',
        'baseline_manifest': base/'optimization_v8/stability_candidates/heading_and_bus_path/manifest.json',
        'baseline_trajectories': base/'optimization_v8/stability_candidates/heading_and_bus_path/trajectories.npz',
        'asset_registration': base/'optimization_v6/source/coherent_rear_go/run.json',
        'registered_scene': base/'optimization_v7/registered_data/2021.10.05.04.38.41_veh-50_00014_00429__a89c1931e7b25976/nexus_scene.json',
        'original_scene': base/'real_vs_nexus/inputs/scene/nexus_scene.json',
        'nexus_checkpoint': ROOT.parent/'checkpoints/nuplan.ckpt',
        'source_config': ROOT/'artifacts/nuplan_evaluation/independent256_v1/config.yaml',
        'source_plan': ROOT/'artifacts/nuplan_evaluation/independent256_v1/frozen_plan.json',
        'source_raw_features': ROOT/'artifacts/nuplan_evaluation/independent256_v1/raw_features/096_a89c1931e7b25976.pt',
        'interaction_runner': ROOT/'scripts/pipeline/diagnose_nexus_interaction.py',
        'renderer': ROOT/'scripts/pipeline/render_nexus_registered_prediction.py',
    }
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], capture_output=True, text=True, check=True).stdout
    records = {}
    for name, path in inputs.items():
        records[name] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
    expected = {'visual_checkpoint':'1debd88b7c7f9172a64c65b00df69257007bb1a6595001ff38ef9305a94a188b',
                'nexus_checkpoint':'679f6ccfbd5563f696218311566b76bb62f73c7fbfb04b39f2714fb80aefacfc'}
    for name, digest in expected.items():
        assert records[name]['sha256'] == digest, name
    snap = OUT/'input_snapshots'
    snap.mkdir()
    for name, path in inputs.items():
        if name not in ('visual_checkpoint', 'nexus_checkpoint'):
            shutil.copy2(path, snap/(name + path.suffix))
    # Freeze the full small pose/calibration/identity input; preserve existing
    # image and LiDAR links instead of duplicating shared sensor observations.
    scene = inputs['registered_scene'].parent
    scene_files = {}
    for path in sorted(scene.rglob('*')):
        if path.is_file() and not path.is_symlink():
            rel = str(path.relative_to(scene))
            if rel.split('/')[0] not in ('images','lidar','sky_masks','road_masks','dynamic_masks','vehicle_interior_masks'):
                scene_files[rel] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    record = {'status':'frozen','accepted_visual_baseline':'v9; user acceptance supersedes historical candidate decision',
              'inputs':records, 'scene_metadata_files':scene_files, 'gpu_at_freeze':gpu,
              'experiment':{'index':96,'bus_slot':1,'seed':0,'variants':['bus_stop','bus_follow_logged'],
                            'other_agent_futures_erased':True,'original_outputs_preserved':True},
              'baseline_trajectory_note':'v8 derived display baseline; not reused as new raw paired predictions',
              'new_scene_scope':'3-5 additional clips for engineering validation; no unseen-scene benchmark claim'}
    (OUT/'frozen_inputs.json').write_text(json.dumps(record,indent=2)+'\n')
    (OUT/'progress.json').write_text(json.dumps({'1_freeze':'complete','2_pair':'pending','3_analysis':'pending','4_visualization':'pending','5_new_scenes':'pending'},indent=2)+'\n')
    print('V9_INPUTS_FROZEN',OUT, 'scene_files',len(scene_files), flush=True)


if __name__ == '__main__':
    main()
