"""Inventory local inputs for the Nexus -> Omniscenegraph integration (read only)."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nexus_3dgs.nuplan import SensorLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('../runs/demo_20261020_120001_008217'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/nexus_3dgs/input_audit.json'))
    parser.add_argument('--probe-runtime', action='store_true')
    parser.add_argument('--find-sensor-scene', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / 'run.json').read_text())
    database = Path(manifest['database'])
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        schemas = {name: [dict(r) for r in db.execute(f'PRAGMA table_info({name})')]
                   for name in ('lidar_pc', 'ego_pose', 'lidar_box', 'track', 'category', 'image', 'camera', 'lidar', 'log') if name in tables}
        token = bytes.fromhex(manifest['scenario_token'])
        anchor = dict(db.execute('SELECT * FROM lidar_pc WHERE token=?', (token,)).fetchone())
        camera_rows = [dict(r) for r in db.execute('SELECT channel, translation, rotation, intrinsic, distortion, width, height FROM camera')]
        images = [dict(r) for r in db.execute('SELECT filename_jpg, timestamp FROM image WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp LIMIT 8', (anchor['timestamp'] - 2000000, anchor['timestamp'] + 8000000))]
    shared = Path('/inspire/hdd/global_public/public_datas/nuPlan')
    locator = SensorLocator([shared/'sensor_blobs_full', shared/'sensor_blobs_mini'])
    selection = None
    if args.find_sensor_scene:
        # Select only by sensor availability, without reading prediction values.
        for run in (Path('artifacts/nuplan_evaluation/independent256_v1'), Path('artifacts/nuplan_evaluation/multicity128_official_v2')):
            plan_path = run/'frozen_plan.json'
            if not plan_path.is_file():
                continue
            plan = json.loads(plan_path.read_text())
            records = json.loads((run/'run.json').read_text()).get('records', {})
            for scenario in plan['scenarios']:
                with sqlite3.connect(Path(scenario['database']).resolve().as_uri()+'?mode=ro', uri=True) as db:
                    db.row_factory = sqlite3.Row
                    sample = db.execute('SELECT filename,timestamp FROM lidar_pc WHERE token=?', (bytes.fromhex(scenario['token']),)).fetchone()
                    image = db.execute('SELECT im.filename_jpg FROM image im JOIN camera c ON c.token=im.camera_token '
                        'WHERE c.channel=? AND im.timestamp BETWEEN ? AND ? ORDER BY im.timestamp LIMIT 1',
                        ('CAM_F0', sample['timestamp']-50000, sample['timestamp']+50000)).fetchone()
                if image is None or locator.resolve(image['filename_jpg']) is None or locator.resolve(sample['filename']) is None:
                    continue
                record = next((r for r in records.values() if r['index'] == scenario['index'] and r['seed'] == 0), None)
                if record:
                    selection = {'run_dir': str(run.resolve()), 'prediction': str((run/record['prediction_file']).resolve()),
                        'database': scenario['database'], 'scenario_token': scenario['token'],
                        'image': str(locator.resolve(image['filename_jpg'])), 'lidar': str(locator.resolve(sample['filename'])),
                        'selection_rule': 'first existing seed-0 prediction with local front camera and lidar at anchor; no model values used'}
                    break
            if selection:
                break
    for row in images:
        found = locator.resolve(row['filename_jpg'])
        row['existing_paths'] = [str(found)] if found else []
    packages = {}
    for name in ('torch', 'numpy', 'omegaconf', 'gsplat', 'kornia', 'open3d', 'torchmetrics', 'nerfview', 'smplx', 'pyquaternion', 'nuplan-devkit'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    def serial(value):
        if isinstance(value, bytes):
            return {'blob_bytes': len(value), 'hex': value.hex() if len(value) <= 16 else None}
        raise TypeError(type(value).__name__)
    result = {'python': sys.executable, 'run': str(args.run_dir.resolve()), 'database': str(database),
              'scenario_token': manifest['scenario_token'], 'anchor': anchor, 'schemas': schemas,
              'cameras': camera_rows, 'sample_images': images, 'packages': packages,
              'copied_checkpoint_files': [str(p) for p in Path('Omniscenegraph/ckpt').glob('*') if p.is_file()]}
    result['sensor_scene_selection'] = selection
    if args.probe_runtime:
        import torch
        result['cuda_available'] = torch.cuda.is_available()
        result['gpu'] = torch.cuda.get_device_name(0) if result['cuda_available'] else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, default=serial, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'image_count': len(images), 'existing_image_count': sum(bool(r['existing_paths']) for r in images), 'cuda_available': result.get('cuda_available'), 'sensor_scene_selection': selection, 'packages': packages}, indent=2))


if __name__ == '__main__':
    main()
