"""Render fixed seed 0 with frozen v9 assets; keep raw sampler trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

DEFAULT_BRIDGE = Path('/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--bridge-root', type=Path, default=DEFAULT_BRIDGE)
    p.add_argument('--methods', nargs='+', default=['anchor','omega'])
    args = p.parse_args()
    root, bridge = args.experiment.resolve(), args.bridge_root.resolve()
    run = json.loads((root/'run.json').read_text())
    assert run['status'] == 'complete'
    frozen = json.loads((bridge/'artifacts/nexus_3dgs/interaction_v9_20261025/frozen_inputs.json').read_text())['inputs']
    keys = ['visual_checkpoint','visual_config','asset_registration','registered_scene','original_scene']
    for key in keys:
        assert digest(frozen[key]['path']) == frozen[key]['sha256'], key
    renders = root/'renders'
    renders.mkdir(exist_ok=True)
    progress = {}
    for method in args.methods:
        for variant in ('normal','event'):
            row = next(r for r in run['records'] if r['method']==method and r['seed']==0 and r['variant']==variant)
            out = renders/f'{method}_{variant}'
            if (out/'run.json').exists():
                prior = json.loads((out/'run.json').read_text())
                assert prior['status']=='passed'
                assert prior['source_prediction_sha256']==digest(root/row['path']/'prediction.pt')
                continue
            if out.exists():
                raise ValueError('Incomplete rendering preserved: '+str(out))
            command = ['bash',str(bridge/'scripts/pipeline/nexus_3dgs.sh'),'script',
                str(bridge/'scripts/pipeline/render_nexus_registered_prediction.py'),
                '--checkpoint',frozen['visual_checkpoint']['path'],
                '--scene',str(Path(frozen['registered_scene']['path']).parent),
                '--bundle',str(root/row['path']),
                '--asset-registration',frozen['asset_registration']['path'],
                '--original-scene',str(Path(frozen['original_scene']['path']).parent),
                '--output',str(out),'--save-actor-metadata']
            (renders/f'{method}_{variant}_command.json').write_text(json.dumps(command,indent=2)+'\n')
            print('OMEGA_RENDER_START',method,variant,flush=True)
            with (renders/f'{method}_{variant}.log').open('w') as log:
                result = subprocess.run(command,cwd=bridge,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                print((renders/f'{method}_{variant}.log').read_text()[-5000:],flush=True)
                raise RuntimeError('Render failed')
            progress[f'{method}_{variant}'] = 'complete'
            (renders/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            print('OMEGA_RENDER_COMPLETE',method,variant,flush=True)
    for key in keys:
        assert digest(frozen[key]['path']) == frozen[key]['sha256'], key
    print('OMEGA_RENDERS_COMPLETE',flush=True)


if __name__=='__main__':
    main()
