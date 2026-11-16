"""Render the predeclared seed with frozen v9 assets, preserving failed attempts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();a.root=a.root.resolve()
    run=json.loads((a.root/'raw/run.json').read_text());assert run['status']=='complete'
    frozen=json.loads((ROOT/'artifacts/nexus_3dgs/interaction_v9_20261025/frozen_inputs.json').read_text())['inputs']
    for key in ('visual_checkpoint','visual_config','asset_registration','registered_scene','original_scene'):
        path=Path(frozen[key]['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=frozen[key]['sha256']:raise ValueError(f'Frozen input changed: {key}')
    seed=run['request']['render_seed'];folder=a.root/'renders';folder.mkdir(exist_ok=True)
    progress={}
    for name in ('normal','event'):
        out=folder/name
        if (out/'run.json').exists():
            old=json.loads((out/'run.json').read_text())
            if old.get('status')=='passed' and old['source_prediction_sha256']==run['pairs'][str(seed)][name]['prediction_sha256']:
                progress[name]='complete';continue
        if out.exists():
            attempt=1
            while (folder/f'{name}_incomplete_{attempt:02d}').exists():attempt+=1
            out.rename(folder/f'{name}_incomplete_{attempt:02d}')
        command=['bash',str(ROOT/'scripts/pipeline/nexus_3dgs.sh'),'script',str(ROOT/'scripts/pipeline/render_nexus_registered_prediction.py'),
                 '--checkpoint',frozen['visual_checkpoint']['path'],'--scene',str(Path(frozen['registered_scene']['path']).parent),
                 '--bundle',str(a.root/'raw'/run['pairs'][str(seed)][name]['path']),
                 '--asset-registration',frozen['asset_registration']['path'],'--original-scene',str(Path(frozen['original_scene']['path']).parent),
                 '--output',str(out),'--save-actor-metadata']
        (folder/f'{name}_command.json').write_text(json.dumps(command,indent=2)+'\n')
        print('TIMED_EVENT_RENDER_START',name,flush=True)
        with (folder/f'{name}.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            print((folder/f'{name}.log').read_text()[-5000:],flush=True)
            raise RuntimeError(f'Render failed: {name}, code={result.returncode}')
        progress[name]='complete';(folder/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print('TIMED_EVENT_RENDER_COMPLETE',name,flush=True)


if __name__=='__main__':main()
