"""Freeze the existing v9 recorded replay actors as explicit extra obstacles."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    source=ROOT/'artifacts/nexus_omega/pilot_v4_kl20'
    manifest=json.loads((source/'anchor/seed_000/event/manifest.json').read_text())
    W=np.asarray(manifest['world_from_nexus'])
    with np.load(source/'anchor/seed_000/event/trajectories.npz',allow_pickle=False) as arrays:
        current=arrays['timestamps_us'][manifest['current_index']]
    runs=[json.loads((source/f'renders/anchor_{name}/run.json').read_text()) for name in ('normal','event')]
    assert all(r['checkpoint_sha256']=='1debd88b7c7f9172a64c65b00df69257007bb1a6595001ff38ef9305a94a188b' for r in runs)
    tokens=None;centers=[];heading=[];valid=[];times=[];sizes=None
    for f,g in zip(runs[0]['frames'],runs[1]['frames']):
        assert f['timestamp_us']==g['timestamp_us']
        actors=[x for x in f['scene_metadata']['actors'] if x['nexus_slot'] is None]
        others=[x for x in g['scene_metadata']['actors'] if x['nexus_slot'] is None]
        assert actors==others
        names=[x['track_token'] for x in actors]
        if tokens is None:
            tokens=names;sizes=np.array([x['size_lwh'][:2] for x in actors])
        assert names==tokens
        xy=np.array([x['center_world'][:2] for x in actors])
        centers.append((xy-W[:2,3])@W[:2,:2])
        yaw=np.array([x['yaw_world_rad'] for x in actors])-np.arctan2(W[1,0],W[0,0])
        heading.append(np.stack([np.cos(yaw),np.sin(yaw)],-1))
        valid.append([x['valid'] for x in actors]);times.append((f['timestamp_us']-current)/1e6)
    path=a.output/'background_replay.npz'
    np.savez_compressed(path,times_s=np.asarray(times,dtype='float32'),
        centers=np.asarray(centers,dtype='float32'),heading=np.asarray(heading,dtype='float32'),
        sizes=sizes.astype('float32'),valid=np.asarray(valid,dtype=bool))
    info={'schema_version':1,'status':'frozen','actors':tokens,'frames':len(times),
        'source':'recorded unpredicted actors from existing frozen v9 scene replay',
        'extra_information':'Known recorded future of unpredicted vehicles is supplied to guidance; not history-only prediction.',
        'same_unpredicted_actors_in_old_pair_exactly':True,'coordinates':'Nexus local, physical centers',
        'extrapolation':False,'start_s':times[0],'end_s':times[-1],
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (a.output/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    print('BACKGROUND_REPLAY_FROZEN',len(tokens),len(times),flush=True)


if __name__=='__main__':main()
