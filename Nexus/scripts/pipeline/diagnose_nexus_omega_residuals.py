"""Locate residual collisions and kinematic violations without rerunning Nexus."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from nexus_3dgs.bundle import decode
    torch.set_num_threads(2)
    binding=json.loads((ROOT/'artifacts/nexus_3dgs/real_vs_nexus/inputs/goal/manifest.json').read_text())
    tokens={0:'ego',**{b['nexus_slot']:b['track_token'] for b in binding['agent_bindings']}}
    groups={'baseline':ROOT/'artifacts/nexus_omega/pilot_v3/baseline',
            'candidate':ROOT/'artifacts/nexus_omega/pilot_v4_kl20/anchor'}
    report={'status':'complete','records':[]}
    for method,root in groups.items():
        for seed in (0,1,2):
            for variant in ('normal','event'):
                folder=root/f'seed_{seed:03d}'/variant
                data=torch.load(folder/'prediction.pt',map_location='cpu',weights_only=False)
                states=decode(data['sampled_tensor'][0].numpy())
                valid=data['valid_mask'][0].bool().numpy()
                keep=data['task_mask'][0].bool().numpy().all(-1)
                vel=np.diff(states[...,:2],axis=1)/.5
                acc=np.diff(vel,axis=1)/.5
                av=valid[:,:-2]&valid[:,1:-1]&valid[:,2:]
                av[:,:3]=False
                magnitude=np.linalg.norm(acc,axis=-1)
                violations=[]
                for slot,t in np.argwhere((magnitude>5)&av):
                    violations.append({'slot':int(slot),'token':tokens.get(int(slot),'unknown'),
                        'arrival_time_s':float((t+2-4)*.5),
                        'acceleration_mps2':float(magnitude[slot,t]),
                        'all_three_states_conditioned':bool(keep[slot,t:t+3].all()),
                        'condition_mask_triplet':keep[slot,t:t+3].tolist()})
                tracked=[]
                for slot in np.flatnonzero(valid.any(-1)):
                    mask=av[slot]
                    tracked.append({'slot':int(slot),'token':tokens.get(int(slot),'unknown'),
                        'peak_acceleration_mps2':float(magnitude[slot,mask].max()) if mask.any() else 0.,
                        'violating_samples':int(((magnitude[slot]>5)&mask).sum())})
                row={'method':method,'seed':seed,'variant':variant,'violations':violations,'agents':tracked}
                if method=='candidate':
                    guidance=json.loads((folder/'guidance.json').read_text())
                    row['last_guidance_steps']=guidance[-5:]
                report['records'].append(row)
                print(method,seed,variant,[(x['slot'],x['violating_samples'],round(x['peak_acceleration_mps2'],2)) for x in tracked if x['violating_samples']],flush=True)
    for name,folder in {'baseline':ROOT/'artifacts/nexus_3dgs/timed_event_20261026/renders',
                        'candidate':ROOT/'artifacts/nexus_omega/pilot_v4_kl20/renders'}.items():
        for variant in ('normal','event'):
            path=folder/(variant if name=='baseline' else 'anchor_'+variant)/'run.json'
            run=json.loads(path.read_text());pairs={}
            for frame in run['frames']:
                meta=frame['scene_metadata']
                for i,j in meta['overlapping_actor_pairs']:
                    x,y=meta['actors'][i],meta['actors'][j]
                    key=f'{x["track_token"]}:{x["nexus_slot"]} / {y["track_token"]}:{y["nexus_slot"]}'
                    pairs.setdefault(key,[]).append(frame['frame'])
            print('RENDER',name,variant,[(k,len(v)) for k,v in pairs.items()],flush=True)
            report.setdefault('render_pairs',{})[name+'_'+variant]=pairs
    (a.output/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
