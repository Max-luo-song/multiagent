"""Separate collisions already forced by supplied A and recorded background."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_multiscene_20261027'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
def main():
    import numpy as np
    from nexus_3dgs.bundle import decode
    from audit_nexus_omega_v2 import sample_positions
    p=json.loads((OUT/'protocol.json').read_text());rows=[];times=np.arange(81)*.1
    for info in p['cases']:
        idx=info['scenario']['index'];case=OUT/f'case_{idx:03d}';A=info['roles']['A']
        with np.load(case/'reference/inputs/background_replay.npz') as z:bg={k:z[k] for k in z.files}
        for v in ('normal','event'):
            with np.load(case/f'conditions_{v}.npz') as z:states=decode(z['values'])[A:A+1];valid=z['valid'][A:A+1]
            xy,yaw,seen=sample_positions(states,valid,times,np);xy=xy[0];yaw=yaw[0]
            h=np.c_[np.cos(yaw),np.sin(yaw)];s=np.c_[-h[:,1],h[:,0]];bh=bg['heading'];bs=np.stack([-bh[...,1],bh[...,0]],-1)
            aa=np.broadcast_to(h[:,None],bh.shape);ab=np.broadcast_to(s[:,None],bh.shape);axes=np.stack([aa,ab,bh,bs],-2)
            delta=bg['centers']-xy[:,None];size=states[0,4,6:8]
            ra=abs((axes*aa[:,:,None]).sum(-1))*size[0]/2+abs((axes*ab[:,:,None]).sum(-1))*size[1]/2
            rb=abs((axes*bh[:,:,None]).sum(-1))*bg['sizes'][None,:,0:1]/2+abs((axes*bs[:,:,None]).sum(-1))*bg['sizes'][None,:,1:2]/2
            sep=(abs((delta[:,:,None]*axes).sum(-1))-ra-rb).max(-1);hits=(sep<0)&bg['valid'];hits[0]=False
            rows.append({'index':idx,'variant':v,'fixed_A_vs_recorded_background_overlap_samples':int(hits.sum()),'overlap_times_s':times[hits.any(-1)].tolist(),'background_tokens':[p['cases'][0]['scenario']['token']] if False else []})
            reference=json.loads((case/'reference/protocol.json').read_text());rows[-1]['background_tokens']=[reference['background_tokens'][i] for i in np.flatnonzero(hits.any(0))]
    result={'scope':'given A and immutable recorded background; no generated E/C used','records':rows}
    (OUT/'fixed_A_background_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
