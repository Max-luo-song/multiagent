"""Independent observed-history and initial-geometry audit, no trajectory edits."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_multiscene_20261027'
sys.path.insert(0,str(ROOT))
def main():
    import numpy as np,torch
    from shapely import wkb
    from shapely.geometry import Polygon
    from nexus_3dgs.bundle import decode
    from nexus_3dgs.scene_projection import pair_separations
    protocol=json.loads((OUT/'protocol.json').read_text());result=[]
    for info in protocol['cases']:
        idx=info['scenario']['index'];case=OUT/f'case_{idx:03d}';role=info['roles'];slots=list(role.values());z={}
        for variant in ('normal','event'):
            with np.load(case/f'conditions_{variant}.npz') as data:z[variant]={k:data[k] for k in data.files}
        normal=z['normal'];event=z['event'];states=decode(normal['values']);scene=torch.load(case/'scene.pt',weights_only=False)
        assert np.array_equal(normal['values'][:,:5],event['values'][:,:5]);assert not normal['keep'][[0,role['C']],5:].any()
        assert (normal['values'][~normal['keep']]==0).all() and (event['values'][~event['keep']]==0).all()
        channels=[0,1,2,3,6,7]
        assert np.array_equal(normal['values'][:,:5,channels],scene['reference'][:,:5,channels])
        W=np.array(json.loads((case/'reference/protocol.json').read_text())['world_from_nexus']);road=wkb.loads((case/'reference/inputs/drivable.wkb').read_bytes()).buffer(.02)
        initial=[];bodychecks={}
        for name,i in role.items():
            s=states[i,4];a=np.arctan2(s[3],s[2]);q=s[:2].copy()
            if i==0:q+=1.461*s[2:4]
            initial.append([*q,a]);rot=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]);corn=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*s[6:8]/2@rot+q;poly=Polygon(corn@W[:2,:2].T+W[:2,3]);lane=wkb.loads((case/'map/lanes'/f'{i}.wkb').read_bytes()).buffer(.02)
            bodychecks[name]={'initial_body_onroad':bool(road.covers(poly)),'initial_body_inside_selected_lane':bool(lane.covers(poly))}
        sep,_=pair_separations(np.array(initial)[None],np.ones((1,3),bool),states[slots,4,6:8])
        result.append({'index':idx,'normal_event_history_exactly_equal':True,'observed_pose_heading_size_unchanged':True,'only_history_velocity_reconstructed':True,'E_C_future_conditions_empty':True,'initial_body':bodychecks,'initial_role_gaps_m':{'EA':float(sep[0,0,1]),'EC':float(sep[0,0,2]),'AC':float(sep[0,1,2])},'given_A_usable':info['given_A_input_usable']})
    output={'status':'passed','cases':result,'future_reference_scope':'GT future retained solely for audit bookkeeping/identities/presence; E/C condition channels are zero'}
    (OUT/'input_audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2),flush=True)
if __name__=='__main__':main()
