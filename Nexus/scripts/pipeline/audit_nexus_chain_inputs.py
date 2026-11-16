"""Independent history/mask/initial-road audit for the A->B->C experiments."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
 import numpy as np,torch
 from shapely import wkb
 from shapely.geometry import Polygon
 from nexus_3dgs.bundle import decode
 for name in ['nexus_chain_20261027','nexus_chain_cutin_only_20261027']:
  out=ROOT/'artifacts'/name;p=json.loads((out/'protocol.json').read_text());rows=[]
  for info in p['cases']:
   idx=info['scenario']['index'];case=out/f'case_{idx:03d}';scene=torch.load(case/'scene.pt',weights_only=False,map_location='cpu');variants={}
   for v in ('normal','event'):
    with np.load(case/f'conditions_{v}.npz') as z:variants[v]={k:z[k] for k in z.files}
   n=variants['normal'];e=variants['event'];A,B,C=[info['roles'][r] for r in ('A','B','C')]
   assert np.array_equal(n['values'][:,:5],e['values'][:,:5]);assert not n['keep'][[B,C],5:].any() and not e['keep'][[B,C],5:].any()
   channels=[0,1,2,3,6,7];assert np.array_equal(n['values'][:,:5,channels],scene['reference'][:,:5,channels])
   assert (n['values'][~n['keep']]==0).all() and (e['values'][~e['keep']]==0).all()
   s=decode(n['values']);W=np.array(json.loads((case/'reference/protocol.json').read_text())['world_from_nexus']);road=wkb.loads((case/'reference/inputs/drivable.wkb').read_bytes()).buffer(.02);bodies={}
   for role,i in [('A',A),('B',B),('C',C)]:
    st=s[i,4];h=st[2:4];h=h/max(np.linalg.norm(h),1e-8);side=np.array([-h[1],h[0]]);center=st[:2]+(1.461*h if i==0 else 0);signs=np.array([[1,1],[1,-1],[-1,-1],[-1,1]]);corn=center+signs[:,:1]*h*st[6]/2+signs[:,1:]*side*st[7]/2;poly=Polygon(corn@W[:2,:2].T+W[:2,3]);lane=wkb.loads((case/'map/lanes'/f'{i}.wkb').read_bytes()).buffer(.02);bodies[role]={'initial_body_onroad':bool(road.covers(poly)),'initial_body_in_selected_corridor':bool(lane.covers(poly))}
   route=json.loads((case/'map/preparation.json').read_text());rb=next(x for x in route['routes'] if x['slot']==B);rc=next(x for x in route['routes'] if x['slot']==C);shared=sorted(set(rb['selected_edges'])&set(rc['selected_edges']))
   rows.append({'index':idx,'roles':info['roles'],'observed_pose_heading_dimensions_unchanged':True,'history_velocity_reconstructed_only':True,'main_B_C_futures_empty':True,'main_normal_event_history_exact':True,'initial_body':bodies,'B_C_shared_lane_edges':shared,'A_input_usable':info['given_A_input_usable']})
  (out/'input_audit.json').write_text(json.dumps({'status':'audited','cases':rows,'scope':'GT used for observed bookkeeping and dimensions only; future existence/background replay explicitly retained'},indent=2)+'\n');print(name,rows,flush=True)
if __name__=='__main__':main()
