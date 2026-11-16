"""History-only geometry and SDK lane screening for a C lane-change pilot."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/nexus_lane_response_20261027'
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/nuplan-devkit')]

def main():
 import numpy as np,torch
 from nexus_3dgs.bundle import decode,export_arrays
 from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
 from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
 from nuplan.common.actor_state.state_representation import Point2D
 from shapely.geometry import Point
 torch.set_num_threads(2);OUT.mkdir(exist_ok=False)
 src=ROOT/'artifacts/nuplan_evaluation/independent256_v1'
 rows=json.loads((src/'frozen_plan.json').read_text())['scenarios'];ranked=[];screened=[]
 apis={}
 for row in rows:
  idx=row['index'];raw=torch.load(src/'raw_features'/f'{idx:03d}_{row["token"]}.pt',weights_only=False,map_location='cpu')
  s=decode(raw['tensor'].numpy()[...,:8]);valid=raw['validity'].numpy().astype(bool)
  speed=np.linalg.norm((s[:,4,:2]-s[:,3,:2])/.5,axis=-1)
  ok=valid.all(-1)&(s[:,4,6]>3)&(s[:,4,7]>1.4)&(speed>3)
  triples=[]
  for B in np.flatnonzero(ok&(speed>5)):
   h=s[B,4,2:4];h=h/max(np.linalg.norm(h),1e-8);side=np.array([-h[1],h[0]])
   rel=s[:,4,:2]-s[B,4,:2];x=rel@h;y=rel@side;aligned=(s[:,4,2:4]@h)>.97
   aa=np.flatnonzero(ok&aligned&(x>4)&(x<32)&(abs(y)>2.5)&(abs(y)<4.8)&(speed<speed[B]-.5))
   cc=np.flatnonzero(ok&aligned&(x<-6)&(x>-43)&(abs(y)<1)&(speed>=speed[B]-1.5))
   for C in cc:
    gap=-x[C]-(s[B,4,6]+s[C,4,6])/2+(1.461 if B==0 else 0)-(1.461 if C==0 else 0)
    if gap<3 or gap/max(speed[C],.1)>3:continue
    for A in aa:
     if A==0 or A==C:continue
     triples.append((int(A),int(B),int(C),float(gap),float(x[A]),float(y[A])))
  if not triples:continue
  ar,meta=export_arrays({'sampled_tensor':raw['tensor'][...,:8][None],'scene_tensor_gt':raw['tensor'][...,:8][None],'valid_mask':raw['validity'][None]},{'database':row['database'],'scenario_token':row['token']})
  W=np.array(meta['world_from_nexus']);R=W[:2,:2];off=W[:2,3]
  name=row['map_name']
  if name not in apis:apis[name]=get_maps_api(str(ROOT.parent/'data/nuplan/maps'),'nuplan-maps-v1.0',name)
  api=apis[name]
  def lane(slot):
   xy=s[slot,4,:2]@R.T+off;h=s[slot,4,2:4]@R.T;p=Point(xy);cs=[]
   for o in api.get_proximal_map_objects(Point2D(*xy),4,[L.LANE])[L.LANE]:
    line=o.baseline_path.linestring;st=line.project(p)
    d=np.array(line.interpolate(min(st+.3,line.length)).coords[0])-np.array(line.interpolate(max(st-.3,0)).coords[0]);dot=float(d@h/max(np.linalg.norm(d),1e-8))
    if dot>.95 and o.polygon.covers(p):cs.append((line.distance(p),o,st))
   return min(cs,key=lambda x:x[0]) if cs else None
  cache={i:lane(i) for tr in triples for i in tr[:3]}
  for A,B,C,gap,ax,ay in triples:
   la,lb,lc=[cache[i] for i in (A,B,C)]
   reason=None
   if not all(x is not None for x in (la,lb,lc)):reason='not all initially within heading-compatible lane polygons'
   elif lb[1].id!=lc[1].id:reason='B/C not initially same SDK lane'
   elif la[1].id not in [o.id for o in lc[1].adjacent_edges if o is not None]:reason='A not in SDK adjacent lane'
   elif lc[1].baseline_path.linestring.length-lc[2]<max(35.,speed[C]*4.5):reason='insufficient straight-lane distance before connector'
   if reason:screened.append({'index':idx,'roles':{'A':A,'B':B,'C':C},'reason':reason});continue
   score=50-abs(gap/max(speed[C],.1)-1.5)*6-abs(ax-12)+(speed[B]-speed[A])*2
   rec={'scenario':row,'roles':{'A':A,'B':B,'C':C},'score':float(score),'BC_initial_body_gap_m':gap,'A_relative_xy_m':[ax,ay],'speeds_mps':{r:float(speed[i]) for r,i in [('A',A),('B',B),('C',C)]},'source_lane_id':lc[1].id,'adjacent_lane_id':la[1].id,'C_lane_remaining_m':float(lc[1].baseline_path.linestring.length-lc[2]),'world_from_nexus':W.tolist()}
   ranked.append(rec);print('LANE_CANDIDATE',idx,rec['roles'],'remaining',round(rec['C_lane_remaining_m'],1),'speeds',rec['speeds_mps'],flush=True)
 ranked.sort(key=lambda x:(-x['score'],x['scenario']['index']))
 (OUT/'selection.json').write_text(json.dumps({'selection_uses_future_XY':False,'future_information':'recorded existence only','scanned_scenes':len(rows),'ranked':ranked,'rejected':screened},indent=2)+'\n')
 print('SELECT_COMPLETE',len(ranked),flush=True)
if __name__=='__main__':main()
