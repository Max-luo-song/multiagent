"""Find real road layouts with an adjacent same-direction escape lane."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_hazard_response_v2_20261110';SOURCE=ROOT/'artifacts/nuplan_evaluation/independent256_v1'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline'),str(ROOT/'third_party/nuplan-devkit'),str(ROOT/'third_party/alf')]
os.environ.update(NUPLAN_DATA_STORE='local',MPLBACKEND='Agg')
def main():
 import numpy as np,torch
 from shapely.geometry import Point,Polygon
 from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
 from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
 from nuplan.common.actor_state.state_representation import Point2D
 from nexus_3dgs.bundle import decode
 from nexus_3dgs.geometry import pose
 from nexus_3dgs import nuplan
 OUT.mkdir(exist_ok=True);torch.set_num_threads(2);maps={};records=[]
 def head(line,s):
  q=np.array(line.interpolate(max(0,s-.2)).coords[0]);p=np.array(line.interpolate(min(line.length,s+.2)).coords[0]);return (p-q)/max(np.linalg.norm(p-q),1e-8)
 plan=json.loads((SOURCE/'frozen_plan.json').read_text())['scenarios']
 # Check previous roles first, then every cached ego history; synthetic A may be inserted.
 plan.sort(key=lambda r:(r['index'] not in [174,170,9,33,190],r['index']))
 for row in plan:
  raw=torch.load(SOURCE/'raw_features'/f"{row['index']:03d}_{row['token']}.pt",map_location='cpu',weights_only=False);valid=raw['validity'].numpy().astype(bool)
  if not valid[0].all():continue
  state=decode(raw['tensor'].numpy()[...,:8]);e=state[0,4];ve=float(np.linalg.norm((e[:2]-state[0,3,:2])/.5))
  if not 5<ve<12:continue
  with nuplan.connect(row['database']) as db:anchor=nuplan.anchor_frame(db,row['token'])
  W=pose(anchor['x'],anchor['y'],yaw=nuplan.ego_yaw(anchor))@np.linalg.inv(pose(*e[:2],yaw=np.arctan2(e[3],e[2])));R=W[:2,:2];offset=W[:2,3];world=e[:2]@R.T+offset;eh=e[2:4]@R.T;side=np.array([-eh[1],eh[0]])
  if row['map_name'] not in maps:maps[row['map_name']]=get_maps_api(str(ROOT.parent/'data/nuplan/maps'),'nuplan-maps-v1.0',row['map_name'])
  api=maps[row['map_name']];objects=api.get_proximal_map_objects(Point2D(*world),10,[L.LANE])[L.LANE];own=[]
  for obj in objects:
   line=obj.baseline_path.linestring;s=line.project(Point(world));hh=head(line,s)
   if hh@eh>.97 and obj.polygon.covers(Point(world)):own.append((line.distance(Point(world)),obj))
  if not own:continue
  _,el=min(own,key=lambda a:(a[0],str(a[1].id)));body=e[:2]+1.461*e[2:4]+np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*e[6:8]/2@np.array([[e[2],e[3]],[-e[3],e[2]]]);poly=Polygon(body@R.T+offset)
  if not el.polygon.buffer(.02).covers(poly):continue
  same=[];opposite=[]
  for obj in objects:
   if str(obj.id)==str(el.id) or obj.polygon.distance(el.polygon)>.12:continue
   line=obj.baseline_path.linestring;s=line.project(Point(world));q=np.array(line.interpolate(s).coords[0]);hh=head(line,s);lat=float((q-world)@side)
   if not 2.5<abs(lat)<4.8 or abs((q-world)@eh)>2:continue
   rec={'id':str(obj.id),'lateral_m':lat,'initial_baseline_length_m':line.length,'heading_dot':float(hh@eh)}
   if hh@eh>.97:same.append(rec)
   elif hh@eh<-.97:opposite.append(rec)
  if not same:continue
  rec={'scenario':row,'E_lane':str(el.id),'E_speed_mps':ve,'same_direction':same,'opposite_direction':opposite,'world_from_nexus':W.tolist()};records.append(rec)
  print('LAYOUT',row['index'],'same',[(x['id'],round(x['lateral_m'],2)) for x in same],'opposite',[(x['id'],round(x['lateral_m'],2)) for x in opposite],flush=True)
  (OUT/'road_layouts.json').write_text(json.dumps({'selection':'observed ego history and actual SDK lane topology only','layouts':records},ensure_ascii=False,indent=2)+'\n')
 print('ROAD_SCAN_COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
