"""Add a shared map-only adjacent-lane option to the selected real scene."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_lane_response_20261027'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline'),str(ROOT/'third_party/nuplan-devkit')]
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 import numpy as np
 from shapely import wkb
 from shapely.geometry import LineString,Point
 from shapely.ops import unary_union
 from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
 from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
 from prepare_nexus_chain_cases import raster
 import audit_nexus_multiscene_fixed_A as bg
 p=json.loads((OUT/'protocol.json').read_text());selection=json.loads((OUT/'selection.json').read_text())['ranked']
 for info in p['cases']:
  idx=info['scenario']['index'];C=info['roles']['C'];B=info['roles']['B'];case=OUT/f'case_{idx:03d}'
  chosen=next(r for r in selection if r['scenario']['index']==idx and r['roles']['A']==info['roles']['A'])
  W=np.array(json.loads((case/'reference/protocol.json').read_text())['world_from_nexus']);R=W[:2,:2];off=W[:2,3]
  api=get_maps_api(str(ROOT.parent/'data/nuplan/maps'),'nuplan-maps-v1.0',info['scenario']['map_name'])
  lane=api.get_map_object(chosen['source_lane_id'],L.LANE);target=api.get_map_object(chosen['adjacent_lane_id'],L.LANE)
  assert target.id in [o.id for o in lane.adjacent_edges if o]
  points=[np.array(target.baseline_path.linestring.coords)];polys=[target.polygon];ids=[target.id];obj=target;length=target.baseline_path.linestring.length
  def tangent(line,end):
   st=line.length if end else 0;delta=np.array(line.interpolate(min(st+.3,line.length)).coords[0])-np.array(line.interpolate(max(st-.3,0)).coords[0]);return delta/max(np.linalg.norm(delta),1e-8)
  while length<220:
   opts=[];prev=tangent(obj.baseline_path.linestring,True)
   for e in obj.outgoing_edges:
    if e.id in ids:continue
    line=e.baseline_path.linestring;h=tangent(line,False);q=tangent(line,True)
    angle=lambda a,b:abs(float(np.arctan2(a[0]*b[1]-a[1]*b[0],a@b)))
    opts.append((angle(prev,h)+angle(h,q),e.id,e))
   if not opts:break
   _,_,obj=min(opts,key=lambda x:(x[0],x[1]));ids.append(obj.id);points.append(np.array(obj.baseline_path.linestring.coords));polys.append(obj.polygon);length+=obj.baseline_path.linestring.length
  xy=np.concatenate(points);xy=xy[np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=-1)>1e-5]];line=LineString(xy);st=np.arange(0,line.length,.75)
  target_points=(np.array([line.interpolate(float(s)).coords[0] for s in st])-off)@R
  origin=wkb.loads((case/'map/lanes'/f'{C}.wkb').read_bytes());adj=unary_union(polys);allowed=origin.union(adj)
  (case/'map/adjacent_lane.wkb').write_bytes(adj.wkb);(case/'map/allowed_C_lanes.wkb').write_bytes(allowed.wkb)
  with np.load(case/'map/routes.npz') as z:source=z[f'route_{C}'].copy()
  np.savez_compressed(case/'map/lane_choices.npz',slot=C,source_route=source,target_route=target_points.astype('float32'),change_windows=np.array([[.5,4.],[1.5,5.]],np.float32))
  # Keep the historical map raster unchanged; put the union in a new map dir.
  maps=case/'choice_map';maps.mkdir();(maps/'routes.npz').symlink_to(case/'map/routes.npz')
  grids=[raster(wkb.loads((case/'map/lanes'/f'{B}.wkb').read_bytes()),W,None,[-100.,-100.],.1,(2001,2801)),raster(allowed,W,None,[-100.,-100.],.1,(2001,2801))]
  np.savez_compressed(maps/'lane_corridors_sdf.npz',sdf=np.stack(grids),slots=np.array([B,C]),origin=np.array([-100.,-100.]),resolution=.1)
  info['lane_choice']={'source_lane':lane.id,'adjacent_lane':target.id,'adjacent_edges':ids,'C_initial_lane_remaining_m':chosen['C_lane_remaining_m'],'choice_times_are_initializations_only':True,'boundary_marking_rules_verified':False,'map_scope':'same-direction adjacent SDK lanes; crossing corridor checked separately; not full traffic-law validation'}
 bg.OUT=OUT;bg.main();audit=json.loads((OUT/'fixed_A_background_audit.json').read_text())
 for info in p['cases']:
  conflicts=[r for r in audit['records'] if r['index']==info['scenario']['index'] and r['fixed_A_vs_recorded_background_overlap_samples']]
  info['given_A_background_conflicts']=conflicts;info['given_A_input_usable'] &=not bool(conflicts)
  write(OUT/f"case_{info['scenario']['index']:03d}"/'preparation.json',info)
 p.update(event_kind='pure_cut_in',variants=['normal','event'],lane_choice={'candidate_seeds':['keep','adjacent_0.5_4','adjacent_1.5_5'],'selection':'common joint physical/cruise/collision objective at first guided denoising anchor; same seeds and costs in both variants','choice_iterations':100,'alternative_center_weight':4.,'lane_change_cost':25.,'C_future_conditioned':False,'weights_updated':False},success_criteria={'B_extra_speed_reduction_mps':.5,'C_event_target_lane_body_hold_s':1.,'C_normal_stays_source_lane':True,'actual_three_physical_pass':True,'C_future_free':True,'crossing_outside_lane_connectors':True},status='prepared',all_outcomes_retained=True)
 write(OUT/'protocol.json',p)
 print('LANE_PREPARED',[(i['scenario']['index'],i['given_A_input_usable'],i['input_precheck']) for i in p['cases']],flush=True)
if __name__=='__main__':main()
