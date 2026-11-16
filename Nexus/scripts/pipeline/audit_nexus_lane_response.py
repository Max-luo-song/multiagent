"""Independent saved-output checks for complete C lane changes and propagation."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_lane_response_20261027'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
def main():
 import numpy as np,torch
 from shapely import wkb
 from shapely.geometry import Polygon,LineString,Point
 from shapely.ops import unary_union
 from nexus_3dgs.bundle import decode
 from audit_nexus_omega_v2 import sample_positions
 p=json.loads((OUT/'protocol.json').read_text());stage=OUT/'pilot_seed0';run=json.loads((stage/'run.json').read_text());results=[]
 for info in p['cases']:
  idx=info['scenario']['index'];case=OUT/f'case_{idx:03d}';C=info['roles']['C'];B=info['roles']['B'];stats={};speeds={}
  W=np.array(json.loads((case/'reference/protocol.json').read_text())['world_from_nexus'])
  origin=wkb.loads((case/'map/lanes'/f'{C}.wkb').read_bytes());target=wkb.loads((case/'map/adjacent_lane.wkb').read_bytes());allowed=origin.union(target)
  # The initial common roadblock lane pair is narrower than the full routes:
  # every straddling transition sample must stay in it, outside connectors.
  from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
  from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
  api=get_maps_api(str(ROOT.parent/'data/nuplan/maps'),'nuplan-maps-v1.0',info['scenario']['map_name'])
  lc=info['lane_choice'];crossing_region=unary_union([api.get_map_object(lc[k],L.LANE).polygon for k in ('source_lane','adjacent_lane')])
  with np.load(case/'map/lane_choices.npz') as z:line=LineString(z['source_route'])
  for rec in [r for r in run['records'] if r['index']==idx]:
   v=rec['variant'];payload=torch.load(stage/rec['path']/'prediction.pt',weights_only=False,map_location='cpu');s=decode(payload['sampled_tensor'][0].numpy());valid=payload['valid_mask'][0].bool().numpy();tt=np.arange(81)*.1;xy,yaw,seen=sample_positions(s,valid,tt,np)
   size=s[C,4,6:8];source_inside=[];target_inside=[];outside=[];crossing=[];cross_bad=[];lateral=[]
   for k,t in enumerate(tt):
    angle=yaw[C,k];rot=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]);center=xy[C,k].copy()
    if C==0:center+=1.461*np.array([np.cos(angle),np.sin(angle)])
    body=Polygon((np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*size/2@rot+center)@W[:2,:2].T+W[:2,3])
    source_inside.append(origin.buffer(.02).covers(body));target_inside.append(target.buffer(.02).covers(body));outside.append(not allowed.buffer(.02).covers(body))
    straddle=not source_inside[-1] and not target_inside[-1];crossing.append(straddle)
    cross_bad.append(straddle and not crossing_region.buffer(.02).covers(body))
    st=line.project(Point(center));q=np.array(line.interpolate(st).coords[0]);a=np.array(line.interpolate(max(st-.1,0)).coords[0]);b=np.array(line.interpolate(min(st+.1,line.length)).coords[0]);h=(b-a)/max(np.linalg.norm(b-a),1e-8);lateral.append(float((center-q)@np.array([-h[1],h[0]])))
   count=0;maxcount=0;start=None;hold=None
   for k,inside in enumerate(target_inside):
    count=count+1 if inside else 0;maxcount=max(maxcount,count)
    if count>=11 and hold is None:hold=float(tt[k-10])
   hb=np.c_[np.cos(yaw[B]),np.sin(yaw[B])];cb=xy[B].copy();cc=xy[C].copy()
   if B==0:cb+=1.461*hb
   if C==0:cc+=1.461*np.c_[np.cos(yaw[C]),np.sin(yaw[C])]
   along=((cc-cb)*hb).sum(-1);turn=yaw[C]-yaw[B]
   extent=s[B,4,6]/2+(s[C,4,6]*np.abs(np.cos(turn))+s[C,4,7]*np.abs(np.sin(turn)))/2
   ahead=along>extent;complete=np.flatnonzero(ahead&np.array(target_inside))
   speeds[v]={r:np.linalg.norm(np.diff(s[slot,4:,:2],axis=0)/.5,axis=-1) for r,slot in [('B',B),('C',C)]}
   stat={'C_future_free':not bool(payload['task_mask'][0,C,5:].any()),'C_source_lane_body_outside_samples':int(np.sum(~np.array(source_inside)[1:])),'C_target_lane_body_samples':int(np.sum(target_inside[1:])),'C_target_lane_held_at_least_1s':hold is not None,'C_target_lane_1s_hold_start_s':hold,'C_max_continuous_target_lane_s':max(0,maxcount-1)*.1,'C_final_in_target_lane':bool(target_inside[-1]),'C_union_body_outside_samples':int(np.sum(outside[1:])),'transition_samples_in_connector_or_outside_initial_lane_pair':int(np.sum(cross_bad[1:])),'C_lateral_from_source_m':lateral,'C_max_lateral_from_source_m':float(max(map(abs,lateral))),'primary_three_accepted':rec['acceptance']['primary_three_accepted'],'full_scene_accepted':rec['acceptance']['full_scene_accepted'],'speed_mps':{r:x.tolist() for r,x in speeds[v].items()},'tactical_search':json.loads((stage/rec['path']/'lane_choice.json').read_text())}
   stat.update(C_initial_along_B_m=float(along[0]),C_final_along_B_m=float(along[-1]),C_rear_fully_ahead_of_B_front_in_target_lane=bool(len(complete)),C_first_full_pass_B_time_s=float(tt[complete[0]]) if len(complete) else None)
   stats[v]=stat
  result={'index':idx,'roles':info['roles'],'variants':stats,'boundary_marking_rules_verified':False,'scope':'full-body target-lane completion and 0.1s physical checks; no real-world online reaction-delay proof'}
  if all(k in stats for k in ['normal','event']):
   bdrop=float(np.max(speeds['normal']['B']-speeds['event']['B']));n=stats['normal'];e=stats['event']
   success=bool(bdrop>=.5 and n['primary_three_accepted'] and e['primary_three_accepted'] and n['C_source_lane_body_outside_samples']==0 and e['C_target_lane_held_at_least_1s'] and e['C_final_in_target_lane'] and e['C_union_body_outside_samples']==0 and e['transition_samples_in_connector_or_outside_initial_lane_pair']==0)
   result.update(B_max_additional_speed_reduction_mps=bdrop,main_behavior_criteria_passed=success)
  if all(k in stats for k in ['transfer_control','transfer_B_response']):
   a=stats['transfer_control'];b=stats['transfer_B_response'];result['B_transfer_lane_choice_criteria_passed']=bool(a['primary_three_accepted'] and b['primary_three_accepted'] and a['C_source_lane_body_outside_samples']==0 and b['C_target_lane_held_at_least_1s'] and b['C_final_in_target_lane'] and not b['C_union_body_outside_samples'] and not b['transition_samples_in_connector_or_outside_initial_lane_pair'])
  results.append(result)
 result={'run_status':run['status'],'cases':results,'strict_causal_chain_proven':False,'Nexus_native_learned_lane_choice_proven':False}
 (OUT/'lane_response_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 for r in results:print('LANE_AUDIT',r['index'],'main',r.get('main_behavior_criteria_passed'),'B-transfer',r.get('B_transfer_lane_choice_criteria_passed'),'variants',{k:(v['C_final_in_target_lane'],v['C_target_lane_1s_hold_start_s'],v['primary_three_accepted']) for k,v in r['variants'].items()},flush=True)
if __name__=='__main__':main()
