"""Verify requested responses, independently of the sampler's cost values."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_hazard_response_v2_20261110'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
def main():
 import numpy as np,torch
 from shapely import wkb
 from shapely.geometry import Point,Polygon,LineString
 from nexus_3dgs.bundle import decode
 from audit_nexus_omega_v2 import sample_positions
 parser=argparse.ArgumentParser();parser.add_argument('--kind');args=parser.parse_args();stage=OUT/'response_seed0';run=json.loads((stage/'run.json').read_text());records=[]
 for info in run['protocol']['cases']:
  kind=info['kind']
  if args.kind and kind!=args.kind:continue
  if not any(r['kind']==kind and r['variant']=='event' for r in run['records']):continue
  case=OUT/info['case'];payload={v:torch.load(stage/kind/v/'prediction.pt',map_location='cpu',weights_only=False) for v in ['normal','event']};states={v:decode(p['sampled_tensor'][0].numpy()) for v,p in payload.items()};ev=states['event'];nv=states['normal'];audits={v:json.loads((stage/kind/v/'acceptance.json').read_text()) for v in ['normal','event']};times=np.arange(81)*.1;W=np.array(info['world_from_nexus']);R=W[:2,:2];offset=W[:2,3]
  checks={'geometry_and_motion_pass':all(a['accepted'] for a in audits.values()),'E_future_not_conditioned_in_both_plans':all(not p['task_mask'][0,0,5:].any() for p in payload.values())};details={};A=info['roles']['A']
  if kind in ['brake','reveal']:
   raw=torch.load(stage/kind/'event/raw_stage_prediction.pt',map_location='cpu',weights_only=False);checks['replanned_E_future_is_free']=not bool(raw['task_mask'][0,0,5:].any())
   n=round(info['causal_replan_trigger_s']/.5);prefix=bool(torch.equal(payload['event']['sampled_tensor'][0,0,:5+n],payload['normal']['sampled_tensor'][0,0,:5+n]));controls=np.load(stage/kind/'event/controls.npy');speed=np.linalg.norm(ev[0,:,4:6],axis=-1);after=speed[4+n:];xy_speed=np.linalg.norm(np.diff(ev[0,:,:2],axis=0)/.5,axis=-1)
   stop=next((float((j-4)*.5) for j in range(5+n,21) if speed[j]<.1 and (speed[j:]<.1).all()),None)
   checks.update(executed_prefix_identical_to_normal=prefix,no_positive_acceleration_after_detection=bool((controls[0,:,0]<=1e-7).all()),E_stops_and_stays_stopped=bool(stop is not None and speed[-1]<.1 and xy_speed[-1]<.1),stops_before_deadline=bool(stop is not None and stop<=info['stop_deadline_global_s']+.5))
   details.update(trigger_s=info['causal_replan_trigger_s'],E_first_persistent_stop_s=stop,E_final_speed_mps=float(speed[-1]),E_last_interval_speed_mps=float(xy_speed[-1]),E_max_speed_increase_after_detection_mps=float(max(0,np.diff(after).max())),max_future_speed_before_detection_mps=float(speed[5:5+n].max()))
   if kind=='brake':
    prefix_given=decode(payload['normal']['conditioned_scene_tensor'][0].numpy());checks['prefix_model_does_not_receive_future_emergency_brake']=bool(np.linalg.norm(prefix_given[A,-1,4:6])>8)
    h=ev[0,4,2:4];gap=float((ev[A,4,:2]-ev[0,4,:2]-1.461*h)@h-(ev[0,4,6]+ev[A,4,6])/2);checks['close_initial_following_gap']=gap<=12.2;checks['substantial_braking_response']=speed[4]>8 and speed[-1]<.1 and audits['event']['E_motion']['max_xy_acceleration_mps2']>3;details['initial_net_gap_m']=gap
   else:
    checks['prefix_model_does_not_receive_S']=not bool(payload['normal']['valid_mask'][0,info['roles']['S']].any())
    hazard=audits['event']['hazard'];checks['stationary_car_initially_occluded_then_revealed']=hazard['static_S_initial_blocked_fraction']>=.95 and hazard['first_at_least_90percent_visible_s'] is not None;details['first_90percent_visible_s']=hazard['first_at_least_90percent_visible_s'];S=info['roles']['S'];h=ev[0,-1,2:4];gap=float((ev[S,-1,:2]-ev[0,-1,:2]-1.461*h)@h-(ev[0,-1,6]+ev[S,-1,6])/2);checks['stops_behind_S_with_at_least_2m_gap']=gap>=2;details['final_net_gap_to_S_m']=gap
  else:
   target=wkb.loads((case/'map/target_E_corridor.wkb').read_bytes()).buffer(.02);original=wkb.loads((case/'map/original_E_corridor.wkb').read_bytes()).buffer(.02)
   xy,yaw,seen=sample_positions(ev[[0,A]],payload['event']['valid_mask'][0].numpy().astype(bool)[[0,A]],times,np);inside=[];normal_inside=[]
   for t in range(81):
    h=np.array([np.cos(yaw[0,t]),np.sin(yaw[0,t])]);side=np.array([-h[1],h[0]]);center=xy[0,t]+1.461*h;corners=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*ev[0,4,6:8]/2@np.array([h,side])+center;inside.append(bool(target.covers(Polygon(corners@R.T+offset))))
   with np.load(case/'map/routes.npz') as z:source_route=LineString(z['route_0'])
   deviations=[source_route.distance(Point(q)) for q in xy[0]];normal_dev=[source_route.distance(Point(q)) for q in nv[0,4:,:2]]
   settled=next((float(t) for j,t in enumerate(times) if t>=1 and all(inside[j:])),None);checks.update(E_finishes_in_target_lane=bool(all(inside[-20:])) ,E_moves_at_least_2m_from_source_route=bool(max(deviations)>2),normal_E_keeps_source_lane=bool(audits['normal']['E_body_outside_original_lane_samples']==0));details.update(E_persistent_target_lane_entry_s=settled,E_final_distance_from_source_route_m=float(deviations[-1]),E_max_distance_from_source_route_m=float(max(deviations)))
   if kind=='rear':
    given_inside=[original.covers(Point(q@R.T+offset)) for q in xy[1]];checks['rear_A_continues_source_lane']=all(given_inside);details['initial_speed_difference_mps']=info['observed_speeds_mps']['A']-info['observed_speeds_mps']['E']
   else:
    insideA=[original.covers(Point(q@R.T+offset)) for q in xy[1]];late=(times>=2.5)&(times<=8);checks['A_remains_in_invaded_lane_without_return']=bool(np.array(insideA)[late].mean()>.95);details['A_inside_invaded_lane_fraction_after_2p5s']=float(np.array(insideA)[late].mean())
  row={'kind':kind,'title':info['title'],'checks':{k:bool(v) for k,v in checks.items()},'details':details,'requested_behavior_accepted':bool(all(checks.values())),'physical_audits':audits};records.append(row);print('RESPONSE_AUDIT',kind,row['requested_behavior_accepted'],details,'failed',[k for k,v in checks.items() if not v],flush=True)
 result={'status':'complete' if len(records)==4 else 'partial','all_requested_responses_accepted':bool(len(records)==4 and all(r['requested_behavior_accepted'] for r in records)),'cases':records,'method_scope':'explicit stop/escape-lane intent + Nexus control-space sampling; E future values never conditioned; brake/reveal use one event-triggered replan, not a fully online perception/control stack'}
 (OUT/('response_verification_'+args.kind+'.json' if args.kind else 'response_verification.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
