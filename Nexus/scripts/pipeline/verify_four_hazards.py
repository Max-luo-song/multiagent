"""Final selected-run verification, including event definitions and provenance."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/nexus_four_hazards_20261110';REPORT=ROOT/'report/nexus_four_hazards_20261110'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
def main():
 import numpy as np,torch
 from shapely import wkb
 from shapely.geometry import Point
 from nexus_3dgs.bundle import decode
 from audit_nexus_omega_v2 import sample_positions
 selections={'brake':BASE/'brake_stopping_v3','reveal':BASE/'reveal_stopping_v2','rear':BASE,'oncoming':BASE/'oncoming_yield_v3'};rows=[]
 for kind,root in selections.items():
  run=json.loads((root/'frozen_v4_seed0/run.json').read_text());assert run['status']=='complete' and run['weights_unchanged'] and run['protected_source_unchanged'];info=next(c for c in run['protocol']['cases'] if c['kind']==kind);case=root/info['case']
  for path,h in {**run['source_sha256'],**run['input_sha256']}.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
  payloads={};audits={}
  for variant in ['normal','event']:
   folder=root/'frozen_v4_seed0'/kind/variant;file=folder/'prediction.pt';row=next(r for r in run['records'] if r['kind']==kind and r['variant']==variant);assert hashlib.sha256(file.read_bytes()).hexdigest()==row['prediction_sha256']
   payloads[variant]=torch.load(file,map_location='cpu',weights_only=False);audits[variant]=json.loads((folder/'acceptance.json').read_text());p=payloads[variant];keep=p['task_mask'].bool();assert not keep[0,0,5:].any() and not p['conditioned_scene_tensor'][~keep].any();assert torch.equal(p['sampled_tensor'][keep],p['conditioned_scene_tensor'][keep]);assert audits[variant]['accepted'],(kind,variant,audits[variant])
  p=payloads['event'];s=decode(p['sampled_tensor'][0].numpy());A=info['roles']['A'];a=audits['event'];definition={}
  if kind=='brake':definition={'given_A_stops':a['hazard']['A_final_speed_mps']<.1,'given_A_emergency_deceleration':a['hazard']['A_peak_speed_drop_mps2']>5.,'brake_start_s':info['parameters']['brake_start_s']};assert definition['given_A_stops'] and definition['given_A_emergency_deceleration']
  elif kind=='reveal':
   h=a['hazard'];definition={'initial_S_completely_occluded':h['static_S_initial_blocked_fraction']>=.95,'S_becomes_visible':h['first_at_least_90percent_visible_s'] is not None,'first_90percent_visible_s':h['first_at_least_90percent_visible_s'],'S_is_inserted_stationary_vehicle':h['static_S_motion_m']==0.};assert definition['initial_S_completely_occluded'] and definition['S_becomes_visible'] and definition['S_is_inserted_stationary_vehicle']
  elif kind=='rear':definition={'initial_speed_difference_mps':a['hazard']['initial_speed_difference_mps'],'initial_rear_TTC_s':a['hazard']['initial_rear_TTC_s'],'actual_minimum_gap_m':a['pairs']['EA']['minimum_signed_gap_m']};assert definition['initial_speed_difference_mps']>3 and definition['initial_rear_TTC_s']<6
  else:
   W=np.array(info['world_from_nexus']);lane=wkb.loads((case/'map/lanes/0.wkb').read_bytes());times=np.arange(81)*.1;inside={}
   for variant in ['normal','event']:
    ss=decode(payloads[variant]['sampled_tensor'][0].numpy());valid=payloads[variant]['valid_mask'][0].numpy().astype(bool);xy,_,_=sample_positions(ss[[A]],valid[[A]],times,np);q=xy[0]@W[:2,:2].T+W[:2,3];inside[variant]=times[[lane.covers(Point(x)) for x in q]].tolist()
   definition={'initial_opposing_lanes_boundary_gap_m':info['parameters']['opposing_lane_boundary_gap_m'],'normal_A_center_inside_E_continuation_times_s':inside['normal'],'event_A_center_inside_E_continuation_times_s':inside['event'],'note':'Use continuous E SDK corridor: the initial short lane segment ends before the intrusion location.'};assert len(inside['event'])>0 and len(inside['normal'])==0
  margins=[];eh=s[0,-1,2:4];eh=eh/np.linalg.norm(eh);ec=s[0,-1,:2]+1.461*eh;ev=float(np.linalg.norm(s[0,-1,4:6]))
  for role,slot in info['roles'].items():
   if role=='E' or np.linalg.norm(s[slot,-1,4:6])>.2:continue
   delta=s[slot,-1,:2]-ec;along=float(delta@eh);lateral=abs(float(delta@np.array([-eh[1],eh[0]])));width=(s[0,4,7]+s[slot,4,7])/2
   if along<=0 or lateral>width+.3 or s[slot,-1,2:4]@eh<.7:continue
   gap=along-float((s[0,4,6]+s[slot,4,6])/2);required=.5+ev*ev/(2*4.5)+.3*ev;margins.append({'lead_role':role,'gap_m':gap,'E_endpoint_speed_mps':ev,'required_stopping_gap_m':required,'margin_m':gap-required,'accepted_with_0p1m_numeric_tolerance':gap-required>=-.1})
  baseline=torch.load(BASE/'frozen_v4_seed0'/kind/'normal/prediction.pt',map_location='cpu',weights_only=False)
  same=bool(torch.equal(baseline['sampled_tensor'],payloads['normal']['sampled_tensor']));assert same or kind=='reveal','Unexpected normal change: '+kind
  rows.append({'kind':kind,'selected_experiment_root':str(root),'normal_and_event_accepted':True,'all_original_source_and_input_hashes_match':True,'normal_prediction_identical_to_first_frozen_run':same,'event_definition':definition,'endpoint_stationary_lead_stopping_margins':margins,'E_body_outside_original_lane_samples':a['E_body_outside_original_lane_samples'],'min_gap_m':min(x['minimum_signed_gap_m'] for x in a['pairs'].values()),'E_min_speed_mps':a['E_motion']['minimum_speed_mps'],'runtime_refiner_adapter':run['protocol'].get('runtime_refiner_adapter')})
  print('VERIFIED',kind,'min_gap',rows[-1]['min_gap_m'],'endpoint_margins',margins,flush=True)
 result={'status':'complete','selected_cases':rows,'physical_scope':'80 future checks at0.1s; not continuous-time proof or safety beyond forecast','interpretation':'A/S full future known; E free; minimal edited population; Nexus+physical guidance, with extra stopping-gap and opposing-yield adapters in brake/oncoming; schematic camera is not3DGS'}
 (REPORT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('FOUR_HAZARDS_VERIFIED',flush=True)
if __name__=='__main__':main()
