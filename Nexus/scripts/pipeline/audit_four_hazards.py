"""Independent geometry audit of the minimal four-hazard population."""
import json
from pathlib import Path

def evaluate(payload,case,info,variant):
 import numpy as np
 from shapely import wkb
 from shapely.geometry import Polygon,LineString,Point
 from nexus_3dgs.bundle import decode
 from nexus_3dgs.scene_projection import pair_separations
 from audit_nexus_omega_v2 import sample_positions
 case=Path(case);W=np.array(info['world_from_nexus']);roles=info['roles'];active=info['active_slots']
 pred=payload['sampled_tensor'][0].numpy();cond=payload['conditioned_scene_tensor'][0].numpy();keep=payload['task_mask'][0].numpy().astype(bool);valid=payload['valid_mask'][0].numpy().astype(bool)
 assert np.array_equal(pred[keep],cond[keep]) and not keep[0,5:].any() and (cond[~keep]==0).all()
 states=decode(pred);sizes=states[active,4,6:8];times=np.arange(81)*.1
 xy,yaw,seen=sample_positions(states[active],valid[active],times,np)
 boxes=np.concatenate([xy.transpose(1,0,2),yaw.T[...,None]],-1)
 boxes[:,0,:2]+=1.461*np.c_[np.cos(boxes[:,0,2]),np.sin(boxes[:,0,2])]
 sep,vp=pair_separations(boxes,seen.T,sizes);pairs={};role_indices={r:active.index(s) for r,s in roles.items()}
 for r,i in role_indices.items():
  for s,j in role_indices.items():
   if i>=j:continue
   hits=(sep[1:,i,j]<-1e-5)&vp[1:,i,j]
   pairs[r+s]={'overlap_samples':int(hits.sum()),'overlap_times_s':times[1:][hits].tolist(),'minimum_signed_gap_m':float(sep[:,i,j].min())}
 road=wkb.loads((case/'reference/inputs/drivable.wkb').read_bytes()).buffer(.02)
 lane=wkb.loads((case/'map/lanes/0.wkb').read_bytes()).buffer(.02)
 def body(q,size):
  x,y,a=q;c,s=np.cos(a),np.sin(a)
  return Polygon(np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*size/2@np.array([[c,s],[-s,c]])+[x,y])
 def world(poly):return Polygon(np.array(poly.exterior.coords)@W[:2,:2].T+W[:2,3])
 off={r:sum(not road.covers(world(body(b,sizes[i]))) for b in boxes[1:,i]) for r,i in role_indices.items()}
 outside=sum(not lane.covers(world(body(b,sizes[0]))) for b in boxes[1:,0])
 velocity=np.diff(states[0,:,:2],axis=0)/.5;speed=np.linalg.norm(velocity[4:],axis=-1);acc=np.linalg.norm(np.diff(velocity,axis=0)[3:]/.5,axis=-1)
 head=states[0,5:,2:4];head=head/np.maximum(np.linalg.norm(head,axis=-1,keepdims=True),1e-8)
 along=np.sum(head*velocity[4:],axis=-1);cross=velocity[4:,0]*head[:,1]-velocity[4:,1]*head[:,0];err=np.degrees(np.arctan2(abs(cross),along));moving=speed>.5
 slip=float(err[moving].max()) if moving.any() else 0.;reverse=float(np.maximum(-along,0).sum()*.5);sd=float(abs(speed-np.linalg.norm(states[0,5:,4:6],axis=-1)).max())
 motion={'xy_interval_speeds_mps':speed.tolist(),'max_xy_acceleration_mps2':float(acc.max()),'max_heading_error_deg':slip,'reverse_distance_m':reverse,'max_xy_vs_speed_channel_difference_mps':sd,'minimum_speed_mps':float(speed.min()),'final_speed_mps':float(speed[-1]),'progress_m':float(speed.sum()*.5),'accepted':bool(acc.max()<=5 and slip<=10 and reverse<=.2 and sd<=1.6)}
 details={};A=roles['A'];ai=role_indices['A']
 if info['kind']=='reveal':
  si=role_indices['S'];blocked=[]
  for t in range(81):
   h=np.array([np.cos(yaw[0,t]),np.sin(yaw[0,t])]);eye=xy[0,t]+(1.461+sizes[0,0]/2-.5)*h
   sh=np.array([np.cos(yaw[si,t]),np.sin(yaw[si,t])]);side=np.array([-sh[1],sh[0]]);front=boxes[t,si,:2]-sizes[si,0]/2*sh
   blocker=body(boxes[t,ai],sizes[ai]);blocked.append(float(np.mean([LineString([eye,front+u*side*sizes[si,1]/2]).intersects(blocker) for u in np.linspace(-1,1,41)])))
  details={'static_S_initial_blocked_fraction':blocked[0],'static_S_blocked_fraction_0p1s':blocked,'first_at_least_90percent_visible_s':next((float(t) for t,f in zip(times,blocked) if f<=.1),None),'static_S_motion_m':float(np.linalg.norm(states[roles['S'],5:,:2]-states[roles['S'],4,:2],axis=-1).max())}
 if info['kind']=='oncoming':
  initial=wkb.loads((case/'map/initial_E_lane.wkb').read_bytes());inside=[]
  for q in boxes[:,ai,:2]:inside.append(bool(initial.covers(Point(q@W[:2,:2].T+W[:2,3]))))
  details={'A_center_inside_initial_E_lane_samples':sum(inside),'A_center_inside_initial_E_lane_times_s':times[inside].tolist()}
 if info['kind']=='brake':
  av=np.diff(states[A,4:,:2],axis=0)/.5;asp=np.linalg.norm(av,axis=-1)
  details={'A_final_speed_mps':float(asp[-1]),'A_peak_speed_drop_mps2':float(max(0,-np.diff(np.r_[info['observed_speeds_mps']['A'],asp]).min()/.5))}
 if info['kind']=='rear':details={'initial_rear_TTC_s':info['parameters']['initial_rear_ttc_s'],'initial_speed_difference_mps':info['observed_speeds_mps']['A']-info['observed_speeds_mps']['E']}
 accepted=not any(p['overlap_samples'] for p in pairs.values()) and not any(off.values()) and motion['accepted'] and (variant!='normal' or outside==0)
 return {'scope':'80 future samples at 0.1s, minimal edited population only; not a continuous-time collision proof','variant':variant,'conditions_exact':True,'E_future_free':True,'pairs':pairs,'body_offroad_samples':off,'E_body_outside_original_lane_samples':outside,'E_motion':motion,'hazard':details,'accepted':bool(accepted)}

def compare(normal,event,info):
 import numpy as np
 from nexus_3dgs.bundle import decode
 from nexus_3dgs.scene_projection import pair_separations
 from audit_nexus_omega_v2 import sample_positions
 n=decode(normal['sampled_tensor'][0].numpy());e=decode(event['sampled_tensor'][0].numpy());A=info['roles']['A'];times=np.arange(1,81)*.1
 mixed=n[[0,A]].copy();mixed[1]=e[A];valid=normal['valid_mask'][0].numpy().astype(bool)[[0,A]]
 xy,yaw,seen=sample_positions(mixed,valid,times,np);b=np.concatenate([xy.transpose(1,0,2),yaw.T[...,None]],-1);b[:,0,:2]+=1.461*np.c_[np.cos(b[:,0,2]),np.sin(b[:,0,2])]
 sep,vp=pair_separations(b,seen.T,n[[0,A],4,6:8]);ns=np.linalg.norm(np.diff(n[0,4:,:2],axis=0)/.5,axis=-1);es=np.linalg.norm(np.diff(e[0,4:,:2],axis=0)/.5,axis=-1)
 return {'same_noise_and_warm_start':True,'E_max_position_change_m':float(np.linalg.norm(n[0,5:,:2]-e[0,5:,:2],axis=-1).max()),'normal_min_E_speed_mps':float(ns.min()),'event_min_E_speed_mps':float(es.min()),'max_event_E_slowdown_vs_normal_mps':float((ns-es).max()),'max_event_E_speedup_vs_normal_mps':float((es-ns).max()),'normal_E_plus_event_A_overlap_samples':int(((sep[:,0,1]<-1e-5)&vp[:,0,1]).sum()),'normal_E_plus_event_A_min_gap_m':float(sep[:,0,1].min()),'strict_closed_loop_causality_proven':False}
