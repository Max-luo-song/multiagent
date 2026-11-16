"""Saved-output measurements for main joint generation and B-transfer probes."""
import numpy as np
from .bundle import decode
from .scene_projection import pair_separations

def states(payload):return decode(payload['sampled_tensor'][0].numpy())
def contrast(normal,event,slot):
 from shapely.geometry import LineString,Point
 n=states(normal)[slot,4:,:2];e=states(event)[slot,4:,:2]
 vn=np.linalg.norm(np.diff(n,axis=0)/.5,axis=-1);ve=np.linalg.norm(np.diff(e,axis=0)/.5,axis=-1)
 # Extend the reference endpoints so different progress does not look like a lane change.
 tail=n[-1]-n[-2];tail=tail/max(np.linalg.norm(tail),1e-8)*30
 line=LineString(np.concatenate([n[:1]-(n[1:2]-n[:1])*10,n,n[-1:]+tail]))
 cross=max(line.distance(Point(q)) for q in e)
 delta=abs(vn-ve);visible=np.flatnonzero(delta>=.5)
 return {'maximum_position_difference_m':float(np.linalg.norm(n-e,axis=-1).max()),'maximum_speed_difference_mps':float(delta.max()),'maximum_speed_reduction_mps':float((vn-ve).max()),'maximum_cross_path_difference_m':float(cross),'first_speed_difference_over0p5_interval_midpoint_s':float(.25+.5*visible[0]) if len(visible) else None,'normal_xy_interval_speeds_mps':vn.tolist(),'event_xy_interval_speeds_mps':ve.tolist(),'event_minimum_speed_mps':float(ve.min()),'event_final_speed_mps':float(ve[-1])}
def mixed_pair(first,first_slot,second,second_slot):
 from shapely.geometry import Polygon
 a=states(first);b=states(second);valid1=first['valid_mask'][0].bool().numpy();valid2=second['valid_mask'][0].bool().numpy()
 times=np.arange(1,81)*.1;idx=times/.5+4;lo=np.minimum(np.floor(idx).astype(int),19);f=idx-lo
 boxes=[];seen=[];sizes=[]
 for s,slot,valid,payload in [(a,first_slot,valid1,first),(b,second_slot,valid2,second)]:
  p=s[slot];xy=p[lo,:2]*(1-f[:,None])+p[lo+1,:2]*f[:,None];ang=np.arctan2(p[:,3],p[:,2]);da=np.arctan2(np.sin(ang[lo+1]-ang[lo]),np.cos(ang[lo+1]-ang[lo]));yaw=ang[lo]+da*f
  if slot==0:xy+=1.461*np.c_[np.cos(yaw),np.sin(yaw)]
  boxes.append(np.c_[xy,yaw]);seen.append(valid[slot,lo]&valid[slot,lo+1]);sizes.append(decode(payload['scene_tensor_gt'][0].numpy())[slot,4,6:8])
 sep,vp=pair_separations(np.stack(boxes,axis=1),np.stack(seen,axis=1),np.stack(sizes));hits=(sep[:,0,1]<0)&vp[:,0,1]
 return {'overlap_samples':int(hits.sum()),'minimum_signed_gap_m':float(sep[:,0,1].min()),'overlap_times_s':times[hits].tolist(),'scope':'offline diagnostic splice only; never used to set a free future'}
def summarize(payloads,acceptance,roles):
 A,B,C=[roles[r] for r in ('A','B','C')];n=payloads['normal'];e=payloads['event'];t0=payloads['transfer_control'];t1=payloads['transfer_B_response']
 bc=contrast(n,e,B);cc=contrast(n,e,C);tc=contrast(t0,t1,C)
 def visible(x,threshold):return x['maximum_position_difference_m']>=1 and (x['maximum_speed_difference_mps']>=threshold or x['maximum_cross_path_difference_m']>=.5)
 pressure=mixed_pair(e,B,n,C);ab=mixed_pair(e,A,n,B)
 main=all(acceptance[k]['primary_three_accepted'] for k in ('normal','event'));transfer=all(acceptance[k]['primary_three_accepted'] for k in ('transfer_control','transfer_B_response'))
 return {'B_response':bc,'C_response':cc,'C_response_to_B_only_transfer':tc,'event_A_plus_normal_B':ab,'event_B_plus_normal_C':pressure,'main_primary_pair_accepted':main,'transfer_primary_pair_accepted':transfer,'visible_B_response':visible(bc,.75),'visible_C_response':visible(cc,.75),'visible_C_response_to_B_transfer':visible(tc,.5),'chain_supported_in_this_generator':bool(main and transfer and visible(bc,.75) and visible(cc,.75) and visible(tc,.5) and pressure['overlap_samples']>0),'causal_scope':'matched conditional B-trajectory intervention inside Nexus plus guidance; sufficiency evidence, not uniqueness or real-world online causal proof','strict_causal_chain_proven':False}
