"""Compile the predeclared A-only cut-in + slowdown; never set B/C futures."""
import json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_chain_20261027'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
def main():
 import numpy as np
 from shapely import wkb
 from shapely.geometry import Polygon,LineString,Point
 from nexus_3dgs.bundle import decode,FEATURE_MEANS,FEATURE_STD
 p=json.loads((OUT/'protocol.json').read_text());assert p['status']=='prepared'
 p.update(event_kind='cut_in_then_slowdown',braking={'start_s':3.,'deceleration_mps2':2.,'minimum_speed_mps':2.},event_description_zh='A在0.5–3秒切入B车道，3秒起以2m/s²减速至2m/s；B/C未来自由。',status='compiling_event')
 for info in p['cases']:
  idx=info['scenario']['index'];case=OUT/f'case_{idx:03d}';A=info['roles']['A'];B=info['roles']['B'];C=info['roles']['C']
  shutil.copy2(case/'conditions_event.npz',case/'conditions_cutin_only.npz')
  with np.load(case/'conditions_event.npz') as z:values=z['values'].copy();keep=z['keep'];valid=z['valid']
  state=decode(values);given=state[A].copy();speed=info['observed_speeds_mps']['A'];initial=given[4]
  with np.load(case/'map/routes.npz') as z:points=z[f'route_{A}']
  station=np.r_[0,np.linalg.norm(np.diff(points,axis=0),axis=-1).cumsum()];start=LineString(points).project(Point(initial[:2]));t=np.arange(17)*.5;tau=np.maximum(t-3,0);dur=max((speed-2)/2,0);d=np.minimum(tau,dur);distance=speed*np.minimum(t,3)+speed*d-d*d+2*np.maximum(tau-dur,0)
  if station[-1]-start<distance[-1]+2:raise ValueError('A route too short; reject endpoint clamping')
  xy=np.c_[np.interp(start+distance,station,points[:,0]),np.interp(start+distance,station,points[:,1])];xy+=initial[:2]-xy[0]
  before=np.c_[np.interp(start+distance-.1,station,points[:,0]),np.interp(start+distance-.1,station,points[:,1])];after=np.c_[np.interp(start+distance+.1,station,points[:,0]),np.interp(start+distance+.1,station,points[:,1])];h=after-before;h/=np.maximum(np.linalg.norm(h,axis=-1,keepdims=True),1e-8)
  bh=state[B,4,2:4];lateral=(state[A,4,:2]-state[B,4,:2])@np.array([-bh[1],bh[0]]);u=np.clip((t-.5)/2.5,0,1);shift=-lateral*(10*u**3-15*u**4+6*u**5);xy+=shift[:,None]*np.c_[-h[:,1],h[:,0]]
  given[4:,:2]=xy;given[:11]=state[A,:11];given[5:,4:6]=np.diff(given[4:,:2],axis=0)/.5;angle=np.arctan2(given[5:,5],given[5:,4]);given[5:,2:4]=np.c_[np.cos(angle),np.sin(angle)];given[:11]=state[A,:11]
  values[A,11:]=((given[11:]-FEATURE_MEANS)/(2*FEATURE_STD)).astype(values.dtype);assert not keep[[B,C],5:].any()
  np.savez_compressed(case/'conditions_event.npz',values=values,keep=keep,valid=valid)
  W=np.array(json.loads((case/'reference/protocol.json').read_text())['world_from_nexus']);road=wkb.loads((case/'reference/inputs/drivable.wkb').read_bytes()).buffer(.02);tt=np.arange(1,81)*.1;native=np.arange(-2,8.01,.5);pos=np.c_[np.interp(tt,native,given[:,0]),np.interp(tt,native,given[:,1])];angles=np.interp(tt,native,np.unwrap(np.arctan2(given[:,3],given[:,2])));off=0
  for q,a in zip(pos,angles):
   rot=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]);corn=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*initial[6:8]/2@rot+q;off+=not road.covers(Polygon(corn@W[:2,:2].T+W[:2,3]))
  acc=np.linalg.norm(np.diff(given[:,:2],n=2,axis=0)/.25,axis=-1)[3:];info['input_precheck']['event']={'A_box_offroad_samples':int(off),'A_max_xy_acceleration_mps2':float(acc.max())};info['given_A_input_usable']=all(v['A_box_offroad_samples']==0 and v['A_max_xy_acceleration_mps2']<=5 for v in info['input_precheck'].values());(case/'preparation.json').write_text(json.dumps(info,indent=2)+'\n');print('CHAIN_EVENT',idx,info['input_precheck'],flush=True)
 p['status']='prepared';(OUT/'protocol.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n')
 import audit_nexus_multiscene_fixed_A as bg;bg.OUT=OUT;bg.main()
 audit=json.loads((OUT/'fixed_A_background_audit.json').read_text())
 for info in p['cases']:
  conflicts=[r for r in audit['records'] if r['index']==info['scenario']['index'] and r['fixed_A_vs_recorded_background_overlap_samples']]
  info['given_A_background_conflicts']=conflicts;info['given_A_input_usable'] &=not bool(conflicts)
  (OUT/f"case_{info['scenario']['index']:03d}"/'preparation.json').write_text(json.dumps(info,indent=2)+'\n')
 p['status']='prepared';(OUT/'protocol.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
