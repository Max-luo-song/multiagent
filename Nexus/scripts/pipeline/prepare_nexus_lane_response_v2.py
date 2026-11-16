"""V2 inputs: identical A progress cap from observed stopped lead in both arms."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/nexus_lane_response_20261027';OUT=ROOT/'artifacts/nexus_lane_response_v2_20261027'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline'),str(ROOT/'third_party/nuplan-devkit')]
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
 import numpy as np
 from shapely import wkb
 from shapely.geometry import Point,LineString,Polygon
 from nexus_3dgs.bundle import decode,FEATURE_MEANS,FEATURE_STD
 from prepare_nexus_chain_cases import raster
 import audit_nexus_multiscene_fixed_A as bg
 OUT.mkdir(exist_ok=False)
 for f in ['selection.json','chosen.json']:shutil.copy2(BASE/f,OUT/f)
 p=json.loads((BASE/'protocol.json').read_text())
 for info in p['cases']:
  idx=info['scenario']['index'];case=OUT/f'case_{idx:03d}';shutil.copytree(BASE/f'case_{idx:03d}',case,symlinks=False)
  A=info['roles']['A'];B=info['roles']['B'];C=info['roles']['C']
  with np.load(case/'conditions_normal.npz') as z:values=z['values'].copy();keep=z['keep'];valid=z['valid']
  s=decode(values);ini=s[A,4];h=ini[2:4];side=np.array([-h[1],h[0]]);rel=s[:,4,:2]-ini[:2];along=rel@h;lat=np.abs(rel@side);speed=np.linalg.norm(s[:,4,4:6],axis=-1);v=float(speed[A]);extent=(s[:,4,6]+ini[6])/2
  leads=np.flatnonzero(valid[:,4]&(speed<.5)&(along>extent+3)&(lat<(s[:,4,7]+ini[7])/2+.3)&((s[:,4,2:4]@h)>.9))
  assert len(leads),'This adapter requires an observed stopped lead, not future replay inference'
  lead=int(leads[np.argmin(along[leads]-extent[leads])]);distance=float(along[lead]-extent[lead]-2.)
  deceleration=1.8;begin=(distance-v*v/(2*deceleration))/v;assert begin>.5
  tt=np.arange(17)*.5;tau=np.clip(tt-begin,0,v/deceleration);progress=v*np.minimum(tt,begin)+v*tau-deceleration*tau**2/2
  with np.load(case/'map/routes.npz') as z:route=z[f'route_{A}']
  station=np.r_[0,np.linalg.norm(np.diff(route,axis=0),axis=-1).cumsum()];start=LineString(route).project(Point(ini[:2]));st=start+progress
  center=np.c_[np.interp(st,station,route[:,0]),np.interp(st,station,route[:,1])];center+=ini[:2]-center[0]
  lo=np.c_[np.interp(st-.1,station,route[:,0]),np.interp(st-.1,station,route[:,1])];hi=np.c_[np.interp(st+.1,station,route[:,0]),np.interp(st+.1,station,route[:,1])];direction=hi-lo;direction/=np.maximum(np.linalg.norm(direction,axis=-1,keepdims=True),1e-8)
  lateral=float((ini[:2]-s[B,4,:2])@np.array([-s[B,4,3],s[B,4,2]]));u=np.clip((tt-.5)/2.5,0,1);shift=-lateral*(10*u**3-15*u**4+6*u**5)
  W=np.array(json.loads((case/'reference/protocol.json').read_text())['world_from_nexus']);road=wkb.loads((case/'reference/inputs/drivable.wkb').read_bytes()).buffer(.02);prechecks={}
  for variant in ['normal','event']:
   xy=center.copy()
   if variant=='event':xy+=shift[:,None]*np.c_[-direction[:,1],direction[:,0]]
   given=s[A].copy();given[4:,:2]=xy;vel=np.diff(xy,axis=0)/.5;mag=np.linalg.norm(vel,axis=-1,keepdims=True);head=np.where(mag>.01,vel/np.maximum(mag,.01),direction[1:]);given[5:,2:4]=head;given[5:,4:6]=vel
   vv=values.copy();vv[A,5:]=((given[5:]-FEATURE_MEANS)/(2*FEATURE_STD)).astype(vv.dtype);assert np.array_equal(vv[:,:5],values[:,:5]);np.savez_compressed(case/f'conditions_{variant}.npz',values=vv,keep=keep,valid=valid)
   times=np.arange(1,81)*.1;grid=np.arange(-2,8.01,.5);pos=np.c_[np.interp(times,grid,given[:,0]),np.interp(times,grid,given[:,1])];angle=np.interp(times,grid,np.unwrap(np.arctan2(given[:,3],given[:,2])));off=0
   for q,a in zip(pos,angle):
    rot=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]);body=Polygon((np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*ini[6:8]/2@rot+q)@W[:2,:2].T+W[:2,3]);off+=not road.covers(body)
   acc=np.linalg.norm(np.diff(given[:,:2],n=2,axis=0)/.25,axis=-1)[3:];prechecks[variant]={'A_box_offroad_samples':int(off),'A_max_xy_acceleration_mps2':float(acc.max())}
  source=wkb.loads((case/'map/lanes'/f'{C}.wkb').read_bytes());target=wkb.loads((case/'map/adjacent_lane.wkb').read_bytes())
  grids=[raster(g,W,None,[-100.,-100.],.1,(2001,2801)) for g in (source,target)]
  np.savez_compressed(case/'map/lane_modes_sdf.npz',sdf=np.stack(grids),origin=np.array([-100.,-100.]),resolution=.1)
  info.update(input_precheck=prechecks,given_A_input_usable=all(x['A_box_offroad_samples']==0 and x['A_max_xy_acceleration_mps2']<=5 for x in prechecks.values()),A_progress_policy={'source':'current observed stationary lead, no future lead states','observed_lead_slot':lead,'common_both_variants':True,'stop_progress_m':distance,'deceleration_mps2':deceleration,'brake_start_s':begin,'reason':'prevent the prescribed normal A from driving through a stationary observed lead'})
  write(case/'preparation.json',info)
 p.update(previous_failed_experiment=str(BASE),event_kind='cut_in_with_common_observed_lead_stop',A_normal_event_longitudinal_schedule_identical=True,status='prepared')
 p['lane_choice'].update(maneuver_body_weight=10000.,selection='shared provisional joint traffic; optimize only C for each realized spatial mode; compare common objective; retain chosen mode as a soft spatial prior during denoising',C_path_or_speed_given=False)
 write(OUT/'protocol.json',p);bg.OUT=OUT;bg.main();audit=json.loads((OUT/'fixed_A_background_audit.json').read_text())
 for info in p['cases']:
  bad=[r for r in audit['records'] if r['index']==info['scenario']['index'] and r['fixed_A_vs_recorded_background_overlap_samples']]
  info['given_A_background_conflicts']=bad;info['given_A_input_usable'] &=not bool(bad);write(OUT/f"case_{info['scenario']['index']:03d}"/'preparation.json',info)
 write(OUT/'protocol.json',p);print('V2_PREPARED',[(i['given_A_input_usable'],i['input_precheck'],i['A_progress_policy']) for i in p['cases']],flush=True)
if __name__=='__main__':main()
