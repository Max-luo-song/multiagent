"""Extend intrusion through the encounter estimated from observed speeds."""
import copy,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/nexus_four_hazards_20261110';OUT=BASE/'oncoming_timing_v2'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
def main():
 import numpy as np
 from shapely import wkb
 from shapely.geometry import Polygon
 from nexus_3dgs.bundle import decode,FEATURE_MEANS,FEATURE_STD
 protocol=json.loads((BASE/'protocol.json').read_text());info=copy.deepcopy(next(c for c in protocol['cases'] if c['kind']=='oncoming'))
 OUT.mkdir(exist_ok=False);case=OUT/'case_oncoming';shutil.copytree(BASE/info['case'],case)
 info['previous_case']=str(BASE/info['case']);info['case']='case_oncoming';t=np.arange(17)*.5
 def smooth(t,start,end):
  u=np.clip((t-start)/(end-start),0,1);return 10*u**3-15*u**4+6*u**5
 encounter=info['selection_candidate']['closing_time_s'];start=encounter+.6;end=start+1.8
 info['parameters'].update(return_start_s=start,return_end_s=end,timing_source='observed initial center separation divided by observed closing speed, plus0.6s hold; no generated E future used',historical_estimated_encounter_s=encounter)
 with np.load(case/'conditions_normal.npz') as d:values=d['values'].copy();keep=d['keep'];valid=d['valid']
 A=info['roles']['A'];given=decode(values[A]);heads=given[4:,2:4].copy();shift=info['parameters']['lateral_shift_m'];weight=smooth(t,.5,2)*(1-smooth(t,start,end))
 given[4:,:2]+=shift*weight[:,None]*np.c_[-heads[:,1],heads[:,0]]
 velocity=np.diff(given[4:,:2],axis=0)/.5;speed=np.linalg.norm(velocity,axis=-1);angle=np.arctan2(velocity[:,1],velocity[:,0]);moving=speed>.05;heads=given[5:,2:4].copy();heads[moving]=np.c_[np.cos(angle[moving]),np.sin(angle[moving])];given[5:,2:4]=heads;given[5:,4:6]=velocity
 values[A,5:]=((given[5:]-FEATURE_MEANS)/(2*FEATURE_STD)).astype(values.dtype)
 W=np.array(info['world_from_nexus']);road=wkb.loads((case/'reference/inputs/drivable.wkb').read_bytes()).buffer(.02);dense=np.arange(81)*.1;angles=np.interp(dense,np.arange(-2,8.01,.5),np.unwrap(np.arctan2(given[:,3],given[:,2])));xy=np.c_[np.interp(dense,np.arange(-2,8.01,.5),given[:,0]),np.interp(dense,np.arange(-2,8.01,.5),given[:,1])];off=0
 for q,a in zip(xy,angles):
  c,s=np.cos(a),np.sin(a);p=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*given[4,6:8]/2@np.array([[c,s],[-s,c]])+q;off+=not road.covers(Polygon(p@W[:2,:2].T+W[:2,3]))
 acc=float(np.linalg.norm(np.diff(given[:,:2],n=2,axis=0)/.25,axis=-1)[3:].max());assert off==0 and acc<=9,(off,acc)
 np.savez_compressed(case/'conditions_event.npz',values=values,keep=keep,valid=valid)
 info['input_precheck']['event'].update(A_offroad_samples=off,A_max_native_acceleration_mps2=acc);(case/'preparation.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
 protocol.update(status='prepared',cases=[info],parent_protocol=str(BASE/'protocol.json'),revision_reason='v1 A returned before estimated encounter and E changed less than0.08m; preserve v1 and extend intrusion through encounter using historical geometry only',attempts=[])
 (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n');print('ONCOMING_V2_PREPARED',start,end,'offroad',off,'acc',acc,flush=True)
if __name__=='__main__':main()
