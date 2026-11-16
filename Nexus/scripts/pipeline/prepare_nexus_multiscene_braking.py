"""Predeclare stronger A-only events; reuse original normal inputs unchanged."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/nexus_multiscene_20261027';OUT=ROOT/'artifacts/nexus_multiscene_braking_20261027'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
def main():
    import numpy as np
    from shapely import wkb
    from shapely.geometry import Polygon,LineString,Point
    from nexus_3dgs.bundle import decode,FEATURE_MEANS,FEATURE_STD
    p=json.loads((BASE/'protocol.json').read_text());p=json.loads(json.dumps(p));p['indices']=[15,18];p['cases']=[c for c in p['cases'] if c['scenario']['index'] in p['indices']];p['roles']={k:v for k,v in p['roles'].items() if int(k) in p['indices']}
    p.update(status='preparing',selection='follow-up severity check on the two initially clean low-speed cases; selected after scene015 mild cut-in showed no threatening overlap',normal_prediction_reuse=str(BASE/'frozen_v4_seed0'),new_inference_samples_planned=2,
        event_description_zh='A在0.5–2.5秒切入，3秒起以2m/s²减速至2m/s；只编辑A，E/C未来仍自由。正常视频复用本轮原始正常输出。',
        report_title='Nexus：切入后减速的强化威胁测试',braking={'start_s':3.,'deceleration_mps2':2.,'minimum_speed_mps':2.},no_E_C_future_used_for_event_design=True,
        purpose='test stronger threat, separately from the frozen pure cut-in batch; no solver/weight tuning')
    OUT.mkdir(exist_ok=False)
    for info in p['cases']:
        idx=info['scenario']['index'];source=BASE/f'case_{idx:03d}';case=OUT/f'case_{idx:03d}';case.mkdir()
        for name in ('reference','map','scene.pt','conditions_normal.npz'):(case/name).symlink_to(source/name,target_is_directory=(source/name).is_dir())
        with np.load(source/'conditions_event.npz') as z:values=z['values'].copy();keep=z['keep'];valid=z['valid']
        states=decode(values);A=info['roles']['A'];given=states[A].copy();initial=given[4];speed=info['observed_speeds_mps']['A']
        with np.load(source/'map/routes.npz') as z:points=z[f'route_{A}']
        station=np.r_[0,np.linalg.norm(np.diff(points,axis=0),axis=-1).cumsum()];start=LineString(points).project(Point(initial[:2]));t=np.arange(17)*.5;tau=np.maximum(t-3,0);duration=max((speed-2)/2,0);decel=np.minimum(tau,duration)
        distance=speed*np.minimum(t,3)+speed*decel-decel**2+2*np.maximum(tau-duration,0)
        xy=np.c_[np.interp(start+distance,station,points[:,0]),np.interp(start+distance,station,points[:,1])];xy+=initial[:2]-xy[0]
        # Road-relative normal heading at the new sampled stations.
        eps=.1;before=np.c_[np.interp(start+distance-eps,station,points[:,0]),np.interp(start+distance-eps,station,points[:,1])];after=np.c_[np.interp(start+distance+eps,station,points[:,0]),np.interp(start+distance+eps,station,points[:,1])];h=after-before;h/=np.maximum(np.linalg.norm(h,axis=-1,keepdims=True),1e-8)
        # Exactly preserve the original cut-in through the braking onset.
        u=np.clip((t-.5)/2,0,1);shift=3*(10*u**3-15*u**4+6*u**5);side=-1 if info['cut_in_side']=='right' else 1
        xy+=side*shift[:,None]*np.c_[-h[:,1],h[:,0]]
        given[4:,:2]=xy;given[:11]=states[A,:11] # history and future through t=3s unchanged
        given[5:,4:6]=np.diff(given[4:,:2],axis=0)/.5;angle=np.arctan2(given[5:,5],given[5:,4]);given[5:,2:4]=np.c_[np.cos(angle),np.sin(angle)]
        given[:11]=states[A,:11]
        np.testing.assert_array_equal(given[:11],states[A,:11])
        values[A,11:]=((given[11:]-FEATURE_MEANS)/(2*FEATURE_STD)).astype(values.dtype)
        np.savez_compressed(case/'conditions_event.npz',values=values,keep=keep,valid=valid)
        W=np.array(json.loads((source/'reference/protocol.json').read_text())['world_from_nexus']);road=wkb.loads((source/'reference/inputs/drivable.wkb').read_bytes()).buffer(.02);times=np.arange(1,81)*.1;native=np.arange(-2,8.01,.5);pos=np.c_[np.interp(times,native,given[:,0]),np.interp(times,native,given[:,1])];angles=np.interp(times,native,np.unwrap(np.arctan2(given[:,3],given[:,2])));off=0
        for q,a in zip(pos,angles):
            rot=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]);corn=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*initial[6:8]/2@rot+q;off+=not road.covers(Polygon(corn@W[:2,:2].T+W[:2,3]))
        acc=np.linalg.norm(np.diff(given[:,:2],n=2,axis=0)/.25,axis=-1)[3:];info['input_precheck']['event']={'A_box_offroad_samples':int(off),'A_max_xy_acceleration_mps2':float(acc.max())};info['given_A_input_usable']=off==0 and float(acc.max())<=5
        info['event_kind']='cut_in_then_brake';(case/'preparation.json').write_text(json.dumps(info,indent=2)+'\n');print('STRONG_INPUT',idx,info['input_precheck'],info['given_A_input_usable'],flush=True)
    p['status']='prepared';(OUT/'protocol.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n')
    import audit_nexus_multiscene_inputs as audit;audit.OUT=OUT;audit.main()
    import audit_nexus_multiscene_fixed_A as bg;bg.OUT=OUT;bg.main()
    (OUT/'adapter_parity_check.json').write_bytes((BASE/'adapter_parity_check.json').read_bytes())
if __name__=='__main__':main()
