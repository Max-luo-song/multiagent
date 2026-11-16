"""Behavioral checks of the event compiler, including future-value poisoning."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from nexus_3dgs.bundle import FEATURE_MEANS,FEATURE_STD,decode
from nexus_3dgs.timed_event import compile_event


def main():
    states=np.zeros((3,21,8));states[...,2]=1;states[...,6:]=[4.5,2.]
    states[1,:,0]=15
    original=((states-FEATURE_MEANS)/(2*FEATURE_STD)).astype(np.float32)
    valid=np.ones((3,21),bool)
    request={'start_s':3.,'end_s':6.,'behavior':'reverse_approach','parameters':{'distance_m':4.5}}
    tensor,mask,audit=compile_event(original,valid,1,request)
    expected=np.zeros_like(mask);expected[:,:5]=True;expected[1,10:17]=True
    np.testing.assert_array_equal(mask,expected)
    np.testing.assert_allclose(decode(tensor[1,16])[:2],[10.5,0],atol=2e-6)
    poisoned=original.copy();poisoned[:,5:]=np.nan
    other=compile_event(poisoned,valid,1,request)[0]
    np.testing.assert_array_equal(tensor,other)
    normal,_,_=compile_event(original,valid,1,request,'normal')
    np.testing.assert_allclose(decode(normal[1,10:17])[:,0],15,atol=2e-6)
    rejected=[]
    for change in ({'start_s':3.1},{'end_s':9.},{'behavior':'unimplemented'}, {'parameters':{'distance_m':float('nan')}}):
        try:compile_event(original,valid,1,{**request,**change})
        except ValueError:rejected.append(change)
        else:raise AssertionError(f'Invalid request accepted: {change}')
    missing=valid.copy();missing[1,12]=False
    try:compile_event(original,missing,1,request)
    except ValueError:pass
    else:raise AssertionError('Missing actor accepted')
    moving=states.copy();moving[1,:,0]=15+(np.arange(21)-4)*2
    moving=((moving-FEATURE_MEANS)/(2*FEATURE_STD)).astype(np.float32)
    lateral=compile_event(moving,valid,1,{'start_s':2.,'end_s':4.,'behavior':'lateral_shift','parameters':{'distance_m':2.,'side':'left'}})[0]
    np.testing.assert_allclose(decode(lateral[1,12])[:2],[31,2],atol=3e-6)
    braking=compile_event(moving,valid,1,{'start_s':2.,'end_s':4.,'behavior':'brake','parameters':{'deceleration_mps2':2.}})[0]
    np.testing.assert_allclose(decode(braking[1,12])[:2],[27,0],atol=3e-6)
    result={'status':'passed','future_value_poisoning_invariance':True,'local_interval_mask_correct':True,
            'ego_and_other_futures_free':True,'normal_reverse_lateral_brake_endpoints_checked':True,
            'invalid_time_behavior_distance_and_presence_rejected':True}
    path=ROOT/'artifacts/nexus_3dgs/timed_event_20261026/compiler_checks.json'
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
