"""Check masks, matched histories and absence of hidden future state values."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from nexus_3dgs.bundle import FEATURE_MEANS,FEATURE_STD,decode
from nexus_3dgs.input_mode_conditions import compile_mode,MODES


def main():
    states=np.zeros((3,21,8));states[...,2]=1;states[...,6:]=[4.8,2.]
    times=(np.arange(21)-4)*.5
    states[:,:,0]=times[None,:]*5;states[1,:,0]+=12;states[1,:,1]=3.2
    original=((states-FEATURE_MEANS)/(2*FEATURE_STD)).astype(np.float32);valid=np.ones((3,21),bool)
    compiled={(m,v):compile_mode(original,valid,1,m,v) for m in MODES for v in ('normal','event')}
    for (mode,variant),(values,keep,audit) in compiled.items():
        assert (values[~keep]==0).all()
        assert (not keep[0,5:].any())==(mode!='official_sg_fixed_ego')
        assert keep[1,:5].all()
        if mode=='bp_history':assert not keep[:,5:].any()
        else:assert keep[1].all()
        if mode in ('bp_history','history_full_target','future_full_target'):assert keep[:,:5].all()
        if mode in ('pure_sg_free_ego','pure_sg_bad_schedule'):assert not keep[0].any() and not keep[2].any()
        if mode!='official_sg_fixed_ego':
            poisoned=original.copy();poisoned[:,5:]=np.nan
            np.testing.assert_array_equal(values,compile_mode(poisoned,valid,1,mode,variant)[0])
    for variant in ('normal','event'):
        np.testing.assert_array_equal(compiled[('bp_history',variant)][0][:,:5],compiled[('history_full_target',variant)][0][:,:5])
        np.testing.assert_array_equal(compiled[('pure_sg_free_ego',variant)][0][1],compiled[('future_full_target',variant)][0][1])
    normal=decode(compiled[('bp_history','normal')][0][1,4]);event=decode(compiled[('bp_history','event')][0][1,4])
    np.testing.assert_allclose(normal[:2],[14,3.2],atol=2e-6)
    np.testing.assert_allclose(event[:2],[14,.2],atol=2e-6)
    output={'status':'passed','matched_history_pair':True,'ego_future_free_except_official_reference':True,
            'pure_sg_other_histories_unknown':True,'future_poisoning_invariance_except_supplied_official_ego':True,
            'history_cut_in_endpoint_and_common_longitudinal_offset':True}
    print(json.dumps(output))


if __name__=='__main__':main()
