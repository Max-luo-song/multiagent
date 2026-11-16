"""Independent geometric regression against Shapely and the upstream SAT code."""

import json
from pathlib import Path
import sys

import numpy as np

from collision_audit import rectangle_overlaps, collision_counts, add_counts, finalize_counts, scene_scaling, decode_scene


def box(x=0.,y=0.,heading=0.,length=4.,width=2.):
    return np.array([x,y,np.cos(heading),np.sin(heading),0,0,length,width],dtype=np.float64)


def main():
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'third_party/nuplan-devkit'))
    from shapely.geometry import Polygon
    from nuplan_extent.planning.training.modeling.models.tokenizers.base_tokenizer_utils import check_collision
    fixtures = [
        (box(),box(10),False),
        (box(),box(4),True),  # boundary contact follows the released convention
        (box(),box(.2,.3,np.pi/3),True),
        (box(length=20),box(14,length=20),True),  # 10 m culling would miss this overlap
    ]
    for a,b,expected in fixtures:
        assert bool(rectangle_overlaps(np.stack([a,b]),[True,True])[0,1])==expected
        assert not rectangle_overlaps(np.stack([a,b]),[True,False]).any()
    rng=np.random.default_rng(20261021)
    checked=0
    def polygon(s):
        h=np.arctan2(s[3],s[2]);c,t=np.cos(h),np.sin(h)
        corners=np.array([[1,1],[-1,1],[-1,-1],[1,-1]])*s[[6,7]]/2
        return Polygon(corners@np.array([[c,t],[-t,c]])+s[:2])
    for _ in range(30):
        states=np.stack([box(*rng.normal(0,4,2),rng.uniform(-np.pi,np.pi),rng.uniform(2,12),rng.uniform(1,3)) for _ in range(8)])
        actual=rectangle_overlaps(states,np.ones(8,dtype=bool))
        assert np.array_equal(actual,actual.T) and not actual.diagonal().any()
        for i in range(8):
            for j in range(i+1,8):
                a,b=states[i],states[j]
                oracle=polygon(a).intersects(polygon(b))
                upstream=bool(check_collision(np.array([*a[:2],np.arctan2(a[3],a[2]),a[7],a[6]]),
                                              np.array([*b[:2],np.arctan2(b[3],b[2]),b[7],b[6]])))
                assert bool(actual[i,j])==oracle==upstream,(a,b,actual[i,j],oracle,upstream)
                checked+=1
    pileup=np.tile(box(),(3,21,1));valid=np.ones((3,21),dtype=bool)
    counts=collision_counts(pileup,valid,goal_conditioned=True)
    assert counts['all_agents']['pair_events']==48
    assert counts['all_agents']['colliding_vehicle_frames']==48
    assert counts['all_agents']['generated_vehicle_frames']==45
    assert counts['non_ego_only']['pair_events']==16
    assert counts['non_ego_only']['colliding_vehicle_frames']==32
    # Each of two cars collides; pair/vehicle ratio is 1/2, collided-vehicle fraction is 1.
    two=collision_counts(pileup[:2],valid[:2])
    total={};add_counts(total,two);add_counts(total,two);out=finalize_counts(total)
    assert out['all_agents']['vehicle_frame_collision_fraction']==1
    assert out['all_agents']['upstream_integer_horizon_pair_ratio']==.5
    bus=np.stack([np.tile(box(length=20),(21,1)),np.tile(box(14,length=20),(21,1))])
    assert collision_counts(bus,valid[:2])['all_agents']['pair_events_missed_by_10m']==16
    # A short track's provided goal is excluded at its own last valid time.
    short=valid.copy();short[1,11:]=False
    expected=int(short[:,5:].sum())-3
    assert collision_counts(pileup,short,True)['all_agents']['generated_vehicle_frames']==expected
    invalid=valid.copy();invalid[2]=False
    assert collision_counts(pileup,invalid)['all_agents']['valid_vehicles']==2
    assert not rectangle_overlaps(np.empty((0,8)),np.empty(0,dtype=bool)).any()
    mean,std=scene_scaling()
    values=np.ones((2,21,8),dtype=np.float32)
    import torch
    expected=torch.from_numpy(values)*2*torch.tensor(std)+torch.tensor(mean)
    assert np.array_equal(decode_scene(values,(mean,std)),expected.numpy())
    print('COLLISION_AUDIT_TEST_OK: '+str(checked)+' random pairs matched Shapely and upstream; '
          'contact, invalid masks, pileups, long vehicles, short goals, aggregation and decode checked',flush=True)


if __name__=='__main__':
    main()
