"""Analytical motion and independent geometry checks for road/kinematic audit."""

import numpy as np

from collision_audit import scene_scaling,add_counts
from road_kinematics_audit import RoadGeometry,common_road_counts,native_road_counts,physical_kinematics,load_native_kinematics,native_kinematic_counts


def main():
    from shapely.geometry import LineString,Point
    import torch
    torch.set_num_threads(2)
    scaling=scene_scaling()
    graph=np.zeros((1,2,7),dtype=np.float32)
    graph[0,:,0]=[-5/104,5/104];graph[...,2]=1
    valid_map=np.ones_like(graph)
    geometry=RoadGeometry(graph,valid_map,scaling,True)
    states=np.zeros((1,21,8),dtype=np.float32);valid=np.ones((1,21),dtype=bool)
    states[...,2]=1
    point=geometry.distances(states,valid)
    segments=geometry.distances(states,valid,segments=True)
    assert np.allclose(point,5,atol=1e-6) and np.allclose(segments,0,atol=1e-6)
    assert common_road_counts(point,valid,[True])['distant_future_frames']==16
    assert common_road_counts(segments,valid,[True])['distant_future_frames']==0
    states[:,5:,1]=3
    assert native_road_counts(geometry.distances(states,valid,True),valid)['scene_ratio_sum']==1
    rng=np.random.default_rng(20261022)
    q=rng.uniform(-9,9,(5,21,2)).astype(np.float32)
    probe=np.zeros((5,21,8),dtype=np.float32);probe[...,:2]=q
    actual=geometry.distances(probe,np.ones((5,21),dtype=bool),True)
    line=LineString([geometry.points[0],geometry.points[1]])
    expected=np.array([Point(p).distance(line) for p in q.reshape(-1,2)]).reshape(5,21)
    assert np.allclose(actual,expected,atol=1e-8)
    # Invalid vertices must not create a phantom bridge across a masked gap.
    gap=np.zeros((1,3,7),dtype=np.float32);gap[...,2]=1;gap[0,:,0]=[-5/104,0,5/104]
    mask=np.ones_like(gap);mask[0,1]=0
    gap_geometry=RoadGeometry(gap,mask,scaling,True)
    origin=np.zeros((1,21,8),dtype=np.float32)
    assert np.allclose(gap_geometry.distances(origin,valid,True),5,atol=1e-6)
    t=np.arange(21)*.5
    def trajectory(x,y):
        out=np.zeros((2,21,8),dtype=np.float64);out[...,0]=x;out[...,1]=y;out[...,2]=1
        return out
    def mean(result,key):
        d=result['all_agents'][key];return d['abs_sum']/d['count']
    straight=trajectory(3*t,0*t)
    for v in physical_kinematics(straight,np.ones((2,21),dtype=bool))['all_agents'].values():
        assert abs(v['abs_sum'])<1e-10
    accelerating=trajectory(t*t,0*t)
    result=physical_kinematics(accelerating,np.ones((2,21),dtype=bool))
    assert np.isclose(mean(result,'longitudinal_acceleration_mps2'),2)
    assert np.isclose(mean(result,'longitudinal_jerk_mps3'),0)
    lateral=physical_kinematics(trajectory(0*t,t*t),np.ones((2,21),dtype=bool))
    assert np.isclose(mean(lateral,'lateral_acceleration_mps2'),2)
    jerk=physical_kinematics(trajectory(t**3/6,0*t),np.ones((2,21),dtype=bool))
    assert np.isclose(mean(jerk,'longitudinal_jerk_mps3'),1)
    assert np.isclose(mean(jerk,'longitudinal_acceleration_mps2'),5.75)
    phi=.73;rotated=accelerating.copy()
    rotated[...,:2]=accelerating[...,:2]@np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
    rotated[...,2]=np.cos(phi);rotated[...,3]=np.sin(phi)
    rotated_result=physical_kinematics(rotated,np.ones((2,21),dtype=bool))
    assert np.isclose(mean(rotated_result,'longitudinal_acceleration_mps2'),2)
    missing=np.ones((2,21),dtype=bool);missing[:,6]=False
    assert physical_kinematics(accelerating,missing)['all_agents']['longitudinal_acceleration_mps2']['count']==26
    assert physical_kinematics(accelerating,missing)['all_agents']['longitudinal_jerk_mps3']['count']==24
    functions=load_native_kinematics()
    native=native_kinematic_counts(straight,np.ones((2,21),dtype=bool),functions)
    assert all(abs(v['sum'])<1e-6 for v in native.values())
    total={};add_counts(total,result);add_counts(total,result)
    assert total['all_agents']['longitudinal_acceleration_mps2']['count']==64
    print('ROAD_KINEMATICS_TEST_OK: 105 Shapely distances; sparse points, gaps, departures; '
          'constant speed/acceleration/jerk, rotation, invalid stencils, aggregation and upstream functions',flush=True)


if __name__=='__main__':
    main()
