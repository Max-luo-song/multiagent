"""CPU road-distance and position-derived kinematic diagnostic definitions."""

import ast
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


TRAJECTORY_SOURCE=Path('nuplan_extent/planning/training/modeling/aggregated_metrics/traj_metric.py')


class RoadGeometry:
    def __init__(self,graph,validity,scaling,centerlines_only=False):
        graph=np.asarray(graph,dtype=np.float32)
        valid=np.asarray(validity[...,0],dtype=bool).copy()
        if centerlines_only:
            valid &= (graph[...,2]+graph[...,3])==1
        means,std=scaling
        xy=graph[...,:2]*np.float32(2)*std[:2]+means[:2]
        self.points=xy[valid].copy()
        self.tree=cKDTree(self.points) if len(self.points) else None
        paired=valid[:,:-1] & valid[:,1:]
        # Every retained point remains available, including isolated valid vertices.
        self.starts=np.concatenate((xy[:,:-1][paired],self.points)).astype(np.float64)
        ends=np.concatenate((xy[:,1:][paired],self.points)).astype(np.float64)
        self.directions=ends-self.starts
        length2=(self.directions*self.directions).sum(axis=-1)
        self.inverse_length2=np.divide(1.,length2,out=np.zeros_like(length2),where=length2>0)

    def distances(self,states,valid,segments=False,native_float32=False):
        valid=np.asarray(valid,dtype=bool)
        result=np.full(valid.shape,1e6,dtype=np.float64)
        query=np.asarray(states[...,:2][valid],dtype=np.float32)
        if not len(query) or self.tree is None:
            return result
        if not segments and not native_float32:
            result[valid]=self.tree.query(query,k=1)[0]
        elif not segments:
            import torch
            # Same float32 norm and map-point definition as the released code.
            points=torch.from_numpy(self.points)
            values=[]
            for offset in range(0,len(query),128):
                delta=torch.from_numpy(query[offset:offset+128])[:,None,:]-points[None,:,:]
                values.append(torch.norm(delta,dim=-1).min(dim=-1).values.numpy())
            result[valid]=np.concatenate(values)
        else:
            values=[]
            for offset in range(0,len(query),128):
                delta=query[offset:offset+128,None,:].astype(np.float64)-self.starts[None,:,:]
                position=np.clip(np.einsum('qsd,sd->qs',delta,self.directions)*self.inverse_length2,0,1)
                residual=delta-position[...,None]*self.directions[None,:,:]
                values.append(np.sqrt((residual*residual).sum(axis=-1).min(axis=-1)))
            result[valid]=np.concatenate(values)
        return result


def native_road_counts(distances,valid,threshold=2.75):
    valid=np.asarray(valid,dtype=bool)
    initially_near=((distances[:,:5]<=threshold)&valid[:,:5]).any(axis=-1)
    departed=((distances[:,5:]>threshold)&valid[:,5:]).any(axis=-1)&initially_near
    denominator=int(valid.any(axis=-1).sum())
    return dict(scene_ratio_sum=float(departed.sum()/denominator) if denominator else 0.,
                scenes=int(denominator>0),departing_vehicles=int(departed.sum()),valid_vehicles=denominator)


def common_road_counts(distances,valid,eligible,threshold=2.75):
    valid=np.asarray(valid,dtype=bool)
    selected=valid[:,5:] & np.asarray(eligible,dtype=bool)[:,None]
    distant=(distances[:,5:]>threshold)&selected
    return dict(valid_future_frames=int(selected.sum()),distant_future_frames=int(distant.sum()),
                eligible_vehicles=int(np.asarray(eligible).sum()),departing_vehicles=int(distant.any(axis=-1).sum()),
                distance_sum_m=float(distances[:,5:][selected].sum()))


def physical_kinematics(states,valid,dt=.5):
    """Backward finite differences projected onto the vehicle's current heading.

    Jerk here differentiates the projected acceleration components. Future
    samples may use observed history for their finite-difference stencil;
    every required position must be valid. Units are m/s² and m/s³.
    """
    states=np.asarray(states,dtype=np.float64)
    valid=np.asarray(valid,dtype=bool)
    xy=states[...,:2]
    acceleration=np.diff(xy,n=2,axis=1)/(dt*dt)
    mask_a=valid[:,2:]&valid[:,1:-1]&valid[:,:-2]
    heading=np.arctan2(states[:,2:,3],states[:,2:,2])
    longitudinal=np.stack((np.cos(heading),np.sin(heading)),axis=-1)
    lateral=np.stack((-longitudinal[...,1],longitudinal[...,0]),axis=-1)
    a_lon=(acceleration*longitudinal).sum(axis=-1)
    a_lat=(acceleration*lateral).sum(axis=-1)
    j_lon=np.diff(a_lon,axis=1)/dt
    j_lat=np.diff(a_lat,axis=1)/dt
    mask_j=mask_a[:,1:]&mask_a[:,:-1]
    result={}
    for scope,start in (('all_agents',0),('non_ego_only',1)):
        entries={}
        for name,values,mask in (
            ('longitudinal_acceleration_mps2',a_lon[:,3:],mask_a[:,3:]),
            ('lateral_acceleration_mps2',a_lat[:,3:],mask_a[:,3:]),
            ('longitudinal_jerk_mps3',j_lon[:,2:],mask_j[:,2:]),
            ('lateral_jerk_mps3',j_lat[:,2:],mask_j[:,2:])):
            selected=mask[start:]
            entries[name]=dict(abs_sum=float(np.abs(values[start:])[selected].sum()),count=int(selected.sum()))
        result[scope]=entries
    return result


def load_native_kinematics(path=TRAJECTORY_SOURCE):
    """Load only the three local upstream functions, without its heavy imports."""
    import torch
    names={'mertic_AAE','mertic_ASV','mertic_comfort'}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==3
    namespace={'torch':torch,'math':math}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def native_kinematic_counts(states,valid,functions):
    import torch
    selected=np.asarray(valid[:,5:],dtype=bool).all(axis=-1)
    count=int(selected.sum())
    names=('M_AAE','M_ASV','M_LonACC','M_LonJerk','LatAcc','LatJerk')
    if not count:
        return {name:dict(sum=0.,count=0) for name in names}
    tensor=torch.from_numpy(np.asarray(states[:,5:][selected],dtype=np.float32).copy())
    values=(functions['mertic_AAE'](tensor),functions['mertic_ASV'](tensor),
            *functions['mertic_comfort'](tensor))
    return {name:dict(sum=float(value*count),count=count) for name,value in zip(names,values)}


def finalize_road_kinematics(counts):
    import copy
    result=copy.deepcopy(counts)
    n=result['native_road']
    n['scene_mean_ratio']=n['scene_ratio_sum']/n['scenes'] if n['scenes'] else None
    for name in ('common_points','common_segments'):
        values=result[name]
        values['frame_deviation_fraction']=(values['distant_future_frames']/values['valid_future_frames']
                                             if values['valid_future_frames'] else None)
        values['vehicle_deviation_fraction']=(values['departing_vehicles']/values['eligible_vehicles']
                                               if values['eligible_vehicles'] else None)
        values['mean_distance_m']=(values['distance_sum_m']/values['valid_future_frames']
                                    if values['valid_future_frames'] else None)
    for entries in result['physical_kinematics'].values():
        for v in entries.values():
            v['mean_abs']=v['abs_sum']/v['count'] if v['count'] else None
    for v in result['native_kinematics'].values():
        v['mean']=v['sum']/v['count'] if v['count'] else None
    return result
