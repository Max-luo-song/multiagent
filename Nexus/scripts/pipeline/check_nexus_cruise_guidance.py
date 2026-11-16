"""Checks that a soft observed-speed target neither creeps nor overrides risk."""
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    from nuplan_extent.planning.training.modeling.models.diffusion.omega_cruise import CruiseConfig,CruiseRefiner
    from nuplan_extent.planning.training.modeling.models.diffusion.omega_structural import sat_separation
    torch.set_num_threads(2)
    scale=np.array([104,104,1,1,4,4,5,1.6],np.float32)
    mean=np.array([0,0,0,0,0,0,4.5,2],np.float32)
    times=np.arange(-2,8.01,.5)
    reports=[]
    with tempfile.TemporaryDirectory(prefix='nexus_cruise_check_') as directory:
        path=Path(directory)/'routes.npz';xx=np.linspace(-40,180,221)
        np.savez(path,route_0=np.c_[xx,xx*0],route_2=np.c_[xx,xx*0-3.8])
        for scenario in ('slow_neural_prior','cutin','stop_and_go'):
            states=np.zeros((3,21,8),np.float32)
            initial=[(0,0,12),(8,3.3,10),(4,-3.8,12)]
            if scenario=='stop_and_go':initial[1]=(22,0,8)
            for i,(x,y,v) in enumerate(initial):
                states[i,:,:2]=np.c_[x+times*v,np.full(21,y)]
                states[i,:,2]=1;states[i,:,4]=v;states[i,:,6:8]=[4.5,1.8]
            if scenario=='cutin':
                u=np.clip((times[5:]-.5)/2,0,1)
                states[1,5:,1]-=3.3*(10*u**3-15*u**4+6*u**5)
                vv=np.diff(states[1,4:,:2],axis=0)/.5
                angle=np.arctan2(vv[:,1],vv[:,0]);states[1,5:,2:4]=np.c_[np.cos(angle),np.sin(angle)]
                states[1,5:,4:6]=vv
            if scenario=='stop_and_go':
                tt=np.arange(17)*.5
                vv=np.where(tt<2,np.maximum(8-4*tt,0),np.where(tt<4,0,2*(tt-4)))
                xx=22+np.r_[0,np.cumsum((vv[:-1]+vv[1:])/2*.5)]
                states[1,4:,0]=xx;states[1,4:,4]=vv
            keep=np.zeros_like(states,bool);keep[:,:5]=True;keep[1]=True
            encoded=torch.from_numpy((states-mean)/scale)[None]
            mask=torch.from_numpy(keep)[None];valid=torch.ones(1,3,21)
            conditions=torch.where(mask,encoded,0);anchor=encoded.clone()
            if scenario=='slow_neural_prior':
                # An erroneously slow network proposal must not turn free
                # traffic into a queue when there is no conflicting lead.
                for actor in (0,2):
                    anchor[0,actor,5:,0]=torch.tensor((states[actor,4,0]+times[5:]*5)/104,dtype=anchor.dtype)
                    anchor[0,actor,5:,4]=5/4
            config=CruiseConfig(start_call=1,control_iterations=300,final_control_iterations=300,
                                following_weight=0,motion_qp=False,collision_weight=3000,
                                road_weight=1000,acceleration_weight=100,jerk_weight=2)
            refiner=CruiseRefiner(config,conditions,valid,mask,route_path=path)
            out=refiner(anchor,torch.ones_like(anchor),torch.ones_like(anchor),{})
            assert torch.equal(out[mask],conditions[mask])
            physical=out*torch.tensor(scale)+torch.tensor(mean)
            xy,head,seen=refiner.interpolate(physical,valid.bool())
            center=xy.clone();center[:,0]+=1.461*head[:,0]
            gaps=[]
            for j in (1,2):
                gap=sat_separation(center[:,0],head[:,0],refiner.sizes[:,0,None],
                                   center[:,j],head[:,j],refiner.sizes[:,j,None])
                gaps.append(float(gap.min()))
            assert min(gaps)>=0,(scenario,gaps)
            speed=physical[0,0,5:,4:6].norm(dim=-1)
            c_speed=physical[0,2,5:,4:6].norm(dim=-1)
            if scenario=='slow_neural_prior':
                assert float(speed.mean())>10.2 and float(speed[-1])>10.2,speed.tolist()
                assert float(c_speed.mean())>10.2 and float(c_speed[-1])>10.2,c_speed.tolist()
                assert float(physical[0,0,5:,1].abs().max())<.6
            elif scenario=='cutin':
                assert float(speed[:5].min())<11.5,speed.tolist()
                assert float(speed[-1])>8,speed.tolist()
            else:
                assert float(speed[:8].min())<3,speed.tolist()
                assert float(speed[-1])>float(speed.min())+2,speed.tolist()
            reports.append({'scenario':scenario,'minimum_EA_EC_gap_m':gaps,
                            'ego_speeds_mps':speed.tolist(),'C_speeds_mps':c_speed.tolist(),
                            'given_inputs_exact':True,'E_C_future_free':True})
    print(json.dumps({'status':'passed','checks':reports},indent=2))


if __name__=='__main__':main()
