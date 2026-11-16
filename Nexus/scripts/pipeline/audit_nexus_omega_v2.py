"""Independent native/dense SAT, kinematics, replay-obstacle and road audit."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SCALE=[104,104,1,1,4,4,5,1.6]
MEAN=[0,0,0,0,0,0,4.5,2]


def sample_positions(states,valid,times,np):
    A,T=valid.shape
    pos=np.zeros((A,len(times),2));yaw=np.zeros((A,len(times)));seen=np.zeros((A,len(times)),bool)
    angles=np.arctan2(states[...,3],states[...,2])
    for k,time in enumerate(times):
        u=time/.5+4;lo=min(int(np.floor(u)),T-2);f=u-lo;hi=lo+1
        pos[:,k]=states[:,lo,:2]*(1-f)+states[:,hi,:2]*f
        d=angles[:,hi]-angles[:,lo];d=np.arctan2(np.sin(d),np.cos(d))
        yaw[:,k]=angles[:,lo]+f*d
        seen[:,k]=valid[:,lo]&valid[:,hi]
        if abs(f)<1e-7:seen[:,k]=valid[:,lo]
        if abs(f-1)<1e-7:seen[:,k]=valid[:,hi]
    return pos,yaw,seen


def box_sat(pos,yaw,size,np):
    h=np.stack([np.cos(yaw),np.sin(yaw)],-1)
    centers=pos.copy();centers[0]+=1.461*h[0]
    basis=np.stack([h,np.stack([-h[...,1],h[...,0]],-1)],-2)
    i,j=np.triu_indices(len(pos),1)
    axes=np.concatenate([basis[i],basis[j]],-2)
    ri=(np.abs((basis[i,:,None]*axes[...,None,:]).sum(-1))*size[i,None,None]/2).sum(-1)
    rj=(np.abs((basis[j,:,None]*axes[...,None,:]).sum(-1))*size[j,None,None]/2).sum(-1)
    sep=(np.abs(((centers[j]-centers[i])[...,None,:]*axes).sum(-1))-ri-rj).max(-1)
    return sep,i,j,centers,basis


def evaluate(payload,background,road,world,np):
    raw=payload['sampled_tensor'][0].numpy()
    fullvalid=payload['valid_mask'][0].bool().numpy()
    slots=np.flatnonzero(fullvalid.any(-1))
    states=raw[slots]*np.asarray(SCALE)+np.asarray(MEAN)
    gt=payload['scene_tensor_gt'][0].numpy()[slots]*np.asarray(SCALE)+np.asarray(MEAN)
    valid=fullvalid[slots];sizes=gt[:,4,6:8]
    metrics={}
    for name,times in [('native',np.arange(1,17)*.5),('dense',np.arange(1,81)*.1)]:
        pos,yaw,seen=sample_positions(states,valid,times,np)
        sep,i,j,centers,basis=box_sat(pos,yaw,sizes,np)
        pairs=seen[i]&seen[j];overlap=(sep<0)&pairs
        bus=(i==0)&(j==1)
        collisions=[]
        for p in np.flatnonzero(overlap.any(-1)):
            collisions.append({'slots':[int(slots[i[p]]),int(slots[j[p]])],
                'times_s':times[overlap[p]].tolist(),'min_signed_sat_m':float(sep[p,pairs[p]].min())})
        item={'samples':len(times),'predicted_pair_samples_overlapping':int(overlap.sum()),
            'ego_bus_overlapping_samples':int(overlap[bus].sum()),
            'minimum_ego_bus_signed_sat_m':float(sep[bus].min()),'overlaps':collisions}
        if background is not None:
            bt=background['times_s'];M=len(background['sizes']);B=len(times)
            bc=np.zeros((M,B,2));bh=np.zeros((M,B));bv=np.zeros((M,B),bool)
            for k,t in enumerate(times):
                if not bt[0]<=t<=bt[-1]:continue
                right=np.searchsorted(bt,t);right=max(1,min(right,len(bt)-1));left=right-1
                f=(t-bt[left])/(bt[right]-bt[left])
                bc[:,k]=(1-f)*background['centers'][left]+f*background['centers'][right]
                angle=np.arctan2(background['heading'][...,1],background['heading'][...,0])
                d=angle[right]-angle[left];d=np.arctan2(np.sin(d),np.cos(d))
                bh[:,k]=angle[left]+f*d
                bv[:,k]=background['valid'][left]&background['valid'][right]
            allpos=np.concatenate([centers,bc]);allyaw=np.concatenate([yaw,bh]);allsizes=np.concatenate([sizes,background['sizes']])
            # box_sat adds ego axle offset itself, so undo it in this input.
            allpos[0]=pos[0]
            bs,bi,bj,_,_=box_sat(allpos,allyaw,allsizes,np)
            vis=np.concatenate([seen,bv]);mask=(bi<len(slots))&(bj>=len(slots))
            ov=(bs<0)&(vis[bi]&vis[bj])&mask[:,None]
            item['predicted_vs_recorded_overlap_samples']=int(ov.sum())
            item['background_coverage_s']=[float(bt[0]),float(bt[-1])]
            item['all_vehicle_pair_samples_overlapping']=int(overlap.sum()+ov.sum())
        if road is not None:
            from shapely.geometry import Point,Polygon
            full_boxes=0;center_off=0;eligible_off=0;ego_off=0
            observed_pos=gt[:,4,:2].copy()
            observed_yaw=np.arctan2(gt[:,4,3],gt[:,4,2]);observed_pos[0]+=1.461*np.array([np.cos(observed_yaw[0]),np.sin(observed_yaw[0])])
            observed_world=observed_pos@world[:2,:2].T+world[:2,3]
            eligible=np.array([road.covers(Point(p)) for p in observed_world])
            for agent,time in np.argwhere(seen):
                center=centers[agent,time]@world[:2,:2].T+world[:2,3]
                off=not road.covers(Point(center))
                center_off+=off;eligible_off+=off and eligible[agent]
                ego_off+=off and agent==0
                corners=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*sizes[agent]/2
                corners=corners@basis[agent,time]+centers[agent,time]
                polygon=Polygon(corners@world[:2,:2].T+world[:2,3])
                full_boxes+=not road.covers(polygon)
            item.update(drivable_center_off_samples=int(center_off),
                initially_onroad_center_off_samples=int(eligible_off),ego_drivable_center_off_samples=int(ego_off),
                drivable_full_box_off_samples=int(full_boxes),
                drivable_scope='SDK road/intersection/generic-drivable/carpark union, zero tolerance; not a traffic-legality test')
        metrics[name]=item
    vel=np.diff(states[...,:2],axis=1)/.5;vv=valid[:,1:]&valid[:,:-1]
    acc=np.diff(vel,axis=1)/.5;av=vv[:,1:]&vv[:,:-1];av[:,:3]=False
    jerk=np.diff(acc,axis=1)/.5;jv=av[:,1:]&av[:,:-1];jv[:,2]=valid[:,2:6].all(-1)
    norms=np.linalg.norm(acc,axis=-1);jnorm=np.linalg.norm(jerk,axis=-1)
    agents=[]
    for n,slot in enumerate(slots):
        agents.append({'slot':int(slot),'acceleration_over_5_samples':int(((norms[n]>5)&av[n]).sum()),
            'max_acceleration_mps2':float(norms[n,av[n]].max()) if av[n].any() else 0.,
            'max_jerk_mps3':float(jnorm[n,jv[n]].max()) if jv[n].any() else 0.})
    metrics['kinematics']={'all_agent_acceleration_over_5_samples':int(((norms>5)&av).sum()),'agents':agents}
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from shapely import wkb
    torch.set_num_threads(2)
    inputs=ROOT/'artifacts/nexus_omega_v2'
    with np.load(inputs/'background_inputs/background_replay.npz',allow_pickle=False) as data:
        background={k:data[k] for k in data.files}
    road=wkb.loads((inputs/'drivable_inputs/drivable.wkb').read_bytes())
    world=np.array(json.loads((inputs/'drivable_inputs/manifest.json').read_text())['world_from_nexus'])
    result={'status':'complete','records':[],'scope':'native 16 and dense 80 future points; physical boxes; explicit recorded replay obstacles; SDK drivable polygons'}
    for root in a.runs:
        run=json.loads((root/'run.json').read_text())
        if run['status']!='complete':raise ValueError('Incomplete input: '+str(root))
        for record in run['records']:
            folder=root/record['path'];path=folder/'prediction.pt'
            payload=torch.load(path,map_location='cpu',weights_only=False)
            keep=payload['task_mask'].bool();pred=payload['sampled_tensor'];cond=payload['conditioned_scene_tensor']
            torch.testing.assert_close(pred[keep],cond[keep],rtol=0,atol=0)
            assert not keep[:,0,5:].any() and (cond[~keep]==0).all()
            metrics=evaluate(payload,background,road,world,np)
            row={'run':root.name,'method':record['method'],'seed':record['seed'],'variant':record['variant'],
                'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'metrics':metrics,
                'noise_sha256':record['noise_sha256'],'conditions_exact':True,'ego_future_free':True}
            result['records'].append(row)
            (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
            print('DENSE_AUDIT',root.name,record['method'],record['seed'],record['variant'],
                'native',metrics['native']['ego_bus_overlapping_samples'],
                'dense_all',metrics['dense']['all_vehicle_pair_samples_overlapping'],
                'acc',metrics['kinematics']['all_agent_acceleration_over_5_samples'],
                'road',metrics['dense']['initially_onroad_center_off_samples'],flush=True)
    print('DENSE_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
