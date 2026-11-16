"""Independent behavior audit of saved trajectories, without trajectory edits."""
import argparse
import json
from pathlib import Path
import hashlib


def measure(payload,np):
    scale=np.array([104,104,1,1,4,4,5,1.6]);mean=np.array([0,0,0,0,0,0,4.5,2])
    states=payload['sampled_tensor'][0].cpu().numpy()*scale+mean
    valid=payload['valid_mask'][0].cpu().numpy().astype(bool)
    velocity=np.diff(states[...,:2],axis=1)/.5
    mask=(valid[:,1:]&valid[:,:-1])[:,4:]
    v=velocity[:,4:];head=states[:,5:,2:4]
    head=head/np.maximum(np.linalg.norm(head,axis=-1,keepdims=True),1e-8)
    along=(v*head).sum(-1);side=v[...,0]*head[...,1]-v[...,1]*head[...,0]
    residual=np.linalg.norm(v-states[:,5:,4:6],axis=-1)
    speed=np.linalg.norm(v,axis=-1);moving=mask&(speed>.5)
    angle=np.degrees(np.arctan2(np.abs(side),along))
    ego=states[0];yaw=np.arctan2(ego[:,3],ego[:,2]);start=yaw[4]
    direction=np.array([np.cos(start),np.sin(start)]);normal=np.array([-direction[1],direction[0]])
    displacement=ego[5:,:2]-ego[4,:2]
    longitudinal=displacement@direction;lateral=displacement@normal
    yaw_delta=np.degrees(np.arctan2(np.sin(yaw[5:]-start),np.cos(yaw[5:]-start)))
    em=mask[0];reverse=np.maximum(-along[0],0)*em
    yaw_rate=np.abs(np.arctan2(np.sin(np.diff(yaw[4:])),np.cos(np.diff(yaw[4:]))))/.5
    curvature_excess=np.maximum(yaw_rate-.2*speed[0],0)
    def rms(a):return float(np.sqrt(np.mean(a*a))) if len(a) else None
    per_agent=[]
    for slot in np.flatnonzero(valid[:,5:].any(-1)):
        m=mask[slot];mm=moving[slot]
        per_agent.append({'slot':int(slot),'xy_velocity_head_rms_mps':rms(residual[slot,m]),
            'xy_lateral_velocity_rms_mps':rms(side[slot,m]),
            'moving_heading_error_max_deg':float(angle[slot,mm].max()) if mm.any() else None})
    return {'ego':{'final_forward_progress_m':float(longitudinal[em][-1]),
        'final_left_displacement_m':float(lateral[em][-1]),'max_left_displacement_m':float(lateral[em].max()),
        'max_abs_yaw_delta_deg':float(np.abs(yaw_delta[em]).max()),
        'max_backward_displacement_from_start_m':float(max(0,-longitudinal[em].min())),
        'reverse_distance_body_axis_m':float(reverse.sum()*.5),
        'reverse_speed_max_mps':float(reverse.max()),'unrequested_reverse_over_0p2m':bool(reverse.sum()*.5>.2),
        'curvature_bound_excess_max_rps':float(curvature_excess[em].max()),
        'curvature_bound_excess_over_0p01_samples':int((curvature_excess[em]>.01).sum()),
        'xy_velocity_head_rms_mps':rms(residual[0,em]),'xy_lateral_velocity_rms_mps':rms(side[0,em]),
        'moving_heading_error_max_deg':float(angle[0,moving[0]].max()) if moving[0].any() else None},
        'all_agents':per_agent,'scope':'backward XY differences at 0.5 s; body heading at interval end; not bicycle dynamics or route legality'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    a.output.mkdir(parents=True,exist_ok=False)
    result={'status':'running','records':[],'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for root in a.runs:
        run=json.loads((root/'run.json').read_text());assert run['status']=='complete'
        for record in run['records']:
            path=root/record['path']/'prediction.pt'
            payload=torch.load(path,map_location='cpu',weights_only=False)
            metrics=measure(payload,np)
            result['records'].append({'run':root.name,'method':record['method'],'seed':record['seed'],
                'variant':record['variant'],'seconds':record['seconds'],
                'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'metrics':metrics})
            print(root.name,record['seed'],record['variant'],json.dumps(metrics['ego']),flush=True)
    result['status']='complete'
    (a.output/'behavior.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
