"""Check actual temporary renderer poses, including its unbound actor policy."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from nexus_3dgs.runtime import TrajectoryController
from nexus_3dgs.grounding import GroundedRegistration
from nexus_3dgs.geometry import quaternion_matrix
from nexus_3dgs.clearance import rear_axle_to_box
from nexus_3dgs.scene_projection import pair_separations


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--render',type=Path,required=True)
    p.add_argument('--ground-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run=json.loads((a.render/'run.json').read_text())
    if run['status']!='complete':raise ValueError('Incomplete render')
    state=torch.load(run['checkpoint'],map_location='cpu',weights_only=False)['models']['RigidNodes']
    rigid=SimpleNamespace(instance_ids=state['dataset_instance_ids'],instances_trans=state['instances_trans'],
                          instances_quats=state['instances_quats'],instances_fv=state['instances_fv'])
    rigid.num_frames,rigid.num_instances=rigid.instances_fv.shape
    ground_record=run['ground_registration'];scene=Path(ground_record['ground_model']['scene'])
    ctrl=TrajectoryController(run['bundle'],scene,rigid)
    ctrl.grounding=GroundedRegistration(ctrl,a.ground_model,stabilize_all=ground_record['stabilize_all_rigid_instances'])
    for k,v in ground_record['object_mounts'].items():np.testing.assert_allclose(ctrl.grounding.object_mounts[int(k)],v,atol=1e-10)
    original=[v.clone() for v in (rigid.instances_trans,rigid.instances_quats,rigid.instances_fv)]
    sizes=np.r_[np.array([[5.176,2.297]]),state['instances_size'].numpy()[:,:2]]
    all_boxes=[];all_valid=[]
    for frame in run['frames']:
        image=next(i for i in frame['images'] if i['camera_id']==0)
        with ctrl.edit_frame(frame['frame_idx'],image['timestamp_us']) as ego:
            row=np.zeros((1+rigid.num_instances,3));mask=np.r_[True,rigid.instances_fv[frame['frame_idx']].numpy()]
            row[0]=[ego[0,3],ego[1,3],np.arctan2(ego[1,0],ego[0,0])];row[0]=rear_axle_to_box(row[0],1.461)
            for m in range(rigid.num_instances):
                pose=np.eye(4);pose[:3,3]=rigid.instances_trans[frame['frame_idx'],m].numpy()
                pose[:3,:3]=quaternion_matrix(rigid.instances_quats[frame['frame_idx'],m].numpy())
                world=ctrl.grounding.world_from_scene@pose
                row[m+1]=[world[0,3],world[1,3],np.arctan2(world[1,0],world[0,0])]
            all_boxes.append(row);all_valid.append(mask)
    for actual,expected in zip((rigid.instances_trans,rigid.instances_quats,rigid.instances_fv),original):
        if not torch.equal(actual,expected):raise ValueError('Renderer pose edit did not restore checkpoint state')
    sep,pairs=pair_separations(np.array(all_boxes),np.array(all_valid),sizes)
    overlap=np.argwhere((sep<=0)&pairs)
    report={'status':'passed' if not len(overlap) else 'failed','frames':len(all_boxes),
            'all_scene_actors_including_ego':len(sizes),'overlapping_pair_frame_samples':len(overlap),
            'minimum_sat_separation_m':float(sep[pairs].min()),
            'checkpoint_state_restored':True,'scope':'actual TrajectoryController.edit_frame poses at rendered camera exposures; physical 2D boxes',
            'overlap_frame_actor_indices':overlap.tolist()}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if len(overlap):raise ValueError('Actual rendered scene poses still overlap')


if __name__=='__main__':main()
