"""Sample all reconstructed actors with the renderer's physical registration."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .runtime import TrajectoryController
from .grounding import GroundedRegistration
from .geometry import resample_poses


def xyyaw(values):
    return np.concatenate((values[..., :2,3],np.arctan2(values[...,1,0],values[...,0,0])[...,None]),axis=-1)


class SceneTraffic:
    def __init__(self,checkpoint,bundle,ground_model):
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)['models']['RigidNodes']
        rigid=SimpleNamespace(instance_ids=state['dataset_instance_ids'],instances_trans=state['instances_trans'],
                              instances_quats=state['instances_quats'],instances_fv=state['instances_fv'])
        rigid.num_frames,rigid.num_instances=rigid.instances_fv.shape
        scene=Path(json.loads(Path(ground_model).read_text())['scene'])
        self.controller=TrajectoryController(bundle,scene,rigid)
        self.ground=GroundedRegistration(self.controller,ground_model,stabilize_all=True)
        self.sizes=np.r_[np.array([[5.176,2.297]]),state['instances_size'].numpy()[:,:2]]
        info=json.loads((scene/'instances/instances_info.json').read_text())
        self.tokens=['__ego__']+[info[str(int(i))]['id'] for i in rigid.instance_ids]
        self.agents={m+1:a for a,m in self.controller.bindings.items()}
        self.agents[0]=0
        self.reference_times=np.array([f['timestamp_us'] for f in self.controller.scene['frames']])
        T,M=len(self.reference_times),rigid.num_instances
        self.reference=np.broadcast_to(np.eye(4),(T,M,4,4)).copy()
        self.reference_valid=np.zeros((T,M),bool)
        for m,refs in self.ground.reference_poses.items():
            for f,p in refs.items():self.reference[f,m]=p;self.reference_valid[f,m]=True

    def sample(self,timestamps):
        ctrl=self.controller;ts=np.asarray(timestamps,dtype=np.int64)
        predicted,pv=resample_poses(ctrl.bundle['timestamps_us'],ctrl.bundle['predicted_poses_world'],ctrl.bundle['valid'],ts)
        refs,rv=resample_poses(self.reference_times,self.reference,self.reference_valid,ts)
        N,M=len(ts),len(self.tokens)
        boxes=np.zeros((N,M,3));valid=np.zeros((N,M),bool)
        boxes[:,0]=xyyaw(predicted[:,0]);valid[:,0]=pv[:,0]
        boxes[:,0,:2]+=1.461*np.stack((np.cos(boxes[:,0,2]),np.sin(boxes[:,0,2])),axis=-1)
        for i in range(1,M):
            agent=self.agents.get(i)
            poses,mask=(predicted[:,agent],pv[:,agent]) if agent is not None else (refs[:,i-1],rv[:,i-1])
            for t in np.flatnonzero(mask):
                world=self.ground.world_from_scene@self.ground.object_scene_pose(i-1,poses[t])
                boxes[t,i]=xyyaw(world)
            valid[:,i]=mask
        return boxes,valid
