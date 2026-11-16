"""Remove static-background Gaussian centers inside observed vehicle volumes.

Writes a new checkpoint. RGB is never painted or filled; unobserved regions
can remain incomplete. Ground-adjacent points are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text())
    scene=Path(cfg['data']['data_root'])/cfg['data']['scene_idx']
    source=json.loads((scene/'nexus_scene.json').read_text())
    info=json.loads((scene/'instances/instances_info.json').read_text())
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    bg=state['models']['Background'];means=bg['_means'].numpy()
    start=cfg['data']['start_timestep'];world_from_scene=np.array(source['frames'][start]['ego_to_world'])
    world=means@world_from_scene[:3,:3].T+world_from_scene[:3,3]
    remove=np.zeros(len(means),bool);counts={}
    for item in info.values():
        if item['class_name']!='vehicle':continue
        ann=item['frame_annotations'];poses=np.array(ann['obj_to_world']);sizes=np.array(ann['box_size'])
        # All observations, with a broad phase; a point must enter an actual
        # occupied volume, not merely an actor's XY trajectory envelope.
        radius=np.linalg.norm(sizes,axis=-1).max()/2
        low=poses[:,:3,3].min(axis=0)-radius;high=poses[:,:3,3].max(axis=0)+radius
        candidates=np.flatnonzero(((world>=low)&(world<=high)).all(axis=-1))
        marked=np.zeros(len(candidates),bool)
        for pose,size in zip(poses,sizes):
            local=(world[candidates]-pose[:3,3])@pose[:3,:3]
            inside=(np.abs(local)<np.maximum(size/2-.05,0)).all(axis=-1)
            inside &= local[:,2]>-size[2]/2+.25
            marked |= inside
        remove[candidates[marked]]=True;counts[item['id']]=int(marked.sum())
    n=len(means)
    for key,value in list(bg.items()):
        if torch.is_tensor(value) and value.ndim and value.shape[0]==n:
            bg[key]=value[torch.from_numpy(~remove)].clone()
    checkpoint=a.output/'checkpoint_final.pth';torch.save(state,checkpoint)
    shutil.copyfile(a.checkpoint.parent/'config.yaml',a.output/'config.yaml')
    report={'status':'complete','method':'background center exclusion from observed 3D vehicle interiors; bottom .25 m retained',
            'source_checkpoint':str(a.checkpoint.resolve()),'source_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            'output_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'background_before':n,'removed':int(remove.sum()),'background_after':int((~remove).sum()),
            'counts_by_track':counts,'rigid_appearance_and_poses_changed':False,
            'scope':'geometry cleanup; does not reconstruct hidden background or establish novel-view quality'}
    (a.output/'cleanup.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='counts_by_track'}),flush=True)


if __name__=='__main__':main()
