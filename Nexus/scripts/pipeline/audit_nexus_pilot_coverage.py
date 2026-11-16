"""Measure geometric camera coverage and real LiDAR support for predicted cars."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from nexus_3dgs.bundle import load_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    arrays,manifest=load_bundle(a.bundle);meta=json.loads((a.scene/'nexus_scene.json').read_text());info=json.loads((a.scene/'instances/instances_info.json').read_text())
    tokens={str(token):int(slot) for token,slot in zip(arrays['track_tokens'],arrays['nexus_slots']) if slot}
    tracks={r['id']:r for r in info.values() if r['id'] in tokens}
    byframe={}
    rows={token:{'nexus_slot':tokens[token],'track_token':token,'lidar_points_total':0,'frames_with_5_lidar_points':0,'geometric_camera_observations':0,'cameras':set()} for token in tracks}
    for token,r in tracks.items():
        annotations=r['frame_annotations']
        for f,pose,size in zip(annotations['frame_idx'],annotations['obj_to_world'],annotations['box_size']):byframe.setdefault(f,[]).append((token,np.array(pose),np.array(size)))
    calibration={c:(np.loadtxt(a.scene/'intrinsics'/f'{c}.txt'),np.loadtxt(a.scene/'extrinsics'/f'{c}.txt')) for c in meta['identity']['cameras']}
    for frame in meta['frames']:
        f=frame['frame_idx'];points=np.fromfile(a.scene/'lidar'/f'{f:03d}.bin',np.float32).reshape(-1,6)[:,:3]
        ego=np.array(frame['ego_to_world']);world=points@ego[:3,:3].T+ego[:3,3]
        for token,body,size in byframe.get(f,[]):
            delta=world-body[:3,3];near=np.max(np.abs(delta),axis=-1)<float(np.linalg.norm(size)/2+.2)
            local=delta[near]@body[:3,:3];count=int((np.abs(local)<size/2).all(axis=-1).sum());row=rows[token]
            row['lidar_points_total']+=count;row['frames_with_5_lidar_points']+=int(count>=5)
            for camera in frame['images']:
                c=camera['camera_id'];intr,extr=calibration[c];C=np.array(camera['ego_to_world'])@extr;point=(body[:3,3]-C[:3,3])@C[:3,:3]
                if point[2]<=.1:continue
                uv=point[:2]/point[2]*intr[:2]+intr[2:4]
                if 0<=uv[0]<1920 and 0<=uv[1]<1080:
                    row['geometric_camera_observations']+=1;row['cameras'].add(c)
    for row in rows.values():row['cameras']=sorted(row['cameras'])
    output={'status':'complete','scene':str(a.scene),'scenario_token':manifest['scenario_token'],'predicted_non_ego_tracks':len(tokens),
            'tracks':list(rows.values()),'not_in_scene_annotations':sorted(set(tokens)-set(tracks)),
            'limitations':'Camera observations are centre projections, not segmentation or occlusion proof. LiDAR support and available viewpoints do not guarantee full appearance coverage.'}
    a.output.write_text(json.dumps(output,indent=2)+'\n');print('PILOT_COVERAGE_COMPLETE',len(rows),'tracks',flush=True)


if __name__=='__main__':main()
