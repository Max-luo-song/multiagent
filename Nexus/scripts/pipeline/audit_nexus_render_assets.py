"""Locate reconstructed actors in the image and audit pose/appearance support."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from nexus_3dgs.runtime import TrajectoryController
from nexus_3dgs.grounding import GroundedRegistration
from nexus_3dgs.geometry import quaternion_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.render/'run.json').read_text())
    state = torch.load(run['checkpoint'], map_location='cpu', weights_only=False)['models']['RigidNodes']
    ground = run['ground_registration']
    scene_dir = Path(ground['ground_model']['scene'])
    info = json.loads((scene_dir/'instances/instances_info.json').read_text())
    rigid = SimpleNamespace(instance_ids=state['dataset_instance_ids'], instances_trans=state['instances_trans'],
                            instances_quats=state['instances_quats'], instances_fv=state['instances_fv'])
    rigid.num_frames, rigid.num_instances = rigid.instances_fv.shape
    ctrl = TrajectoryController(run['bundle'], scene_dir, rigid)
    # Match the exact saved camera and body registration, rather than re-fit.
    body_mounts = {int(k): np.array(v) for k,v in ground['object_mounts'].items()}
    camera_mount = np.array(ground['camera_mounts']['0'])
    plane = np.array(ground['ground_model']['coefficients_z_ax_by_c'])
    plane_scene_from_world = np.linalg.inv(np.array(ground['ground_model'].get('world_from_scene', ctrl.scene['frames'][0]['ego_to_world'])))
    coeff = plane_scene_from_world.T @ np.r_[-plane[:2], 1., -plane[2]]
    coeff /= np.linalg.norm(coeff[:3])
    helper = SimpleNamespace(world_plane=coeff)
    road_pose = lambda p: GroundedRegistration.road_pose(helper, p)
    width, height = Image.open(args.render/'060_0.png').size
    source_width, source_height = Image.open(scene_dir/'images/060_0.jpg').size
    intr = np.loadtxt(scene_dir/'intrinsics/0.txt')
    K = np.array([[intr[0]*width/source_width,0,intr[2]*width/source_width],
                  [0,intr[1]*height/source_height,intr[3]*height/source_height],[0,0,1]])
    observations = json.loads((REPO/'artifacts/nexus_3dgs/improvements/step01_all_camera_support.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_index, dataset_id in enumerate(rigid.instance_ids.numpy()):
        record = info[str(int(dataset_id))]; token = record['id']
        visible = rigid.instances_fv[:, model_index].numpy().astype(bool)
        trans = rigid.instances_trans[:, model_index].numpy()[visible]
        world = trans @ np.linalg.inv(ctrl.scene_from_world)[:3,:3].T+np.linalg.inv(ctrl.scene_from_world)[:3,3]
        ids = state['points_ids'][:,0].numpy()==model_index
        scales = state['_scales'].numpy()[ids]
        scales = np.exp(scales)
        means = state['_means'].numpy()[ids]
        size = state['instances_size'][model_index].numpy()
        rows.append({'model_index':model_index,'dataset_id':int(dataset_id),'track_token':token,
                     'nexus_agent':next((int(a) for a,m in ctrl.bindings.items() if m==model_index),None),
                     'grounded_in_saved_render':model_index in body_mounts,'size_lwh':size.tolist(),
                     'learned_world_height_range_m':float(np.ptp(world[:,2])),
                     'gaussians':int(ids.sum()),'scale_std_m_quantiles':np.quantile(scales.max(axis=1),[.5,.9,.99,1]).tolist(),
                     'center_outside_box_fraction':float((np.abs(means)>size/2).any(axis=1).mean()),
                     'support':next((v for v in observations['actors'].values() if v['track_token']==token),None), 'projections':[]})
    for frame in (20,60,99):
        image = Image.open(args.render/f'{frame:03d}_0.png').convert('RGB');draw=ImageDraw.Draw(image)
        record = ctrl.scene['frames'][frame]
        camera = next(c for c in record['images'] if c['camera_id']==0)
        poses, valid = ctrl.sample(camera['timestamp_us'])
        c2w = ctrl.scene_from_world @ road_pose(poses[0]) @ camera_mount
        with ctrl.edit_frame(frame,camera['timestamp_us']):
            for item in rows:
                m = item['model_index']; agent=item['nexus_agent']
                if not rigid.instances_fv[frame,m]:continue
                inside=False
                if agent is not None:
                    p = plane_scene_from_world @ poses[agent,:,3]
                    low,high=np.array(ground['ground_model']['supported_xy_bounds_scene'])
                    inside=bool(np.all(p[:2]>=low)&np.all(p[:2]<=high))
                if m in body_mounts and inside:
                    obj=ctrl.scene_from_world @ road_pose(poses[agent]) @ body_mounts[m]
                else:
                    obj=np.eye(4);obj[:3,:3]=quaternion_matrix(rigid.instances_quats[frame,m].numpy())
                    obj[:3,3]=rigid.instances_trans[frame,m].numpy()
                center=(np.linalg.inv(c2w)@obj)[:3,3]
                pixel=K@center
                if center[2]<=0:continue
                uv=pixel[:2]/pixel[2]
                if not(0<=uv[0]<width and 0<=uv[1]<height):continue
                label=f'{m} / id {item["dataset_id"]}'
                draw.text((uv[0],uv[1]),label,fill='red',stroke_width=1,stroke_fill='white')
                item['projections'].append({'frame':frame,'uv':uv.tolist(),'depth_m':float(center[2]),'ground_applied':inside and m in body_mounts})
        image.save(args.output/f'actors_{frame:03d}.png')
    (args.output/'assets.json').write_text(json.dumps({'status':'complete','actors':rows},indent=2)+'\n')
    for item in rows:
        if item['projections']:
            print(json.dumps({k:item[k] for k in ['model_index','track_token','nexus_agent','learned_world_height_range_m','scale_std_m_quantiles','projections']}))


if __name__=='__main__':main()
