"""Create a native Gaussian road surface from actual observations and LiDAR plane.

No generated image is used as a texture. This is a local planar-road candidate,
not a general solution for buildings, sky, or unobserved road texture.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'Omniscenegraph')]
from nexus_3dgs.surface_lifting import lift_pixels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scene', 'checkpoint', 'reference', 'ground', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--local-paint',action='store_true',help='Conservative single raw-observation foreground road-marking patches')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(4)
    import torch
    import yaml
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text())
    scene = json.loads((a.scene/'nexus_scene.json').read_text())
    S = np.linalg.inv(np.asarray(scene['frames'][0]['ego_to_world']))
    ground = json.loads(a.ground.read_text())
    x, y, z = ground['coefficients_z_ax_by_c']
    plane = np.array([-x, -y, 1., -z])
    bounds = np.array(ground['supported_xy_bounds_scene'])
    records = {f['frame_idx']: f['images'][0]['scene_metadata'] for f in json.loads((a.reference/'run.json').read_text())['frames']}
    from models.modules import AffineTransform
    aff = state['models']['Affine']
    affine = AffineTransform('Affine', len(aff['embedding.weight']), device=torch.device('cpu'))
    affine.load_state_dict(aff)
    all_points, all_scales, all_quats, all_colors, footprints = [], [], [], [], []
    report = []
    for f in ((20,) if a.local_paint else (20,99)):
        record = records[f]
        C, K = S @ np.asarray(record['camera_world']), np.asarray(record['intrinsics'])
        intr = np.loadtxt(a.scene/'intrinsics/0.txt')
        source_image=Image.open(a.scene/'images'/f'{f:03d}_0.jpg').convert('RGB')
        W,H=source_image.size if a.local_paint else (960,540)
        K=K.copy();K[0]*=W/960;K[1]*=H/540
        image = np.array(source_image.resize((W,H),Image.Resampling.BILINEAR))
        image = cv2.undistort(image, K, intr[4:]).astype(np.float32)/255
        with np.load(a.scene/'scene_semantics'/f'{f:03d}_0.npz') as m:
            mask = m['road'].astype(np.uint8)
        if mask.shape!=(H,W):mask=cv2.resize(mask,(W,H),interpolation=cv2.INTER_NEAREST)
        vehicle = np.array(Image.open(a.scene/'vehicle_interior_masks'/f'{f:03d}_0.png')) > 0
        if vehicle.shape != mask.shape:
            vehicle = cv2.resize(vehicle.astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST)>0
        mask[cv2.dilate(vehicle.astype(np.uint8),np.ones((61,61) if a.local_paint else (11,11),np.uint8))>0] = 0
        mask = cv2.erode(mask,np.ones((3,3),np.uint8)) > 0
        if a.local_paint:
            yy,xx=np.indices(mask.shape);mask &= (yy>.62*H)&(xx>.15*W)&(xx<.85*W)
            hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV);white=mask&(image.mean(-1)>.43)&(hsv[...,1]<.22)
            count,labels,stats,_=cv2.connectedComponentsWithStats(white.astype(np.uint8),8)
            accepted=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>120);accepted=accepted[accepted!=0];paint=np.isin(labels,accepted)
            mask &= cv2.dilate(paint.astype(np.uint8),np.ones((13,13),np.uint8))>0
            if not mask.any():raise ValueError('No supported near-road paint patch')
        v, u = np.where(mask)
        uv = np.column_stack((u, v))
        points, scales, quats, valid = lift_pixels(uv,K,C,plane)
        valid &= (points[:,:2]>=bounds[0]).all(-1) & (points[:,:2]<=bounds[1]).all(-1)
        # The earlier observation owns texture wherever it exists. The later
        # observation only fills disocclusions; never layer two road markings.
        for old_C, old_K, old_coverage in footprints:
            xyz = (points-old_C[:3,3]) @ old_C[:3,:3]
            pixels = xyz[:,:2]/np.maximum(xyz[:,2,None],.1) @ old_K[:2,:2].T + old_K[:2,2]
            ix,iy = np.rint(pixels).astype(int).T
            inside = (xyz[:,2]>.1)&(ix>=0)&(ix<W)&(iy>=0)&(iy<H)
            duplicate = np.zeros(len(points),bool)
            duplicate[inside] = cv2.dilate(old_coverage.astype(np.uint8),np.ones((3,3),np.uint8))[iy[inside],ix[inside]]>0
            valid &= ~duplicate
        coverage = np.zeros((H,W),bool)
        coverage[v[valid],u[valid]] = True
        footprints.append((C,K,coverage))
        with torch.no_grad():
            A = affine({'img_idx':torch.tensor([f*3])})[0].numpy()
        rgb = (image[v[valid],u[valid]]-A[:,3]) @ np.linalg.inv(A[:,:3]).T
        all_points.append(points[valid]);all_scales.append(scales[valid]);all_quats.append(quats[valid]);all_colors.append(np.clip(rgb,.001,.999))
        report.append({'frame':f,'native_road_gaussians':int(valid.sum())})
    bg = state['models']['Background']
    means = bg['_means'].numpy()
    suppress = np.zeros(len(means),bool)
    for C,K,coverage in footprints:
        xyz = (means-C[:3,3]) @ C[:3,:3]
        pixels = xyz[:,:2]/np.maximum(xyz[:,2,None],.1) @ K[:2,:2].T + K[:2,2]
        ix,iy = np.rint(pixels).astype(int).T
        inside = (xyz[:,2]>.1)&(ix>=0)&(ix<W)&(iy>=0)&(iy<H)
        owned = np.zeros(len(means),bool)
        owned[inside] = coverage[iy[inside],ix[inside]]
        # The legacy full-road candidate uses a narrow height envelope. The
        # local-paint diagnostic also clears source-owned buried texture points.
        height = means[:,2] - (x*means[:,0]+y*means[:,1]+z)
        suppress |= owned & ((height<1.5) if a.local_paint else (np.abs(height)<1.5))
    bg['_opacities'][suppress] = -15.
    road = {'_means':np.concatenate(all_points),'_scales':np.concatenate(all_scales),
            '_quats':np.concatenate(all_quats),'_features_dc':np.log(np.concatenate(all_colors)/(1-np.concatenate(all_colors)))}
    N = len(road['_means'])
    road['_features_rest'] = np.empty((N,0,3));road['_opacities'] = np.full((N,1),4.6)
    state['models']['RoadNodes'] = {k:torch.tensor(v,dtype=torch.float32) for k,v in road.items()}
    protected = {k: all(torch.equal(v,state['models'][k][name]) for name,v in values.items())
                 for k,values in torch.load(a.checkpoint,map_location='cpu',weights_only=False)['models'].items() if k!='Background'}
    assert all(protected.values())
    cfg['model']['RoadNodes'] = copy.deepcopy(cfg['model']['Background'])
    cfg['model']['RoadNodes']['ctrl'] = {**cfg['model']['RoadNodes'].get('ctrl',{}),'sh_degree':0,'gaussian_2d':False,'ball_gaussians':False}
    if not a.local_paint:
        cfg['data']['data_root'] = str(a.scene.parent)
    else:
        original_scene=Path(cfg['data']['data_root'])/str(cfg['data']['scene_idx'])
        original_meta=json.loads((original_scene/'nexus_scene.json').read_text())
        assert original_meta['identity']['scenario_token']==scene['identity']['scenario_token']
        assert original_meta['frames'][0]['ego_to_world']==scene['frames'][0]['ego_to_world']
    torch.save(state,a.output/'checkpoint_final.pth')
    (a.output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    result = {'status':'candidate','source_frames':report,'road_gaussians':N,'local_paint_only':a.local_paint,'suppressed_old_road_gaussians':int(suppress.sum()),'protected_models':protected,
              'scope':'native 3D Gaussian surface, real observed RGB, LiDAR-supported local plane; no generated-ego pose changes; buildings and actors unchanged'}
    (a.output/'surface.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
