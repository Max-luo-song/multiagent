"""Experimental native facade Gaussians on three-view triangulated support.

Only small triangles inside confident building regions are filled. Sparse,
low-parallax triangulation is an estimate, not measured facade ground truth.
"""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]
from nexus_3dgs.surface_lifting import lift_pixels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','checkpoint','reference','support','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4)
    import torch
    import yaml
    from models.modules import AffineTransform
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text())
    first=np.asarray(json.loads((a.scene/'nexus_scene.json').read_text())['frames'][0]['ego_to_world'])
    meta=json.loads((a.reference/'run.json').read_text())['frames'][0]['images'][0]['scene_metadata']
    C=np.linalg.inv(first)@np.array(meta['camera_world']);K=np.array(meta['intrinsics']);f=20
    points=np.load(a.support)['points'];xyz=(points-C[:3,3])@C[:3,:3]
    uv=xyz[:,:2]/xyz[:,2,None]@K[:2,:2].T+K[:2,2]
    inside=(xyz[:,2]>60)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<540)
    uv,points=uv[inside],points[inside]
    with np.load(a.scene/'scene_semantics'/f'{f:03d}_0.npz') as masks:
        mask=np.isin(masks['labels'],[1,25,48])&(masks['confidence']>=204)
    vehicle=np.array(Image.open(a.scene/'vehicle_interior_masks'/f'{f:03d}_0.png'))>0
    if vehicle.shape!=mask.shape:vehicle=cv2.resize(vehicle.astype(np.uint8),(960,540),interpolation=cv2.INTER_NEAREST)>0
    mask[cv2.dilate(vehicle.astype(np.uint8),np.ones((9,9),np.uint8))>0]=False
    mask=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8))>0
    image=np.array(Image.open(a.scene/'images'/f'{f:03d}_0.jpg').convert('RGB').resize((960,540),Image.Resampling.BILINEAR))
    image=cv2.undistort(image,K,np.loadtxt(a.scene/'intrinsics/0.txt')[4:]).astype(np.float32)/255
    aff=state['models']['Affine'];affine=AffineTransform('Affine',len(aff['embedding.weight']),device=torch.device('cpu'));affine.load_state_dict(aff)
    with torch.no_grad():A=affine({'img_idx':torch.tensor([f*3])})[0].numpy()
    triangulation=Delaunay(uv)
    vv,uu=np.where(mask);pixel=np.column_stack((uu,vv));simplex=triangulation.find_simplex(pixel)
    all_points,all_scales,all_quats,colors=[],[],[],[];coverage=np.zeros_like(mask);accepted=0
    for tri in np.unique(simplex[simplex>=0]):
        ids=triangulation.simplices[tri];p3=points[ids];p2=uv[ids]
        if np.linalg.norm(p2[:,None]-p2[None],axis=-1).max()>70:continue
        depth=(p3-C[:3,3])@C[:3,2]
        if depth.max()/depth.min()>1.5:continue
        n=np.cross(p3[1]-p3[0],p3[2]-p3[0]);norm=np.linalg.norm(n)
        if norm<.01:continue
        n/=norm
        # Road-like and grazing triangles are not facade evidence.
        if abs(n[2])>.6 or abs(n@((p3.mean(0)-C[:3,3])/np.linalg.norm(p3.mean(0)-C[:3,3])))<.2:continue
        px=pixel[simplex==tri]
        if not len(px):continue
        lifted,scales,quats,valid=lift_pixels(px,K,C,np.r_[n,-n@p3[0]],pixel_std=.7,normal_std=.002)
        valid &= np.exp(scales).max(-1)<1.5
        px=px[valid]
        all_points.append(lifted[valid]);all_scales.append(scales[valid]);all_quats.append(quats[valid])
        rgb=(image[px[:,1],px[:,0]]-A[:,3])@np.linalg.inv(A[:,:3]).T
        colors.append(np.clip(rgb,.001,.999));coverage[px[:,1],px[:,0]]=True;accepted+=1
    if not all_points:raise ValueError('No supported facade triangles')
    bg=state['models']['Background'];means=bg['_means'].numpy();p=(means-C[:3,3])@C[:3,:3]
    pix=np.rint(p[:,:2]/np.maximum(p[:,2,None],.1)@K[:2,:2].T+K[:2,2]).astype(int)
    visible=(p[:,2]>10)&(pix[:,0]>=0)&(pix[:,0]<960)&(pix[:,1]>=0)&(pix[:,1]<540)
    suppress=np.zeros(len(means),bool);suppress[visible]=coverage[pix[visible,1],pix[visible,0]]
    bg['_opacities'][suppress]=-15.
    count=sum(map(len,all_points));degree=bg['_features_rest'].shape[1]
    additions={'_means':np.concatenate(all_points),'_scales':np.concatenate(all_scales),'_quats':np.concatenate(all_quats),
               '_features_dc':(np.concatenate(colors)-.5)/.28209479177387814,'_features_rest':np.zeros((count,degree,3)),
               '_opacities':np.full((count,1),4.6)}
    for key,value in additions.items():bg[key]=torch.cat((bg[key],torch.tensor(value,dtype=torch.float32)))
    torch.save(state,a.output/'checkpoint_final.pth');(a.output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    cv2.imwrite(str(a.output/'source_coverage.png'),coverage.astype(np.uint8)*255)
    report={'status':'candidate','support_points':len(points),'accepted_triangles':accepted,'facade_gaussians':count,
            'suppressed_old_facade_gaussians':int(suppress.sum()),'source_frame':20,
            'scope':'real-image texture on sparse three-view estimated facade triangles; low-parallax uncertainty remains; no camera or actor changes'}
    (a.output/'facades.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
