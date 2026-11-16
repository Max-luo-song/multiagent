"""Conservative camera-ray depth rectification of the accepted original scene.

Keep original colors/opacity and source projected Gaussian footprints. Use
measured road/LiDAR depth, sparse estimated facade planes, and far-away sky.
No generated RGB, temporal smoothing or new vehicle model is introduced.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]
from nexus_3dgs.surface_lifting import rectify_gaussians
from scripts.pipeline.correct_nexus_far_geometry import fit_inverse_depth


def project(points,C,K):
    xyz=(points-C[:3,3])@C[:3,:3]
    uv=xyz[:,:2]/np.maximum(xyz[:,2,None],.1)@K[:2,:2].T+K[:2,2]
    valid=(xyz[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<540)
    return xyz,uv,valid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','scene','reference','ground','cloud','sfm','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4)
    import torch,yaml
    from scipy.spatial.transform import Rotation
    from scripts.pipeline.merge_nexus_vehicle_asset import equal
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    bg=state['models']['Background'];original={k:v.clone() for k,v in bg.items()}
    means=bg['_means'].numpy().astype(float);logs=bg['_scales'].numpy().astype(float);quats=bg['_quats'].numpy().astype(float)
    source_means=means.copy();assigned=np.zeros(len(means),np.uint8)
    scene=json.loads((a.scene/'nexus_scene.json').read_text());S=np.linalg.inv(scene['frames'][0]['ego_to_world'])
    ground=json.loads(a.ground.read_text());aa,bb,cc=ground['coefficients_z_ax_by_c'];bounds=np.array(ground['supported_xy_bounds_scene'])
    cloud=np.load(a.cloud);P=cloud['points'];normals=cloud['normals'];planar=cloud['planar'];sfm=np.load(a.sfm)['points']
    records={f['frame_idx']:i['scene_metadata'] for f in json.loads((a.reference/'run.json').read_text())['frames'] for i in f['images'] if i['camera_id']==0}
    reports=[];max_pixel_error=0.;max_cov_error=0.
    for f in [20,99]:
        metadata=records[f];C=S@np.array(metadata['camera_world']);K=np.array(metadata['intrinsics'])
        xyz,uv,in_image=project(source_means,C,K);pix=np.clip(uv,[0,0],[959,539]).astype(int)
        with np.load(a.scene/'scene_semantics'/f'{f:03d}_0.npz') as m:
            masks={k:m[k].copy() for k in m.files}
        vehicle=np.array(Image.open(a.scene/'vehicle_interior_masks'/f'{f:03d}_0.png'))>0
        if vehicle.shape!=(540,960):vehicle=cv2.resize(vehicle.astype(np.uint8),(960,540),interpolation=cv2.INTER_NEAREST)>0
        vehicle=cv2.dilate(vehicle.astype(np.uint8),np.ones((9,9),np.uint8))>0
        available=in_image&(assigned==0)&~vehicle[pix[:,1],pix[:,0]]
        sky=available&(masks['sky'][pix[:,1],pix[:,0]]>0)
        # Infinity represented by distant native Gaussians. Scaling centre and
        # covariance together preserves the source image rather than deleting
        # sky layers and exposing holes at the skyline.
        ids=np.flatnonzero(sky);ratio=10000/xyz[ids,2]
        means[ids]=C[:3,3]+(source_means[ids]-C[:3,3])*ratio[:,None]
        logs[ids]+=np.log(ratio[:,None]);assigned[ids]=1
        available &= ~sky
        target=np.zeros(len(means));normal=np.zeros_like(means);kind=np.zeros(len(means),np.uint8)
        road=available&(masks['road'][pix[:,1],pix[:,0]]>0)
        ids=np.flatnonzero(road);rays=(source_means[ids]-C[:3,3])/xyz[ids,2,None]
        n=np.array([-aa,-bb,1.]);depth=-(C[:3,3]@n-cc)/(rays@n)
        hit=C[:3,3]+rays*depth[:,None]
        keep=(depth>1)&(hit[:,:2]>=bounds[0]).all(-1)&(hit[:,:2]<=bounds[1]).all(-1)
        ids=ids[keep];target[ids]=depth[keep];normal[ids]=n;kind[ids]=2
        building=cv2.erode((np.isin(masks['labels'],[1,25,48])&(masks['confidence']>=204)).astype(np.uint8),np.ones((5,5),np.uint8))
        _,components=cv2.connectedComponents(building)
        sx,su,sv=project(sfm,C,K);valid=sv&(sx[:,2]>60);sx,su=sx[valid],su[valid]
        sp=su.astype(int);labels=components[sp[:,1],sp[:,0]];fitted=[]
        for label in np.unique(labels):
            support=np.flatnonzero(labels==label)
            if label==0 or len(support)<30:continue
            result=fit_inverse_depth(su[support],sx[support,2])
            if result is None:continue
            coef,inliers,error=result
            ids=np.flatnonzero(available&(kind==0)&(components[pix[:,1],pix[:,0]]==label))
            X=np.column_stack(((uv[ids]-[480,270])/500,np.ones(len(ids))))
            depth=1/np.maximum(X@coef,1e-9);keep=(depth>60)&(depth<350);ids=ids[keep]
            cuv=np.array([coef[0]/500,coef[1]/500,coef[2]-coef[0]*480/500-coef[1]*270/500])
            n=C[:3,:3]@(K.T@cuv);n/=np.linalg.norm(n)
            rays=(source_means[ids]-C[:3,3])/xyz[ids,2,None]
            nongrazing=np.abs(rays@n)>1e-3;ids=ids[nongrazing]
            target[ids]=depth[keep][nongrazing];normal[ids]=n;kind[ids]=3
            fitted.append({'component':int(label),'support':len(support),'inliers':int(inliers.sum()),'fit_residual_median_m':float(np.median(error)),'gaussians':len(ids)})
        px,pu,pv=project(P,C,K);cloud_ids=np.flatnonzero(pv&(px[:,2]<250))
        # A local depth agreement test rejects mixed foreground/background
        # samples along silhouette boundaries.
        tree=cKDTree(pu[cloud_ids]);ids=np.flatnonzero(available&(kind==0)&(masks['occupied'][pix[:,1],pix[:,0]]>0))
        distance,neighbors=tree.query(uv[ids],k=8,workers=8);support_ids=cloud_ids[neighbors];depth=px[support_ids,2]
        spread=np.quantile(depth,.8,axis=1)-np.quantile(depth,.2,axis=1);median=np.median(depth,axis=1)
        keep=(distance[:,0]<1.25)&(distance[:,-1]<4)&(spread<np.maximum(.5,.05*median))
        ids=ids[keep];support_ids=support_ids[keep];distance=distance[keep];depth=depth[keep]
        weight=1/(distance**2+.25);target[ids]=weight.sum(-1)/(weight/depth).sum(-1)
        nearest=support_ids[:,0];normal[ids]=normals[nearest]
        normal[ids[~planar[nearest]]]=C[:3,2];kind[ids]=4
        rays=(source_means[ids]-C[:3,3])/xyz[ids,2,None]
        grazing=np.abs((rays*normal[ids]).sum(-1))/np.linalg.norm(rays,axis=-1)<.15
        normal[ids[grazing]]=C[:3,2]
        ids=np.flatnonzero(kind>0);ratio=target[ids]/xyz[ids,2]
        rays=(source_means[ids]-C[:3,3])/xyz[ids,2,None]
        good=(ratio>.05)&(ratio<20)&(np.abs((rays*normal[ids]).sum(-1))/np.linalg.norm(normal[ids],axis=-1)>1e-3)
        ids=ids[good]
        points,scales,q=rectify_gaussians(source_means[ids],original['_scales'][ids].numpy(),original['_quats'][ids].numpy(),K,C,target[ids],normal[ids])
        _,pu_new,_=project(points,C,K);pixel_error=np.linalg.norm(pu_new-uv[ids],axis=-1)
        max_pixel_error=max(max_pixel_error,float(pixel_error.max()) if len(ids) else 0.)
        means[ids]=points;logs[ids]=scales;quats[ids]=q;assigned[ids]=kind[ids]
        reports.append({'frame':f,'sky':int(sky.sum()),'road':int((kind[ids]==2).sum()),'estimated_facade':int((kind[ids]==3).sum()),'local_lidar':int((kind[ids]==4).sum()),'facade_fits':fitted})
        print('RAY_RECTIFICATION '+json.dumps(reports[-1]),flush=True)
    bg['_means']=torch.tensor(means,dtype=torch.float32);bg['_scales']=torch.tensor(logs,dtype=torch.float32);bg['_quats']=torch.tensor(quats,dtype=torch.float32)
    output=a.output/'checkpoint_final.pth';torch.save(state,output)
    import shutil
    shutil.copyfile(a.checkpoint.parent/'config.yaml',a.output/'config.yaml')
    saved=torch.load(output,map_location='cpu',weights_only=False)
    before=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    for name,value in before['models'].items():
        if name!='Background':equal(value,saved['models'][name])
    for key in ('_features_dc','_features_rest','_opacities'):equal(original[key],saved['models']['Background'][key])
    report={'status':'candidate','counts':{name:int((assigned==i).sum()) for i,name in enumerate(['unchanged','sky_far','road','estimated_facade','local_lidar'])},'source_mean_reprojection_max_px_float64':max_pixel_error,'frames':reports,
            'original_colors_opacity_traffic_camera_sky_environment_preserved':True,'scope':'source pixel/first-order footprint preserving ray geometry correction; local plane/LiDAR and uncertain three-view facade support; skyline sorting and disocclusion still require visual review'}
    (a.output/'rectification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
