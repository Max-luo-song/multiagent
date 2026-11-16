"""Recover sparse static colour-view geometry with known nuPlan cameras."""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minimum-ray-angle',type=float,default=.5)
    p.add_argument('--maximum-range',type=float,default=300.)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    cv2.setNumThreads(4);meta=json.loads((a.scene/'nexus_scene.json').read_text())
    intr=np.loadtxt(a.scene/'intrinsics/0.txt');K=np.array([[intr[0]/2,0,intr[2]/2],[0,intr[1]/2,intr[3]/2],[0,0,1.]])
    extr=np.loadtxt(a.scene/'extrinsics/0.txt');first=np.linalg.inv(np.array(meta['frames'][0]['ego_to_world']))
    sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.02);views={}
    for f in (20,75,80,85,90,95,99,100):
        frame=meta['frames'][f];cam=next(i for i in frame['images'] if i['camera_id']==0)
        C=first@np.array(cam['ego_to_world'])@extr;P=K@np.linalg.inv(C)[:3]
        image=np.array(Image.open(a.scene/'images'/f'{f:03d}_0.jpg').convert('RGB').resize((960,540),Image.Resampling.BILINEAR));image=cv2.undistort(image,K,intr[4:])
        with np.load(a.scene/'scene_semantics'/f'{f:03d}_0.npz') as masks:
            mask=(np.isin(masks['labels'],[0,1,6,11,25,48])&(masks['confidence']>=180)).astype(np.uint8)*255
        vehicle=cv2.imread(str(a.scene/'vehicle_interior_masks'/f'{f:03d}_0.png'),0)>0;mask[vehicle]=0
        mask=cv2.erode(mask,np.ones((5,5),np.uint8));keys,desc=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),mask)
        xy=np.array([k.pt for k in keys],np.float32)
        views[f]={'C':C,'P':P,'xy':xy,'desc':desc,'tree':cKDTree(xy),'image':image}
    matcher=cv2.BFMatcher();cloud=[];rows=[]
    for f,g in itertools.combinations(views,2):
        left,right=views[f],views[g];baseline=np.linalg.norm(left['C'][:3,3]-right['C'][:3,3])
        if baseline<1:continue
        matched=matcher.knnMatch(left['desc'],right['desc'],k=2);pairs=[m for m,n in matched if m.distance<.7*n.distance]
        if len(pairs)<20:continue
        i=np.array([m.queryIdx for m in pairs]);j=np.array([m.trainIdx for m in pairs]);p1=left['xy'][i];p2=right['xy'][j]
        _,inlier=cv2.findFundamentalMat(p1,p2,cv2.USAC_MAGSAC,1.0,.999)
        if inlier is None:continue
        good=inlier.ravel()>0;i=i[good];j=j[good];p1=p1[good];p2=p2[good]
        X=cv2.triangulatePoints(left['P'],right['P'],p1.T,p2.T).T
        finite=np.abs(X[:,3])>1e-10;X=X[:,:3]/np.where(finite,X[:,3],1)[:,None]
        ray1=X-left['C'][:3,3];ray2=X-right['C'][:3,3]
        cosine=(ray1*ray2).sum(-1)/np.maximum(np.linalg.norm(ray1,axis=-1)*np.linalg.norm(ray2,axis=-1),1e-12)
        angle=np.rad2deg(np.arccos(np.clip(cosine,-1,1)))
        error=np.zeros(len(X));valid=finite&(angle>=a.minimum_ray_angle)&(np.linalg.norm(X,axis=-1)<a.maximum_range)
        for view,target in ((left,p1),(right,p2)):
            uv=np.c_[X,np.ones(len(X))]@view['P'].T;valid&=uv[:,2]>.1
            error=np.maximum(error,np.linalg.norm(uv[:,:2]/np.maximum(uv[:,2,None],1e-8)-target,axis=-1))
        valid&=error<1.5
        # Confirm a compatible feature in a third, intermediate camera.
        check_frames=[h for h in views if f<h<g]
        if not check_frames:continue
        h=min(check_frames,key=lambda t:abs(t-(f+g)/2));third=views[h]
        uv=np.c_[X,np.ones(len(X))]@third['P'].T;valid&=uv[:,2]>.1;uv=uv[:,:2]/np.maximum(uv[:,2,None],1e-8)
        d,k=third['tree'].query(uv,k=1)
        descriptor=np.linalg.norm(left['desc'][i]-third['desc'][k],axis=-1)
        valid&=(d<2.5)&(descriptor<180)
        cloud.append(X[valid].astype(np.float32));rows.append({'frames':[f,g,h],'baseline_m':float(baseline),
                         'ratio_matches':len(pairs),'accepted_three_view_points':int(valid.sum()),
                         'median_ray_angle_deg':float(np.median(angle[valid])) if valid.any() else None})
    if not cloud:raise ValueError('No triangulated support')
    points=np.concatenate(cloud)
    if not len(points):raise ValueError('No points passed three-view checks')
    _,ids=np.unique(np.floor(points/.2).astype(np.int32),axis=0,return_index=True);points=points[ids]
    np.savez_compressed(a.output/'static_sfm.npz',points=points)
    report={'status':'complete','points':len(points),'minimum_ray_angle_deg':a.minimum_ray_angle,'maximum_range_m':a.maximum_range,
            'depth_scene_norm_quantiles_m':np.quantile(np.linalg.norm(points,axis=-1),[.1,.5,.9]).tolist(),
            'pairs':rows,'scope':'sparse image-derived geometry, known calibrated poses, vehicle/sky exclusion, three-view check; not ground truth'}
    (a.output/'triangulation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
