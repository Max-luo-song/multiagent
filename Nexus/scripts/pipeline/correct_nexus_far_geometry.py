"""Separate infinity sky and test source-pixel-preserving facade depth correction."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def fit_inverse_depth(uv, depth):
    """Robust image inverse-depth plane, with inlier residual diagnostics."""
    X=np.column_stack(((uv-[480,270])/500,np.ones(len(uv))))
    target=1/depth;rng=np.random.default_rng(20261024)
    best=np.zeros(len(X),bool)
    # Depth uncertainty grows quickly at low parallax. Keep the uncertainty
    # visible in the report instead of claiming millimetric facade accuracy.
    for _ in range(600):
        sample=rng.choice(len(X),3,replace=False)
        if np.linalg.cond(X[sample])>1e4:continue
        coef=np.linalg.solve(X[sample],target[sample])
        predicted=X@coef
        inlier=(predicted>0)&(np.abs(predicted-target)<target*.12)
        if inlier.sum()>best.sum():best=inlier
    if best.sum()<25:return None
    coef=np.linalg.lstsq(X[best],target[best],rcond=None)[0]
    predicted=X@coef
    if (predicted[best]<=0).any():return None
    return coef,best, np.abs(1/predicted[best]-depth[best])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','initialized','trained','scene','reference','support','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(4)
    import torch,yaml
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    initialized=torch.load(a.initialized,map_location='cpu',weights_only=False)
    trained=torch.load(a.trained,map_location='cpu',weights_only=False)
    cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text())
    bg=state['models']['Background'];sky_ids=initialized['models']['Background']['_opacities'][:,0]<-9
    bg['_opacities'][sky_ids]=-18.
    state['models']['Sky']=trained['models']['Sky']
    cfg['model']['Sky']['params']['resolution']=128
    sky_dir=a.output/'road_sky';sky_dir.mkdir(exist_ok=True)
    torch.save(state,sky_dir/'checkpoint_final.pth');(sky_dir/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    scene=json.loads((a.scene/'nexus_scene.json').read_text());S=np.linalg.inv(scene['frames'][0]['ego_to_world'])
    meta=json.loads((a.reference/'run.json').read_text())['frames'][0]['images'][0]['scene_metadata']
    C=S@meta['camera_world'];K=np.array(meta['intrinsics'])
    with np.load(a.scene/'scene_semantics/020_0.npz') as masks:
        mask=(np.isin(masks['labels'],[1,25,48])&(masks['confidence']>=204)).astype(np.uint8)
    mask=cv2.erode(mask,np.ones((5,5),np.uint8));_,components,component_stats,_=cv2.connectedComponentsWithStats(mask)
    support=np.load(a.support)['points'];sp=(support-C[:3,3])@C[:3,:3]
    su=sp[:,:2]/sp[:,2,None]@K[:2,:2].T+K[:2,2]
    valid=(sp[:,2]>60)&(su[:,0]>=0)&(su[:,0]<960)&(su[:,1]>=0)&(su[:,1]<540)
    sp,su=sp[valid],su[valid];support_component=components[su[:,1].astype(int),su[:,0].astype(int)]
    means=bg['_means'].numpy().copy();old_points=means.copy();xyz=(means-C[:3,3])@C[:3,:3]
    uv=xyz[:,:2]/np.maximum(xyz[:,2,None],.1)@K[:2,:2].T+K[:2,2]
    in_image=(xyz[:,2]>5)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<540)
    component=np.zeros(len(means),int);pix=uv[in_image].astype(int);component[in_image]=components[pix[:,1],pix[:,0]]
    scales=bg['_scales'].numpy().copy();changed=np.zeros(len(means),bool);fits=[]
    for label in np.unique(support_component):
        ids=np.flatnonzero(support_component==label)
        if label==0 or len(ids)<30:continue
        fitted=fit_inverse_depth(su[ids],sp[ids,2])
        if fitted is None:continue
        coef,inliers,error=fitted
        hull=ConvexHull(su[ids[inliers]])
        coverage=np.zeros((540,960),np.uint8)
        cv2.fillConvexPoly(coverage,su[ids[inliers]][hull.vertices].astype(np.int32),1)
        coverage=cv2.dilate(coverage,np.ones((21,21),np.uint8))
        selected=np.flatnonzero(component==label)
        up=uv[selected].astype(int);selected=selected[coverage[up[:,1],up[:,0]]>0]
        X=np.column_stack(((uv[selected]-[480,270])/500,np.ones(len(selected))))
        inv_depth=X@coef;depth=1/np.maximum(inv_depth,1e-6);ratio=depth/xyz[selected,2]
        good=(depth>60)&(depth<350)&(ratio>1.2)&(ratio<8)
        selected=selected[good];ratio=ratio[good]
        means[selected]=C[:3,3]+(old_points[selected]-C[:3,3])*ratio[:,None]
        scales[selected]+=np.log(ratio[:,None]);changed[selected]=True
        fits.append({'component':int(label),'support_points':len(ids),'inliers':int(inliers.sum()),
                     'support_depth_median_m':float(np.median(sp[ids,2])),
                     'fit_residual_median_m':float(np.median(error)),
                     'corrected_gaussians':len(selected),'median_depth_scale':float(np.median(ratio)) if len(ratio) else None})
    # Exact original mean-pixel preservation at the source view is an analytic
    # property of ray scaling; verify it independently before saving.
    new_xyz=(means[changed]-C[:3,3])@C[:3,:3]
    pixels=new_xyz[:,:2]/new_xyz[:,2,None]@K[:2,:2].T+K[:2,2]
    reprojection=float(np.max(np.linalg.norm(pixels-uv[changed],axis=-1))) if changed.any() else 0.
    assert reprojection<1e-3, f'Float32 source reprojection drift: {reprojection} px'
    bg['_means']=torch.tensor(means,dtype=torch.float32);bg['_scales']=torch.tensor(scales,dtype=torch.float32)
    far_dir=a.output/'road_sky_far';far_dir.mkdir(exist_ok=True)
    torch.save(state,far_dir/'checkpoint_final.pth');(far_dir/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    report={'status':'candidates','suppressed_sky_gaussians':int(sky_ids.sum()),'facade_gaussians_corrected':int(changed.sum()),
            'source_mean_reprojection_max_px':reprojection,'facade_fits':fits,
            'scope':'sky learned from confident actual observations; facade depths estimated from low-parallax three-view support, not ground truth; source pixel centres/first-order covariance retained; other background and actor poses unchanged'}
    (a.output/'corrections.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
