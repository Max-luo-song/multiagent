"""Seed static facade detail on conservative three-view estimated planes."""
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]
from nexus_3dgs.surface_lifting import lift_pixels
from scripts.pipeline.correct_nexus_far_geometry import fit_inverse_depth


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','scene','reference','sfm','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4)
    import torch,yaml
    from models.modules import AffineTransform
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text())
    scene=json.loads((a.scene/'nexus_scene.json').read_text());S=np.linalg.inv(scene['frames'][0]['ego_to_world'])
    record=next(f for f in json.loads((a.reference/'run.json').read_text())['frames'] if f['frame_idx']==20)['images'][0]['scene_metadata']
    C=S@record['camera_world'];K=np.array(record['intrinsics'])
    with np.load(a.scene/'scene_semantics/020_0.npz') as m:
        mask=cv2.erode((np.isin(m['labels'],[1,25,48])&(m['confidence']>=204)).astype(np.uint8),np.ones((5,5),np.uint8))
    _,components=cv2.connectedComponents(mask)
    P=np.load(a.sfm)['points'];xyz=(P-C[:3,3])@C[:3,:3];uv=xyz[:,:2]/np.maximum(xyz[:,2,None],.1)@K[:2,:2].T+K[:2,2]
    valid=(xyz[:,2]>60)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<540)
    xyz,uv=xyz[valid],uv[valid];pix=uv.astype(int);labels=components[pix[:,1],pix[:,0]]
    image=np.array(Image.open(a.scene/'images/020_0.jpg').convert('RGB').resize((960,540),Image.Resampling.BILINEAR))
    image=cv2.undistort(image,K,np.loadtxt(a.scene/'intrinsics/0.txt')[4:]).astype(np.float32)/255
    weights=state['models']['Affine'];aff=AffineTransform('Affine',len(weights['embedding.weight']),device=torch.device('cpu'));aff.load_state_dict(weights)
    with torch.no_grad():A=aff({'img_idx':torch.tensor([60])})[0].numpy()
    bg=state['models']['Background'];degree=bg['_features_rest'].shape[1];pieces={key:[] for key in bg};coverage=np.zeros((540,960),bool);report=[]
    for label in np.unique(labels):
        ids=np.flatnonzero(labels==label)
        if label==0 or len(ids)<30:continue
        fitted=fit_inverse_depth(uv[ids],xyz[ids,2])
        if fitted is None:continue
        coef,inlier,error=fitted
        v,u=np.where(components==label);px=np.column_stack((u,v))
        cuv=np.array([coef[0]/500,coef[1]/500,coef[2]-coef[0]*480/500-coef[1]*270/500]);n=C[:3,:3]@(K.T@cuv)
        plane=np.r_[n,-1-n@C[:3,3]]
        means,scales,quats,ok=lift_pixels(px,K,C,plane,pixel_std=.7,normal_std=.002)
        dep=((means-C[:3,3])@C[:3,2]);ok &= (dep>60)&(dep<350)&(np.exp(scales).max(-1)<2)
        u,v=u[ok],v[ok];count=int(ok.sum());coverage[v,u]=True
        color=(image[v,u]-A[:,3])@np.linalg.inv(A[:,:3]).T;color=np.clip(color,.001,.999)
        values={'_means':means[ok],'_scales':scales[ok],'_quats':quats[ok],'_features_dc':(color-.5)/.28209479177387814,
                '_features_rest':np.zeros((count,degree,3)),'_opacities':np.full((count,1),3.)}
        for key,value in values.items():pieces[key].append(value)
        report.append({'component':int(label),'support_points':len(ids),'inliers':int(inlier.sum()),'fit_residual_median_m':float(np.median(error)),'gaussians':count})
    mu=bg['_means'].numpy();camera=(mu-C[:3,3])@C[:3,:3];pixels=np.rint(camera[:,:2]/np.maximum(camera[:,2,None],.1)@K[:2,:2].T+K[:2,2]).astype(int)
    valid=(camera[:,2]>5)&(pixels[:,0]>=0)&(pixels[:,0]<960)&(pixels[:,1]>=0)&(pixels[:,1]<540);ids=np.flatnonzero(valid)
    suppress=ids[coverage[pixels[ids,1],pixels[ids,0]]];bg['_opacities'][suppress]=-18.
    count=0
    for key,value in bg.items():
        if pieces[key]:
            added=torch.tensor(np.concatenate(pieces[key]),dtype=value.dtype);bg[key]=torch.cat((value,added))
            if key=='_means':count=len(added)
    torch.save(state,a.output/'checkpoint_final.pth');(a.output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    result={'status':'candidate','native_facade_gaussians':count,'suppressed_old_facade_centres':len(suppress),'source_frame':20,'planes':report,
            'scope':'native Gaussians initialized from actual RGB on uncertain three-view estimated planes; followed by appearance-only fitting; no frame repainting'}
    (a.output/'seeds.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
