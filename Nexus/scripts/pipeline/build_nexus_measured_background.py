"""Lift real static RGB into native Gaussians at measured LiDAR-supported depth.

Depth is interpolated only within the same observed semantic class, within
three pixels of measured returns, with a bounded local depth spread. Later
real views fill previously unsupported/occluded surfaces rather than stacking
another texture over an already represented surface.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['observations','scene','calibration','static-points','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--radius',type=float,default=2.5)
    p.add_argument('--depth-spread',type=float,default=1.12)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cv2,numpy as np
    from PIL import Image
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    cv2.setNumThreads(4);meta=json.loads((a.scene/'nexus_scene.json').read_text());S=np.linalg.inv(meta['frames'][0]['ego_to_world']);ext=np.loadtxt(a.scene/'extrinsics/0.txt');correction=np.array(json.loads(a.calibration.read_text())['fixed_correction']);P=np.load(a.static_points);annotations=json.loads((a.scene/'instances/instances_info.json').read_text());boxes={};sign=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    for item in annotations.values():
        ann=item['frame_annotations']
        for f,B,size in zip(ann['frame_idx'],ann['obj_to_world'],ann['box_size']):boxes.setdefault(f,[]).append((S@np.array(B),np.array(size)))
    all_means=[];all_colors=[];all_scales=[];all_quats=[];footprints=[];rows=[]
    for rec in json.loads((a.observations/'observations.json').read_text())['frames']:
        f=rec['frame'];d=dict(np.load(a.observations/f'{f:03d}.npz'));image=np.array(Image.open(a.observations/f'{f:03d}.png')).astype(np.float32)/255;H,W=image.shape[:2];C=S@np.array(meta['frames'][f]['images'][0]['ego_to_world'])@ext@correction;K=d['K'];vehicle=d['vehicle'].copy().astype(np.uint8)
        for B,size in boxes.get(f,[]):
            world=(sign*(size+.2)/2)@B[:3,:3].T+B[:3,3];q=(world-C[:3,3])@C[:3,:3]
            if q[:,2].min()<=.1:continue
            uv=q[:,:2]/q[:,2,None]@K[:2,:2].T+K[:2,2];cv2.fillConvexPoly(vehicle,cv2.convexHull(np.rint(uv).astype(np.int32)),1)
        vehicle=cv2.dilate(vehicle,np.ones((9,9),np.uint8))>0
        q=(P-C[:3,3])@C[:3,:3];uv=q[:,:2]/np.maximum(q[:,2,None],.1)@K[:2,:2].T+K[:2,2]-.5;ix,iy=np.rint(uv).astype(int).T;inside=(q[:,2]>1)&(ix>=0)&(ix<W)&(iy>=0)&(iy<H);ids=np.flatnonzero(inside);visible=~vehicle[iy[ids],ix[ids]];ids=ids[visible];lab=d['labels'][iy[ids],ix[ids]];supported=[];depths=[]
        for label in np.unique(lab):
            if label in [2,6]:continue
            selected=ids[lab==label]
            if len(selected)<20:continue
            # Frontmost real return at each image pixel prevents hidden walls
            # from supplying the apparent depth of foreground vegetation.
            zbuf=np.full(H*W,np.inf);flat=iy[selected]*W+ix[selected];np.minimum.at(zbuf,flat,q[selected,2]);selected=selected[q[selected,2]<=zbuf[flat]+.3]
            valid=(d['labels']==label)&(d['confidence']>.85)&(~vehicle);v,u=np.where(valid)
            if not len(u):continue
            distance,near=cKDTree(uv[selected]).query(np.column_stack((u,v)),k=4,workers=8);z=q[selected[near],2];good=(distance[:,0]<a.radius)&(distance[:,-1]<2*a.radius)&(z.max(-1)/np.maximum(z.min(-1),.1)<a.depth_spread)
            weight=1/np.maximum(distance,.35)**2;depth=(weight*z).sum(-1)/weight.sum(-1);supported.append(np.column_stack((u[good],v[good])));depths.append(depth[good])
        if not supported:continue
        pixels=np.concatenate(supported);depth=np.concatenate(depths);rays=np.column_stack(((pixels[:,0]+.5-K[0,2])/K[0,0],(pixels[:,1]+.5-K[1,2])/K[1,1],np.ones(len(pixels))));means=(rays*depth[:,None])@C[:3,:3].T+C[:3,3];keep=np.ones(len(means),bool)
        for old_C,old_K,coverage,old_depth in footprints:
            q0=(means-old_C[:3,3])@old_C[:3,:3];uv0=q0[:,:2]/np.maximum(q0[:,2,None],.1)@old_K[:2,:2].T+old_K[:2,2]-.5;x0,y0=np.rint(uv0).astype(int).T;ok=(q0[:,2]>.1)&(x0>=0)&(x0<W)&(y0>=0)&(y0<H);ind=np.flatnonzero(ok);same=coverage[y0[ind],x0[ind]]&(abs(q0[ind,2]-old_depth[y0[ind],x0[ind]])<np.maximum(.5,.08*q0[ind,2]));keep[ind[same]]=False
        pixels,depth,means=pixels[keep],depth[keep],means[keep];coverage=np.zeros((H,W),bool);depthmap=np.zeros((H,W));coverage[pixels[:,1],pixels[:,0]]=True;depthmap[pixels[:,1],pixels[:,0]]=depth;footprints.append((C,K,coverage,depthmap))
        scales=np.column_stack((depth/K[0,0]*.72,depth/K[1,1]*.72,depth/max(K[0,0],K[1,1])*.08));quat=Rotation.from_matrix(C[:3,:3]).as_quat()[[3,0,1,2]];all_means.append(means.astype(np.float32));all_colors.append(image[pixels[:,1],pixels[:,0]]);all_scales.append(scales.astype(np.float32));all_quats.append(np.broadcast_to(quat.astype(np.float32),(len(means),4)));rows.append({'frame':f,'native_gaussians':len(means)});print('MEASURED_STATIC '+str(rows[-1]),flush=True)
    means=np.concatenate(all_means);colors=np.concatenate(all_colors);scales=np.concatenate(all_scales);quats=np.concatenate(all_quats);np.savez_compressed(a.output/'static_gaussians.npz',means=means,colors=colors,scales=scales,quats=quats,opacity=np.full(len(means),.995,np.float32));result={'status':'candidate_not_accepted','native_static_gaussians':len(means),'source_frames':rows,'depth_source':'actual static LiDAR returns; image-space same-class local interpolation with depth-spread rejection','rgb_source':'actual undistorted photos; moving vehicle boxes excluded','scope':'measured and observed static surfaces only; unsupported regions retain original scene'};(a.output/'static.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
