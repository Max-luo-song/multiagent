"""Seed static Gaussians only from visible, non-traffic training observations."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['scene','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--max-points',type=int,default=400000);p.add_argument('--voxel',type=float,default=.1);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cv2,numpy as np
    from PIL import Image
    cv2.setNumThreads(4);scene=json.loads((a.scene/'nexus_scene.json').read_text());S=np.linalg.inv(scene['frames'][0]['ego_to_world']);info=json.loads((a.scene/'instances/instances_info.json').read_text());byframe={}
    for item in info.values():
        ann=item['frame_annotations']
        for f,T,size in zip(ann['frame_idx'],ann['obj_to_world'],ann['box_size']):byframe.setdefault(f,[]).append((S@T,np.array(size)+.6))
    test=set(range(10,len(scene['frames']),10));parts=[];counts=[]
    for f,row in enumerate(scene['frames']):
        if f in test:continue
        raw=np.fromfile(a.scene/'lidar'/f'{f:03d}.bin',np.float32).reshape(-1,6)[:,:3];L=S@np.array(row['ego_to_world']);P=raw@L[:3,:3].T+L[:3,3];keep=(np.linalg.norm(raw,axis=1)>2.5)&(np.linalg.norm(raw,axis=1)<220)&(P[:,2]>-3)&(P[:,2]<80)
        keep &= ~((raw[:,0]>-1.45)&(raw[:,0]<4.4)&(np.abs(raw[:,1])<1.5)&(raw[:,2]>-.8)&(raw[:,2]<2.4))
        for B,size in byframe.get(f,[]):
            extent=np.abs(B[:3,:3])@(size/2);ids=np.flatnonzero(keep&(np.abs(P-B[:3,3])<=extent).all(1));local=(P[ids]-B[:3,3])@B[:3,:3];keep[ids[(np.abs(local)<size/2).all(1)]]=False
        best=np.full(len(P),-np.inf);rgb=np.zeros((len(P),3),np.float32)
        for camera in row['images']:
            c=camera['camera_id'];intr=np.loadtxt(a.scene/'intrinsics'/f'{c}.txt');K=np.array([[intr[0]/2,0,intr[2]/2],[0,intr[1]/2,intr[3]/2],[0,0,1.]]);C=S@np.array(camera['ego_to_world'])@np.loadtxt(a.scene/'extrinsics'/f'{c}.txt');q=(P-C[:3,3])@C[:3,:3];z=q[:,2];uv=q[:,:2]/np.maximum(z[:,None],1e-6)@K[:2,:2].T+K[:2,2];valid=(z>.2)&(uv[:,0]>=1)&(uv[:,0]<959)&(uv[:,1]>=1)&(uv[:,1]<539);ids=np.flatnonzero(valid);pix=np.floor(uv[ids]).astype(int);flat=pix[:,1]*960+pix[:,0];nearest=np.full(540*960,np.inf);np.minimum.at(nearest,flat,z[ids]);visible=z[ids]<=nearest[flat]+.15
            mask=cv2.imread(str(a.scene/'vehicle_interior_masks'/f'{f:03d}_{c}.png'),0)>0
            with np.load(a.scene/'scene_semantics'/f'{f:03d}_{c}.npz') as d:sky=d['sky']>0
            available=visible&keep[ids]&~mask[pix[:,1],pix[:,0]]&~sky[pix[:,1],pix[:,0]];ids=ids[available];pix=pix[available]
            if not len(ids):continue
            path=a.scene/'images'/f'{f:03d}_{c}.jpg';image=cv2.undistort(np.array(Image.open(path).convert('RGB').resize((960,540),Image.Resampling.BILINEAR)),K,intr[4:]);valid_pixels=cv2.undistort(np.full((540,960),255,np.uint8),K,intr[4:])>254
            score=z[ids]/np.linalg.norm(q[ids],axis=1)-.0001*z[ids];better=(score>best[ids])&valid_pixels[pix[:,1],pix[:,0]];selected=ids[better];pixels=pix[better];best[selected]=score[better];rgb[selected]=image[pixels[:,1],pixels[:,0]]/255
        selected=np.isfinite(best);parts.append(np.c_[P[selected],rgb[selected]].astype(np.float32));counts.append({'frame':f,'colored_static_points':int(selected.sum())})
        if f%10==1:print(f'STATIC_SEED {f} points={selected.sum()}',flush=True)
    allpoints=np.concatenate(parts);voxel=np.floor(allpoints[:,:3]/a.voxel).astype(np.int32);keys,inverse,number=np.unique(voxel,axis=0,return_inverse=True,return_counts=True);values=np.stack([np.bincount(inverse,weights=allpoints[:,i])/number for i in range(6)],axis=1).astype(np.float32)
    if len(values)>a.max_points:values=values[np.random.default_rng(0).choice(len(values),a.max_points,replace=False)]
    np.savez_compressed(a.output/'static_seed.npz',points=values[:,:3],colors=values[:,3:]);report={'status':'complete','scene':str(a.scene.resolve()),'source_observations':int(len(allpoints)),'voxel_m':a.voxel,'gaussian_seed_points':len(values),'held_out_timesteps':sorted(test),'heldout_rgb_not_used_in_gaussian_seed':True,'heldout_lidar_not_directly_sampled_as_gaussians':True,'pose_refinement_uses_all_lidar_including_rgb_validation_times':True,'validation_scope':'RGB holdout; not an independent sensor or unseen-scene split','policy':'real static returns only; all annotated traffic and conservative per-exposure vehicle/shadow masks excluded before assigning RGB; frontmost per-pixel LiDAR visibility; no high-confidence sky seeds','frames':counts};(a.output/'seed_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('STATIC_SEED_COMPLETE '+str(len(values)),flush=True)


if __name__=='__main__':main()
