"""Check static LiDAR alignment across time, independent of RGB and 3DGS."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['scene','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--all-frames',action='store_true')
    p.add_argument('--save-clouds',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    scene=json.loads((a.scene/'nexus_scene.json').read_text());S=np.linalg.inv(scene['frames'][0]['ego_to_world']);info=json.loads((a.scene/'instances/instances_info.json').read_text());boxes={}
    for item in info.values():
        ann=item['frame_annotations']
        for f,B,size in zip(ann['frame_idx'],ann['obj_to_world'],ann['box_size']):boxes.setdefault(f,[]).append((S@B,np.array(size)+.6))
    def cloud(f):
        raw=np.fromfile(a.scene/'lidar'/f'{f:03d}.bin',np.float32).reshape(-1,6)[:,:3];L=S@np.array(scene['frames'][f]['ego_to_world']);P=raw@L[:3,:3].T+L[:3,3];keep=(np.linalg.norm(raw,axis=1)>3)&(np.linalg.norm(raw,axis=1)<80)&(P[:,2]>-2)&(P[:,2]<25)
        keep &= ~((raw[:,0]>-1.45)&(raw[:,0]<4.4)&(np.abs(raw[:,1])<1.5)&(raw[:,2]>-.8)&(raw[:,2]<2.4))
        for B,size in boxes.get(f,[]):keep &= ~np.all(np.abs((P-B[:3,3])@B[:3,:3])<size/2,axis=1)
        P=P[keep];_,index=np.unique(np.floor(P/.2).astype(np.int32),axis=0,return_index=True);return P[index]
    anchor=cloud(20);tree=cKDTree(anchor);d,nn=tree.query(anchor,k=20,workers=8);groups=anchor[nn];offset=groups-groups.mean(1)[:,None];cov=np.einsum('nki,nkj->nij',offset,offset)/20;e,R=np.linalg.eigh(cov);normal=R[:,:,0];planar=(e[:,0]/np.maximum(e.sum(1),1e-10)<.025)&(d[:,-1]<1.3);anchor=anchor[planar];normal=normal[planar];tree=cKDTree(anchor);print(f'LIDAR_REGISTRATION anchor_planar_points={len(anchor)}',flush=True)
    np.savez_compressed(a.output/'anchor.npz',points=anchor,normals=normal)
    rows=[]
    for f in (range(len(scene['frames'])) if a.all_frames else [40,55,60,63,65,70,75,80,85,90,99,110,130]):
        P=cloud(f);T=np.eye(4);initial=None;final_H=np.eye(6);step=-1
        for step in range(0 if f==20 else 70):
            X=P@T[:3,:3].T+T[:3,3];distance,index=tree.query(X,k=1,workers=8);n=normal[index];r=((X-anchor[index])*n).sum(1);keep=(distance<(1.2 if step<10 else .6))&(np.abs(r)<(.35 if step<10 else .2));X,n,r=X[keep],n[keep],r[keep]
            if len(X)<200:raise ValueError('Insufficient static registration support')
            if initial is None:initial={'median_plane_distance_m':float(np.median(np.abs(r))),'rms_plane_distance_m':float(np.sqrt(np.mean(r*r))),'matched_points':len(X)}
            w=1/np.maximum(1,np.abs(r)/.03);w*=np.where(np.abs(n[:,2])>.95,.25,1.);A=np.c_[np.cross(X,n),n];H=A.T@(w[:,None]*A);b=A.T@(w*r);delta=np.linalg.solve(H+np.eye(6)*1e-6,-b);D=np.eye(4);D[:3,:3]=Rotation.from_rotvec(delta[:3]).as_matrix();D[:3,3]=delta[3:];T=D@T;final_H=H
            if np.linalg.norm(delta[:3])<1e-7 and np.linalg.norm(delta[3:])<1e-5:break
        X=P@T[:3,:3].T+T[:3,3];distance,index=tree.query(X,k=1,workers=8);r=((X-anchor[index])*normal[index]).sum(1);keep=(distance<.6)&(np.abs(r)<.2);after={'median_plane_distance_m':float(np.median(np.abs(r[keep]))),'rms_plane_distance_m':float(np.sqrt(np.mean(r[keep]**2))),'matched_points':int(keep.sum())}
        if initial is None:initial=after.copy()
        before_dist,before_nn=tree.query(P,k=1,workers=8);before_r=((P-anchor[before_nn])*normal[before_nn]).sum(1);common=keep&(before_dist<.6)&(np.abs(before_r)<.2)
        row={'frame':f,'timestamp_us':scene['frames'][f]['timestamp_us'],'iterations':step+1,'scene_correction_matrix':T.tolist(),'rotation_correction_deg':float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(T[:3,:3]).as_rotvec()))),'rotation_vector_scene_deg':np.rad2deg(Rotation.from_matrix(T[:3,:3]).as_rotvec()).tolist(),'translation_component_m':T[:3,3].tolist(),'initial':initial,'after':after,'common_correspondence_points':int(common.sum()),'common_before_rms_m':float(np.sqrt(np.mean(before_r[common]**2))),'common_after_rms_m':float(np.sqrt(np.mean(r[common]**2))),'normal_equation_condition_number':float(np.linalg.cond(final_H))};rows.append(row)
        if a.save_clouds:np.savez_compressed(a.output/f'{f:03d}.npz',before=P,after=X)
        print(f'LIDAR_REGISTRATION {f} rotation={row["rotation_correction_deg"]:.4f} common_rms={row["common_before_rms_m"]:.4f}->{row["common_after_rms_m"]:.4f}',flush=True)
    report={'status':'complete','method':'robust static point-to-plane ICP to measured frame 20; dynamic annotation boxes and ego returns excluded, 0.2 m voxels, locally planar support','scope':'cross-time consistency of existing LiDAR world transforms; no RGB or Gaussian model; estimated corrections are diagnostic, not applied automatically','rows':rows};(a.output/'lidar_registration.json').write_text(json.dumps(report,indent=2)+'\n');print('LIDAR_REGISTRATION_COMPLETE',flush=True)


if __name__=='__main__':main()
