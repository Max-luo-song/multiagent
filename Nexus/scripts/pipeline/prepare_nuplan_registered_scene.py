"""Build a separate scene with common LiDAR/camera/annotation pose refinement."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['scene','semantics','registration','output-root']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();out=a.output_root/a.scene.name;out.mkdir(parents=True,exist_ok=False)
    import cv2,numpy as np
    from nexus_3dgs.geometry import resample_poses
    cv2.setNumThreads(4);source_path=a.scene/'nexus_scene.json';source_hash=hashlib.sha256(source_path.read_bytes()).hexdigest();source=json.loads(source_path.read_text());scene=copy.deepcopy(source);registered=json.loads(a.registration.read_text());assert registered['status']=='complete'
    byframe={r['frame']:r for r in registered['rows']};assert set(byframe)==set(range(len(scene['frames'])));np.testing.assert_array_equal(byframe[20]['scene_correction_matrix'],np.eye(4))
    S=np.linalg.inv(source['frames'][0]['ego_to_world']);worldS=np.linalg.inv(S);times=np.array([f['timestamp_us'] for f in source['frames']],np.int64);changes=np.array([byframe[f]['scene_correction_matrix'] for f in range(len(times))]);assert np.all(np.diff(times)>0)
    def correction(timestamp):
        # Hold only the correction (never the measured body motion) over the
        # <50 ms image/point-cloud acquisition offset at the two boundaries.
        clipped=int(np.clip(timestamp,times[0],times[-1]));assert abs(clipped-timestamp)<=50000
        value,valid=resample_poses(times,changes[:,None],np.ones((len(times),1),bool),[clipped]);assert valid[0,0]
        return worldS@value[0,0]@S,abs(clipped-timestamp)
    for name in ['images','lidar','intrinsics','extrinsics']:(out/name).symlink_to((a.scene/name).resolve(),target_is_directory=True)
    for name in ['ego_pose','camera_ego_pose','instances','vehicle_interior_masks']:(out/name).mkdir()
    (out/'scene_semantics').symlink_to(a.semantics.resolve(),target_is_directory=True)
    endpoint_holds=[]
    for f,row in enumerate(scene['frames']):
        Tw=worldS@changes[f]@S;row['source_ego_to_world']=row['ego_to_world'];row['ego_to_world']=(Tw@row['ego_to_world']).tolist();np.savetxt(out/'ego_pose'/f'{f:03d}.txt',row['ego_to_world'])
        for camera in row['images']:
            Cw,held=correction(camera['timestamp_us']);camera['source_ego_to_world']=camera['ego_to_world'];camera['ego_to_world']=(Cw@camera['ego_to_world']).tolist();np.savetxt(out/'camera_ego_pose'/f'{f:03d}_{camera["camera_id"]}.txt',camera['ego_to_world'])
            if held:endpoint_holds.append({'frame':f,'camera_id':camera['camera_id'],'held_correction_us':held})
    info=json.loads((a.scene/'instances/instances_info.json').read_text())
    for item in info.values():
        ann=item['frame_annotations'];ann['source_obj_to_world']=copy.deepcopy(ann['obj_to_world'])
        for j,f in enumerate(ann['frame_idx']):ann['obj_to_world'][j]=(worldS@changes[f]@S@ann['obj_to_world'][j]).tolist()
    (out/'instances/instances_info.json').write_text(json.dumps(info,indent=2)+'\n');(out/'instances/frame_instances.json').write_text((a.scene/'instances/frame_instances.json').read_text())
    # Conservative per-exposure traffic/shadow exclusion uses interpolated
    # refined annotation poses, covering all classes rather than modeled IDs.
    items=list(info.values());pose=np.broadcast_to(np.eye(4),(len(times),len(items),4,4)).copy();valid=np.zeros(pose.shape[:2],bool);sizes=[]
    for j,item in enumerate(items):
        ann=item['frame_annotations'];sizes.append(np.median(ann['box_size'],axis=0)+.4)
        for f,T in zip(ann['frame_idx'],ann['obj_to_world']):pose[f,j]=T;valid[f,j]=True
    sizes=np.array(sizes);signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]);mask_rows=[]
    for f,row in enumerate(scene['frames']):
        for camera in row['images']:
            c=camera['camera_id'];intr=np.loadtxt(out/'intrinsics'/f'{c}.txt');K=np.array([[intr[0]/2,0,intr[2]/2],[0,intr[1]/2,intr[3]/2],[0,0,1.]]);C=np.array(camera['ego_to_world'])@np.loadtxt(out/'extrinsics'/f'{c}.txt');sample,ok=resample_poses(times,pose,valid,[int(np.clip(camera['timestamp_us'],times[0],times[-1]))]);mask=np.zeros((540,960),np.uint8)
            for j in np.flatnonzero(ok[0]):
                B=sample[0,j];xyz=((signs*sizes[j]/2)@B[:3,:3].T+B[:3,3]-C[:3,3])@C[:3,:3];points=[q for q in xyz if q[2]>.1]
                for i in range(8):
                    for k in range(i+1,8):
                        if np.count_nonzero(signs[i]!=signs[k])!=1:continue
                        u,v=xyz[i],xyz[k]
                        if (u[2]>.1)!=(v[2]>.1):points.append(u+(v-u)*((.10001-u[2])/(v[2]-u[2])))
                if not points:continue
                q=np.array(points);uv=q[:,:2]/q[:,2,None]@K[:2,:2].T+K[:2,2];lo=uv.min(0);hi=uv.max(0)
                if hi[0]<0 or hi[1]<0 or lo[0]>=960 or lo[1]>=540:continue
                w=min(960.,hi[0]-lo[0]);h=min(540.,hi[1]-lo[1]);x0=max(0,int(lo[0]-max(4,.08*w)));x1=min(960,int(hi[0]+max(4,.08*w)+1));y0=max(0,int(lo[1]-4));y1=min(540,int(hi[1]+max(6,.3*h)+1));mask[y0:y1,x0:x1]=255
            cv2.imwrite(str(out/'vehicle_interior_masks'/f'{f:03d}_{c}.png'),mask);mask_rows.append({'frame':f,'camera_id':c,'excluded_fraction':float((mask>0).mean())})
        if f%20==0:print('REGISTERED_SCENE '+str(f),flush=True)
    (out/'vehicle_interior_masks/manifest.json').write_text(json.dumps({'status':'complete','is_semantic_ground_truth':False,'use':'static-training exclusion only, conservative per-exposure refined annotation boxes and shadow halo','images':len(mask_rows)},indent=2)+'\n')
    scene['world_from_scene']=scene['frames'][0]['ego_to_world'];scene['identity']['pose_refinement']='static_lidar_icp_anchor_original_frame_20';scene['pose_refinement']={'source_scene':str(a.scene.resolve()),'source_scene_sha256':source_hash,'registration':str(a.registration.resolve()),'registration_sha256':hashlib.sha256(a.registration.read_bytes()).hexdigest(),'common_world_correction_applied_to':['lidar ego poses','camera exposure ego poses','object world annotations'],'anchor_frame':20,'camera_correction_time_interpolation':'SE3 rotation SLERP and translation interpolation at exposure timestamps','boundary_correction_holds':endpoint_holds,'source_images_and_pointclouds_unchanged':True,'nexus_predictions_changed':False};(out/'nexus_scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()==source_hash
    report={'status':'complete','scene':str(out.resolve()),'source_unmodified':True,'frames':len(times),'image_masks':len(mask_rows),'boundary_correction_holds':endpoint_holds,'masks':mask_rows,'scope':'separate aligned reconstruction input; no 3DGS or Nexus training performed'};(out/'registration_preparation.json').write_text(json.dumps(report,indent=2)+'\n');print('REGISTERED_SCENE_COMPLETE '+str(out.resolve()),flush=True)


if __name__=='__main__':main()
