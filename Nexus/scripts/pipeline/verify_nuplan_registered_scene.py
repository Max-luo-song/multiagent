"""Check geometric invariants on the actual independently registered dataset."""
import argparse, hashlib, json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);a=p.parse_args()
    import numpy as np
    meta=json.loads((a.scene/'nexus_scene.json').read_text());source=Path(meta['pose_refinement']['source_scene'])
    old=json.loads((source/'nexus_scene.json').read_text());before=json.loads((source/'instances/instances_info.json').read_text());after=json.loads((a.scene/'instances/instances_info.json').read_text())
    checksum=hashlib.sha256((source/'nexus_scene.json').read_bytes()).hexdigest();assert checksum==meta['pose_refinement']['source_scene_sha256']
    for name in ['images','lidar','intrinsics','extrinsics']:assert (a.scene/name).resolve()==(source/name).resolve()
    errors=[];pairs=0
    for key in before:
        raw=before[key]['frame_annotations'];new=after[key]['frame_annotations'];assert raw['frame_idx']==new['frame_idx']
        for f,B0,B1 in zip(raw['frame_idx'],raw['obj_to_world'],new['obj_to_world']):
            # A common world correction must preserve the relative pose of
            # every measured body to its point cloud at that same timestamp.
            A=np.linalg.inv(np.array(old['frames'][f]['ego_to_world']))@B0
            B=np.linalg.inv(np.array(meta['frames'][f]['ego_to_world']))@B1
            errors.append(float(np.abs(A-B).max()));pairs+=1
    maximum=max(errors);assert maximum<1e-8,maximum
    static=json.loads((a.scene/'vehicle_interior_masks/manifest.json').read_text());assert 'static-training exclusion only' in static['use']
    report={'status':'passed','frames':len(meta['frames']),'body_pointcloud_relative_pose_pairs_checked':pairs,'max_relative_pose_matrix_error':maximum,
            'source_metadata_sha256_verified':checksum,'raw_images_pointclouds_intrinsics_extrinsics_unchanged':True,
            'camera_and_lidar_exposure_times_preserved':all(x['timestamp_us']==y['timestamp_us'] and [i['timestamp_us'] for i in x['images']]==[i['timestamp_us'] for i in y['images']] for x,y in zip(old['frames'],meta['frames']))}
    assert report['camera_and_lidar_exposure_times_preserved']
    (a.scene/'geometry_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
