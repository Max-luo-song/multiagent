"""Freeze three additional sensor-complete clips before reconstruction scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from nexus_3dgs import nuplan


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=ROOT/'artifacts/nuplan_evaluation/independent256_v1';plan=json.loads((source/'frozen_plan.json').read_text())
    cameras=[0,1,2,4];roots=[Path('/inspire/hdd/global_public/public_datas/nuPlan/sensor_blobs_full'),Path('/inspire/hdd/global_public/public_datas/nuPlan/sensor_blobs_mini')]
    locator=nuplan.SensorLocator(roots);selected=[];used_logs=set()
    cache_path=a.output.with_name(a.output.name+'_scan.json')
    cached=json.loads(cache_path.read_text()) if cache_path.exists() else {'candidates':[],'rejected':[],'done':[]}
    candidates=cached['candidates'];rejected=cached['rejected'];done=set(cached['done'])
    for row in plan['scenarios']:
        if row['index']==96 or row['log_name']=='2021.10.05.04.38.41_veh-50_00014_00429':continue
        if row['index'] in done:continue
        try:
            with nuplan.connect(row['database']) as db:
                anchor=nuplan.anchor_frame(db,row['token']);frames=nuplan.nearest_frames(db,[anchor['lidar_timestamp'],anchor['lidar_timestamp']+8000000])
                motion=float(np.linalg.norm(np.array([frames[-1]['x']-frames[0]['x'],frames[-1]['y']-frames[0]['y']])))
                dyaw=float(np.rad2deg(abs(np.arctan2(np.sin(nuplan.ego_yaw(frames[-1])-nuplan.ego_yaw(frames[0])),np.cos(nuplan.ego_yaw(frames[-1])-nuplan.ego_yaw(frames[0]))))))
                calibration=nuplan.cameras(db,[nuplan.CAMERAS[i] for i in cameras]);front=calibration['CAM_F0']
                probe=[locator.resolve(f['filename']) for f in frames]
                probe += [locator.resolve(nuplan.nearest_image(db,front['token'],f['lidar_timestamp'])['filename_jpg']) for f in frames]
                if not all(probe):raise FileNotFoundError('missing endpoint RGB/LiDAR')
                objects=nuplan.boxes(db,anchor['lidar_token']);E=nuplan.ego_pose(anchor)
                forward=[o for o in objects if o['category_name']=='vehicle' and 5<((np.array([o['x'],o['y']])-E[:2,3])@E[:2,0])<50 and abs((np.array([o['x'],o['y']])-E[:2,3])@E[:2,1])<4]
            candidates.append({**row,'ego_8s_displacement_m':motion,'ego_8s_heading_change_deg':dyaw,'forward_vehicles_at_anchor':len(forward)})
        except (ValueError,FileNotFoundError) as exc:rejected.append({'index':row['index'],'reason':str(exc)})
        done.add(row['index']);cached['done']=sorted(done)
        cache_path.write_text(json.dumps(cached,indent=2)+'\n')
        if len(done)%8==0:print('PILOT_SCAN',len(done),'sensor_available',len(candidates),flush=True)
    rules=[('turning',lambda s:s['ego_8s_heading_change_deg']>=15 and s['ego_8s_displacement_m']>3),
           ('following',lambda s:s['ego_8s_heading_change_deg']<10 and s['forward_vehicles_at_anchor']>0 and s['ego_8s_displacement_m']>3),
           ('intersection',lambda s:('intersection' in s['type'] or 'traffic_light' in s['type']) and s['ego_8s_displacement_m']>3)]
    for category,predicate in rules:
        pool=sorted([s for s in candidates if predicate(s)],key=lambda s:(s['log_name'],s['index']))
        for row in pool:
            if row['log_name'] in used_logs:continue
            try:
                with nuplan.connect(row['database']) as db:
                    targets=row['timestamp_us']+np.arange(-2000000,8000001,100000)
                    frames=nuplan.nearest_frames(db,targets);calibration=nuplan.cameras(db,[nuplan.CAMERAS[i] for i in cameras]);files=[];exposure_offsets=[]
                    for f in frames:
                        path=locator.resolve(f['filename'])
                        if path is None:raise FileNotFoundError(f['filename'])
                        files.append(str(path))
                        for c in calibration.values():
                            image=nuplan.nearest_image(db,c['token'],f['lidar_timestamp']);path=locator.resolve(image['filename_jpg'])
                            if path is None:raise FileNotFoundError(image['filename_jpg'])
                            files.append(str(path));exposure_offsets.append(image['image_timestamp']-f['lidar_timestamp'])
                selected.append({**row,'pilot_category':category,'frames':len(frames),'cameras':cameras,'sensor_file_count':len(files),
                                 'sensor_paths_sha256':hashlib.sha256('\n'.join(files).encode()).hexdigest(),
                                 'maximum_camera_lidar_offset_ms':max(abs(v) for v in exposure_offsets)/1000,
                                 'all_sensor_files_exist':True,'sensor_files':files})
                used_logs.add(row['log_name']);print('PILOT_SELECTED',category,row['index'],row['type'],flush=True);break
            except (ValueError,FileNotFoundError) as exc:rejected.append({'index':row['index'],'reason':'full preflight: '+str(exc)})
        else:raise RuntimeError('No complete, distinct-log clip for '+category)
    result={'status':'frozen','selection_before_reconstruction':True,'scope':'engineering pilot using existing independent256 scenes; not an unseen-trip generalization benchmark',
            'source_plan_sha256':hashlib.sha256((source/'frozen_plan.json').read_bytes()).hexdigest(),'cameras':cameras,
            'selection_rule':'one turning, one following, one intersection; distinct logs; sorted log/index; complete 101-frame RGB/LiDAR preflight',
            'scenarios':selected,'rejected_candidates':rejected,'available_endpoint_candidates':len(candidates)}
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'frozen_scenes.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PILOT_SCENES_FROZEN',a.output,flush=True)


if __name__=='__main__':main()
