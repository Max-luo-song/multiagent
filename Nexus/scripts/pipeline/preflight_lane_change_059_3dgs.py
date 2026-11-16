"""Check local 059 sensors and export existing predictions; no download/training."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
 import numpy as np
 import torch
 from nexus_3dgs import nuplan
 from nexus_3dgs.bundle import export_arrays,save_bundle
 source=ROOT/'artifacts/nexus_lane_response_v2_20261027'
 protocol=json.loads((source/'protocol.json').read_text())
 scene=protocol['cases'][0]['scenario'];roles=protocol['cases'][0]['roles']
 out=ROOT/'artifacts/nexus_3dgs/lane_change_059_render_handoff_20261028'
 out.mkdir(exist_ok=False)
 roots=[Path('/inspire/hdd/global_public/public_datas/nuPlan/sensor_blobs_full'),Path('/inspire/hdd/global_public/public_datas/nuPlan/sensor_blobs_mini'),ROOT.parent/'data/nuplan']
 locator=nuplan.SensorLocator(roots);requests=[]
 with nuplan.connect(scene['database']) as db:
  anchor=nuplan.anchor_frame(db,scene['token'])
  frames=nuplan.nearest_frames(db,anchor['lidar_timestamp']+np.arange(-2000000,8000001,100000))
  calibration=nuplan.cameras(db)
  for frame in frames:
   requests.append({'kind':'lidar','filename':frame['filename']})
   for channel,camera in calibration.items():
    image=nuplan.nearest_image(db,camera['token'],frame['lidar_timestamp'])
    requests.append({'kind':channel,'filename':image['filename_jpg']})
 for request in requests:
  path=locator.resolve(request['filename']);request['resolved']=str(path) if path else None
 missing=[x for x in requests if x['resolved'] is None]
 report={'status':'missing_sensor_files' if missing else 'sensor_files_available','scenario_token':scene['token'],'log_name':scene['log_name'],'database':scene['database'],'frames_at_10fps':len(frames),'sensor_roots_searched':list(map(str,roots)),'requested_files':len(requests),'available_files':len(requests)-len(missing),'missing_lidar':sum(x['kind']=='lidar' for x in missing),'missing_images':sum(x['kind']!='lidar' for x in missing),'camera_calibrations_available':list(calibration),'trajectory_bundles':[],'photorealistic_video_generated':False,'original_predictions_unchanged':True,'missing_files':missing}
 def save():
  (out/'sensor_preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 save();(out/'missing_files.txt').write_text('\n'.join(x['filename'] for x in missing)+'\n')
 for variant in ('normal','event'):
  pred=source/'pilot_seed0/case_059'/variant/'prediction.pt'
  checksum=hashlib.sha256(pred.read_bytes()).hexdigest()
  payload=torch.load(pred,map_location='cpu',weights_only=False)
  arrays,manifest=export_arrays(payload,{'database':scene['database'],'scenario_token':scene['token'],'prediction':str(pred)})
  manifest.update(prediction_sha256=checksum,experiment_source=str(source),variant=variant,roles=roles,trajectory_scope='saved Nexus plus lane-choice guidance; no render-stage trajectory edits',full_scene_accepted=False,approved_for_downstream=False)
  save_bundle(out/variant,arrays,manifest)
  assert hashlib.sha256(pred.read_bytes()).hexdigest()==checksum
  report['trajectory_bundles'].append({'variant':variant,'path':str(out/variant),'source_prediction_sha256':checksum,'agents':len(arrays['track_tokens']),'role_bindings':{role:next((x for x in manifest['agent_bindings'] if x['nexus_slot']==slot),None) for role,slot in roles.items() if role!='E'}})
  save()
 print(json.dumps({k:v for k,v in report.items() if k not in ('missing_files','trajectory_bundles')},ensure_ascii=False,indent=2),flush=True)
 print('exported',[(x['variant'],x['agents']) for x in report['trajectory_bundles']],flush=True)

if __name__=='__main__':main()
