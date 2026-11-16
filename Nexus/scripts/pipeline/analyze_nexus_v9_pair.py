"""Audit unfiltered paired responses; optionally summarize actual render poses."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from shapely.geometry import Polygon
from nexus_3dgs.bundle import load_bundle
from nexus_3dgs.clearance import rear_axle_to_box
from nexus_3dgs.scene_projection import pair_separations
from nexus_3dgs import nuplan


def polygon(box, size):
    c, s = np.cos(box[2]), np.sin(box[2])
    return Polygon((np.array([[1,1],[1,-1],[-1,-1],[-1,1]]) * np.asarray(size)/2) @ np.array([[c,s],[-s,c]]) + box[:2])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--renders', type=Path)
    a = p.parse_args()
    run = json.loads((a.experiment/'run.json').read_text())
    assert run['status'] == 'complete'
    variants = {name: load_bundle(a.experiment/name) for name in ('bus_stop','bus_follow_logged')}
    left, lm = variants['bus_stop']; right, rm = variants['bus_follow_logged']
    for key in ('timestamps_us','track_tokens','nexus_slots','valid','reference_poses_world'):
        np.testing.assert_array_equal(left[key], right[key])
    assert len({v['initial_noise_sha256'] for v in run['variants'].values()}) == 1
    ci = lm['current_index']; bus = list(left['nexus_slots']).index(run['conditioned_slot'])
    with nuplan.connect(lm['database']) as db:
        anchor = nuplan.anchor_frame(db, lm['scenario_token'])
        annotations = nuplan.boxes(db, anchor['lidar_token'])
        sizes = []
        for i, token in enumerate(left['track_tokens']):
            if i == 0:
                sizes.append([5.176,2.297]); continue
            obj = next((o for o in annotations if o['track_token'] == token), None)
            sizes.append([obj['length'],obj['width']] if obj else left['predicted_size_lw'][np.flatnonzero(left['valid'][:,i])[0],i])
    sizes = np.array(sizes)
    output = {'status':'passed','raw_predictions_only':True,'trajectory_filtering_applied':False,
              'same_history_map_validity_noise_weights':True, 'bus_slot':run['conditioned_slot'],
              'raw_native_geometry':run['geometry_audit'],'variants':{},'responses':run['actors'],
              'notes':['Bus futures are supplied in both variants; all other future values were erased.',
                       'Logged future validity is retained. This is a single-scene conditional-dependence diagnostic.',
                       'Native predicted-size boxes, fixed physical boxes and actual rendered asset boxes are distinct diagnostics.']}
    for name, (arr, manifest) in variants.items():
        poses = arr['predicted_poses_world']; xy = poses[:,:,:2,3]
        yaw = np.arctan2(poses[:,:,1,0],poses[:,:,0,0])
        boxes = np.concatenate((xy,yaw[...,None]),axis=-1)
        boxes[:,0] = rear_axle_to_box(boxes[:,0],1.461)
        future = slice(ci+1,None)
        sep,pairs = pair_separations(boxes[future],arr['valid'][future],sizes)
        distances = [polygon(b[0],sizes[0]).distance(polygon(b[bus],sizes[bus])) for b in boxes[future]]
        ego_speed = np.linalg.norm(np.diff(xy[ci:,0],axis=0),axis=-1)/.5
        bus_speed = np.linalg.norm(np.diff(xy[ci:,bus],axis=0),axis=-1)/.5
        output['variants'][name] = {
            'bundle_sha256':manifest['bundle_sha256'],'all_valid_predicted_values_finite':bool(np.isfinite(poses[arr['valid']]).all()),
            'ego_future_path_length_m':float(ego_speed.sum()*.5),'ego_speed_mps':ego_speed.tolist(),
            'bus_speed_mps':bus_speed.tolist(),'minimum_ego_bus_polygon_gap_m':float(min(distances)),
            'minimum_ego_bus_signed_sat_separation_m':float(sep[:,0,bus].min()),
            'ego_bus_overlapping_future_samples':int((sep[:,0,bus]<=0).sum()),
            'all_vehicle_overlapping_pair_samples':int(((sep<=0)&pairs).sum()),
            'overlap_future_sample_actor_pairs':np.argwhere((sep<=0)&pairs).tolist(),
            'geometry_scope':'16 future half-second samples; fixed logged vehicle dimensions, ego 5.176x2.297m with rear-axle offset 1.461m'}
    if a.renders:
        output['render_geometry']={}
        for name in variants:
            render=json.loads((a.renders/name/'run.json').read_text())
            assert render['bundle_sha256']==output['variants'][name]['bundle_sha256']
            frames=render['frames']; gaps=[]
            for frame in frames:
                actors=frame['scene_metadata']['actors']; e=actors[0]; b=next(x for x in actors if x['nexus_slot']==run['conditioned_slot'])
                box=lambda x: np.array([*x['center_world'][:2],x['yaw_world_rad']])
                gaps.append(polygon(box(e),e['size_lwh'][:2]).distance(polygon(box(b),b['size_lwh'][:2])))
            events=[{'frame':f['frame'],'time_s':(f['timestamp_us']-int(left['timestamps_us'][ci]))/1e6,
                     'actor_pairs':f['scene_metadata']['overlapping_actor_pairs']} for f in frames if f['scene_metadata']['overlapping_actor_pairs']]
            bus_actor=next(i for i,x in enumerate(frames[0]['scene_metadata']['actors']) if x['nexus_slot']==run['conditioned_slot'])
            ego_bus_events=[event for event in events if [0,bus_actor] in event['actor_pairs']]
            output['render_geometry'][name]={'frames':len(frames),'minimum_ego_bus_polygon_gap_m':float(min(gaps)),
                'minimum_ego_bus_signed_sat_separation_m':min(f['scene_metadata']['ego_bus_sat_separation_m'] for f in frames),
                'frames_with_any_overlap':len(events),'overlapping_pair_frame_samples':sum(len(e['actor_pairs']) for e in events),
                'ego_bus_overlapping_frames':len(ego_bus_events),'first_ego_bus_overlap_s':ego_bus_events[0]['time_s'] if ego_bus_events else None,
                'overlap_events':events,'scope':'all assets including unpredicted recorded vehicles; actual camera exposure poses; 2D physical boxes; no continuous-time guarantee'}
    (a.experiment.parent/'paired_analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k in ('variants','render_geometry')},indent=2),flush=True)


if __name__=='__main__':main()
