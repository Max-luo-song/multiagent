"""Screen history geometry for three-car demonstrations before generating responses."""
import json
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    from nexus_3dgs.bundle import decode
    source=ROOT/'artifacts/nuplan_evaluation/independent256_v1'
    cached=ROOT/'artifacts/nexus_3dgs/interaction_v9_20261025/new_scene_pilot_scan.json'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--all-scenes',action='store_true')
    args=p.parse_args()
    available_rows=json.loads(cached.read_text())['candidates']
    available={r['index'] for r in available_rows}
    rows=json.loads((source/'frozen_plan.json').read_text())['scenarios'] if args.all_scenes else available_rows
    output=ROOT/'artifacts/nexus_3dgs'/('multicar_selection_all_20261026' if args.all_scenes else 'multicar_selection_20261026');output.mkdir(exist_ok=False)
    results=[];rejected=[]
    for row in rows:
        path=source/'raw_features'/f'{row["index"]:03d}_{row["token"]}.pt'
        raw=torch.load(path,map_location='cpu',weights_only=False)
        s=decode(raw['tensor'].numpy()[...,:8]);v=raw['validity'].numpy().astype(bool)
        yaw=np.arctan2(s[0,4,3],s[0,4,2]);R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        xy=(s[:,4,:2]-s[0,4,:2])@R
        relative=np.arctan2(np.sin(np.arctan2(s[:,4,3],s[:,4,2])-yaw),np.cos(np.arctan2(s[:,4,3],s[:,4,2])-yaw))
        velocity=(s[:,4,:2]-s[:,3,:2])@R/.5
        available=v[:,:5].all(axis=1)&v[:,5:13].all(axis=1)
        same=(np.abs(relative)<np.deg2rad(30))&(velocity[:,0]>1)&available
        same[0]=False
        actors=[]
        for slot in np.flatnonzero(v[:,4]):
            actors.append({'slot':int(slot),'xy_m':xy[slot].tolist(),'relative_yaw_deg':float(np.rad2deg(relative[slot])),
                           'history_velocity_mps':velocity[slot].tolist(),'future_valid_samples':int(v[slot,5:].sum()),
                           'same_direction_moving_and_present_4s':bool(same[slot])})
        combos=[]
        if velocity[0,0]>=2:
            for actor in np.flatnonzero(same):
                x,y=xy[actor]
                front=5<x<30 and abs(y)<2.2
                adjacent=1<x<20 and 2.3<abs(y)<5.8
                if not (front or adjacent):continue
                for third in np.flatnonzero(same):
                    if third==actor:continue
                    tx,ty=xy[third]
                    rear=-22<tx<-4 and abs(ty)<2.2
                    beside=-15<tx<12 and 2.3<abs(ty)<5.8
                    # For cut-in, prefer a third car on the opposite side of ego.
                    if adjacent:beside=beside and ty*y<0
                    if not (rear or beside):continue
                    score=50-(abs(x-12)+abs(tx+5))*.5+(10 if beside else 0)+(5 if adjacent else 0)
                    combos.append({'event_slot':int(actor),'third_slot':int(third),
                                   'event_kind':'lateral_shift' if adjacent else 'brake',
                                   'third_role':'adjacent' if beside else 'following','score':float(score)})
        item={k:row[k] for k in ('index','token','database','log_name','type')}
        item.update(ego_speed_mps=float(np.linalg.norm(velocity[0])),actors=actors,combinations=sorted(combos,key=lambda c:-c['score']),
                    sensor_endpoint_preflight_passed=row['index'] in available,
                    sensor_status='cached endpoint camera0/LiDAR available' if row['index'] in available else 'endpoint RGB/LiDAR missing in prior scan; data availability must be resolved before video')
        if combos:results.append(item)
        else:rejected.append({'index':row['index'],'reason':'no moving same-direction three-car geometry meeting declared thresholds'})
    results.sort(key=lambda item:(-item['combinations'][0]['score'],item['index']))
    report={'status':'screened','candidate_source':str(source/'frozen_plan.json') if args.all_scenes else str(cached),'scanned_candidates':len(rows),
            'selection_uses_generated_responses':False,'history_values_only_for_geometry_and_speed':True,
            'future_presence_requirement':'all selected actors valid through first 4 future seconds',
            'rules':'ego >=2m/s; other selected actors heading within30deg, longitudinal speed>1m/s; front or adjacent trigger and adjacent/opposite-side or following third',
            'ranked_candidates':results,'rejected':rejected,
            'limitations':['Dataset actor presence mask is retained. Endpoint availability is not full reconstruction readiness.',
                           'Geometric lane relation is only an approximation; map, sensor coverage and visual review are still required.',
                           'Suitable initial geometry does not guarantee Nexus will generate a valid interaction.']}
    (output/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'scanned':len(rows),'qualifying':len(results),'top':[{'index':r['index'],'ego_speed_mps':r['ego_speed_mps'],'combination':r['combinations'][0]} for r in results[:5]]},indent=2))


if __name__=='__main__':main()
