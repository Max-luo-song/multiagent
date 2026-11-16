"""Audit an existing scene bundle against another reconstruction checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from nexus_3dgs.scene_traffic import SceneTraffic
from nexus_3dgs.scene_projection import pair_separations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--ground-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report={'status':'passed','checkpoint':str(a.checkpoint.resolve()),'variants':{},
                           'scope':'physical 2D vehicle boxes, 100 Hz, valid trajectories only'}
    for name in ('bus_stop','bus_follow_logged'):
        traffic=SceneTraffic(a.checkpoint,a.experiment/name,a.ground_model)
        bundle=traffic.controller.bundle;meta=traffic.controller.manifest
        t0=int(bundle['timestamps_us'][meta['current_index']]);t1=int(bundle['timestamps_us'][-1])
        timestamps=np.arange(t0,t1+1,10000,dtype=np.int64)
        boxes,valid=traffic.sample(timestamps);sep,pairs=pair_separations(boxes,valid,traffic.sizes)
        overlap=np.argwhere((sep<=0)&pairs)
        item={'vehicle_participants_including_ego':len(traffic.tokens),'samples':len(timestamps),
              'overlapping_pair_samples':len(overlap),'minimum_sat_separation_m':float(sep[pairs].min()),
              'ego_minimum_sat_separation_m':float(sep[:,0,1:][valid[:,0,None]&valid[:,1:]].min()),
              'bundle_sha256':meta['bundle_sha256'],'overlap_time_actor_indices':overlap.tolist()}
        report['variants'][name]=item
        if len(overlap):report['status']='failed'
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if report['status']!='passed':raise ValueError('New reconstruction changes physical registration enough to cause overlap')


if __name__=='__main__':main()
