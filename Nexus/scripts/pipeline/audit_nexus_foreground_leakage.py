"""Measure vehicle opacity in observed local non-vehicle rings (training views)."""
import argparse
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[]
    for directory in (a.old,a.new):
        run=json.loads((directory/'run.json').read_text())
        if run['status']!='complete' or not run['reference'] or run['camera_mode']!='logged' or run['ground_registration'] is not None:
            raise ValueError('Leakage comparison requires observed reference cameras without ground substitution')
    files=sorted((a.old/'diagnostic_layers').glob('*_0_raw.npz'))
    if not files:raise ValueError('No paired diagnostic layers')
    for file in files:
        frame=int(file.stem.split('_')[0]);record={'frame':frame}
        for label,directory in (('old',a.old),('new',a.new)):
            alpha=np.load(directory/'diagnostic_layers'/file.name)['Dynamic_opacity'].squeeze()
            ext=np.array(Image.open(a.scene/'vehicle_exterior_masks'/f'{frame:03d}_0.png'))>0
            pos=np.array(Image.open(a.scene/'vehicle_interior_masks'/f'{frame:03d}_0.png'))>0
            if ext.shape!=alpha.shape:raise ValueError('Diagnostic and observed masks have different resolution')
            # The near central bus is the dominant connected observed vehicle
            # in this F0 clip; isolate its local ring from the side-car rings.
            count,labels,stats,_=cv2.connectedComponentsWithStats(pos.astype(np.uint8),8)
            if count<2:raise ValueError('No observed vehicle component')
            component=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));x,y,w,h,_=stats[component]
            if not (300<x+w/2<700 and stats[component,cv2.CC_STAT_AREA]>1000):
                raise ValueError('Dominant component does not match the expected near bus')
            window=np.zeros_like(ext);window[max(0,y-40):min(len(ext),y+h+40),max(0,x-40):min(ext.shape[1],x+w+40)]=True
            bus_ext=ext&window;bus_pos=labels==component
            roof_ext=bus_ext&(np.indices(ext.shape)[0] < y+max(8,int(h*.15)))
            record[label]={'exterior_mean_opacity':float(alpha[ext].mean()),
                           'exterior_fraction_alpha_gt_0_1':float((alpha[ext]>.1).mean()),
                           'interior_mean_opacity':float(alpha[pos].mean()),
                           'bus_exterior_mean_opacity':float(alpha[bus_ext].mean()),
                           'bus_roof_exterior_mean_opacity':float(alpha[roof_ext].mean()),
                           'bus_interior_mean_opacity':float(alpha[bus_pos].mean())}
        rows.append(record)
    summary={label:{key:float(np.mean([r[label][key] for r in rows])) for key in rows[0][label]} for label in ('old','new')}
    report={'status':'complete','frames':rows,'summary':summary,
            'scope':'observed training-camera local-ring opacity; image/box-derived masks, not semantic ground truth or generated-view quality',
            'uncertain_edge_px':5}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
