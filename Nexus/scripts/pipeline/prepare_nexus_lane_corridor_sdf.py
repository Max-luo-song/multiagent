"""Rasterize exact SDK lane corridors for body-clearance preferences."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'artifacts/nexus_multicar_upgrade_check_20261027'
OUT=ROOT/'artifacts/nexus_kinematic_20261027'


def main():
    import numpy as np
    from PIL import Image,ImageDraw
    from scipy.ndimage import distance_transform_edt
    from shapely import wkb
    p=json.loads((OLD/'protocol.json').read_text());W=np.array(p['world_from_nexus'])
    resolution=.1;origin=np.array([-30.,-20.]);width=2001;height=501
    grids=[]
    for slot in [0,12]:
        corridor=wkb.loads((OUT/'lanes'/f'{slot}.wkb').read_bytes())
        mask=Image.new('L',(width,height),0);draw=ImageDraw.Draw(mask)
        for polygon in (corridor.geoms if hasattr(corridor,'geoms') else [corridor]):
            points=(np.array(polygon.exterior.coords)-W[:2,3])@W[:2,:2]
            points=(points-origin)/resolution
            draw.polygon([tuple(v) for v in points],fill=1)
            for ring in polygon.interiors:
                points=((np.array(ring.coords)-W[:2,3])@W[:2,:2]-origin)/resolution
                draw.polygon([tuple(v) for v in points],fill=0)
        inside=np.asarray(mask).astype(bool)
        distance_in=distance_transform_edt(inside)*resolution
        distance_out=distance_transform_edt(~inside)*resolution
        sdf=np.where(inside,distance_in-resolution/2,-distance_out+resolution/2)
        grids.append(sdf.astype(np.float32))
    target=OUT/'lane_corridors_sdf.npz'
    if target.exists():raise FileExistsError(target)
    np.savez_compressed(target,sdf=np.stack(grids),slots=np.array([0,12]),origin=origin,resolution=resolution)
    (OUT/'lane_corridors_sdf.json').write_text(json.dumps({'slots':[0,12],'resolution_m':resolution,
       'source':'exact SDK initial lane and connected continuation corridors; no future actor states',
       'purpose':'soft whole-body lane preference; prescribed cut-in can relax ego preference',
       'extent_local_xy_m':[[-30,170],[-20,30]]},indent=2)+'\n')
    print('LANE_CORRIDOR_SDF_PREPARED',target,flush=True)


if __name__=='__main__':main()
