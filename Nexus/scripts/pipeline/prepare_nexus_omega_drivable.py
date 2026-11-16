"""Freeze the SDK drivable-area union for an independent physical-road audit."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/nuplan-devkit')]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sdf-from',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.sdf_from:
        import hashlib
        import numpy as np
        from shapely import wkb,affinity,vectorized
        from scipy.ndimage import distance_transform_edt
        source=a.sdf_from/'drivable.wkb'
        info=json.loads((a.sdf_from/'manifest.json').read_text())
        W=np.asarray(info['world_from_nexus']);R=W[:2,:2];t=W[:2,3]
        # Convert the already frozen world polygon to Nexus-local XY.
        shift=-R.T@t
        region=affinity.affine_transform(wkb.loads(source.read_bytes()),
            [R[0,0],R[1,0],R[0,1],R[1,1],shift[0],shift[1]])
        origin=np.array([-220.,-220.]);resolution=.25
        xx,yy=np.meshgrid(np.arange(1761)*resolution+origin[0],np.arange(1761)*resolution+origin[1])
        inside=vectorized.contains(region,xx,yy)
        sdf=(distance_transform_edt(inside)-distance_transform_edt(~inside))*resolution
        np.savez_compressed(a.output/'drivable_sdf.npz',sdf=sdf.astype('float32'),
            origin=origin.astype('float32'),resolution=np.float32(resolution))
        (a.output/'manifest.json').write_text(json.dumps({'status':'frozen','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'units':'metres; positive inside SDK drivable union','resolution_m':resolution,
            'coordinates':'Nexus local','grid_shape':list(sdf.shape),'purpose':'differentiable static-map constraint; independent polygon audit retained'},indent=2)+'\n')
        print('DRIVABLE_SDF_FROZEN',sdf.shape,flush=True)
        return
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from shapely.geometry import box
    from shapely.ops import unary_union
    scene=json.loads((ROOT/'artifacts/nexus_3dgs/real_vs_nexus/inputs/goal/manifest.json').read_text())
    W=scene['world_from_nexus'];x,y=W[0][3],W[1][3]
    maps=ROOT.parent/'data/nuplan/maps'
    assert not maps.is_symlink()
    api=get_maps_api(str(maps),'nuplan-maps-v1.0','sg-one-north')
    frame=api._load_vector_map_layer('drivable_area')
    patch=box(x-250,y-250,x+250,y+250)
    selected=frame[frame.geometry.intersects(patch)]
    geometry=unary_union([g.intersection(patch) for g in selected.geometry])
    if not geometry.is_valid:geometry=geometry.buffer(0)
    (a.output/'drivable.wkb').write_bytes(geometry.wkb)
    (a.output/'manifest.json').write_text(json.dumps({'status':'frozen','map_name':'sg-one-north',
        'map_version':'nuplan-maps-v1.0','private_map_root':str(maps),'polygons':len(selected),
        'bounds_world':[x-250,y-250,x+250,y+250],
        'definition':'nuPlan SDK union: road segments, intersections, generic drivable areas, car parks',
        'purpose':'independent geometry audit; not a lane-centerline proxy','world_from_nexus':W},indent=2)+'\n')
    print('DRIVABLE_AREA_FROZEN',len(selected),flush=True)


if __name__=='__main__':main()
