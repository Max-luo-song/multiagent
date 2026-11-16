"""Map preparation adapter; same lane-selection algorithm as the frozen case."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'artifacts/nexus_multicar_upgrade_check_20261027'
OUT = ROOT/'artifacts/nexus_kinematic_20261027'
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/nuplan-devkit')]


def prepare_routes(reference_root, output_root, third_slot, reactor_slot):
    OLD = Path(reference_root)
    OUT = Path(output_root)
    import numpy as np
    from shapely.geometry import Point, LineString
    from shapely.ops import unary_union
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as Layer
    from nuplan.common.actor_state.state_representation import Point2D
    protocol = json.loads((OLD/'protocol.json').read_text())
    for name, value in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == value, name
    OUT.mkdir(exist_ok=False)
    (OUT/'lanes').mkdir()
    with np.load(OLD/'normal_route_factorial_v1/conditions_normal.npz') as data:
        values = data['repaired']
        valid = data['valid'].astype(bool)
    states = values*np.array([104,104,1,1,4,4,5,1.6])+np.array([0,0,0,0,0,0,4.5,2])
    W = np.array(protocol['world_from_nexus']);R = W[:2,:2];offset = W[:2,3]
    api = get_maps_api(str(ROOT.parent/'data/nuplan/maps'), 'nuplan-maps-v1.0', protocol['scenario']['map_name'])
    arrays = {}; records = []

    def tangent(line, station):
        a = np.array(line.interpolate(max(0, station-.25)).coords[0])
        b = np.array(line.interpolate(min(line.length, station+.25)).coords[0])
        return (b-a)/max(np.linalg.norm(b-a), 1e-8)

    for slot in np.flatnonzero(valid[:,4]):
        state = states[slot,4]
        xy = state[:2]@R.T+offset;head = state[2:4]@R.T;point = Point(xy)
        candidates = []
        for layer in (Layer.LANE, Layer.LANE_CONNECTOR):
            for obj in api.get_proximal_map_objects(Point2D(*xy), 8, [layer])[layer]:
                line = obj.baseline_path.linestring;s = line.project(point)
                dot = float(tangent(line,s)@head/max(np.linalg.norm(head),1e-8))
                if dot < .7:
                    continue
                distance = line.distance(point)
                score = distance+2*(1-dot)+(0 if obj.polygon.covers(point) else 10)
                candidates.append((score,obj,line))
        if not candidates or np.linalg.norm(state[4:6]) < .5:
            records.append({'slot':int(slot),'route':False,'reason':'stationary or no heading-compatible lane'})
            continue
        _,obj,line = min(candidates,key=lambda x:(x[0],str(x[1].id)))
        if line.distance(point)>4:
            records.append({'slot':int(slot),'route':False,'reason':'too far from compatible lane'})
            continue
        points=[np.array(line.coords)];polygons=[obj.polygon]
        ids=[type(obj).__name__+':'+str(obj.id)];seen=set(ids);length=line.length
        while length-line.project(point)<220:
            previous=tangent(obj.baseline_path.linestring,obj.baseline_path.linestring.length)
            options=[]
            for edge in obj.outgoing_edges:
                key=type(edge).__name__+':'+str(edge.id)
                if key in seen:
                    continue
                candidate=edge.baseline_path.linestring
                hs=tangent(candidate,0);he=tangent(candidate,candidate.length)
                turn=lambda a,b:abs(float(np.arctan2(a[0]*b[1]-a[1]*b[0],a@b)))
                options.append((turn(previous,hs)+turn(hs,he),key,edge,candidate))
            if not options:
                break
            _,key,obj,candidate=min(options,key=lambda x:(x[0],x[1]))
            seen.add(key);ids.append(key);points.append(np.array(candidate.coords));polygons.append(obj.polygon);length+=candidate.length
        coordinates=np.concatenate(points)
        coordinates=coordinates[np.r_[True,np.linalg.norm(np.diff(coordinates,axis=0),axis=-1)>1e-5]]
        route=LineString(coordinates);start=route.project(point);h=tangent(route,start)
        lateral_offset=float((xy-np.array(route.interpolate(start).coords[0]))@np.array([-h[1],h[0]]))
        stations=np.arange(0,route.length,.75)
        shifted=[]
        for s in stations:
            h=tangent(route,float(s))
            shifted.append(np.array(route.interpolate(float(s)).coords[0])+lateral_offset*np.array([-h[1],h[0]]))
        arrays[f'route_{slot}']=((np.array(shifted)-offset)@R).astype(np.float32)
        corridor=unary_union(polygons)
        (OUT/'lanes'/f'{slot}.wkb').write_bytes(corridor.wkb)
        records.append({'slot':int(slot),'route':True,'selected_edges':ids,'lateral_offset_m':lateral_offset})
    assert any(x['slot']==reactor_slot and x['route'] for x in records)
    assert any(x['slot']==third_slot and x['route'] for x in records)
    np.savez_compressed(OUT/'routes.npz',**arrays)
    result={'status':'prepared','source_protocol':str(OLD/'protocol.json'),
            'history_source':'unshifted observed history; historical velocities reconstructed',
            'route_source':'current pose/heading and SDK lane topology; minimum-turn continuation',
            'future_E_C_values_used':False,'routes':records,
            'source_sha256':protocol['source_sha256']}
    (OUT/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('KINEMATIC_CASE_PREPARED',len(arrays),'routes',flush=True)
