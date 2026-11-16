"""Build native metric road Gaussians with exclusive observed surface ownership.

Each material sample comes from actual road observations of the same plane
point. Moving-vehicle occlusions are excluded. Brighter repeated observations
can replace dark neutral cast shadows, but cannot invent unseen texture.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def measured_static_guard(scene,meta,data,S,plane):
    import cv2,numpy as np
    aa,bb,cc=plane;annotations=json.loads((scene/'instances/instances_info.json').read_text());byframe={}
    for item in annotations.values():
        ann=item['frame_annotations']
        for f,B,size in zip(ann['frame_idx'],ann['obj_to_world'],ann['box_size']):byframe.setdefault(f,[]).append((np.array(B),np.array(size)+.3))
    static=[]
    for d in data:
        f=d['frame'];raw=np.fromfile(scene/'lidar'/f'{f:03d}.bin',np.float32).reshape(-1,6)[:,:3];E=np.array(meta['frames'][f]['ego_to_world']);world=raw@E[:3,:3].T+E[:3,3]
        ego=(raw[:,0]>-1.45)&(raw[:,0]<4.4)&(abs(raw[:,1])<1.5)&(raw[:,2]>-.8)&(raw[:,2]<2.4);keep=~ego
        for B,size in byframe.get(f,[]):
            local=(world-B[:3,3])@B[:3,:3];keep &= ~(abs(local)<size/2).all(-1)
        points=world[keep]@S[:3,:3].T+S[:3,3];height=points[:,2]-(aa*points[:,0]+bb*points[:,1]+cc);points=points[(height>.12)&(height<30)]
        C,K=d['C'],d['K'];q=(points-C[:3,3])@C[:3,:3];uv=q[:,:2]/np.maximum(q[:,2,None],.1)@K[:2,:2].T+K[:2,2]-.5;ix,iy=np.rint(uv).astype(int).T;h,w=d['labels'].shape;inside=(q[:,2]>.1)&(ix>=0)&(ix<w)&(iy>=0)&(iy<h);keep=np.ones(len(points),bool);ids=np.flatnonzero(inside);label=d['labels'][iy[ids],ix[ids]];keep[ids]=~np.isin(label,[2,6]);static.append(points[keep])
    points=np.concatenate(static);_,unique=np.unique(np.floor(points/.08).astype(np.int32),axis=0,return_index=True)
    return points[unique].astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['observations','scene','source','calibration','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--spacing',type=float,default=.035)
    p.add_argument('--occluders-only',action='store_true')
    p.add_argument('--uncertainty-aware',action='store_true',help='Unknown semantic observations do not veto repeatedly observed road')
    p.add_argument('--harmonize-color',action='store_true',help='Normalize actual source exposures using common diffuse road observations')
    p.add_argument('--feather-m',type=float,default=0.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.occluders_only)
    import cv2,numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    cv2.setNumThreads(4);meta=json.loads((a.scene/'nexus_scene.json').read_text());S=np.linalg.inv(meta['frames'][0]['ego_to_world']);ext=np.loadtxt(a.scene/'extrinsics/0.txt');g=json.loads(a.source.read_text())['ground_registration']['ground_model'];aa,bb,cc=g['coefficients_z_ax_by_c'];normal=np.array([-aa,-bb,1.]);normal/=np.linalg.norm(normal)
    correction=np.array(json.loads(a.calibration.read_text())['fixed_correction']);frames=json.loads((a.observations/'observations.json').read_text())['frames'];data=[]
    for rec in frames:
        f=rec['frame'];d=dict(np.load(a.observations/f'{f:03d}.npz'));d['frame']=f;d['C']=S@np.array(meta['frames'][f]['images'][0]['ego_to_world'])@ext@correction;d['rgb']=np.array(Image.open(a.observations/f'{f:03d}.png')).astype(np.float32)/255;data.append(d)
    color_gains=[]
    if a.harmonize_color:
        ref=data[0];gray=ref['rgb'].mean(-1);grad=np.hypot(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1));good=ref['road']&(gray>.25)&(gray<.5)&(grad<.035);v,u=np.where(good);rng=np.random.default_rng(19);take=rng.choice(len(u),min(len(u),30000),replace=False);u,v=u[take],v[take];K,C=ref['K'],ref['C'];rays=np.column_stack(((u+.5-K[0,2])/K[0,0],(v+.5-K[1,2])/K[1,1],np.ones(len(u))))@C[:3,:3].T;n=np.array([-aa,-bb,1.]);depth=-(C[:3,3]@n-cc)/(rays@n);points=C[:3,3]+rays*depth[:,None];target=ref['rgb'][v,u]
        for d in data:
            C,K=d['C'],d['K'];q=(points-C[:3,3])@C[:3,:3];uv=q[:,:2]/np.maximum(q[:,2,None],.1)@K[:2,:2].T+K[:2,2]-.5;ix,iy=np.rint(uv).astype(int).T;valid=(q[:,2]>.1)&(ix>=0)&(ix<960)&(iy>=0)&(iy<540);ids=np.flatnonzero(valid);ids=ids[d['road'][iy[ids],ix[ids]]];source=d['rgb'][iy[ids],ix[ids]];difference=abs(source-target[ids]).mean(-1);ok=(source.mean(-1)>.2)&(source.mean(-1)<.55)&(difference<.12);source=source[ok];dest=target[ids[ok]];gain=np.clip(np.median(dest/np.maximum(source,.05),axis=0),.7,1.3) if len(source)>500 else np.ones(3);d['rgb']=np.clip(d['rgb']*gain,0,1).astype(np.float32);color_gains.append({'frame':d['frame'],'rgb_gain':gain.tolist(),'matched_real_points':len(source)})
    if a.occluders_only:
        points=measured_static_guard(a.scene,meta,data,S,[aa,bb,cc]);np.save(a.output/'static_occluders_semantic.npy',points);(a.output/'static_guard.json').write_text(json.dumps({'points':len(points),'excluded':'annotated dynamic objects, source ego body, observed road and sky returns','source':'actual LiDAR'},indent=2));print('SEMANTIC_STATIC_GUARD '+str(len(points)),flush=True);return
    # Restrict to the validated near-road corridor. No texture is emitted for
    # cells lacking agreement among real, non-vehicle observations.
    x=np.arange(-2.,42.,a.spacing);y=np.arange(-18.,18.,a.spacing);X,Y=np.meshgrid(x,y,indexing='ij');P=np.stack((X,Y,aa*X+bb*Y+cc),-1).reshape(-1,3);N=len(P);H,W=len(x),len(y)
    source=np.full(N,-1,np.int16);seen=np.zeros(N,np.uint8);positive=np.zeros(N,np.uint8);colors=np.zeros((N,3),np.float32);best=np.zeros((N,3),np.float32);best_l=np.full(N,-1.,np.float32);second=np.zeros((N,3),np.float32);second_l=best_l.copy();all_samples=[];negative=np.zeros(N,np.uint8)
    for j,d in enumerate(data):
        C,K=d['C'],d['K'];q=(P-C[:3,3])@C[:3,:3];u=(q[:,0]/np.maximum(q[:,2],.1)*K[0,0]+K[0,2]-.5).astype(np.float32).reshape(H,W);v=(q[:,1]/np.maximum(q[:,2],.1)*K[1,1]+K[1,2]-.5).astype(np.float32).reshape(H,W)
        rgb=cv2.remap(d['rgb'],u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(-1,3);vehicle=cv2.remap(d['vehicle'].astype(np.uint8),u,v,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).reshape(-1)>0;road=cv2.remap(d['road'].astype(np.uint8),u,v,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).reshape(-1)>0
        valid=(q[:,2]>.1)&(u.reshape(-1)>2)&(u.reshape(-1)<957)&(v.reshape(-1)>2)&(v.reshape(-1)<537)&(~vehicle);seen+=valid
        if a.uncertainty_aware:
            labels=cv2.remap(d['labels'],u,v,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).reshape(-1);conf=cv2.remap(d['confidence'],u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(-1);road=(labels==6)&(conf>.85);negative+=valid&(labels!=6)&(conf>.95)
        positive+=valid&road
        available=valid&road;choose=available&(source<0);colors[choose]=rgb[choose];source[choose]=j
        lum=rgb.mean(-1);neutral=available&(lum<.55)&((rgb.max(-1)-rgb.min(-1))<.1);higher=neutral&(lum>best_l);middle=neutral&(~higher)&(lum>second_l);second[middle]=rgb[middle];second_l[middle]=lum[middle];second[higher]=best[higher];second_l[higher]=best_l[higher];best[higher]=rgb[higher];best_l[higher]=lum[higher]
        print(f'ROAD_ATLAS_OBSERVATION {d["frame"]}',flush=True)
    valid=(positive>=2)&(source>=0)
    valid &= (negative==0) if a.uncertainty_aware else (positive/np.maximum(seen,1)>.98)
    # Erode and feather in metric ground coordinates to protect uncertain
    # curb/obstacle boundaries under small camera translations.
    mask=valid.reshape(H,W).astype(np.uint8);mask=cv2.erode(mask,np.ones((3,3),np.uint8));distance=cv2.distanceTransform(mask,cv2.DIST_L2,5);confidence=np.minimum(distance/3.,1.).reshape(-1);valid=confidence>0
    original_l=colors.mean(-1);deshadow=valid&(original_l<.3)&(second_l>original_l+.055)&(second_l>original_l*1.25);colors[deshadow]=(best[deshadow]+second[deshadow])/2
    tangent=np.array([1.,0.,aa]);tangent/=np.linalg.norm(tangent);R=np.column_stack((tangent,np.cross(normal,tangent),normal));quat=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
    field={'means':P[valid].astype(np.float32),'colors':colors[valid],'opacity':(.995*confidence[valid]).astype(np.float32),'scales':np.broadcast_to(np.array([a.spacing*.72,a.spacing*.72,.001],np.float32),(valid.sum(),3)),'quats':np.broadcast_to(quat.astype(np.float32),(valid.sum(),4))}
    if a.feather_m>0:
        soft=np.minimum(distance.reshape(-1)[valid]*a.spacing/a.feather_m,1.);soft[colors[valid].mean(-1)>.5]=1.;field['confidence']=soft.astype(np.float32)
    np.savez_compressed(a.output/'road_gaussians.npz',**field)
    # Real static returns guard occlusion. Using the old Gaussian depth here
    # would protect precisely the erroneous floating road/vehicle ghosts.
    annotations=json.loads((a.scene/'instances/instances_info.json').read_text());byframe={}
    for item in annotations.values():
        ann=item['frame_annotations']
        for f,B,size in zip(ann['frame_idx'],ann['obj_to_world'],ann['box_size']):byframe.setdefault(f,[]).append((np.array(B),np.array(size)+.3))
    static=[]
    for d in data:
        f=d['frame'];raw=np.fromfile(a.scene/'lidar'/f'{f:03d}.bin',np.float32).reshape(-1,6)[:,:3];E=np.array(meta['frames'][f]['ego_to_world']);world=raw@E[:3,:3].T+E[:3,3];keep=np.ones(len(world),bool)
        for B,size in byframe.get(f,[]):
            local=(world-B[:3,3])@B[:3,:3];keep &= ~(abs(local)<size/2).all(-1)
        points=world[keep]@S[:3,:3].T+S[:3,3];height=points[:,2]-(aa*points[:,0]+bb*points[:,1]+cc);static.append(points[(height>.12)&(height<30)])
    points=np.concatenate(static);_,unique=np.unique(np.floor(points/.08).astype(np.int32),axis=0,return_index=True);np.save(a.output/'static_occluders.npy',points[unique].astype(np.float32))
    if a.uncertainty_aware:
        clean=measured_static_guard(a.scene,meta,data,S,[aa,bb,cc]);np.save(a.output/'static_occluders_semantic.npy',clean)
    canvas=np.clip(colors.reshape(H,W,3)*255,0,255).astype(np.uint8);canvas[~valid.reshape(H,W)]=0;Image.fromarray(canvas).save(a.output/'material_atlas.png');Image.fromarray((confidence.reshape(H,W)*255).astype(np.uint8)).save(a.output/'confidence.png')
    report={'status':'candidate_not_accepted','native_road_gaussians':int(valid.sum()),'real_static_occlusion_points':len(unique),'spacing_m':a.spacing,'material_shadow_replacements':int(deshadow.sum()),'real_source_frames':[d['frame'] for d in data],'plane_coefficients':g['coefficients_z_ax_by_c'],'road_agreement':'>=2 positive views, no confident non-road contradiction; uncertain views ignored' if a.uncertainty_aware else '>=2 positive views, >98% non-vehicle views agree','texture_source':'earliest valid real road RGB; dark neutral shadows use two brighter real observations of the same ground point','scope':'observed near-road corridor only; no unseen road completion','source_render':str(a.source.resolve()),'fixed_camera_correction':correction.tolist()}
    report['source_color_calibration']=color_gains;report['metric_gray_material_feather_m']=a.feather_m
    (a.output/'road.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
