"""Independent RGB/LiDAR/camera motion consistency, without a 3DGS renderer."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['scene','semantics','native-reference','ground-source','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--registration',type=Path,help='Optional independent LiDAR correction for RGB consistency evaluation')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cv2,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    from scipy.optimize import least_squares
    cv2.setNumThreads(4)
    scene=json.loads((a.scene/'nexus_scene.json').read_text());S=np.linalg.inv(scene['frames'][0]['ego_to_world']);intr=np.loadtxt(a.scene/'intrinsics/0.txt');extr=np.loadtxt(a.scene/'extrinsics/0.txt');K=np.array([[intr[0]/2,0,intr[2]/2],[0,intr[1]/2,intr[3]/2],[0,0,1.]]);H,W=540,960
    native={r['frame_idx']:S@np.array(r['camera_world']) for r in json.loads(a.native_reference.read_text())['frames']}
    registration=None
    if a.registration:
        from nexus_3dgs.geometry import resample_poses
        records=json.loads(a.registration.read_text());assert records['status']=='complete'
        times=np.array([f['timestamp_us'] for f in scene['frames']],np.int64)
        matrices=np.array([r['scene_correction_matrix'] for r in sorted(records['rows'],key=lambda x:x['frame'])])
        def registration(timestamp):
            result,valid=resample_poses(times,matrices[:,None],np.ones((len(times),1),bool),[int(np.clip(timestamp,times[0],times[-1]))]);assert valid[0,0]
            return result[0,0]
    ground=json.loads(a.ground_source.read_text())['ground_registration']['ground_model'];assert ground['status']=='validated'
    groundS=np.linalg.inv(ground.get('world_from_scene',scene['frames'][0]['ego_to_world']));aa,bb,cc=ground['coefficients_z_ax_by_c'];scene_plane=np.linalg.inv(S).T@groundS.T@np.array([-aa,-bb,1.,-cc]);ground_bounds=np.array(ground['supported_xy_bounds_scene'])
    info=json.loads((a.scene/'instances/instances_info.json').read_text());boxes={}
    for item in info.values():
        ann=item['frame_annotations']
        for f,B,size in zip(ann['frame_idx'],ann['obj_to_world'],ann['box_size']):boxes.setdefault(f,[]).append((S@B,np.array(size)))
    frames=[20,40,55,60,63,65,70,75,80,85,90,99,110,130];views={};sift=cv2.SIFT_create(nfeatures=7000,contrastThreshold=.015)
    signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    D20=np.linalg.inv(S@np.array(scene['frames'][20]['images'][0]['ego_to_world'])@extr)@native[20]
    def project(X,C):
        q=(X-C[:3,3])@C[:3,:3];z=q[:,2];uv=q[:,:2]/np.maximum(z[:,None],1e-8)@K[:2,:2].T+K[:2,2];return uv,z
    def stats(x):
        x=np.asarray(x);x=x[np.isfinite(x)];return {'count':len(x),'median_px':float(np.median(x)) if len(x) else None,'p90_px':float(np.quantile(x,.9)) if len(x) else None,'mean_px':float(x.mean()) if len(x) else None}
    for f in frames:
        row=scene['frames'][f];cam=next(c for c in row['images'] if c['camera_id']==0);C=S@np.array(cam['ego_to_world'])@extr;L=S@np.array(row['ego_to_world']);image_path=a.scene/'images'/f'{f:03d}_0.jpg';image_path=image_path if image_path.is_file() else Path(cam['source']);im=cv2.undistort(np.array(Image.open(image_path).convert('RGB').resize((W,H),Image.Resampling.BILINEAR)),K,intr[4:]);imageio_path=a.output/f'{f:03d}_photo.jpg';Image.fromarray(im).save(imageio_path)
        with np.load(a.semantics/f'{f:03d}_0.npz') as masks:
            structure=np.isin(masks['labels'],[0,1,6,11,25,48])&(masks['confidence']>=190)
            road=masks['road'].copy()>0
        occupied=np.zeros((H,W),np.uint8)
        for B,size in boxes.get(f,[]):
            q=((signs*(size+.4)/2)@B[:3,:3].T+B[:3,3]-C[:3,3])@C[:3,:3]
            if q[:,2].min()<=.1:continue
            uv=q[:,:2]/q[:,2,None]@K[:2,:2].T+K[:2,2];cv2.fillConvexPoly(occupied,cv2.convexHull(np.rint(np.clip(uv,-100000,100000)).astype(np.int32)),1)
        structure &= cv2.dilate(occupied,np.ones((17,17),np.uint8))==0;structure=cv2.erode(structure.astype(np.uint8),np.ones((5,5),np.uint8))*255
        kp,desc=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),structure);xy=np.array([k.pt for k in kp],np.float32)
        raw=np.fromfile(a.scene/'lidar'/f'{f:03d}.bin',np.float32).reshape(-1,6);X=raw[:,:3]@L[:3,:3].T+L[:3,3];keep=~((raw[:,0]>-1.45)&(raw[:,0]<4.4)&(np.abs(raw[:,1])<1.5)&(raw[:,2]>-.8)&(raw[:,2]<2.4))
        for B,size in boxes.get(f,[]):
            local=(X-B[:3,3])@B[:3,:3];keep &= ~np.all(np.abs(local)<(size+.6)/2,axis=1)
        X=X[keep];uv,z=project(X,C);ok=(z>.2)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H);points=X[ok];uv,z=uv[ok],z[ok]
        # Per-pixel frontmost hits, before a local surface is used for feature depth.
        pix=np.floor(uv).astype(int);flat=pix[:,1]*W+pix[:,0];order=np.lexsort((z,flat));_,first=np.unique(flat[order],return_index=True);take=order[first];points,uv,z=points[take],uv[take],z[take]
        views[f]={'C':C,'lidar_tick':L@extr,'fixed':C@D20,'native':native.get(f),'image':im,'mask':structure,'road':road,'xy':xy,'desc':desc,'points':points,'uv':uv,'depth':z,'tree':cKDTree(uv),'timestamp':cam['timestamp_us']}
        if f in [20,65,70,85,99,130]:
            panels=[]
            for name,CC in [('factory',C),('native',native.get(f)),('lidar_tick',L@extr)]:
                if CC is None:continue
                overlay=im.copy();pp,zz=project(points,CC);valid=(zz>.2)&(pp[:,0]>=0)&(pp[:,0]<W)&(pp[:,1]>=0)&(pp[:,1]<H);ix,iy=np.floor(pp[valid]).astype(int).T;value=np.uint8(np.clip(np.log1p(zz[valid])/np.log(150)*255,0,255));colors=cv2.applyColorMap(value[:,None],cv2.COLORMAP_TURBO)[:,0,::-1];overlay[iy,ix]=(im[iy,ix]*.25+colors*.75).astype(np.uint8);Image.fromarray(overlay).save(a.output/f'{f:03d}_{name}_lidar.png');panels.append((name,overlay))
            canvas=Image.new('RGB',(W*len(panels),H+45),'#142136');draw=ImageDraw.Draw(canvas)
            for j,(name,panel) in enumerate(panels):draw.text((j*W+10,10),f'frame {f}: {name} / real static LiDAR',fill='white');canvas.paste(Image.fromarray(panel),(j*W,45))
            canvas.save(a.output/f'{f:03d}_projection_comparison.jpg')
        print(f'PROJECTION_FEATURES {f} static_features={len(xy)} static_visible_lidar={len(points)}',flush=True)
    bf=cv2.BFMatcher();pairs=[(20,f) for f in [40,55,60,63,65,70,75,80,85,90,99,110,130]]+[(55,70),(55,85),(55,99),(75,90),(75,110),(85,99),(85,130),(99,130)];rows=[];pose_proposals={}
    for left,right in pairs:
        x,y=views[left],views[right];forward=bf.knnMatch(x['desc'],y['desc'],k=2);backward=bf.knnMatch(y['desc'],x['desc'],k=2);reverse={m.queryIdx:m.trainIdx for m,n in backward if m.distance<.72*n.distance};matches=[m for m,n in forward if m.distance<.72*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
        if len(matches)<30:
            print(f'PROJECTION_SKIP {left}->{right} matches={len(matches)}',flush=True);continue
        ids=np.array([m.queryIdx for m in matches]);p1=x['xy'][ids];p2=y['xy'][[m.trainIdx for m in matches]];baseline=float(np.linalg.norm(x['C'][:3,3]-y['C'][:3,3]))
        if baseline<.1:_,mask=cv2.findHomography(p1,p2,cv2.USAC_MAGSAC,1.5)
        else:_,mask=cv2.findFundamentalMat(p1,p2,cv2.USAC_MAGSAC,1.5,.999)
        if mask is None:continue
        accepted=mask.ravel()>0;p1,p2=p1[accepted],p2[accepted]
        distance,nn=x['tree'].query(p1,k=12,workers=8);depth=x['depth'][nn];median=np.median(depth,axis=1);spread=np.quantile(depth,.9,axis=1)-np.quantile(depth,.1,axis=1);supported=(distance[:,0]<1.8)&(distance[:,-1]<6)&(spread<np.maximum(.5,.06*median))
        # Locally affine inverse depth supplies an actual measured surface,
        # with no Gaussian geometry or generated RGB in the correspondence.
        duv=x['uv'][nn]-p1[:,None,:];design=np.concatenate([duv,np.ones((*duv.shape[:2],1))],axis=-1);weights=1/(distance**2+.25);A=np.einsum('nki,nk,nkj->nij',design,weights,design)+np.eye(3)[None]*1e-6;b=np.einsum('nki,nk,nk->ni',design,weights,1/depth);coeff=np.linalg.solve(A,b);estimated=1/np.maximum(coeff[:,2],1e-6);good=supported&(estimated>.8*median)&(estimated<1.2*median)
        # Road markings need not coincide with a sparse laser beam. Use the
        # independently validated measured plane instead of rejecting them.
        camera_rays=np.c_[(p1[:,0]-K[0,2])/K[0,0],(p1[:,1]-K[1,2])/K[1,1],np.ones(len(p1))];rays_scene=camera_rays@x['C'][:3,:3].T;den=rays_scene@scene_plane[:3];plane_depth=-(x['C'][:3,3]@scene_plane[:3]+scene_plane[3])/np.where(np.abs(den)>1e-8,den,1e-8);hit=x['C'][:3,3]+rays_scene*plane_depth[:,None];hit_ground=(np.c_[hit,np.ones(len(hit))]@np.linalg.inv(S).T@groundS.T)[:,:2];pix=np.clip(np.rint(p1).astype(int),[0,0],[W-1,H-1]);plane_supported=x['road'][pix[:,1],pix[:,0]]&(plane_depth>1)&(plane_depth<100)&(hit_ground>=ground_bounds[0]).all(-1)&(hit_ground<=ground_bounds[1]).all(-1)
        estimated[plane_supported]=plane_depth[plane_supported];good |= plane_supported;plane_count=int(plane_supported.sum());p1,p2,estimated=p1[good],p2[good],estimated[good]
        if len(p1)<25:
            print(f'PROJECTION_SKIP {left}->{right} image_inliers={int(accepted.sum())} depth_support={len(p1)} road={plane_count}',flush=True);continue
        rays=np.c_[(p1[:,0]-K[0,2])/K[0,0],(p1[:,1]-K[1,2])/K[1,1],np.ones(len(p1))];X=rays*estimated[:,None]@x['C'][:3,:3].T+x['C'][:3,3]
        rng=np.random.default_rng(left*1000+right);permutation=rng.permutation(len(X));train=permutation[:int(.75*len(X))];test=permutation[int(.75*len(X)):];C=y['C'];rot0=C[:3,:3].copy()
        def rotated(v):
            out=C.copy();out[:3,:3]=Rotation.from_rotvec(v).as_matrix()@rot0;return out
        def residual(v):uv,z=project(X[train],rotated(v));return (uv-p2[train]).ravel()
        fit=least_squares(residual,np.zeros(3),loss='soft_l1',f_scale=1.,max_nfev=120);visual=rotated(fit.x)
        row={'frames':[left,right],'baseline_m':baseline,'mutual_ratio_matches':len(matches),'image_only_geometric_inliers':int(accepted.sum()),'lidar_supported_matches':len(X),'measured_road_plane_supported_matches':plane_count,'fit_matches':len(train),'evaluation_matches':len(test),'errors':{},'visual_rotation_correction_deg':float(np.rad2deg(np.linalg.norm(fit.x))),'visual_rotation_vector_scene_deg':np.rad2deg(fit.x).tolist(),'measured_target_camera':C.tolist(),'visual_rotation_target_camera':visual.tolist()}
        for name,CC in [('factory',y['C']),('fixed_first_correction',y['fixed']),('lidar_tick',y['lidar_tick']),('native',y['native']),('visual_rotation',visual)]:
            if CC is None:continue
            pp,zz=project(X,CC);e=np.linalg.norm(pp-p2,axis=1);row['errors'][name]={'all':stats(e),'fit':stats(e[train]),'evaluation':stats(e[test]),'evaluation_median_vector_px':np.median(pp[test]-p2[test],axis=0).tolist()}
        if registration is not None:
            left_change=registration(x['timestamp']);right_change=registration(y['timestamp'])
            corrected_X=X@left_change[:3,:3].T+left_change[:3,3]
            pp,_=project(corrected_X,right_change@y['C']);e=np.linalg.norm(pp-p2,axis=1)
            row['errors']['independent_lidar_registration']={'all':stats(e),'evaluation':stats(e[test])}
        rvec,_=cv2.Rodrigues(C[:3,:3].T);tvec=-(C[:3,:3].T@C[:3,3]);success,rvec,tvec,inliers=cv2.solvePnPRansac(X[train],p2[train].astype(float),K,None,rvec,tvec,useExtrinsicGuess=True,iterationsCount=1000,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
        if success and inliers is not None and len(inliers)>15:
            ii=train[inliers.ravel()];rvec,tvec=cv2.solvePnPRefineLM(X[ii],p2[ii].astype(float),K,None,rvec,tvec);RR,_=cv2.Rodrigues(rvec);pnp=np.eye(4);pnp[:3,:3]=RR.T;pnp[:3,3]=-RR.T@tvec.ravel();pp,_=project(X,pnp);row['errors']['visual_pnp']={'evaluation':stats(np.linalg.norm(pp[test]-p2[test],axis=1))};row['pnp_camera_translation_delta_m']=float(np.linalg.norm(pnp[:3,3]-C[:3,3]));row['pnp_camera_rotation_delta_deg']=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(pnp[:3,:3]@C[:3,:3].T).as_rotvec())))
        rows.append(row)
        if left==20:pose_proposals[str(right)]={'camera_scene':visual.tolist(),'source_frame':left,'evaluation':row['errors']['visual_rotation']['evaluation'],'scope':'diagnostic rotation-only fit, not yet a production camera trajectory'}
        # Plot held-out observations (green) and independent projections (red).
        panels=[]
        for name,CC in [('factory',C),('native',y['native']),('visual_rotation',visual)]:
            if CC is None:continue
            canvas=y['image'].copy();pp,_=project(X[test],CC)
            for observed,predicted in zip(p2[test][:150],pp[:150]):
                if not np.isfinite(predicted).all():continue
                a0=tuple(np.rint(observed).astype(int));b0=tuple(np.rint(predicted).astype(int));cv2.line(canvas,a0,b0,(255,220,40),1,cv2.LINE_AA);cv2.circle(canvas,a0,3,(40,255,40),1,cv2.LINE_AA);cv2.circle(canvas,b0,2,(255,40,40),1,cv2.LINE_AA)
            panels.append((name,canvas))
        sheet=Image.new('RGB',(W*len(panels),H+45),'#142136');draw=ImageDraw.Draw(sheet)
        for j,(name,image) in enumerate(panels):draw.text((j*W+10,10),f'{left} -> {right}: {name}; green=observed, red=projected',fill='white');sheet.paste(Image.fromarray(image),(j*W,45))
        sheet.save(a.output/f'matches_{left:03d}_{right:03d}.jpg')
        print(f'PROJECTION_PAIR {left}->{right} n={len(X)} eval_factory={row["errors"]["factory"]["evaluation"]["median_px"]:.3f} eval_visual={row["errors"]["visual_rotation"]["evaluation"]["median_px"]:.3f} rotation={row["visual_rotation_correction_deg"]:.3f}',flush=True)
    report={'status':'complete' if rows else 'insufficient_correspondences','method':'real static SIFT matches + source real LiDAR local inverse-depth surfaces or validated measured road plane; image-only outlier rejection, deterministic 75/25 rotation-fit/evaluation split; no 3DGS geometry or RGB','source_frame_anchor':'source factory camera rays intersect local real LiDAR depth or measured road plane; evaluates cross-frame consistency, not independent absolute calibration truth','camera_variants':{'factory':'actual camera exposure ego pose and original extrinsic','lidar_tick':'intentionally wrong same-frame LiDAR-time ego pose','native':'v5 learned per-frame camera correction','fixed_first_correction':'one stationary-frame correction','visual_rotation':'rotation-only fit on 75% correspondences; fixed measured position, remaining 25% evaluated'},'image_resolution_wh':[W,H],'rows':rows,'diagnostic_rotation_proposals':pose_proposals,'scope':'single-scene independent sensor and image consistency diagnostic; estimates are not automatically applied or a complete bundle adjustment'}
    (a.output/'projection_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('PROJECTION_AUDIT_COMPLETE '+str(a.output.resolve()),flush=True)


if __name__=='__main__':main()
