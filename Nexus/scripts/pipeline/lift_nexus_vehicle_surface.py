"""Build an experimental native Gaussian rear panel from measured shape and real RGB.

The panel remains in the vehicle's body coordinates. This is an explicitly local
single-observation reconstruction, not a screen-space video overlay or an unseen
view ground truth. Frozen original geometry is retained outside the panel.
"""
import argparse,copy,json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree,ConvexHull,Delaunay
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]
from nexus_3dgs.surface_lifting import lift_pixels,rectify_gaussians


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','reference','points','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frame',type=int,default=20)
    p.add_argument('--smooth-surface',action='store_true',help='Fit one coherent robust quadratic rear surface instead of independent local planes')
    p.add_argument('--top-z',type=float,default=.65,help='Upper extent in measured vehicle body coordinates')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4)
    import torch,yaml
    from models.gaussians.basics import quat_to_rotmat
    from models.modules import AffineTransform
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);w=state['models']['RigidNodes'];cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text());scene=Path(cfg['data']['data_root'])/str(cfg['data']['scene_idx'])
    support=json.loads((a.points.parent/'surface_points.json').read_text());assert support['fixed_renderer_body_coordinates'];model=support['model_index'];f=a.frame;means=w['_means'].numpy();ids=w['points_ids'].numpy().reshape(-1);selected=ids==model
    B=np.eye(4);B[:3,:3]=quat_to_rotmat(torch.nn.functional.normalize(w['instances_quats'][f,model][None],dim=-1))[0].numpy();B[:3,3]=w['instances_trans'][f,model].numpy()
    meta=json.loads((scene/'nexus_scene.json').read_text());S=np.linalg.inv(np.asarray(meta['frames'][0]['ego_to_world']))
    run=json.loads((a.reference/'run.json').read_text());assert run['reference'] and run['camera_mode']=='logged'
    record=next(i for r in run['frames'] if r['frame_idx']==f for i in r['images'] if i['camera_id']==0);C=S@np.asarray(record['scene_metadata']['camera_world']);camera=np.linalg.inv(B)@C
    rgb=np.array(Image.open(scene/'images'/f'{f:03d}_0.jpg').convert('RGB'));H,W=rgb.shape[:2];K=np.asarray(record['scene_metadata']['intrinsics']).copy();K[0]*=W/960;K[1]*=H/540
    intr=np.loadtxt(scene/'intrinsics/0.txt');rgb=cv2.undistort(rgb,K,intr[4:]).astype(np.float32)/255
    mask=np.array(Image.open(scene/'vehicle_interior_masks'/f'{f:03d}_0.png'))>0
    if mask.shape!=(H,W):mask=cv2.resize(mask.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0
    # These masks were prepared in the undistorted training image domain.
    P=np.load(a.points)['points'];rear=-6.149869162707331;P=P[(abs(P[:,0]-rear)<.25)&(P[:,2]>-1.15)&(P[:,2]<a.top_z)&(abs(P[:,1])<1.23)]
    if len(P)<100:raise ValueError('Insufficient measured rear-panel support')
    hull=Delaunay(P[ConvexHull(P[:,1:]).vertices,1:]);v,u=np.where(mask);uv=np.column_stack((u,v));mu,logs,quats,valid=lift_pixels(uv,K,camera,np.array([1.,0,0,-rear]),pixel_std=.7,normal_std=.003)
    valid &= hull.find_simplex(mu[:,1:])>=0
    uv=uv[valid];mu=mu[valid];logs=logs[valid];quats=quats[valid]
    d,near=cKDTree(P).query(mu,k=48,workers=8);groups=P[near];anchor=groups.mean(1);delta=groups-anchor[:,None];val,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',delta,delta)/48);normal=vec[:,:,0]
    good=(d[:,0]<.2)&(d[:,-1]<.55)&(abs(normal[:,0])>.8)&(val[:,0]/np.maximum(val.sum(-1),1e-12)<.06)
    normal[normal[:,0]<0]*=-1
    fit_report=None
    if a.smooth_surface:
        def basis(y,z):return np.column_stack((np.ones_like(y),y,z,y*y,y*z,z*z))
        design=basis(P[:,1],P[:,2]);keep=np.ones(len(P),bool)
        for _ in range(8):
            coef=np.linalg.lstsq(design[keep],P[keep,0],rcond=None)[0];res=P[:,0]-design@coef;mad=1.4826*np.median(abs(res[keep]-np.median(res[keep])));keep=abs(res)<max(.025,3*mad)
        # Intersect each source ray with the same continuous measured surface.
        # Independent noisy local fits can leave holes or fold textured regions.
        ray=mu-camera[:3,3];t=np.ones(len(mu))
        for _ in range(8):
            q=camera[:3,3]+ray*t[:,None];y,z=q[:,1],q[:,2];dy=coef[1]+2*coef[3]*y+coef[4]*z;dz=coef[2]+coef[4]*y+2*coef[5]*z
            residual=q[:,0]-basis(y,z)@coef;derivative=ray[:,0]-dy*ray[:,1]-dz*ray[:,2];t-=residual/derivative
        anchor=camera[:3,3]+ray*t[:,None];y,z=anchor[:,1],anchor[:,2];dy=coef[1]+2*coef[3]*y+coef[4]*z;dz=coef[2]+coef[4]*y+2*coef[5]*z;normal=np.column_stack((np.ones(len(mu)),-dy,-dz));normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        good=np.isfinite(t)&(t>0)&(hull.find_simplex(anchor[:,1:])>=0)
        fit_report={'quadratic_x_of_yz_coefficients':coef.tolist(),'inlier_points':int(keep.sum()),'all_point_residual_p90_m':float(np.quantile(abs(res),.9)),'inlier_rmse_m':float(np.sqrt(np.mean(res[keep]**2)))}
    xyz=(mu-camera[:3,3])@camera[:3,:3];den=((mu-camera[:3,3])*normal).sum(-1);depth=xyz[:,2]*((anchor-camera[:3,3])*normal).sum(-1)/den
    new_mu,new_logs,new_q=rectify_gaussians(mu,logs,quats,K,camera,depth,normal,normal_std=.003)
    good &= np.isfinite(new_mu).all(-1)&(np.linalg.norm(new_mu-mu,axis=-1)<.22)
    uv=uv[good];new_mu=new_mu[good];new_logs=new_logs[good];new_q=new_q[good]
    if len(uv)<100:raise ValueError('Insufficient supported source pixels')
    aff=state['models']['Affine'];Amodel=AffineTransform('Affine',len(aff['embedding.weight']),device=torch.device('cpu'));Amodel.load_state_dict(aff)
    with torch.no_grad():A=Amodel({'img_idx':torch.tensor([f*3])})[0].numpy()
    color=np.clip((rgb[uv[:,1],uv[:,0]]-A[:,3])@np.linalg.inv(A[:,:3]).T,.001,.999)
    # Suppress only points actually covered by the observed native panel.
    coverage=np.zeros((H,W),np.uint8);coverage[uv[:,1],uv[:,0]]=1;coverage=cv2.dilate(coverage,np.ones((5,5),np.uint8))
    cam=(means-camera[:3,3])@camera[:3,:3];pix=cam[:,:2]/np.maximum(cam[:,2,None],.1)@K[:2,:2].T+K[:2,2];ix,iy=np.rint(pix).astype(int).T
    inside=(cam[:,2]>.1)&(ix>=0)&(ix<W)&(iy>=0)&(iy<H);covered=np.zeros(len(means),bool);covered[inside]=coverage[iy[inside],ix[inside]]>0
    suppress=selected&(means[:,0]<rear+.45)&covered&(means[:,2]>-1.25)&(means[:,2]<.72)
    n=len(new_mu);added={'_means':new_mu,'_scales':new_logs,'_quats':new_q,'_features_dc':(color-.5)/.28209479177387814,'_features_rest':np.zeros((n,w['_features_rest'].shape[1],3)),'_opacities':np.full((n,1),6.),'points_ids':np.full((n,1),model)}
    original=copy.deepcopy(w);w['_opacities'][torch.from_numpy(suppress)]=-15.
    for key,value in added.items():w[key]=torch.cat((w[key],torch.as_tensor(value,dtype=w[key].dtype)),dim=0)
    # All other vehicle parameters, instance poses, camera, sky, and background
    # must be exactly preserved; only chosen rear opacity and appended rows vary.
    for key,value in original.items():
        if key=='_opacities':assert torch.equal(value[~suppress],w[key][:len(means)][~suppress])
        elif key in added:assert torch.equal(value,w[key][:len(means)])
        else:assert torch.equal(value,w[key])
    torch.save(state,a.output/'checkpoint_final.pth');(a.output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));Image.fromarray((rgb*255).astype(np.uint8)).save(a.output/'source_undistorted.jpg');Image.fromarray(coverage*255).save(a.output/'source_coverage.png')
    np.save(a.output/'added_indices.npy',np.arange(len(means),len(means)+n))
    report={'status':'candidate_not_accepted','frame':f,'source_resolution':[W,H],'native_panel_gaussians':n,'suppressed_original_gaussians':int(suppress.sum()),'body_model_index':model,'coherent_surface_fit':fit_report,'top_z':a.top_z,'real_lidar_support_points':len(P),'source_rgb':'actual undistorted raw observation, inverse original affine','source_pixel_bounds':np.stack((uv.min(0),uv.max(0))).tolist(),'all_other_assets_and_instance_poses_frozen':True,'scope':'observed lower rear panel only; no guarantee for windows, unseen sides, shadows, or static background'}
    (a.output/'surface.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
