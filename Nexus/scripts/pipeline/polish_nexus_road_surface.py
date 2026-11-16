"""Fit only observed road colors/opacity on fixed metric geometry.

Training cameras use sensor calibration plus one constant correction estimated
from stationary observations. Generated evaluation cameras are never changed.
"""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','observations','scene','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--legacy-training-cameras',action='store_true')
    p.add_argument('--remove-recorded-shadows',action='store_true',help='Use actual brighter unoccluded observations of the same ground point for dark neutral shadow interiors')
    p.add_argument('--ground-source',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cv2,numpy as np,torch,yaml
    from PIL import Image
    from gsplat import rasterization
    from nexus_3dgs.replay import load_trainer
    from nexus_3dgs.grounding import mean_pose
    from scripts.pipeline.merge_nexus_vehicle_asset import equal
    torch.set_num_threads(8);cv2.setNumThreads(4);torch.manual_seed(19)
    trainer,cfg=load_trainer(a.checkpoint);bg=trainer.models['Background'];selected=torch.tensor(np.load(a.checkpoint.parent/'road_indices.npy'),device='cuda',dtype=torch.long);assert len(selected)>100
    for model in trainer.models.values():
        for param in model.parameters():param.requires_grad_(False)
    mask=torch.zeros(bg.num_points,device='cuda',dtype=torch.bool);mask[selected]=True
    groups=[]
    for name,lr in [('_features_dc',.005),('_features_rest',.00025),('_opacities',.01)]:
        param=getattr(bg,name);param.requires_grad_(True);gradmask=mask.reshape((-1,)+(1,)*(param.ndim-1)).to(param.dtype)
        if name=='_features_rest':gradmask=gradmask.expand_as(param).clone();gradmask[:,3:]=0
        param.register_hook(lambda g,m=gradmask:g*m);groups.append({'params':[param],'lr':lr})
    optimizer=torch.optim.Adam(groups,eps=1e-15)
    meta=json.loads((a.scene/'nexus_scene.json').read_text());S=np.linalg.inv(meta['frames'][0]['ego_to_world']);ext=np.loadtxt(a.scene/'extrinsics/0.txt');items=[];corrections=[]
    for rec in json.loads((a.observations/'observations.json').read_text())['frames']:
        f=rec['frame'];d=dict(np.load(a.observations/f'{f:03d}.npz'));raw=S@np.array(meta['frames'][f]['images'][0]['ego_to_world'])@ext;d['raw_camera']=raw;d['frame']=f
        if f<=50:corrections.append(np.linalg.inv(raw)@d['C'])
        items.append(d)
    correction=mean_pose(corrections);calibration=[];cut=250
    material_targets={};shadow_audit=[]
    if a.remove_recorded_shadows:
        if not a.ground_source:raise ValueError('Shadow separation needs a validated ground source')
        ground=json.loads(a.ground_source.read_text())['ground_registration']['ground_model'];aa,bb,cc=ground['coefficients_z_ax_by_c'];normal=np.array([-aa,-bb,1.])
        photos={d['frame']:np.array(Image.open(a.observations/f'{d["frame"]:03d}.png')).astype(np.float32)/255 for d in items}
        cameras={d['frame']:(d['C'] if a.legacy_training_cameras else d['raw_camera']@correction) for d in items}
        for d in items:
            f=d['frame'];image=photos[f];H,W=image.shape[:2];y,x=np.indices((H,W));C,K=cameras[f],d['K'];ray=np.stack(((x+.5-K[0,2])/K[0,0],(y+.5-K[1,2])/K[1,1],np.ones((H,W))),-1)@C[:3,:3].T;den=ray@normal;depth=-(C[:3,3]@normal-cc)/np.where(abs(den)>1e-8,den,np.nan);world=C[:3,3]+ray*depth[...,None];colors=[];lights=[]
            for other in items:
                Cf,Kf=cameras[other['frame']],other['K'];q=(world-Cf[:3,3])@Cf[:3,:3];u=(q[...,0]/np.maximum(q[...,2],.1)*Kf[0,0]+Kf[0,2]-.5).astype(np.float32);v=(q[...,1]/np.maximum(q[...,2],.1)*Kf[1,1]+Kf[1,2]-.5).astype(np.float32)
                sample=cv2.remap(photos[other['frame']],u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);valid=cv2.remap(other['road'].astype(np.uint8),u,v,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0;valid &= (q[...,2]>.1)&np.isfinite(depth)&(depth>0)
                lum=sample.mean(-1);valid &= (lum<.55)&((sample.max(-1)-sample.min(-1))<.1)
                colors.append(sample);lights.append(np.where(valid,lum,-1.))
            values=np.stack(lights);rgb=np.stack(colors);order=np.argsort(values,axis=0);yy,xx=np.indices((H,W));top=rgb[order[-1],yy,xx];second=rgb[order[-2],yy,xx];best_lum=values[order[-2],yy,xx];target=(top+second)/2;original_lum=image.mean(-1);gradient=np.hypot(cv2.Sobel(original_lum,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(original_lum,cv2.CV_32F,0,1,ksize=3))
            change=d['road']&(original_lum<.3)&(best_lum>original_lum+.055)&(best_lum>original_lum*1.25)&(gradient<.08)
            image=image.copy();image[change]=target[change];material_targets[f]=image;shadow_audit.append({'frame':f,'pixels':int(change.sum()),'source':'two actual brighter neutral unoccluded views of the same metric ground point'})
            Image.fromarray((image*255).round().astype(np.uint8)).save(a.output/f'material_target_{f:03d}.png')
    for d in items:
        f=d['frame'];C=d['C'] if a.legacy_training_cameras else d['raw_camera']@correction
        K=d['K'].copy();K[1,2]-=cut;gt=material_targets[f] if a.remove_recorded_shadows else np.array(Image.open(a.observations/f'{f:03d}.png')).astype(np.float32)/255
        valid=d['road'][cut:];weight=1+3*(gt[cut:].mean(-1)>.45)
        calibration.append({'frame':f,'training_camera_to_scene':C.tolist(),'legacy_camera_to_scene':d['C'].tolist()})
        d.update(C=torch.tensor(C,dtype=torch.float32,device='cuda'),K=torch.tensor(K,dtype=torch.float32,device='cuda'),gt=torch.tensor(gt[cut:],device='cuda'),mask=torch.tensor(valid,dtype=torch.float32,device='cuda'),weight=torch.tensor(weight,dtype=torch.float32,device='cuda'))
    # Exclude late observations from the appearance optimization. The source
    # checkpoint and geometry already saw them: this is not an unseen benchmark.
    training=items if a.remove_recorded_shadows else [d for d in items if d['frame'] not in [80,90,99]]
    moving=[d for d in training if d['frame']>=60];stationary=[d for d in training if d['frame']<60]
    history=[]
    for step in range(a.steps):
        pool=moving if step%2 else stationary;d=pool[(step//2)%len(pool)];C,K=d['C'],d['K'];camera=type('Camera',(),{'camtoworlds':C})();gs=bg.get_gaussians(camera)
        rgb,alpha,_=rasterization(means=gs['_means'],quats=gs['_quats'],scales=gs['_scales'],opacities=gs['_opacities'].squeeze(-1),colors=gs['_rgbs'],viewmats=torch.linalg.inv(C)[None],Ks=K[None],width=960,height=540-cut,near_plane=.1,far_plane=1e10,packed=False,render_mode='RGB',rasterize_mode='classic')
        rgb=trainer.affine_transformation(rgb[0].clamp(max=1),{'img_idx':torch.tensor(d['frame']*3,device='cuda')});valid=d['mask'];weight=d['weight']*valid
        photo=(abs(rgb-d['gt'])*weight[...,None]).sum()/(weight.sum()*3).clamp_min(1)
        sim=1-trainer.ssim((rgb*valid[...,None]).permute(2,0,1)[None],(d['gt']*valid[...,None]).permute(2,0,1)[None]);loss=.8*photo+.2*sim
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%200==0:
            row={'step':step,'loss':float(loss),'photo':float(photo)};history.append(row);print('ROAD_FIT '+json.dumps(row),flush=True)
    original=torch.load(a.checkpoint,map_location='cpu',weights_only=False);after=copy.deepcopy(original);after['models']['Background']={k:v.detach().cpu() for k,v in bg.state_dict().items()}
    untouched=(~mask).cpu();old,new=original['models']['Background'],after['models']['Background']
    for key,value in old.items():
        if key in ['_features_dc','_features_rest','_opacities']:equal(value[untouched],new[key][untouched])
        else:equal(value,new[key])
    assert torch.count_nonzero(new['_features_rest'][selected.cpu(),3:])==0
    for name,value in original['models'].items():
        if name!='Background':equal(value,after['models'][name])
    torch.save(after,a.output/'checkpoint_final.pth');(a.output/'config.yaml').write_text(yaml.safe_dump(__import__('omegaconf').OmegaConf.to_container(cfg,resolve=True),sort_keys=False))
    result={'status':'candidate_not_accepted','road_appearance_points':len(selected),'steps':a.steps,'all_geometry_traffic_camera_sky_and_other_background_rows_frozen':True,'training_camera_policy':'legacy learned per-frame' if a.legacy_training_cameras else 'one stationary-estimated calibration correction applied to real sensor cameras','fixed_correction':correction.tolist(),'rgb_fit_excluded_frames':[80,90,99],'holdout_scope':'base and geometry saw these frames; appearance-fit exclusion only','history':history}
    if a.remove_recorded_shadows:
        result['rgb_fit_excluded_frames']=[55,85];result['shadow_separation']=shadow_audit
    (a.output/'polish.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'training_cameras.json').write_text(json.dumps(calibration,indent=2)+'\n');print('ROAD_POLISH_COMPLETE',flush=True)

if __name__=='__main__':main()
