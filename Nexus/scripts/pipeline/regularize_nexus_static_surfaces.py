"""Constrain observed static surfaces, preserving every dynamic asset and camera.

Sky suppression uses repeated high-confidence real observations. Optional road
flattening preserves the source pixel centres/covariances before appearance fit.
"""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','observations','source','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--flatten-road',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cv2,numpy as np,torch,yaml
    from PIL import Image
    from models.modules import AffineTransform,EnvLight
    from models.gaussians.basics import spherical_harmonics
    from nexus_3dgs.surface_lifting import rectify_gaussians
    from scripts.pipeline.merge_nexus_vehicle_asset import equal
    cv2.setNumThreads(4);torch.set_num_threads(8)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text());bg=state['models']['Background'];original=copy.deepcopy(bg);mu=bg['_means'].numpy();N=len(mu)
    manifest=json.loads((a.observations/'observations.json').read_text());frames=[x['frame'] for x in manifest['frames']]
    obs=[];seen=np.zeros(N,np.uint16);sky_votes=np.zeros(N,np.uint16);road_votes=np.zeros(N,np.uint16);first=np.full(N,-1,np.int16)
    for j,f in enumerate(frames):
        d=dict(np.load(a.observations/f'{f:03d}.npz'));image=np.array(Image.open(a.observations/f'{f:03d}.png')).astype(np.float32)/255;d['image']=image;d['frame']=f;obs.append(d);C,K=d['C'],d['K'];H,W=d['sky'].shape;xyz=(mu-C[:3,3])@C[:3,:3];uv=xyz[:,:2]/np.maximum(xyz[:,2,None],.1)@K[:2,:2].T+K[:2,2];ix,iy=np.rint(uv).astype(int).T;valid=(xyz[:,2]>.1)&(ix>=0)&(ix<W)&(iy>=0)&(iy<H);ids=np.flatnonzero(valid);ids=ids[~d['vehicle'][iy[ids],ix[ids]]];seen[ids]+=1
        sk=d['sky'][iy[ids],ix[ids]];rd=d['road'][iy[ids],ix[ids]];sky_votes[ids[sk]]+=1;road_votes[ids[rd]]+=1;new=ids[rd&(first[ids]<0)];first[new]=j
    sky=(seen>=5)&(sky_votes/np.maximum(seen,1)>.98)
    # Reject dark/colored Gaussian centres whose sky-labelled projection may
    # be a misleading overlap with an actual object.
    rgb_dc=bg['_features_dc'].numpy()*.28209479177387814+.5
    sky &= (rgb_dc.mean(-1)>.35)&((rgb_dc.max(-1)-rgb_dc.min(-1))<.3)
    road=(seen>=2)&(road_votes/np.maximum(seen,1)>.8)&(first>=0)
    source=json.loads(a.source.read_text());g=source['ground_registration']['ground_model'];aa,bb,cc=g['coefficients_z_ax_by_c'];height=mu[:,2]-(aa*mu[:,0]+bb*mu[:,1]+cc)
    road &= (abs(height)<3.)&(mu[:,0]>-5)&(mu[:,0]<65)&(abs(mu[:,1])<25)
    np.savez_compressed(a.output/'point_classes.npz',sky=sky,road=road,first=first,seen=seen)
    bg['_opacities'][torch.from_numpy(sky)]=-15.
    changes=[]
    if a.flatten_road:
        normal=np.array([-aa,-bb,1.]);normal/=np.linalg.norm(normal)
        for j,d in enumerate(obs):
            ids=np.flatnonzero(road&(first==j));C,K=d['C'],d['K']
            if not len(ids):continue
            points=mu[ids];ray=points-C[:3,3];xyz=ray@C[:3,:3];plane_n=np.array([-aa,-bb,1.]);lam=-(C[:3,3]@plane_n-cc)/(ray@plane_n);depth=xyz[:,2]*lam;valid=(depth>.3)&(lam>.1)&(lam<10)
            ids=ids[valid];points=mu[ids];depth=depth[valid]
            means,logs,quats=rectify_gaussians(points,bg['_scales'][ids].numpy(),bg['_quats'][ids].numpy(),K,C,depth,np.tile(normal,(len(ids),1)),normal_std=.002)
            good=np.isfinite(means).all(-1)&np.isfinite(logs).all(-1);ids=ids[good];means,logs,quats=means[good],logs[good],quats[good]
            # Preserve original color at the exact source viewing direction,
            # then remove only this cohort's higher-degree directional terms.
            with torch.no_grad():
                direction=torch.tensor(points[good]-C[:3,3],dtype=torch.float32,device='cuda');direction=torch.nn.functional.normalize(direction,dim=-1);sh=torch.cat((bg['_features_dc'][ids,None],bg['_features_rest'][ids]),1).cuda()
                delta=spherical_harmonics(3,direction,sh)-spherical_harmonics(1,direction,sh);bg['_features_dc'][ids]+=delta.cpu()/.28209479177387814;bg['_features_rest'][ids,3:]=0
            bg['_means'][ids]=torch.tensor(means,dtype=torch.float32);bg['_scales'][ids]=torch.tensor(logs,dtype=torch.float32);bg['_quats'][ids]=torch.tensor(quats,dtype=torch.float32);changes.extend(ids.tolist())
    # Fit environment radiance to real sky RGB with original exposure removed.
    aff=state['models']['Affine'];exposure=AffineTransform('Affine',len(aff['embedding.weight']),device=torch.device('cpu'));exposure.load_state_dict(aff)
    samples=[]
    for d in obs:
        C,K=d['C'],d['K'];v,u=np.where(d['sky']);rng=np.random.default_rng(d['frame']);take=rng.choice(len(u),min(30000,len(u)),replace=False);u,v=u[take],v[take];r=np.column_stack(((u+.5-K[0,2])/K[0,0],(v+.5-K[1,2])/K[1,1],np.ones(len(u))))@C[:3,:3].T;r/=np.linalg.norm(r,axis=-1,keepdims=True)
        with torch.no_grad():A=exposure({'img_idx':torch.tensor([d['frame']*3])})[0].numpy()
        target=np.clip((d['image'][v,u]-A[:,3])@np.linalg.inv(A[:,:3]).T,0,1);samples.append((torch.tensor(r,dtype=torch.float32,device='cuda'),torch.tensor(target,dtype=torch.float32,device='cuda')))
    sky_model=EnvLight('Sky',**cfg['model']['Sky']['params'],device=torch.device('cuda')).cuda();sky_model.load_state_dict(state['models']['Sky']);opt=torch.optim.Adam(sky_model.parameters(),lr=.02);mean=torch.cat([p[1] for p in samples]).mean(0)
    torch.manual_seed(19)
    for step in range(401):
        rays,target=samples[step%len(samples)];ids=torch.randint(len(rays),(min(12000,len(rays)),),device='cuda');pred=sky_model({'viewdirs':rays[ids]});texture=sky_model.base.sigmoid() if sky_model.bounded else sky_model.base;loss=(pred-target[ids]).square().mean()+.002*(texture-mean).square().mean();opt.zero_grad();loss.backward();opt.step()
        if step%100==0:print(f'SKY_FIT {step} {float(loss):.6f}',flush=True)
    state['models']['Sky']={k:v.detach().cpu() for k,v in sky_model.state_dict().items()};torch.save(state,a.output/'checkpoint_final.pth');(a.output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));np.save(a.output/'road_indices.npy',np.array(changes,dtype=np.int64))
    base=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['models']
    for name,v in base.items():
        if name not in ['Background','Sky']:equal(v,state['models'][name])
    changed=np.zeros(N,bool);changed[sky]=True;changed[changes]=True
    for key,value in original.items():
        if len(value)==N:equal(value[~changed],bg[key][~changed])
    report={'status':'candidate_not_accepted','sky_suppressed_gaussians':int(sky.sum()),'road_supported_gaussians':int(road.sum()),'road_flattened_gaussians':len(changes),'road_height_abs_p90_before_m':float(np.quantile(abs(height[road]),.9)),'all_traffic_camera_affine_frozen':True,'all_other_background_rows_exact':True,'sky_source':'real high-confidence sky RGB, original affine removed','road_source':'real-observation semantic consensus plus validated LiDAR plane','scope':'geometry and native environment texture, no rendered-frame postprocessing'}
    (a.output/'regularization.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
