"""Replay saved physical render states without rerunning or changing Nexus.

Older actor records contain position/yaw and the fixed mounting calibration.
Together with the saved ground normal these determine the full rigid rotation.
"""
import json
from pathlib import Path
import numpy as np
from .geometry import matrix_quaternion


def road_rotation(yaw, normal):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.])
    forward -= n * (n @ forward)
    forward /= np.linalg.norm(forward)
    return np.column_stack((forward, np.cross(n, forward), n))


def recover_rotation(body_yaw, mounting, normal, flat=False):
    """Invert the recorded body yaw under the exact saved mounting model."""
    Rm = np.asarray(mounting)[:3, :3]
    n = np.array([0., 0., 1.]) if flat else np.asarray(normal)
    angle = lambda x: np.arctan2(np.sin(x), np.cos(x))
    def heading(theta):
        R = road_rotation(theta, n) @ Rm
        return np.arctan2(R[1, 0], R[0, 0])
    theta = body_yaw - np.arctan2(Rm[1, 0], Rm[0, 0])
    for _ in range(5):
        derivative = angle(heading(theta+1e-5)-heading(theta-1e-5))/2e-5
        if abs(derivative) < .1:
            raise ValueError('Mounting yaw cannot be inverted reliably')
        theta -= angle(heading(theta)-body_yaw)/derivative
    R = road_rotation(theta, n) @ Rm
    if abs(angle(heading(theta)-body_yaw)) > 1e-8:
        raise ValueError('Saved body yaw recovery failed')
    return R


class RecordedStates:
    def __init__(self, path, original_scene):
        self.path = Path(path)
        self.run = json.loads(self.path.read_text())
        if self.run['status'] != 'complete':
            raise ValueError('Cannot replay an incomplete render')
        self.scene = json.loads((Path(original_scene)/'nexus_scene.json').read_text())
        self.S = np.linalg.inv(self.scene['frames'][0]['ego_to_world'])
        self.records = {f['frame_idx']: next(i for i in f['images'] if i['camera_id']==0)
                        for f in self.run['frames']}
        g = self.run.get('ground_registration')
        if g:
            gm = g['ground_model']
            ground_S = np.linalg.inv(gm.get('world_from_scene', self.scene['frames'][0]['ego_to_world']))
            a,b,c = gm['coefficients_z_ax_by_c']
            world_plane = ground_S.T @ np.array([-a,-b,1.,-c])
            self.normal = world_plane[:3]/np.linalg.norm(world_plane[:3])
            self.mounts = g['object_mounts']
            self.flat = g.get('flat_track_mounts', {})
        else:
            self.normal = self.mounts = self.flat = None

    def set_actors(self, rigid, frame, destination_frame):
        import torch
        if self.mounts is None:
            raise ValueError('Full actor replay requires saved ground/mounting calibration')
        meta = self.records[frame]['scene_metadata']
        for actor in meta['actors']:
            m = actor['model_index']
            if m < 0:
                continue
            rigid.instances_fv[destination_frame,m] = actor['valid']
            if not actor['valid']:
                continue
            key = str(m)
            R = recover_rotation(actor['yaw_world_rad'],
                self.flat.get(key,self.mounts[key]), self.normal, flat=key in self.flat)
            world = np.eye(4)
            world[:3,:3] = R
            world[:3,3] = actor['center_world']
            pose = self.S @ world
            rigid.instances_trans[destination_frame,m].copy_(torch.as_tensor(pose[:3,3],
                device=rigid.instances_trans.device,dtype=rigid.instances_trans.dtype))
            rigid.instances_quats[destination_frame,m].copy_(torch.as_tensor(matrix_quaternion(pose[:3,:3]),
                device=rigid.instances_quats.device,dtype=rigid.instances_quats.dtype))


def camera_inputs(C, K, frame, height, width, num_frames=101, num_cameras=3):
    import torch
    from datasets.base.pixel_source import get_rays
    device = C.device
    y,x = torch.meshgrid(torch.arange(height,device=device),torch.arange(width,device=device),indexing='ij')
    origins,dirs,norms = get_rays(x.flatten(),y.flatten(),C,K)
    images = {'origins':origins.reshape(height,width,3),'viewdirs':dirs.reshape(height,width,3),
              'direction_norm':norms.reshape(height,width,1),
              'normed_time':torch.tensor([frame/(num_frames-1)],device=device),
              'img_idx':torch.tensor([frame*num_cameras],device=device,dtype=torch.long),
              'frame_idx':torch.tensor([frame],device=device,dtype=torch.long),
              'pixel_coords':torch.stack((y/height,x/width),dim=-1)}
    camera = {'camera_to_world':C,'intrinsics':K,'height':height,'width':width}
    return images,camera


def load_trainer(checkpoint, scene_aabb=None):
    import torch
    from omegaconf import OmegaConf
    from utils.misc import import_str
    checkpoint = Path(checkpoint)
    cfg = OmegaConf.load(checkpoint.parent/'config.yaml')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    num_frames=state['models']['RigidNodes']['instances_trans'].shape[0]
    num_images=state['models']['Affine']['embedding.weight'].shape[0]
    assert num_images==num_frames*len(cfg.data.pixel_source.cameras)
    # This box matches the source scene's logged LiDAR AABB. It initializes
    # model scale only; all rendered Gaussian geometry comes from the weights.
    aabb = torch.tensor(scene_aabb if scene_aabb is not None else [[-23.1217,-36.1490,-.4956],[128.6485,34.6414,20.]],device='cuda')
    trainer = import_str(cfg.trainer.type)(**cfg.trainer,num_timesteps=num_frames,
        model_config=cfg.model,num_train_images=num_images,num_full_images=num_images,
        test_set_indices=[],scene_aabb=aabb,device=torch.device('cuda'))
    trainer.load_state_dict(state,load_only_model=True,strict=True)
    trainer.set_eval()
    trainer.test_set_indices = []
    return trainer,cfg


def change_color_basis(rgb, source, target):
    """Keep displayed opaque colors when the background exposure model changes."""
    import torch
    displayed=rgb@source[:,:3].T+source[:,3]
    return torch.linalg.solve(target[:,:3],(displayed-target[:,3]).T).T


def preserve_actor_appearance(trainer, checkpoint):
    """Convert fixed actor radiance into the current background exposure basis.

    Geometry, opacity, and stored actor SH coefficients are untouched. This is
    applied to native Gaussian radiance before ordinary depth compositing.
    """
    import torch
    from omegaconf import OmegaConf
    from models.modules import AffineTransform
    checkpoint=Path(checkpoint);cfg=OmegaConf.load(checkpoint.parent/'config.yaml')
    weights=torch.load(checkpoint,map_location='cpu',weights_only=False)['models']['Affine']
    original=AffineTransform('Affine',len(weights['embedding.weight']),**cfg.model.Affine.params,device=torch.device('cpu'))
    original.load_state_dict(weights)
    with torch.no_grad():
        values=original({'img_idx':torch.arange(len(weights['embedding.weight']))})
        if not torch.equal(values,values[:1].expand_as(values)):
            raise ValueError('Source actor appearance must have a verified constant transform')
        source=values[0].to(trainer.device)
    collect=trainer.collect_gaussians;rigid=trainer.models['RigidNodes'];get=rigid.get_gaussians
    def wrapped_collect(cam,image_ids):
        target=trainer.models['Affine']({'img_idx':image_ids.flatten()[0]})
        def wrapped_get(camera):
            gs=get(camera)
            if gs is not None:gs['_rgbs']=change_color_basis(gs['_rgbs'],source,target)
            return gs
        rigid.get_gaussians=wrapped_get
        try:return collect(cam,image_ids)
        finally:rigid.get_gaussians=get
    trainer.collect_gaussians=wrapped_collect


class ObservedSky:
    """Native distant-sky layer restricted to repeatedly observed empty rays.

Finite static Gaussians are excluded only in directions confirmed by at least
two conservative real sky masks, with no confident static contradiction.
Dynamic Gaussians still occlude the distant sky normally.
Uncertain rays use the original renderer unchanged.
"""
    def __init__(self, checkpoint, observations, device='cuda'):
        import torch
        from omegaconf import OmegaConf
        from models.modules import EnvLight
        import cv2
        checkpoint=Path(checkpoint);observations=Path(observations)
        cfg=OmegaConf.load(checkpoint.parent/'config.yaml')
        weights=torch.load(checkpoint,map_location='cpu',weights_only=False)['models']['Sky']
        self.sky=EnvLight('Sky',**cfg.model.Sky.params,device=torch.device(device)).to(device)
        self.sky.load_state_dict(weights)
        self.observations=[]
        for rec in json.loads((observations/'observations.json').read_text())['frames']:
            with np.load(observations/f'{rec["frame"]:03d}.npz') as d:
                # Feather inside the already eroded high-confidence core;
                # no certificate reaches the semantic uncertainty boundary.
                safe=np.minimum(cv2.distanceTransform(d['sky'].astype(np.uint8),cv2.DIST_L2,5)/8,1.)
                negative=(d['labels']!=2)&(d['confidence']>.95)&(~d['vehicle'])
                self.observations.append((torch.tensor(d['C'][:3,:3],dtype=torch.float32,device=device),
                    torch.tensor(d['K'],dtype=torch.float32,device=device),
                    torch.tensor(safe,dtype=torch.float32,device=device),
                    torch.tensor(negative,dtype=torch.float32,device=device)))

    def __call__(self, trainer, image, outputs):
        import torch
        rays=image['viewdirs'];H,W=rays.shape[:2];first=torch.zeros((H,W),device=rays.device);second=first.clone();contradiction=first.clone()
        for R,K,safe,negative in self.observations:
            q=rays@R;z=q[...,2];u=q[...,0]/z.clamp_min(1e-5)*K[0,0]+K[0,2]-.5;v=q[...,1]/z.clamp_min(1e-5)*K[1,1]+K[1,2]-.5
            h,w=safe.shape;grid=torch.stack((2*u/(w-1)-1,2*v/(h-1)-1),-1)[None]
            confidence=torch.nn.functional.grid_sample(safe[None,None],grid,mode='bilinear',padding_mode='zeros',align_corners=True)[0,0]
            confidence=confidence*(z>0)
            second=torch.maximum(second,torch.minimum(first,confidence));first=torch.maximum(first,confidence)
            veto=torch.nn.functional.grid_sample(negative[None,None],grid,mode='nearest',padding_mode='zeros',align_corners=True)[0,0]*(z>0)
            contradiction=torch.maximum(contradiction,veto)
        coverage=second*(contradiction<.5)
        light=self.sky(image);A=trainer.models['Affine']({'img_idx':image['img_idx'].flatten()[0]})
        # Dynamic_rgb already includes affine bias. Transform the transmitted
        # sky with the linear component only, so bias is applied exactly once.
        corrected=outputs['Dynamic_rgb']+(light*(1-outputs['Dynamic_opacity']))@A[:,:3].T
        return outputs['rgb']*(1-coverage[...,None])+corrected*coverage[...,None],coverage


def actor_volume_guard(rigid,frame,C,K,height,width):
    """Conservative current vehicle ownership, independent of learned opacity."""
    import cv2,torch
    from .geometry import quaternion_matrix
    mask=np.zeros((height,width),np.uint8);C=C.detach().cpu().numpy();K=K.detach().cpu().numpy()
    sign=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(sign[i]!=sign[j])==1]
    trans=rigid.instances_trans[frame].detach().cpu().numpy();quat=rigid.instances_quats[frame].detach().cpu().numpy();size=rigid.instances_size.detach().cpu().numpy();valid=rigid.instances_fv[frame].detach().cpu().numpy()
    for m in np.flatnonzero(valid):
        P=(sign*(size[m]/2+.08))@quaternion_matrix(quat[m]).T+trans[m];q=(P-C[:3,3])@C[:3,:3];points=list(q[q[:,2]>.1])
        for i,j in edges:
            if (q[i,2]>.1)!=(q[j,2]>.1):
                t=(.1-q[i,2])/(q[j,2]-q[i,2]);points.append(q[i]+t*(q[j]-q[i]))
        if len(points)<3:continue
        q=np.array(points);uv=q[:,:2]/q[:,2,None]@K[:2,:2].T+K[:2,2];uv=np.clip(uv,[-4*width,-4*height],[5*width,5*height]);cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(uv).astype(np.int32)),1)
    mask=cv2.dilate(mask,np.ones((5,5),np.uint8))>0
    return torch.tensor(mask,device=rigid.instances_trans.device)


class ObservedRoad:
    """Native Gaussian road surface with metric ownership and real occluders."""
    def __init__(self, directory, trainer, device='cuda', static_guard=None):
        import torch
        directory=Path(directory)
        with np.load(directory/'road_gaussians.npz') as values:
            self.values={k:torch.tensor(values[k],device=device) for k in values.files}
        self.static=torch.tensor(np.load(static_guard or directory/'static_occluders.npy'),device=device)
        with torch.no_grad():
            A=trainer.models['Affine']({'img_idx':torch.tensor(60,device=device)})
            other=trainer.models['Affine']({'img_idx':torch.tensor(297,device=device)})
            torch.testing.assert_close(A,other,atol=1e-7,rtol=0)
            self.values['colors']=((self.values['colors']-A[:,3])@torch.linalg.inv(A[:,:3]).T).clamp(0,1)

    def __call__(self, trainer, image, camera, outputs):
        import torch
        from gsplat import rasterization
        C,K=camera['camera_to_world'],camera['intrinsics'];H,W=camera['height'],camera['width'];g=self.values
        attributes=torch.cat((g['colors'],g['confidence'][:,None]),-1) if 'confidence' in g else g['colors']
        road,alpha,_=rasterization(means=g['means'],quats=g['quats'],scales=g['scales'],opacities=g['opacity'],colors=attributes,viewmats=torch.linalg.inv(C)[None],Ks=K[None],width=W,height=H,packed=False,near_plane=.1,far_plane=1e10,render_mode='RGB+ED',rasterize_mode='classic')
        rgb,depth=road[0,:,:,:3],road[0,:,:,-1];alpha=alpha[0,:,:,0]
        q=(self.static-C[:3,3])@C[:3,:3];z=q[:,2];u=torch.floor(q[:,0]/z.clamp_min(.1)*K[0,0]+K[0,2]).long();v=torch.floor(q[:,1]/z.clamp_min(.1)*K[1,1]+K[1,2]).long();inside=(z>.1)&(u>=0)&(u<W)&(v>=0)&(v<H);buf=torch.full((H*W,),float('inf'),device=C.device);buf.scatter_reduce_(0,v[inside]*W+u[inside],z[inside],reduce='amin',include_self=True);measured=-torch.nn.functional.max_pool2d(-buf.reshape(1,1,H,W),3,stride=1,padding=1)[0,0];occluded=measured<depth-.15
        confidence=road[0,:,:,3]/alpha.clamp_min(1e-6) if 'confidence' in g else torch.ones_like(alpha)
        owner=alpha*(~occluded)*confidence.clamp(0,1);color=rgb/alpha.clamp_min(1e-6)[...,None];A=trainer.models['Affine']({'img_idx':image['img_idx'].flatten()[0]})
        corrected=outputs['Dynamic_rgb']+(color*(1-outputs['Dynamic_opacity']))@A[:,:3].T
        final=outputs['rgb']*(1-owner[...,None])+corrected*owner[...,None]
        return final,owner


class MeasuredBackground:
    """Native measured static surfels; old field remains in unsupported rays."""
    def __init__(self,directory,trainer,device='cuda'):
        import torch
        directory=Path(directory)
        with np.load(directory/'static_gaussians.npz') as data:
            self.values={k:torch.tensor(data[k],device=device) for k in data.files}
        with torch.no_grad():
            A=trainer.models['Affine']({'img_idx':torch.tensor(60,device=device)})
            self.values['colors']=((self.values['colors']-A[:,3])@torch.linalg.inv(A[:,:3]).T).clamp(0,1)

    def __call__(self,trainer,image,camera,outputs):
        import torch
        from gsplat import rasterization
        C,K=camera['camera_to_world'],camera['intrinsics'];H,W=camera['height'],camera['width'];g=self.values
        rendered,alpha,_=rasterization(means=g['means'],quats=g['quats'],scales=g['scales'],opacities=g['opacity'],colors=g['colors'],viewmats=torch.linalg.inv(C)[None],Ks=K[None],width=W,height=H,packed=False,near_plane=.1,far_plane=1e10,render_mode='RGB+ED',rasterize_mode='classic')
        alpha=alpha[0];color=rendered[0,:,:,:3]/alpha.clamp_min(1e-6);depth=rendered[0,:,:,3:4];front=(outputs['Dynamic_depth']<=depth+.15).to(alpha.dtype);A=trainer.models['Affine']({'img_idx':image['img_idx'].flatten()[0]});dynamic=((outputs['Dynamic_rgb']-A[:,3])@torch.linalg.inv(A[:,:3]).T)*front;raw=dynamic+color*(1-outputs['Dynamic_opacity']*front);corrected=raw@A[:,:3].T+A[:,3]
        return outputs['rgb']*(1-alpha)+corrected*alpha,alpha[...,0]
