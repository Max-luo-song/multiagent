"""Read external standard 3DGS PLYs without mutating source assets/models."""
import hashlib
from pathlib import Path
import numpy as np


class GaussianVehicle:
    def __init__(self,path,target_lw,device='cuda'):
        import torch
        from plyfile import PlyData
        from models.gaussians.basics import quat_to_rotmat
        from models.gaussians.support import confine_gaussian_support
        path=Path(path);v=PlyData.read(path)['vertex'].data
        xyz=np.stack([v[k] for k in ('x','y','z')],-1)
        lower=xyz.min(0);upper=xyz.max(0);center=(lower+upper)/2;extent=upper-lower
        # Preserve the supplied vehicle proportions; fit inside the original
        # prediction footprint, including 3-sigma native Gaussian support.
        scale=float(min(np.asarray(target_lw)/extent[:2])*.97)
        self.size=extent*scale
        self.means=torch.tensor((xyz-center)*scale,dtype=torch.float32,device=device)
        self.quats=torch.nn.functional.normalize(torch.tensor(np.stack([v[f'rot_{i}'] for i in range(4)],-1),device=device),dim=-1)
        self.scales=torch.tensor(np.stack([v[f'scale_{i}'] for i in range(3)],-1),device=device).exp()*scale
        self.opacity=torch.tensor(v['opacity'].copy(),device=device).sigmoid()[:,None]
        self.scales,valid=confine_gaussian_support(self.means,self.scales,quat_to_rotmat(self.quats),self.means.new_tensor(self.size).expand(len(v),-1),sigma=3.,margin=0.)
        self.opacity=self.opacity*valid[:,None]
        dc=torch.tensor(np.stack([v[f'f_dc_{i}'] for i in range(3)],-1),device=device)
        # Supplied assets use the standard channel-major PLY layout, which is
        # different from this repo's interleaved instance exporter.
        rest=torch.tensor(np.stack([v[f'f_rest_{i}'] for i in range(45)],-1),device=device).reshape(-1,3,15).transpose(1,2)
        self.sh=torch.cat([dc[:,None],rest],1)
        self.info={'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'gaussians':len(v),'uniform_scale':scale,'source_center':center.tolist(),'raw_extent_lwh':extent.tolist(),'render_extent_lwh':self.size.tolist(),'SH_layout':'standard channel-major','scale_encoding':'log standard deviation','opacity_encoding':'logit','quat_order':'wxyz','native_support_sigma':3.,'points_omitted_outside_box':int((~valid).sum())}

    def gaussians(self,pose,camera):
        import torch
        from gsplat.cuda._wrapper import spherical_harmonics
        from nexus_3dgs.geometry import matrix_quaternion
        R=self.means.new_tensor(pose[:3,:3]);t=self.means.new_tensor(pose[:3,3])
        xyz=self.means@R.T+t
        local_direction=(xyz-camera)@R
        local_direction=local_direction/local_direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        color=(spherical_harmonics(3,local_direction,self.sh)+.5).clamp(0,1)
        q=self.quats.new_tensor(matrix_quaternion(pose[:3,:3])).expand_as(self.quats)
        a,b=q,self.quats;aw,av=a[:,:1],a[:,1:];bw,bv=b[:,:1],b[:,1:]
        rotated=torch.cat([aw*bw-(av*bv).sum(-1,keepdim=True),aw*bv+bw*av+torch.cross(av,bv,dim=-1)],-1)
        return {'_means':xyz,'_quats':rotated,'_scales':self.scales,'_opacities':self.opacity,'_rgbs':color}
