"""Optional static LiDAR plane priors for a fixed Gaussian layout."""
import numpy as np
import torch


class LidarSurfacePrior:
    def __init__(self,path,device,samples=20000):
        arrays=np.load(path)
        self.indices=torch.from_numpy(np.flatnonzero(arrays['valid'])).long().to(device)
        self.anchors=torch.from_numpy(arrays['anchors'][arrays['valid']]).float().to(device)
        self.normals=torch.from_numpy(arrays['normals'][arrays['valid']]).float().to(device)
        self.samples=samples
        if not len(self.indices):raise ValueError('No supported static surfaces')

    def losses(self,model):
        from models.gaussians.basics import quat_to_rotmat
        sample=torch.randint(len(self.indices),(min(self.samples,len(self.indices)),),device=self.indices.device)
        ids=self.indices[sample];n=self.normals[sample]
        offset=model._means[ids]-self.anchors[sample]
        normal_distance=(offset*n).sum(-1)
        R=quat_to_rotmat(model.quat_act(model._quats[ids]))
        local_normal=(R.transpose(-1,-2)@n[...,None])[...,0]
        sigma=torch.sqrt((local_normal.square()*model._scales[ids].exp().square()).sum(-1).clamp_min(1e-12))
        return {'lidar_surface_distance':.15*normal_distance.abs().mean(),
                'lidar_local_support':.03*torch.relu(offset.norm(dim=-1)-.3).mean(),
                'lidar_surface_thickness':.1*torch.relu(sigma-.03).mean()}
