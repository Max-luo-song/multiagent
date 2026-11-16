"""Lift image samples and their pixel footprints onto a measured plane."""
import numpy as np
from scipy.spatial.transform import Rotation


def lift_pixels(uv, K, camera, plane, pixel_std=.7, normal_std=.001):
    """Return surface points, log scales and wxyz quaternions.

    camera is camera-to-scene; plane is [nx,ny,nz,d]. Pixel covariance
    is preserved to first order, rather than choosing arbitrary world scales.
    Rays parallel to, or pointing away from, the surface are invalid.
    """
    uv = np.asarray(uv, dtype=np.float64)
    plane = np.asarray(plane, dtype=np.float64)
    plane = plane / np.linalg.norm(plane[:3])
    n, d = plane[:3], plane[3]
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(K).T
    rays_scene = rays @ camera[:3, :3].T
    denominator = rays_scene @ n
    depth = -(camera[:3, 3] @ n + d) / np.where(np.abs(denominator) > 1e-9, denominator, np.nan)
    valid = np.isfinite(depth) & (depth > .1)
    depth = np.where(valid, depth, 1.)
    points = camera[:3, 3] + rays_scene * depth[:, None]
    tangent = np.eye(3)[np.argmin(np.abs(n))]
    tangent = tangent - n * (tangent @ n)
    tangent /= np.linalg.norm(tangent)
    T = np.column_stack((tangent, np.cross(n, tangent)))
    xyz = (points - camera[:3, 3]) @ camera[:3, :3]
    J = np.zeros((len(points), 2, 3))
    J[:, 0, 0] = K[0, 0] / xyz[:, 2]
    J[:, 1, 1] = K[1, 1] / xyz[:, 2]
    J[:, 0, 2] = -K[0, 0] * xyz[:, 0] / xyz[:, 2]**2
    J[:, 1, 2] = -K[1, 1] * xyz[:, 1] / xyz[:, 2]**2
    A = J @ camera[:3, :3].T @ T
    inverse = np.linalg.pinv(A)
    covariance = pixel_std**2 * (inverse @ inverse.transpose(0, 2, 1))
    value, vector = np.linalg.eigh(covariance)
    basis = T @ vector
    rotations = np.concatenate((basis, np.broadcast_to(n, (len(points), 3))[..., None]), axis=2)
    negative = np.linalg.det(rotations) < 0
    rotations[negative, :, 0] *= -1
    scales = np.column_stack((np.sqrt(np.maximum(value, 1e-12)), np.full(len(points), normal_std)))
    quat = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
    return points, np.log(scales), quat, valid


def rectify_gaussians(means, logs, quats, K, camera, depths, normals, normal_std=.005):
    """Change depth/thickness while preserving source pixel centres/covariance.

    Unlike a Euclidean snap, this does not shift the observed texture in its
    reference image. A thin tangent covariance carries the original footprint.
    Input/output quaternion convention is wxyz.
    """
    means,logs=np.asarray(means,float),np.asarray(logs,float)
    xyz=(means-camera[:3,3])@camera[:3,:3]
    target=xyz*(depths/xyz[:,2])[:,None]
    points=target@camera[:3,:3].T+camera[:3,3]
    R=Rotation.from_quat(np.asarray(quats)[:,[1,2,3,0]]).as_matrix()
    covariance=(R*np.exp(2*logs)[:,None,:])@R.transpose(0,2,1)
    def jacobian(p):
        J=np.zeros((len(p),2,3));J[:,0,0]=K[0,0]/p[:,2];J[:,1,1]=K[1,1]/p[:,2]
        J[:,0,2]=-K[0,0]*p[:,0]/p[:,2]**2;J[:,1,2]=-K[1,1]*p[:,1]/p[:,2]**2
        return J@camera[:3,:3].T
    J0,J1=jacobian(xyz),jacobian(target)
    footprint=J0@covariance@J0.transpose(0,2,1)
    n=np.asarray(normals,float);n=n/np.linalg.norm(n,axis=1,keepdims=True)
    tangent=np.eye(3)[np.argmin(np.abs(n),axis=1)]
    tangent-=n*(tangent*n).sum(-1,keepdims=True);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    T=np.stack((tangent,np.cross(n,tangent)),axis=-1)
    A=J1@T;inverse=np.linalg.inv(A)
    jn=(J1@n[...,None])[...,0]
    sigma=np.minimum(normal_std,.1*np.sqrt(np.maximum(np.linalg.eigvalsh(footprint)[:,0],1e-12))/np.maximum(np.linalg.norm(jn,axis=-1),1e-12))
    residual=footprint-sigma[:,None,None]**2*jn[:,:,None]*jn[:,None,:]
    planar=inverse@residual@inverse.transpose(0,2,1)
    eigenvalue,eigenvector=np.linalg.eigh(planar)
    basis=T@eigenvector
    rotations=np.concatenate((basis,n[...,None]),axis=2)
    negative=np.linalg.det(rotations)<0;rotations[negative,:,0]*=-1
    scales=np.column_stack((np.sqrt(np.maximum(eigenvalue,1e-12)),sigma))
    quaternion=Rotation.from_matrix(rotations).as_quat()[:,[3,0,1,2]]
    return points,np.log(scales),quaternion
