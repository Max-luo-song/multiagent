"""Verify the native vehicle panel against frozen assets and its real source RGB."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('base','candidate','reference','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 import cv2,numpy as np,torch,yaml
 from PIL import Image
 from models.gaussians.basics import quat_to_rotmat
 from models.gaussians.support import confine_gaussian_support
 from models.modules import AffineTransform
 from scripts.pipeline.merge_nexus_vehicle_asset import equal
 old=torch.load(a.base,map_location='cpu',weights_only=False)['models'];new=torch.load(a.candidate,map_location='cpu',weights_only=False)['models'];report=json.loads((a.candidate.parent/'surface.json').read_text());cfg=yaml.safe_load((a.base.parent/'config.yaml').read_text());m=report['body_model_index'];f=report['frame'];scene=Path(cfg['data']['data_root'])/str(cfg['data']['scene_idx']);N=len(old['RigidNodes']['_means']);w=new['RigidNodes'];ids=torch.arange(N,len(w['_means']));assert len(ids)==report['native_panel_gaussians']
 for k,v in old.items():
  if k!='RigidNodes':equal(v,new[k])
 point_keys={'_means','_scales','_quats','_features_dc','_features_rest','_opacities','points_ids'}
 for k,v in old['RigidNodes'].items():
  if k=='_opacities':
   changed=(v!=w[k][:N]).reshape(-1);assert (old['RigidNodes']['points_ids'][changed,0]==m).all();assert (w[k][:N][changed]==-15).all()
  elif k in point_keys:equal(v,w[k][:N])
  else:equal(v,w[k])
 assert (w['points_ids'][ids]==m).all() and torch.count_nonzero(w['_features_rest'][ids])==0
 B=np.eye(4);B[:3,:3]=quat_to_rotmat(torch.nn.functional.normalize(w['instances_quats'][f,m][None],dim=-1))[0].numpy();B[:3,3]=w['instances_trans'][f,m].numpy()
 S=np.linalg.inv(json.loads((scene/'nexus_scene.json').read_text())['frames'][0]['ego_to_world']);rec=next(i for r in json.loads((a.reference/'run.json').read_text())['frames'] if r['frame_idx']==f for i in r['images'] if i['camera_id']==0);C=np.linalg.inv(B)@S@np.asarray(rec['scene_metadata']['camera_world']);K=np.array(rec['scene_metadata']['intrinsics']);K[:2]*=2
 mu=w['_means'][ids].numpy();q=(mu-C[:3,3])@C[:3,:3];uv=q[:,:2]/q[:,2,None]@K[:2,:2].T+K[:2,2];pixels=np.rint(uv).astype(int);pixel_error=float(abs(uv-pixels).max());assert pixel_error<.001;assert len(np.unique(pixels,axis=0))==len(pixels)
 intr=np.loadtxt(scene/'intrinsics/0.txt');rgb=cv2.undistort(np.array(Image.open(scene/'images'/f'{f:03d}_0.jpg').convert('RGB')),K,intr[4:]).astype(np.float32)/255
 af=old['Affine'];affine=AffineTransform('Affine',len(af['embedding.weight']),device=torch.device('cpu'));affine.load_state_dict(af)
 with torch.no_grad():A=affine({'img_idx':torch.tensor([f*3])})[0].numpy()
 expected=np.clip((rgb[pixels[:,1],pixels[:,0]]-A[:,3])@np.linalg.inv(A[:,:3]).T,.001,.999);actual=w['_features_dc'][ids].numpy()*.28209479177387814+.5;rgb_error=float(abs(expected-actual).max());assert rgb_error<1e-6
 fit=report['coherent_surface_fit'];surface_residual=None
 if fit:
  y,z=mu[:,1],mu[:,2];D=np.column_stack((np.ones_like(y),y,z,y*y,y*z,z*z));surface_residual=float(abs(mu[:,0]-D@np.array(fit['quadratic_x_of_yz_coefficients'])).max());assert surface_residual<1e-5
 scales=w['_scales'][ids].exp();rot=quat_to_rotmat(torch.nn.functional.normalize(w['_quats'][ids],dim=-1));_,valid=confine_gaussian_support(w['_means'][ids],scales,rot,w['instances_size'][m].expand(len(ids),-1),sigma=3.)
 result={'status':'passed','native_panel_gaussians':len(ids),'source_pixel_roundtrip_max_error_px':pixel_error,'source_real_rgb_max_error':rgb_error,'continuous_surface_equation_max_residual_m':surface_residual,'native_centres_inside_original_vehicle_support_fraction':float(valid.float().mean()),'background_sky_camera_affine_other_vehicles_bitwise_preserved':True,'original_vehicle_instance_poses_sizes_visibility_bitwise_preserved':True,'original_gaussian_rows_only_bus_opacity_changed':True,'scope':'construction and frozen-parameter audit; not unseen-view ground truth'}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
