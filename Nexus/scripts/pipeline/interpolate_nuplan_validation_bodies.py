"""Interpolate held-out absolute body poses between fitted neighboring times.

Interpolating only the fitted residual retains the original detector's
frame-specific box jitter at validation times. For two valid neighbors,
interpolate their fitted physical poses instead, without reading validation RGB.
"""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np,torch,yaml
    from scipy.spatial.transform import Rotation,Slerp
    from nexus_3dgs.geometry import quaternion_matrix,matrix_quaternion
    cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text());scene=Path(cfg['data']['data_root'])/cfg['data']['scene_idx'];meta=json.loads((scene/'nexus_scene.json').read_text());times=np.array([next(c['timestamp_us'] for c in r['images'] if c['camera_id']==0) for r in meta['frames']],np.int64)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before_sha=sha(a.checkpoint)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);rigid=state['models']['RigidNodes'];n=len(rigid['instances_trans']);held=np.arange(10,n,10);rows=[]
    assert len(times)==n
    for f in held:
        if f+1>=n:
            rows.append({'frame':int(f),'policy':'keep source residual fallback at sequence endpoint'});continue
        weight=float((times[f]-times[f-1])/(times[f+1]-times[f-1]));count=0;maximum=0.
        for m in range(rigid['instances_trans'].shape[1]):
            if not all(bool(rigid['instances_fv'][g,m]) for g in [f-1,f,f+1]):continue
            initial=rigid['instances_trans'][f,m].clone();rigid['instances_trans'][f,m]=(1-weight)*rigid['instances_trans'][f-1,m]+weight*rigid['instances_trans'][f+1,m]
            rotations=Rotation.from_matrix([quaternion_matrix(rigid['instances_quats'][g,m].numpy()) for g in [f-1,f+1]])
            R=Slerp([0.,1.],rotations)([weight]).as_matrix()[0];rigid['instances_quats'][f,m]=torch.tensor(matrix_quaternion(R),dtype=rigid['instances_quats'].dtype)
            maximum=max(maximum,float(torch.linalg.vector_norm(rigid['instances_trans'][f,m]-initial)));count+=1
        rows.append({'frame':int(f),'bodies_interpolated':count,'maximum_change_from_source_validation_position_m':maximum})
    output=a.output/'checkpoint_final.pth';torch.save(state,output);shutil.copyfile(a.checkpoint.parent/'config.yaml',a.output/'config.yaml')
    if (a.checkpoint.parent/'vehicle_refit_audit.json').is_file():shutil.copyfile(a.checkpoint.parent/'vehicle_refit_audit.json',a.output/'source_training_audit.json')
    reloaded=torch.load(output,map_location='cpu',weights_only=False);source=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    train=np.setdiff1d(np.arange(n),held)
    for name in source['models']:
        for key,value in source['models'][name].items():
            other=reloaded['models'][name][key]
            if name=='RigidNodes' and key in ['instances_trans','instances_quats']:assert torch.equal(value[train],other[train])
            else:assert torch.equal(value,other),(name,key)
    assert sha(a.checkpoint)==before_sha
    report={'status':'passed','source_checkpoint':str(a.checkpoint.resolve()),'source_sha256':before_sha,'checkpoint_sha256':sha(output),
            'policy':'absolute translation interpolation and rotation SLERP using front exposure timestamps; only held-out times with two visible fitted neighbors; source fallback elsewhere',
            'validation_rgb_used':False,'training_frame_body_poses_and_all_other_model_parameters_bit_exact':True,'source_checkpoint_unchanged':True,'rows':rows}
    (a.output/'heldout_body_pose_interpolation.json').write_text(json.dumps(report,indent=2)+'\n');print('ABSOLUTE_BODY_INTERPOLATION_COMPLETE '+str(output),flush=True)


if __name__=='__main__':main()
