"""Build separate bounded body-pose and Nexus-path stability candidates."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','bundle','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np,torch
    from nexus_3dgs.geometry import quaternion_matrix,matrix_quaternion
    from nexus_3dgs.motion_stability import regularize_observed_bodies,bounded_curve_correction
    from nexus_3dgs.bundle import load_bundle,save_bundle
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();source_sha=sha(a.checkpoint)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);rigid=state['models']['RigidNodes'];N,M=rigid['instances_trans'].shape[:2]
    P=np.broadcast_to(np.eye(4),(N,M,4,4)).copy();P[:,:,:3,3]=rigid['instances_trans'].numpy()
    for f in range(N):
        for m in range(M):P[f,m,:3,:3]=quaternion_matrix(rigid['instances_quats'][f,m].numpy())
    corrected,records=regularize_observed_bodies(P,rigid['instances_fv'].numpy())
    rigid['instances_trans']=torch.tensor(corrected[:,:,:3,3],dtype=rigid['instances_trans'].dtype)
    for f in range(N):
        for m in range(M):
            if not np.array_equal(corrected[f,m,:3,:3],P[f,m,:3,:3]):rigid['instances_quats'][f,m]=torch.tensor(matrix_quaternion(corrected[f,m,:3,:3]),dtype=rigid['instances_quats'].dtype)
    model=a.output/'observed_body_stable';model.mkdir();torch.save(state,model/'checkpoint_final.pth');shutil.copyfile(a.checkpoint.parent/'config.yaml',model/'config.yaml')
    after=torch.load(model/'checkpoint_final.pth',map_location='cpu',weights_only=False);before=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    for name,values in before['models'].items():
        for key,value in values.items():
            if name=='RigidNodes' and key in ['instances_trans','instances_quats']:continue
            assert torch.equal(value,after['models'][name][key]),(name,key)
    assert sha(a.checkpoint)==source_sha
    report={'status':'passed','source_checkpoint_sha256':source_sha,'all_gaussian_assets_background_sky_camera_exposure_and_visibility_bit_exact':True,'policy':'per-visible-segment bounded 3D body rotation and lateral refinement; height and reference-longitudinal motion preserved; endpoints locked','maximum_lateral_change_m':.05,'maximum_rotation_change_deg':.75,'tracks':records}
    (model/'stability_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    arrays,manifest=load_bundle(a.bundle);original=arrays['predicted_poses_world'];new=original.copy();bus=next(i for i,t in enumerate(arrays['track_tokens']) if str(t)=='441830199c6752a5');current=manifest['current_index']
    if manifest.get('conditioned_slot')==int(arrays['nexus_slots'][bus]):raise ValueError('Cannot modify a fully conditioned bus track')
    assert arrays['valid'][:,bus].all();yaw=np.unwrap(np.arctan2(original[:,bus,1,0],original[:,bus,0,0]));reference=float(np.median(yaw[current:]));forward=np.array([np.cos(reference),np.sin(reference)]);side=np.array([-forward[1],forward[0]])
    local=original[:,bus,:2,3]-original[current,bus,:2,3];s=local@forward;lateral=local@side;fixed=np.zeros(len(original),bool);fixed[:current+1]=True;fixed[-1]=True
    targets=np.diff(s)*np.tan((yaw[:-1]+yaw[1:])/2-reference)
    refined=bounded_curve_correction(lateral,2.,.35,fixed,targets,80.)
    new[:,bus,:2,3]+=(refined-lateral)[:,None]*side
    other=np.arange(len(arrays['track_tokens']))!=bus
    np.testing.assert_array_equal(new[:current+1],original[:current+1]);np.testing.assert_array_equal(new[-1],original[-1]);np.testing.assert_array_equal(new[:,other],original[:,other])
    np.testing.assert_array_equal(new[:,:,:3,:3],original[:,:,:3,:3]);np.testing.assert_array_equal(new[:,:,2,3],original[:,:,2,3]);np.testing.assert_allclose((new[:,bus,:2,3]-original[:,bus,:2,3])@forward,0,atol=1e-9)
    audit={'source_bundle':str(a.bundle.resolve()),'source_bundle_sha256':manifest['bundle_sha256'],'method':'bus lateral displacement fit with heading-consistent lateral-step prior and bounded corrections; reference-longitudinal progress preserved exactly','track_token':str(arrays['track_tokens'][bus]),'max_position_change_m':float(np.max(np.linalg.norm(new[:,bus,:2,3]-original[:,bus,:2,3],axis=-1))),'positions_given_history_and_goal_preserved_exactly':True,'ego_other_actors_all_headings_and_heights_preserved_exactly':True,'uses_logged_future_motion':False,'nexus_weights_changed':False,'first_future_lateral_step_before_m':float(lateral[current+1]-lateral[current]),'first_future_lateral_step_after_m':float(refined[current+1]-refined[current]),'lateral_step_residual_rms_before_m':float(np.sqrt(np.mean((np.diff(lateral)[current:]-targets[current:])**2))),'lateral_step_residual_rms_after_m':float(np.sqrt(np.mean((np.diff(refined)[current:]-targets[current:])**2)))}
    values={k:np.array(v,copy=True) for k,v in arrays.items()};values['predicted_poses_world']=new;save_bundle(a.output/'heading_and_bus_path',values,{**manifest,'path_stabilization':audit});(a.output/'path_stabilization.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
