"""Isolate moving-vehicle view-dependent color without changing geometry."""
import argparse,copy,json,shutil,sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','render','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np,torch
    from models.gaussians.basics import spherical_harmonics,quat_to_rotmat
    from nexus_3dgs.runtime import TrajectoryController
    from nexus_3dgs.grounding import GroundedRegistration
    from scripts.pipeline.merge_nexus_vehicle_asset import equal
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);weights=state['models']['RigidNodes']
    run=json.loads((a.render/'run.json').read_text());ground=run['ground_registration']
    rigid=SimpleNamespace(instance_ids=weights['dataset_instance_ids'],instances_trans=weights['instances_trans'],instances_quats=weights['instances_quats'],instances_fv=weights['instances_fv'])
    rigid.num_frames,rigid.num_instances=rigid.instances_fv.shape
    controller=TrajectoryController(run['bundle'],Path(ground['ground_model']['scene']),rigid)
    # Reuse the exact recorded mounting calibration, including the original
    # version's choice of temporal calibration samples.
    ground_path=a.output/'ground.json';ground_path.write_text(json.dumps(ground['ground_model'],indent=2)+'\n')
    registration=GroundedRegistration(controller,ground_path,stabilize_all=ground['stabilize_all_rigid_instances'])
    registration.object_mounts={int(k):np.array(v) for k,v in ground['object_mounts'].items()}
    registration.flat_mounts={int(k):np.array(v) for k,v in ground.get('flat_track_mounts',{}).items()}
    controller.grounding=registration
    records={f['frame_idx']:next(i for i in f['images'] if i['camera_id']==0) for f in run['frames']}
    references={}
    for f,image in sorted(records.items()):
        for actor in image['scene_metadata']['actors']:
            m=actor['model_index']
            if m>=0 and actor['valid'] and actor['center_visible'] and m not in references:references[m]=f
    ids=weights['points_ids'].reshape(-1);means=weights['_means'];directions={};reference_dirs=torch.zeros_like(means)
    def direction_at(f):
        image=records[f]
        with controller.edit_frame(f,image['timestamp_us']):
            rotation=quat_to_rotmat(torch.nn.functional.normalize(rigid.instances_quats[f],dim=-1))
            world=torch.bmm(rotation[ids].float(),means[:,:,None].float())[:,:,0]+rigid.instances_trans[f,ids]
            C=controller.scene_from_world@np.array(image['scene_metadata']['camera_world'])
            direction=world-torch.tensor(C[:3,3],dtype=world.dtype)
            return torch.nn.functional.normalize(direction,dim=-1)
    for f in sorted(set(references.values())):
        d=direction_at(f)
        for m,ref in references.items():
            if ref==f:reference_dirs[ids==m]=d[ids==m]
    selected=torch.zeros(len(ids),dtype=torch.bool)
    for m in references:selected |= ids==m
    # Values for unselected actors are never modified or interpreted.
    reference_dirs[~selected]=torch.tensor([0.,0.,1.])
    sh=torch.cat((weights['_features_dc'][:,None,:],weights['_features_rest']),dim=1).cuda()
    d0=reference_dirs.cuda()
    with torch.no_grad():canonical=spherical_harmonics(3,d0,sh)
    outputs={};preservation={}
    for degree in (0,1,2):
        candidate=copy.deepcopy(weights)
        with torch.no_grad():delta=(canonical-spherical_harmonics(degree,d0,sh)).cpu()/.28209479177387814
        candidate['_features_dc'][selected]+=delta[selected]
        first_unused=(degree+1)**2-1
        candidate['_features_rest'][selected,first_unused:]=0
        new_sh=torch.cat((candidate['_features_dc'][:,None,:],candidate['_features_rest']),dim=1).cuda()
        with torch.no_grad():error=(spherical_harmonics(3,d0,new_sh)-canonical).abs()[selected].max().item()
        assert error<3e-6,error
        preservation[degree]=error;outputs[degree]=new_sh
        directory=a.output/f'degree_{degree}';directory.mkdir()
        state['models']['RigidNodes']=candidate;torch.save(state,directory/'checkpoint_final.pth');shutil.copyfile(a.checkpoint.parent/'config.yaml',directory/'config.yaml')
        for key in weights:
            if key not in ('_features_dc','_features_rest'):equal(weights[key],candidate[key])
    per_actor={};sample_frames=[f for f in [20,55,70,85,99] if f in records]
    for f in sample_frames:
        d=direction_at(f).cuda()
        with torch.no_grad():base=(spherical_harmonics(3,d,sh)+.5).clamp(0,1)
        for m,ref in references.items():
            mask=(ids==m).cuda();visible=bool(next(x for x in records[f]['scene_metadata']['actors'] if x['model_index']==m)['valid'])
            if not visible:continue
            row=per_actor.setdefault(str(m),{'reference_frame':ref,'frames':[]});entry={'frame':f}
            for degree in (0,1,2,3):
                coeff=sh if degree==3 else outputs[degree]
                with torch.no_grad():current=(spherical_harmonics(3,d,coeff)+.5).clamp(0,1);ref_rgb=(canonical+.5).clamp(0,1)
                error=(current[mask]-ref_rgb[mask]).abs().mean(-1)
                entry[str(degree)]={'mean_color_change':error.mean().item(),'p90_color_change':torch.quantile(error,.9).item()}
            row['frames'].append(entry)
    state['models']['RigidNodes']=weights
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    for name,value in source['models'].items():
        if name!='RigidNodes':equal(value,state['models'][name])
    report={'status':'diagnostic_candidates','source':str(a.checkpoint.resolve()),'source_bundle_sha256':run['bundle_sha256'],
            'modified_actors':references,'canonical_raw_SH_max_error':preservation,'background_camera_geometry_poses_opacity_unchanged':True,
            'scope':'degree reduction with per-Gaussian DC compensation at each actor first visible generated-camera pose; color variation is a diagnostic, not pixel ghosting ground truth',
            'per_actor':per_actor}
    (a.output/'appearance_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_actor'}),flush=True)


if __name__=='__main__':main()
