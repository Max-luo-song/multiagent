"""Replace one identified 3DGS vehicle, preserving all other scene assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
import yaml


def equal(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    else:
        assert a==b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--vehicle-source',type=Path,required=True)
    p.add_argument('--track-token',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((a.base.parent/'config.yaml').read_text())
    scene=Path(cfg['data']['data_root'])/str(cfg['data']['scene_idx'])
    info=json.loads((scene/'instances/instances_info.json').read_text())
    dataset_id=next(int(i) for i,r in info.items() if r['id']==a.track_token)
    base=torch.load(a.base,map_location='cpu',weights_only=False)
    source=torch.load(a.vehicle_source,map_location='cpu',weights_only=False)
    old=base['models']['RigidNodes'];new=source['models']['RigidNodes']
    equal(old['dataset_instance_ids'],new['dataset_instance_ids'])
    index=int(torch.nonzero(old['dataset_instance_ids']==dataset_id,as_tuple=True)[0].item())
    old_ids=old['points_ids'].reshape(-1);new_ids=new['points_ids'].reshape(-1)
    keep=old_ids!=index;take=new_ids==index
    assert keep.any() and take.any()
    merged={};point_keys=[]
    for key,value in old.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0]==len(old_ids):
            assert new[key].shape[0]==len(new_ids) and value.shape[1:]==new[key].shape[1:]
            merged[key]=torch.cat((value[keep],new[key][take]),dim=0).clone();point_keys.append(key)
        else:
            merged[key]=value.clone() if torch.is_tensor(value) else value
    for key in ('instances_trans','instances_quats'):
        merged[key][:,index]=new[key][:,index]
        mask=torch.arange(old[key].shape[1])!=index
        equal(merged[key][:,mask],old[key][:,mask])
    equal(old['instances_size'],new['instances_size'])
    equal(old['instances_fv'],new['instances_fv'])
    for key in point_keys:equal(merged[key][:int(keep.sum())],old[key][keep])
    base['models']['RigidNodes']=merged
    output=a.output/'checkpoint_final.pth';torch.save(base,output)
    shutil.copyfile(a.base.parent/'config.yaml',a.output/'config.yaml')
    # Reload the written result and verify preservation, not just the in-memory assembly.
    saved=torch.load(output,map_location='cpu',weights_only=False)
    original=torch.load(a.base,map_location='cpu',weights_only=False)
    for name,value in original['models'].items():
        if name!='RigidNodes':equal(value,saved['models'][name])
    for key in point_keys:equal(saved['models']['RigidNodes'][key][:int(keep.sum())],old[key][keep])
    for key in ('instances_trans','instances_quats'):
        mask=torch.arange(old[key].shape[1])!=index
        equal(saved['models']['RigidNodes'][key][:,mask],old[key][:,mask])
    for key in ('instances_fv','instances_size','dataset_instance_ids'):
        equal(saved['models']['RigidNodes'][key],old[key])
    report={'status':'passed','base_checkpoint':str(a.base.resolve()),'vehicle_source_checkpoint':str(a.vehicle_source.resolve()),
            'track_token':a.track_token,'model_index':index,'dataset_instance_id':dataset_id,
            'old_vehicle_points':int((~keep).sum()),'new_vehicle_points':int(take.sum()),
            'other_vehicle_points_bitwise_preserved':True,'other_vehicle_poses_bitwise_preserved':True,
            'background_sky_affine_and_cameras_bitwise_preserved':True,'identity_sizes_and_visibility_unchanged':True,
            'reloaded_output_verified':True,'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'method':'identified vehicle-local Gaussian asset and its fitted body calibration transplanted; no RGB painting'}
    (a.output/'merge_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
