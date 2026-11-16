"""Report measured road geometry and same-camera sky occupancy changes."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('old-checkpoint','new-checkpoint','observations','reference','old-reference','new-reference','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--road-indices',type=Path,required=True)
    a=p.parse_args();import torch
    before=torch.load(a.old_checkpoint,map_location='cpu',weights_only=False)['models'];after=torch.load(a.new_checkpoint,map_location='cpu',weights_only=False)['models']
    ground=json.loads(a.reference.read_text())['ground_registration']['ground_model'];aa,bb,cc=ground['coefficients_z_ax_by_c'];ids=np.load(a.road_indices)
    report={'status':'complete','scope':'actual road-plane support and conservative observed-sky diagnostic; not a novel-view image-quality score','road':{},'sky':[]}
    for label,models in [('old',before),('new',after)]:
        P=models['Background']['_means'].numpy()[ids];e=np.abs(P[:,2]-aa*P[:,0]-bb*P[:,1]-cc)/np.sqrt(aa*aa+bb*bb+1)
        report['road'][label]={'supported_gaussians':len(ids),'normal_distance_mean_m':float(e.mean()),'normal_distance_median_m':float(np.median(e)),'normal_distance_p90_m':float(np.quantile(e,.9))}
    for f in [20,99]:
        d=np.load(a.observations/f'{f:03d}.npz');mask=d['sky']&~d['vehicle']
        old=np.load(a.old_reference/'diagnostic_layers'/f'{f:03d}_0_raw.npz')['opacity'].squeeze()
        new=np.load(a.new_reference/'layers'/f'{f:03d}.npz')['opacity'].squeeze()
        report['sky'].append({'frame':f,'pixels':int(mask.sum()),'old_finite_opacity_mean':float(old[mask].mean()),'new_finite_opacity_mean':float(new[mask].mean())})
    # Shared actor assets are exact, even though the sequence metadata extends.
    keys=['_means','_features_dc','_features_rest','_scales','_quats','_opacities','point_ids','instances_size']
    checked=[]
    for key in keys:
        if key in before['RigidNodes']:
            assert torch.equal(before['RigidNodes'][key],after['RigidNodes'][key]),key
            checked.append(key)
    n=len(before['RigidNodes']['instances_trans'])
    for key in ['instances_trans','instances_quats','instances_fv']:
        assert torch.equal(before['RigidNodes'][key],after['RigidNodes'][key][:n]),key
        checked.append(key)
    report['original_actor_parameters_exact']=checked
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
