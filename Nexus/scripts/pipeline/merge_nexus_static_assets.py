"""Transplant a static reconstruction while preserving original traffic and cameras.

Both scenes must have exactly the same coordinate origin. The global affine
appearance transform is converted into the new static assets' color basis;
dynamic Gaussian geometry, features, poses and visibility remain untouched.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','static-source','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch,yaml
    from models.modules import AffineTransform
    from scripts.pipeline.merge_nexus_vehicle_asset import equal
    old=torch.load(a.base,map_location='cpu',weights_only=False)
    new=torch.load(a.static_source,map_location='cpu',weights_only=False)
    cfg=yaml.safe_load((a.base.parent/'config.yaml').read_text());fresh=yaml.safe_load((a.static_source.parent/'config.yaml').read_text())
    scenes=[Path(c['data']['data_root'])/str(c['data']['scene_idx']) for c in (cfg,fresh)]
    metadata=[json.loads((s/'nexus_scene.json').read_text()) for s in scenes]
    assert metadata[0]['identity']['database']==metadata[1]['identity']['database']
    assert metadata[0]['identity']['scenario_token']==metadata[1]['identity']['scenario_token']
    assert metadata[0]['frames'][0]['ego_to_world']==metadata[1]['frames'][0]['ego_to_world']
    affine=[];variation=[]
    for state,conf in [(old,cfg),(new,fresh)]:
        assert not conf['model']['Affine']['params'].get('pixel_affine',False)
        weights=state['models']['Affine'];n=len(weights['embedding.weight'])
        model=AffineTransform('Affine',n,device=torch.device('cpu'));model.load_state_dict(weights)
        with torch.no_grad():matrices=model({'img_idx':torch.arange(n)})
        drift=float((matrices-matrices[0]).abs().max());variation.append(drift)
        if drift>1e-7:raise ValueError('Appearance is image-dependent; a global static-color conversion is not valid')
        affine.append(matrices[0])
    A0,A1=affine;M=torch.linalg.solve(A0[:,:3],A1[:,:3]);c=torch.linalg.solve(A0[:,:3],A1[:,3]-A0[:,3])
    bg=new['models']['Background'];C0=.28209479177387814
    converted={key:value.clone() for key,value in bg.items()}
    converted['_features_dc']=((bg['_features_dc']*C0+.5)@M.T+c-.5)/C0
    # Work in bounded-size blocks to avoid a second full SH tensor temporary.
    for start in range(0,len(bg['_means']),100000):
        end=start+100000
        converted['_features_rest'][start:end]=bg['_features_rest'][start:end]@M.T
    sky={key:value.clone() for key,value in new['models']['Sky'].items()}
    assert cfg['model']['Sky']['params']['bounded'] and fresh['model']['Sky']['params']['bounded']
    rgb=sky['base'].sigmoid();mapped=rgb@M.T+c
    clipped=((mapped<=1e-6)|(mapped>=1-1e-6)).float().mean().item()
    sky['base']=torch.logit(mapped.clamp(1e-6,1-1e-6))
    old['models']['Background']=converted;old['models']['Sky']=sky
    cfg['model']['Sky']['params']['resolution']=fresh['model']['Sky']['params']['resolution']
    # Keep the original rendering policy for the preserved vehicles. The
    # antialiasing ablation is rendered separately with its own config.
    output=a.output/'checkpoint_final.pth';torch.save(old,output)
    (a.output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    saved=torch.load(output,map_location='cpu',weights_only=False)
    base=torch.load(a.base,map_location='cpu',weights_only=False)
    for name,model in base['models'].items():
        if name not in ('Background','Sky'):equal(model,saved['models'][name])
    for key in ('_means','_scales','_quats','_opacities'):equal(bg[key],saved['models']['Background'][key])
    error=((mapped.clamp(1e-6,1-1e-6)@A0[:,:3].T+A0[:,3])-(rgb@A1[:,:3].T+A1[:,3])).abs().max().item()
    report={'status':'passed','original_traffic_camera_and_affine_bitwise_preserved':True,
            'original_dataset_and_prediction_duration_preserved':True,'static_geometry_bitwise_from_new_model':True,
            'background_gaussians':len(bg['_means']),'affine_image_variation':variation,
            'sky_color_clipping_fraction':clipped,'sky_affine_conversion_max_error':error,
            'color_linear_matrix':M.tolist(),'color_linear_bias':c.tolist(),
            'scope':'native Gaussian geometry and environment texture transplant; SH conversion is exact before RGB clamping; no rendered-frame painting or temporal blending'}
    (a.output/'merge_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
