"""Verify raw generations and reused v9 footage, then deliver under hzy/report."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--render-source',type=Path,required=True)
    p.add_argument('--audits',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experimental-render',type=Path)
    a=p.parse_args();candidate=a.candidate.resolve();source=a.render_source.resolve()
    import torch
    torch.set_num_threads(2)
    run=json.loads((candidate/'run.json').read_text())
    proto=json.loads((candidate/'protocol.json').read_text())
    assert run['status']=='complete' and run['model_parameters_unchanged']
    source_checks={}
    for relative,expected in proto['source_sha256'].items():
        path=candidate/'sources'/relative
        assert digest(path)==expected,relative
        source_checks[relative]=expected
    checkpoint=ROOT.parent/'checkpoints/nuplan.ckpt'
    assert digest(checkpoint)=='679f6ccfbd5563f696218311566b76bb62f73c7fbfb04b39f2714fb80aefacfc'
    background=ROOT/'artifacts/nexus_omega_v2/background_inputs'
    assert digest(background/'background_replay.npz')==json.loads((background/'manifest.json').read_text())['sha256']
    previous=torch.load(ROOT/'artifacts/nexus_omega/pilot_v4_kl20/anchor/seed_000/event/prediction.pt',map_location='cpu',weights_only=False)
    checks=[]
    for row in run['records']:
        path=candidate/row['path']/'prediction.pt'
        d=torch.load(path,map_location='cpu',weights_only=False)
        keep=d['task_mask'].bool();inputs=d['conditioned_scene_tensor'];pred=d['sampled_tensor']
        assert torch.isfinite(pred[d['valid_mask'].bool()]).all()
        torch.testing.assert_close(pred[keep],inputs[keep],atol=0,rtol=0)
        assert not keep[:,0,5:].any() and (inputs[~keep]==0).all()
        # History velocity changes; historical XY/yaw/size do not.
        old=previous['conditioned_scene_tensor']
        torch.testing.assert_close(inputs[:,:,:5,:4],old[:,:,:5,:4],atol=0,rtol=0)
        torch.testing.assert_close(inputs[:,:,:5,6:],old[:,:,:5,6:],atol=0,rtol=0)
        guidance=json.loads((candidate/row['path']/'guidance.json').read_text())
        assert all(x['kl_max_per_agent']<=proto['config']['kl_budget']*(1+1e-4) for x in guidance)
        item={'seed':row['seed'],'variant':row['variant'],'conditions_exact':True,'historical_pose_and_size_exact':True,
              'ego_future_free':True,'free_values_erased':True,'per_agent_kl_within_budget':True}
        if row['seed']==0:
            original=torch.load(source/row['path']/'prediction.pt',map_location='cpu',weights_only=False)
            for key in d:torch.testing.assert_close(d[key],original[key],atol=0,rtol=0)
            item['render_source_payload_exact']=True
        checks.append(item)
    if source!=candidate:
        target=candidate/'renders'
        if target.exists():raise FileExistsError(target)
        shutil.copytree(source/'renders',target)
    rendering={}
    for variant in ('normal','event'):
        folder=candidate/f'renders/anchor_{variant}';path=folder/'run.json'
        meta=json.loads(path.read_text())
        assert meta['status']=='passed' and len(meta['frames'])==80
        assert meta['checkpoint_sha256']=='1debd88b7c7f9172a64c65b00df69257007bb1a6595001ff38ef9305a94a188b'
        assert digest(Path(meta['checkpoint']))==meta['checkpoint_sha256']
        assert meta['appearance_and_checkpoint_unchanged'] and meta['temporary_actor_poses_restored_exactly']
        assert meta['source_prediction_sha256']==digest(source/f'anchor/seed_000/{variant}/prediction.pt')
        meta['sampling_generation']={'kind':'structural OMEGA-inspired constrained sampling',
            'protocol':str(candidate/'protocol.json'),'original_render_source':str(source),
            'candidate_seed0_payload_exact':True,'post_sampling_trajectory_edits':False,
            'recorded_background_future_supplied':True,'history_velocity':'backward position differences'}
        meta['actor_policy']=meta['actor_policy'].replace('saved original Nexus XY/yaw','saved structurally guided Nexus XY/yaw')
        path.write_text(json.dumps(meta,indent=2)+'\n')
        frames=meta['frames']
        rendering[variant]={'ego_bus_overlapping_frames':sum(f['scene_metadata']['ego_bus_sat_separation_m']<0 for f in frames),
            'all_vehicle_overlap_pair_frames':sum(len(f['scene_metadata']['overlapping_actor_pairs']) for f in frames),
            'minimum_ego_bus_signed_sat_m':min(f['scene_metadata']['ego_bus_sat_separation_m'] for f in frames),
            'residual_overlaps':[{'frame':f['frame'],'timestamp_us':f['timestamp_us'],
                'pairs':f['scene_metadata']['overlapping_actor_pairs']} for f in frames if f['scene_metadata']['overlapping_actor_pairs']]}
    audit={'status':'passed','meaning':'technical integrity passed; behavioral acceptance is independent',
        'generations':checks,'source_snapshots':source_checks,'rendered_actual_bodies':rendering,
        'official_checkpoint_unchanged':True,'v9_checkpoint_unchanged':True}
    (candidate/'delivery_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    command=['bash',str(ROOT/'scripts/pipeline/nexus_3dgs.sh'),'script',str(ROOT/'scripts/pipeline/report_nexus_omega_v2.py'),
        '--candidate',str(candidate),'--audits',*[str(path.resolve()) for path in a.audits],'--output',str(a.output.resolve())]
    if a.experimental_render:command+=['--experimental-render',str(a.experimental_render.resolve())]
    subprocess.run(command,cwd=ROOT,check=True)
    shutil.copy2(candidate/'delivery_audit.json',a.output/'assets/delivery_audit.json')
    verification=json.loads((a.output/'verification.json').read_text())
    verification['files']={str(path.relative_to(a.output)):digest(path) for path in a.output.rglob('*') if path.is_file() and path.name!='verification.json'}
    for name,expected in verification['files'].items():assert digest(a.output/name)==expected
    (a.output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print('OMEGA_V2_DELIVERED',a.output.resolve(),flush=True)


if __name__=='__main__':main()
