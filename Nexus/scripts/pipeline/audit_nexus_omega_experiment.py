"""Independently audit the selected sampler, reproducibility and v9 replay."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--confirmed',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    import numpy as np
    import torch
    import imageio.v2 as imageio
    torch.set_num_threads(2)
    old = json.loads((args.candidate/'run.json').read_text())
    confirmed = json.loads((args.confirmed/'run.json').read_text())
    assert old['status']==confirmed['status']=='complete'
    assert confirmed['model_parameters_unchanged']
    rows=[]
    for row in old['records']:
        payload = torch.load(args.candidate/row['path']/'prediction.pt',map_location='cpu',weights_only=False)
        other = torch.load(args.confirmed/row['path']/'prediction.pt',map_location='cpu',weights_only=False)
        for key in payload:
            torch.testing.assert_close(payload[key],other[key],atol=0,rtol=0)
        pred = payload['sampled_tensor']
        keep = payload['task_mask'].bool()
        inputs = payload['conditioned_scene_tensor']
        assert not keep[:,0,5:].any()
        assert (inputs[~keep]==0).all()
        torch.testing.assert_close(pred[keep],inputs[keep],atol=0,rtol=0)
        assert torch.isfinite(pred[payload['valid_mask'].bool()]).all()
        records = json.loads((args.candidate/row['path']/'guidance.json').read_text())
        assert all(r['kl_max']<=20*(1+1e-4) for r in records)
        rows.append({'seed':row['seed'],'variant':row['variant'],
                     'all_saved_tensors_reproduced_exactly':True,'conditions_exact':True,
                     'ego_future_free':True,'free_input_values_zero':True})
    renders={}
    metadata_notes=[]
    for variant in ('normal','event'):
        folder=args.candidate/'renders'/f'anchor_{variant}'
        runpath=folder/'run.json'
        run=json.loads(runpath.read_text())
        assert run['status']=='passed' and len(run['frames'])==80
        assert run['temporary_actor_poses_restored_exactly'] and run['appearance_and_checkpoint_unchanged']
        assert run['checkpoint_sha256']=='1debd88b7c7f9172a64c65b00df69257007bb1a6595001ff38ef9305a94a188b'
        assert sha(Path(run['checkpoint']))==run['checkpoint_sha256']
        assert run['source_prediction_sha256']==sha(args.candidate/'anchor/seed_000'/variant/'prediction.pt')
        # The shared renderer's historical wording predates guided sampling.
        # Correct only this new output's description; numerical poses persist.
        if 'sampling_guidance' not in run:
            previous_hash=sha(runpath)
            run['sampling_guidance']={'method':'independent OMEGA-inspired clean-anchor optimization',
                'kl_budget':20,'model_weights_frozen':True,'post_sampling_trajectory_edits':False}
            run['actor_policy']=run['actor_policy'].replace('saved original Nexus XY/yaw','saved optimization-guided Nexus XY/yaw')
            runpath.write_text(json.dumps(run,indent=2)+'\n')
            metadata_notes.append({'path':str(runpath),'old_sha256':previous_hash,'new_sha256':sha(runpath),
                                   'change':'clarify guided-sampler provenance only; no poses/images changed'})
        with imageio.get_reader(str(folder/'camera_0.mp4')) as reader:
            meta=reader.get_meta_data()
            count=sum(1 for _ in reader)
        assert count==80 and meta['fps']==10
        stats={}
        gaps=[]
        for frame in run['frames']:
            scene=frame['scene_metadata']
            gaps.append(scene['ego_bus_sat_separation_m'])
            for ii,jj in scene['overlapping_actor_pairs']:
                a,b=scene['actors'][ii],scene['actors'][jj]
                key=f'{a["track_token"]} + {b["track_token"]}'
                stats[key]=stats.get(key,0)+1
        renders[variant]={'decoded_frames':count,'fps':meta['fps'],
            'ego_bus_overlap_frames':int((np.array(gaps)<0).sum()),
            'min_ego_bus_separation_m':min(gaps),'overlaps_by_actor_pair':stats,
            'checkpoint_unchanged':True,'temporary_poses_restored':True}
    args.output.write_text(json.dumps({'status':'passed','replicated_predictions':rows,
        'renders':renders,'metadata_clarification':metadata_notes,
        'scope':'engineering improvement; remaining collisions are not suppressed; no full safety acceptance'},indent=2)+'\n')
    print('OMEGA_INDEPENDENT_AUDIT_PASSED',flush=True)


if __name__=='__main__':
    main()
