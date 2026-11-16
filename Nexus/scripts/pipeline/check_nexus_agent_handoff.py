"""Check the actual candidate files, actor identity, time and ego ownership."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
import numpy as np
from export_nexus_agent_handoff import sha,validate_environment,write
from nexus_3dgs.bundle import load_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--handoff',type=Path,required=True);p.add_argument('--experiment',type=Path,required=True);a=p.parse_args()
    result={'status':'passed_contract_and_provenance_checks','cases':{},'closed_loop_executed':False}
    for name in ('bus_stop','bus_follow_logged'):
        directory=a.handoff/name;manifest=json.loads((directory/'case_manifest.json').read_text())
        for filename,digest in manifest['files_sha256'].items():assert sha(directory/filename)==digest,filename
        arrays=dict(np.load(directory/'environment_render_states.npz'));validate_environment(arrays)
        history=dict(np.load(directory/'ego_history.npz'));initial=json.loads((directory/'ego_initial_state.json').read_text())
        assert history['timestamps_us'].max()==initial['timestamp_us']
        assert manifest['ego_controller']=='external_policy'
        assert not manifest['worldengine_ready'] and not manifest['closed_loop_executed'] and not manifest['policy_training_data_ready']
        assert 'diagnostics/nexus_ego_future.npz' not in manifest['policy_input_files']
        raw,raw_manifest=load_bundle(a.experiment/'raw_pair'/name);ci=raw_manifest['current_index']
        np.testing.assert_array_equal(history['poses_world'],raw['reference_poses_world'][:ci+1,0])
        generated=dict(np.load(directory/'native_generated_traffic.npz'));mask=raw['nexus_slots']!=0
        np.testing.assert_array_equal(generated['predicted_poses_world'],raw['predicted_poses_world'][:,mask])
        np.testing.assert_array_equal(generated['valid'],raw['valid'][:,mask])
        rendered=json.loads((a.experiment/'renders'/name/'run.json').read_text())
        for t,frame in enumerate(rendered['frames']):
            actors={v['model_index']:v for v in frame['scene_metadata']['actors']}
            for m,index in enumerate(arrays['model_indices']):
                actor=actors[int(index)];assert actor['track_token']==arrays['track_tokens'][m]
                np.testing.assert_array_equal(arrays['poses_world'][t,m],actor['body_world'])
                assert arrays['valid'][t,m]==actor['valid']
        corrupt={**arrays,'nexus_slots':arrays['nexus_slots'].copy()};corrupt['nexus_slots'][0]=0
        try:validate_environment(corrupt)
        except ValueError as exc:assert 'Ego' in str(exc)
        else:raise AssertionError('Ego leakage was accepted')
        corrupt={**arrays,'poses_world':arrays['poses_world'].copy()};t,m=np.argwhere(arrays['valid'])[0];corrupt['poses_world'][t,m,0,0]=np.nan
        try:validate_environment(corrupt)
        except ValueError as exc:assert 'Non-finite' in str(exc)
        else:raise AssertionError('Non-finite pose was accepted')
        result['cases'][name]={'source_samples_and_poses_exact':True,'policy_owns_ego':True,'ego_future_excluded_from_policy_inputs':True,
            'rejects_ego_in_environment':True,'rejects_nonfinite_valid_pose':True,'file_hashes_verified':len(manifest['files_sha256']),
            'worldengine_ready':False,'scene_validity':manifest['scene_validity']}
    write(a.handoff/'contract_verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
