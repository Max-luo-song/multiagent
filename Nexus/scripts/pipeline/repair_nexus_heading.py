"""Export a separate bundle resolving isolated near-static pi heading flips."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from nexus_3dgs.bundle import load_bundle,save_bundle
from nexus_3dgs.kinematics import repair_isolated_half_turns


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    bundle,meta=load_bundle(a.bundle)
    if a.output.exists():raise FileExistsError(a.output)
    current=meta['current_index'];fixed=np.zeros_like(bundle['valid'],dtype=bool)
    fixed[:current+1]=True;fixed[:,0]=True
    if meta['task']=='intent_conditioned_behavior_pred':
        for agent in range(fixed.shape[1]):
            indices=np.flatnonzero(bundle['valid'][:,agent])
            if len(indices):fixed[indices[-1],agent]=True
    if 'conditioned_slot' in meta:
        fixed[:,bundle['nexus_slots']==meta['conditioned_slot']]=True
    corrected,changes=repair_isolated_half_turns(bundle['predicted_poses_world'],bundle['valid'],fixed)
    for change in changes:
        change['nexus_slot']=int(bundle['nexus_slots'][change['agent_index']])
        change['track_token']=str(bundle['track_tokens'][change['agent_index']])
        change['timestamp_us']=int(bundle['timestamps_us'][change['sample_index']])
    arrays={k:np.array(v,copy=True) for k,v in bundle.items()};arrays['predicted_poses_world']=corrected
    np.testing.assert_array_equal(corrected[...,:3,3],bundle['predicted_poses_world'][...,:3,3])
    np.testing.assert_array_equal(corrected[fixed],bundle['predicted_poses_world'][fixed])
    report={'source_bundle':str(a.bundle.resolve()),'source_bundle_sha256':meta['bundle_sha256'],
            'method':'isolated low-speed half-turn heading branch correction; neighbor poses only',
            'changes':changes,'positions_validity_identity_and_given_conditions_unchanged':True,
            'nexus_model_weights_changed':False,'uses_logged_future_headings':False}
    save_bundle(a.output,arrays,{**meta,'heading_repair':report})
    (a.output/'heading_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
