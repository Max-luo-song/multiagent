"""Check role generalization against saved audited outputs and transfer labels."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
 import torch
 from nexus_3dgs.multiscene_acceptance import evaluate as old
 from nexus_3dgs.chain_acceptance import evaluate
 from nexus_3dgs.chain_metrics import contrast,mixed_pair
 reference=ROOT/'artifacts/nexus_multicar_upgrade_check_20261027';lanes=ROOT/'artifacts/nexus_kinematic_20261027/lanes';base=ROOT/'artifacts/nexus_kinematic_20261027/candidate_v4_cruise_initialized/kinematic/seed_000'
 checks=[]
 for variant in ('normal','event'):
  p=torch.load(base/variant/'prediction.pt',weights_only=False,map_location='cpu')
  a=old(p,reference_root=reference,lane_root=lanes,variant=variant,actor_slot=2,third_slot=12)
  b=evaluate(p,reference_root=reference,lane_root=lanes,variant=variant,actor_slot=2,reactor_slot=0,third_slot=12)
  for k in ['primary_three_accepted','full_scene_accepted','eligible_box_offroad_samples','all_predicted_involved_overlap_samples']:assert a[k]==b[k],k
  for src,dst in [('E','B'),('C','C')]:assert a['motion'][src]==b['motion'][dst]
  assert a['roles']['EA']==b['roles']['BA'] and a['roles']['EC']==b['roles']['BC']
  fixed={k:v.clone() for k,v in p.items()};fixed['task_mask'][0,0,5:]=True;fixed['conditioned_scene_tensor'][0,0,5:]=fixed['sampled_tensor'][0,0,5:]
  d=evaluate(fixed,reference_root=reference,lane_root=lanes,variant=variant,actor_slot=2,reactor_slot=0,third_slot=12,fixed_reactor=True)
  assert not d['B_C_future_free'] and d['C_future_free'] and d['fixed_B_diagnostic']
  assert d['roles']==b['roles'] and d['motion']==b['motion']
  zero=contrast(p,p,12);assert zero['maximum_speed_difference_mps']==0 and zero['maximum_position_difference_m']==0
  direct=mixed_pair(p,0,p,2);assert direct['overlap_samples']==b['roles']['BA']['overlap_samples'];assert abs(direct['minimum_signed_gap_m']-b['roles']['BA']['minimum_signed_gap_m'])<1e-7
  checks.append({'variant':variant,'role_remapping_matches_original':True,'fixed_B_labeled_and_geometrically_identical':True,'same_output_zero_response':True,'mixed_pair_SAT_matches':True})
 out=ROOT/'artifacts/nexus_chain_20261027/audit_adapter_check.json';out.write_text(json.dumps({'status':'passed','checks':checks},indent=2)+'\n');print('CHAIN_AUDIT_CHECK_PASSED',flush=True)
if __name__=='__main__':main()
