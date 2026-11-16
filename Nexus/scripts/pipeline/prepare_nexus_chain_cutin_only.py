"""User-requested pure cut-in control, sharing observed inputs with chain pilot."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/nexus_chain_20261027';OUT=ROOT/'artifacts/nexus_chain_cutin_only_20261027';sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
def main():
 p=json.loads((BASE/'protocol.json').read_text());p['cases']=[x for x in p['cases'] if x['scenario']['index']==33];p['indices']=[33];p['roles']={k:v for k,v in p['roles'].items() if k=='33'};p.pop('braking',None)
 p.update(event_kind='cut_in_only',event_description_zh='A以观测速度沿地图前行，在0.5–3秒切入B前方；A没有额外刹车，B/C未来自由。',comparison_reason='user clarified the pure cut-in -> B slowing -> C slowing chain; fixed before seeing chain pilot outcomes',reuse_predictions=str(BASE/'pilot_seed0'),new_inference_samples_planned=2,status='prepared')
 OUT.mkdir(exist_ok=False)
 for info in p['cases']:
  idx=info['scenario']['index'];source=BASE/f'case_{idx:03d}';case=OUT/f'case_{idx:03d}';case.mkdir()
  for name in ('reference','map','scene.pt','conditions_normal.npz'):(case/name).symlink_to(source/name,target_is_directory=(source/name).is_dir())
  (case/'conditions_event.npz').symlink_to(source/'conditions_cutin_only.npz')
  info['event_kind']='cut_in_only';info['given_A_input_usable']=True
  (case/'preparation.json').write_text(json.dumps(info,indent=2)+'\n')
 (OUT/'protocol.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n')
 import audit_nexus_multiscene_fixed_A as bg;bg.OUT=OUT;bg.main()
 a=json.loads((OUT/'fixed_A_background_audit.json').read_text())
 for info in p['cases']:
  conflicts=[r for r in a['records'] if r['index']==info['scenario']['index'] and r['fixed_A_vs_recorded_background_overlap_samples']]
  info['given_A_background_conflicts']=conflicts;info['given_A_input_usable']=not bool(conflicts)
  (OUT/f"case_{info['scenario']['index']:03d}"/'preparation.json').write_text(json.dumps(info,indent=2)+'\n')
 (OUT/'protocol.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
