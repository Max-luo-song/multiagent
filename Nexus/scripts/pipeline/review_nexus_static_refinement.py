"""Render a static checkpoint at unchanged actual generated/logged states."""
import argparse,runpy,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-root',type=Path,help='Archived v5 checkpoint and saved render states')
    p.add_argument('--scene',type=Path,help='Original coordinate metadata')
    p.add_argument('--full-video',action='store_true')
    p.add_argument('--conditions',nargs='+',choices=['go','stop','reference'],default=['go','reference'])
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    old=a.source_root.resolve() if a.source_root else root/'artifacts/nexus_3dgs/optimization_v5'
    scene=a.scene.resolve() if a.scene else next((root/'artifacts/nexus_3dgs/processed').glob('*/nexus_scene.json')).parent
    for condition in a.conditions:
        sys.argv=['replay_nexus_motion.py','--checkpoint',str(a.checkpoint.resolve()),'--source',str(old/f'coherent_rear_{condition}/run.json'),'--scene',str(scene),'--output',str((a.output/condition).resolve()),'--variants','both','--frames','20','55','70','85','99','--actor-appearance-checkpoint',str(old/'coherent_rear_20/checkpoint_final.pth')]
        if condition=='reference':sys.argv.append('--logged-reference')
        else:sys.argv.extend(['--appearance-frame','20'])
        if a.full_video:sys.argv.append('--full-video')
        runpy.run_path(str(root/'scripts/pipeline/replay_nexus_motion.py'),run_name='__main__')


if __name__=='__main__':main()
