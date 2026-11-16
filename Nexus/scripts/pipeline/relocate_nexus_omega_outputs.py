"""Copy this session's own outputs to the user's explicitly chosen root."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

SOURCE = Path(__file__).resolve().parents[2]
SAMPLER = Path('nuplan_extent/planning/training/modeling/models/diffusion/scenediffusion.py')
CODE = [SAMPLER,Path('nuplan_extent/planning/training/modeling/models/diffusion/omega_guidance.py'),
        *[Path('scripts/pipeline')/name for name in (
            'run_nexus_omega_experiment.py','check_nexus_omega.py',
            'render_nexus_omega_experiment.py','audit_nexus_omega_experiment.py',
            'report_nexus_omega_experiment.py','finalize_nexus_omega_report.py',
            'relocate_nexus_omega_outputs.py')],Path('docs/nexus_omega_experiment_20261026.md')]
PREFERENCE = '''# Nexus 项目文件存放约定

用户于 2026-10-26 明确要求：今后 Nexus、Omniscenegraph／3DGS、OMEGA 相关内容统一放在
`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/` 下。

- 代码与实验结果：`nexus_repro/Nexus/`，实验放在其 `artifacts/` 下。
- 报告、图片和展示视频：`report/` 下，使用明确命名的子目录。
- 交接说明：`others/` 下。
- 临时工作树可以用于编辑，但不能作为成果唯一的保存或交付位置。

该约定仅适用于上述项目。保留旧成果和历史审计中的原始来源路径，不因搬迁改写历史哈希。
'''


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination-root',type=Path,required=True)
    a=p.parse_args()
    dest=a.destination_root.resolve()
    if dest != Path('/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy').resolve():
        raise ValueError('Only the user-authorized project root is supported')
    repo=dest/'nexus_repro/Nexus'
    artifact=repo/'artifacts/nexus_omega'
    report=dest/'report/nexus_omega_20261026'
    # Only the unchanged original sampler may receive the 15-line opt-in
    # integration. All other code and output paths must be new.
    base=subprocess.check_output(['git','show','HEAD:'+str(SAMPLER)],cwd=SOURCE)
    if (repo/SAMPLER).read_bytes()!=base:
        raise ValueError('Original sampler changed; manual merge required')
    for path in CODE:
        if path!=SAMPLER and (repo/path).exists():
            raise FileExistsError(repo/path)
    for path in (artifact,report,dest/'AGENTS.md',dest/'others/Nexus_文件存放约定.md'):
        if path.exists():
            raise FileExistsError(path)
    trees=[(SOURCE/'artifacts/nexus_omega',artifact),
           (SOURCE/'report/nexus_omega_20261026',report)]
    for src,_ in trees:
        if any(p.is_symlink() for p in src.rglob('*')):
            raise ValueError('Unexpected symlink in this session output')
    verified=[]
    for src,target in trees:
        shutil.copytree(src,target)
        for path in src.rglob('*'):
            if path.is_file():
                counterpart=target/path.relative_to(src)
                checksum=sha(path)
                if checksum!=sha(counterpart):
                    raise ValueError('Copy checksum mismatch: '+str(counterpart))
                verified.append({'source':str(path),'destination':str(counterpart),'sha256':checksum})
    backup=artifact/'migration_backup/scenediffusion_before_relocation.py'
    backup.parent.mkdir()
    backup.write_bytes(base)
    for path in CODE:
        target=repo/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(SOURCE/path,target)
        assert sha(SOURCE/path)==sha(target)
        verified.append({'source':str(SOURCE/path),'destination':str(target),'sha256':sha(target)})
    (dest/'AGENTS.md').write_text(PREFERENCE)
    (dest/'others/Nexus_文件存放约定.md').write_text(PREFERENCE)
    note=f'''# 当前保存位置

用户指定主目录：`{dest}`。

- 当前报告：`{report}/index.html`
- 事件对照：`{report}/assets/event_before_after.mp4`
- 正常对照：`{report}/assets/normal_before_after.mp4`
- 实验与审计：`{artifact}`
- 代码：`{repo}`

历史 JSON 中的工作树路径记录了实际运行来源，予以保留。对应文件已复制到上述目录，见 `relocation_checks.json`。
'''
    (report/'LOCATION.md').write_text(note)
    result={'status':'passed','user_authorized_root':str(dest),'verified_file_count':len(verified),
            'files':verified,'preference_recorded':True,'old_results_preserved':True,
            'original_sampler_backup':str(backup),'source_worktree_retained':str(SOURCE)}
    for path in (artifact/'relocation_checks.json',report/'relocation_checks.json'):
        path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'passed','verified_files':len(verified),
                      'report':str(report/'index.html'),'artifacts':str(artifact),
                      'code':str(repo),'preference':str(dest/'AGENTS.md')},ensure_ascii=False))


if __name__=='__main__':
    main()
