"""Finalize local evidence and optionally copy the reviewed report to delivery."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values=[]

    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('href','src') and value and not value.startswith(('http:','https:','#','data:')):
                self.values.append(value)


def verify(report):
    parser=Links()
    parser.feed((report/'index.html').read_text())
    for value in parser.values:
        path=(report/value).resolve()
        path.relative_to(report.resolve())
        assert path.is_file(),value
    return len(parser.values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--publish-to',type=Path)
    args=p.parse_args()
    report=args.report.resolve()
    if not args.publish_to:
        independent=json.loads((ROOT/'artifacts/nexus_omega/candidate_independent_audit.json').read_text())
        assert independent['status']=='passed'
        summary='''当前选择 KL=20 的物理坐标 Adam 作为实验候选，默认 Nexus 采样保持不变。三个固定种子中，自车—公交重叠采样点合计 16→8；正常对照无预测车辆重叠的种子 2→3。
固定种子 0 的 v9 实际车体回放中，事件重叠 32→18 帧，最深重叠约 2.23→0.62 米；正常对照重叠 14→0 帧。候选的六份输出已由整理后的代码逐值复现。
完整避碰尚未通过。三个事件种子的全车加速度超限采样点 24→30，自车峰值加速度最大值 1.559→1.638 m/s²，jerk 最大值 4.852→5.845 m/s³。事件视频另有 14 个无关背景车重叠对×帧。改善存在明确取舍，不能称全面升级成功。'''
        entry=report/'index.html'
        doc=entry.read_text()
        marker='<h2>当前结论</h2>'
        if marker not in doc:
            doc=doc.replace('<h2>三个事件种子的汇总</h2>',marker+'<p>'+summary+'</p><p><a href="assets/independent_audit.json">独立审计</a> · <a href="assets/implementation_notes.md">实现与复跑说明</a> · <a href="assets/selected_candidate.json">候选决策与来源</a></p><h2>三个事件种子的汇总</h2>')
            entry.write_text(doc)
            readme=report/'README.md'
            readme.write_text(readme.read_text().replace('\n\n这是一组','\n\n'+summary+'\n\n这是一组',1))
        shutil.copy2(ROOT/'artifacts/nexus_omega/candidate_independent_audit.json',report/'assets/independent_audit.json')
        shutil.copy2(ROOT/'docs/nexus_omega_experiment_20261026.md',report/'assets/implementation_notes.md')
        codepaths=[ROOT/'nuplan_extent/planning/training/modeling/models/diffusion/scenediffusion.py',
                   ROOT/'nuplan_extent/planning/training/modeling/models/diffusion/omega_guidance.py',
                   ROOT/'scripts/pipeline/run_nexus_omega_experiment.py',ROOT/'scripts/pipeline/check_nexus_omega.py']
        selected={'status':'experimental_partial_improvement','promoted_to_default':False,
            'selected_run':'artifacts/nexus_omega/pilot_v4_kl20',
            'exact_reproduction':'artifacts/nexus_omega/confirmed_candidate_v1',
            'configuration':{'solver':'adam_physical','iterations':40,'kl_budget':20,'collision_weight':30},
            'choice_reason':'KL=20 halves native event overlap samples; increasing to 200 did not reduce the count; retain smaller trust region. Alternative solvers/schedule were not consistently better.',
            'remaining_work':['residual ego-bus collisions','unrelated background overlaps','all-agent kinematic violations','independent scene validation'],
            'training_updates':0,'post_sampling_trajectory_edits':False,
            'source_sha256':{str(f.relative_to(ROOT)):digest(f) for f in codepaths},
            'confirmed_logic_note':'Latest CLI guards/default method selection were tightened after exact replay; sampler computations unchanged.'}
        write(ROOT/'artifacts/nexus_omega/selected_candidate.json',selected)
        write(report/'assets/selected_candidate.json',selected)
        old=json.loads((report/'verification.json').read_text())
        # Existing encoded video hashes must stay exactly the same while
        # adding explanatory text and evidence links.
        for name,expected in old['files'].items():
            if name.endswith('.mp4'):
                assert digest(report/name)==expected
        old.update(local_links_checked=verify(report),
            independent_candidate_audit_passed=True,
            candidate_six_payloads_reproduced_exactly=True,
            files={str(f.relative_to(report)):digest(f) for f in report.rglob('*') if f.is_file() and f.name!='verification.json'})
        write(report/'verification.json',old)
        print('OMEGA_LOCAL_REPORT_FINALIZED',flush=True)
    else:
        destination=args.publish_to.resolve()
        source_manifest=json.loads((report/'verification.json').read_text())
        for name,expected in source_manifest['files'].items():
            assert digest(report/name)==expected,name
        # A new directory protects all previous reports.
        shutil.copytree(report,destination)
        for name,expected in source_manifest['files'].items():
            assert digest(destination/name)==expected,name
        assert digest(destination/'verification.json')==digest(report/'verification.json')
        checked=verify(destination)
        write(destination/'delivery_checks.json',{'status':'passed','files_verified':len(source_manifest['files'])+1,
            'local_links_checked':checked,'source':str(report),'destination':str(destination)})
        print('OMEGA_DELIVERY_COMPLETE',destination,flush=True)


if __name__=='__main__':
    main()
