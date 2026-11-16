"""Publish the documented workflow position and inspected handoff contracts."""
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[2]
STEM='nexus_project_flow_20261026'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    import markdown
    source=ROOT/'docs/nexus_project_workflow_position_20261026.md'
    review=ROOT/'artifacts/project_flow_review_20261026';handoff=ROOT/'artifacts/nexus_3dgs/agent_handoff_20261026'
    checks=json.loads((handoff/'contract_verification.json').read_text())
    assert checks['status']=='passed_contract_and_provenance_checks'
    report=ROOT/'report';assets=report/'assets'/STEM;assets.mkdir(exist_ok=True)
    shutil.copy2(source,report/(STEM+'.md'))
    shutil.copytree(handoff,assets/'agent_handoff',dirs_exist_ok=True)
    for filename in ['slide8_image49.png','slide10_image53.png']:
        shutil.copy2(review/filename,assets/filename)
    svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="470" viewBox="0 0 1200 470">
<defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0 0L10 4L0 8" fill="#667b92"/></marker></defs>
<rect width="1200" height="470" rx="16" fill="#f2f6fa"/>
<g stroke="#667b92" stroke-width="3" fill="none" marker-end="url(#a)">
<path d="M450 82L275 132"/><path d="M750 82L925 132"/><path d="M275 215L450 270"/><path d="M925 215L750 270"/><path d="M600 345L600 384"/></g>
<g font-family="system-ui, sans-serif" text-anchor="middle">
<rect x="350" y="14" width="500" height="72" rx="12" fill="#e1e9f4" stroke="#7b93b5"/>
<text x="600" y="44" font-size="23">上游 Agent：风险需求、场景检索、编辑方案</text><text x="600" y="72" font-size="18">结构化条件＋nuPlan 场景／地图／真实传感器</text>
<rect x="30" y="132" width="490" height="84" rx="12" fill="#daf2e7" stroke="#39866c"/>
<text x="275" y="165" font-size="24">Nexus：条件交通轨迹生成</text><text x="275" y="194" font-size="18">当前为 8 秒候选；公交给定，其余交通采样</text>
<rect x="680" y="132" width="490" height="84" rx="12" fill="#daf2e7" stroke="#39866c"/>
<text x="925" y="165" font-size="24">Omniscenegraph：逐场景 3DGS</text><text x="925" y="194" font-size="18">真实图像、LiDAR、标定与实例监督</text>
<rect x="275" y="270" width="650" height="76" rx="12" fill="#daf2e7" stroke="#39866c"/>
<text x="600" y="301" font-size="24">我们当前：身份／坐标／时间绑定、编辑、渲染与验证</text><text x="600" y="330" font-size="18">输出背景交通状态＋资产＋相机／地图引用＋质量检查</text>
<rect x="245" y="385" width="710" height="70" rx="12" fill="#fff0d4" stroke="#bd8b38"/>
<text x="600" y="415" font-size="23">下游待接通：WorldEngine＋独立自车策略闭环</text><text x="600" y="443" font-size="18">策略控制 ego → 仿真更新／渲染 → 失败筛选 → 后训练</text>
</g></svg>'''
    (assets/'our_workflow.svg').write_text(svg)
    text=source.read_text();text=re.sub(r'```mermaid\n.*?```',f'<img src="assets/{STEM}/our_workflow.svg" alt="项目上下游流程图">',text,flags=re.S)
    body=markdown.markdown(text,extensions=['tables','fenced_code'])
    rel='assets/'+STEM+'/'
    extra='<h2>材料原图与接口文件</h2>'
    extra+=f'<p>PPT 第 10 页：总体 Agent 与工具池。</p><img src="{rel}slide10_image53.png" alt="PPT 总体框架原图">'
    extra+=f'<p>PPT 第 8 页：雨夜高速案例的十步流程。图中数值属于原材料示意，不是本项目实测成绩。</p><img src="{rel}slide8_image49.png" alt="PPT 案例流程原图">'
    for variant in ['bus_stop','bus_follow_logged']:
        extra+=f'<p>{variant}：<a href="{rel}agent_handoff/{variant}/case_manifest.json">候选接口清单</a> · <a href="{rel}agent_handoff/{variant}/export_audit.json">场景审计</a> · <a href="{rel}agent_handoff/{variant}/environment_render_states.npz">非自车状态 NPZ</a></p>'
    extra+=f'<p><a href="{rel}agent_handoff/contract_verification.json">接口与来源的独立验证</a></p>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nexus＋3DGS 项目流程定位与上下游接口</title><style>body{max-width:1380px;padding:28px;margin:auto;font:18px/1.7 system-ui;color:#26384c}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #c5d0de;padding:10px;text-align:left}img{max-width:100%;margin:18px 0}code{background:#eef2f7;padding:2px 4px;overflow-wrap:anywhere}a{color:#1a5d99}h1,h2{color:#154d72}</style>'+body+extra+'</html>'
    page_path=report/(STEM+'.html');page_path.write_text(page)
    inputs=Path('/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/ppt_files')
    consumer=Path('/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagentv1.0')
    engine=Path('/inspire/hdd3/project/intelligent-driving-agent/public/workspace/lch/WorldEngine')
    consumer_files=[consumer/'tools/tool13_policy_runner/real.py',consumer/'mcp_servers/worldengine/service.py',
       engine/'projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py',engine/'projects/SimEngine/worldengine/render/mtgs/mtgs.py',
       engine/'projects/SimEngine/worldengine/scenario/scenarios/scenario_description.py']
    evidence={'source_materials_sha256':{str(p):sha(p) for p in inputs.iterdir() if p.is_file()},
       'inspected_consumer_sources_sha256':{str(p):sha(p) for p in consumer_files},
       'contract_checks_passed':True,'closed_loop_executed':False,'training_started':False,
       'read_only_external_repository_inspection':True}
    (assets/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    resource_hashes={str(path.relative_to(assets)):sha(path) for path in assets.rglob('*') if path.is_file()}
    links=sorted(set(re.findall(r'(?:src|href)="([^"]+)"',page)))
    for link in links:assert (report/link).is_file(),link
    manifest={'status':'verified','resources':resource_hashes,'links_verified':len(links),'asset_hashes_verified':len(resource_hashes)}
    (report/(STEM+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    print('PROJECT_FLOW_REPORT_VERIFIED',page_path,'links',len(links),'assets',len(resource_hashes))


if __name__=='__main__':main()
