"""Export and verify the event-control result, including unsuccessful behavior."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[2]
STEM='nexus_timed_event_20261026'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(report):
    page=(report/(STEM+'.html')).read_text()
    links=sorted(set(re.findall(r'(?:src|href)="([^"]+)"',page)))
    for link in links:
        if not (report/link).is_file():raise ValueError(f'Missing HTML resource: {link}')
    manifest=json.loads((report/(STEM+'_manifest.json')).read_text())
    for relative,sha in manifest['files_sha256'].items():
        if digest(report/relative)!=sha:raise ValueError(f'Changed delivery: {relative}')
    return {'status':'passed','html_links_verified':len(links),'file_hashes_verified':len(manifest['files_sha256'])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--deliver-root',type=Path);a=p.parse_args();a.root=a.root.resolve()
    request=json.loads((a.root/'event_request.json').read_text())
    analysis=json.loads((a.root/'analysis.json').read_text())
    checks=json.loads((a.root/'compiler_checks.json').read_text())
    review=json.loads((a.root/'review/verification.json').read_text())
    assert checks['status']==review['status']=='passed' and analysis['technical_checks_passed']
    assert len(analysis['render_geometry'])==2
    report=ROOT/'report';assets=report/'assets'/STEM;assets.mkdir(parents=True,exist_ok=False)
    for name in ('event_request.json','analysis.json','compiler_checks.json'):
        shutil.copy2(a.root/name,assets/name)
    for name in ('run.json','normal_condition.json','event_condition.json'):
        shutil.copy2(a.root/'raw'/name,assets/name)
    shutil.copytree(a.root/'review',assets/'review')
    (assets/'raw_video').mkdir()
    for name in ('normal','event'):
        shutil.copy2(a.root/'renders'/name/'camera_0.mp4',assets/'raw_video'/f'{name}.mp4')
        shutil.copy2(a.root/'renders'/name/'run.json',assets/f'{name}_render_audit.json')
    code=[ROOT/'nexus_3dgs/timed_event.py']+[ROOT/'scripts/pipeline'/name for name in
          ('run_nexus_timed_event.py','check_nexus_timed_event.py','analyze_nexus_timed_event.py','render_nexus_timed_event.py',
           'review_nexus_timed_event.py','export_nexus_timed_event_report.py','render_nexus_registered_prediction.py')]
    (assets/'code').mkdir()
    for path in code:shutil.copy2(path,assets/'code'/path.name)
    shutil.copy2(ROOT/'docs/nexus_timed_event_implementation_20261026.md',assets/'implementation.md')
    rows=[]
    for seed,data in analysis['seeds'].items():
        g=data['geometry']['event']
        rows.append([seed,round(data['max_ego_position_difference_m'],3),round(data['max_after_event_ego_lateral_difference_m'],3),
                     g['ego_event_actor_overlapping_samples'],','.join(map(str,data['other_responding_slots'])) or '无',
                     ','.join(map(str,data['nearby_other_responding_slots'])) or '无','未通过'])
    header=['种子','自车最大位置差/m','事件后最大横向差/m','自车—公交重叠/80采样','其他响应槽位','10米内响应车','完整交互']
    with (assets/'summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)
    table='| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'
    table+=''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
    render_table='| 视频 | 帧数 | 自车—公交重叠帧 | 全场景重叠对×帧 |\n| --- | --- | --- | --- |\n'
    for name,g in analysis['render_geometry'].items():
        render_table+=f'| {name} | {g["frames"]} | {g["ego_bus_overlapping_frames"]} | {g["overlapping_pair_frame_samples"]} |\n'
    rel=f'assets/{STEM}/'
    md=f'''# 指定车辆、时段与行为：Nexus 事件条件生成

**定时事件控制和视频链路已跑通；首轮完整交互行为未通过。** 自车未来没有指定。三组事件轨迹均出现自车与公交重叠，没有生成可确认的“自车闪躲→第三车交互”。

## 这次实现了什么

1. 上游 JSON 指定场景、车辆 token、行为、时段和参数。
2. 将行为编译为连续轨迹片段，在原生 0.5 秒网格上固定目标车的局部状态。
3. Nexus 自由生成自车、其他车辆与目标车事件外的未来。
4. 使用相同噪声配对生成正常／事件两组；预先固定种子 0、1、2，全部保留。
5. 审计局部条件、输入隔离、响应、物理车框和实际渲染位姿。
6. 复用已验收 v9 资产生成 80 帧、10 fps 视频；编号只放在左侧俯视图。

## 首轮事件

{request['description_zh']}

场景：`{request['scenario_token']}`；车辆：`{request['actor_track_token']}`。第 3、3.5、4、4.5、5、5.5、6 秒共 7 个状态受约束。倒车使用五次平滑位移曲线，端点速度平滑；公交初始停止。正常组在相同时段保持停止。两组都不提供其他车辆的未来状态。

本场景除公交外的其他车距离自车较远，先用于局部事件控制验证。事件外公交也自由生成，因此第 6 秒之后不保证继续停止或倒车。相机跟随 Nexus 自车预测，模型若输出重叠，渲染不会自动阻止穿车。

## 全部种子的结果

{table}

位置差是事件组相对同噪声正常组的差异，不代表合理避让。横向差均不足 0.11 米。种子 0 的槽位 13 出现超过 0.5 米的位置差，但不在自车 10 米内，不能算第三车受到自车闪躲影响。重叠表使用物理尺寸车框，在未来 0.1 秒插值网格的 80 个采样点检查。

Nexus 联合生成整段未来：事件开始之前也可能出现条件差异。所有车辆都能看到条件，第三车的差异也不能直接归因为自车的中介作用。因此本次没有声称严格的顺序因果链。

## 配对视频

展示种子在运行前固定为 0，未按画面效果挑选。视频是原始生成结果的诊断，包含失败。

<video controls preload="metadata" src="{rel}review/paired_event.mp4"></video>

[正常组单独查看]({rel}review/normal_explained.mp4) · [事件组单独查看]({rel}review/event_explained.mp4)

[正常组无标记原始渲染]({rel}raw_video/normal.mp4) · [事件组无标记原始渲染]({rel}raw_video/event.mp4)

![事件发生期间的对照帧]({rel}review/pair_050.jpg)

实际高斯车体安装后，在真实曝光时刻再检查，结果与插值网格口径不同：

{render_table}

已查看起始、事件中和末尾的关键帧。事件中相机接近并进入公交车体，末尾出现未充分观测表面的灰色／模糊区域；这段视频不能作为通过画质与交互验收的展示样例。

## 技术验证与当前限制

- 编译器检查通过，包括非法时间／缺失车辆拒绝、未来数值污染不影响条件、自车与其他车未来自由、三种行为的端点检查。
- 六次推理均精确保留条件；相同种子的两组初始噪声一致；官方 Nexus 权重未更新。没有对输出轨迹做朝向修复、避碰修复或筛选替换。
- 未来存在 mask 沿用记录并在两组间保持一致；不能称为完全不使用未来信息的部署预测实验。
- 已实现倒车靠近、横向移动、制动三类数值条件。实际模型与视频验证目前仅覆盖本次倒车靠近；其余两类只做编译器检查。不是任意自然语言行为理解。
- 当前是离线条件交互生成。WorldEngine 与外部自车策略不是本次视频生成的前置依赖。
- 该场景并不适合验证密集三车连锁响应；后续需要选择自车侧后方存在相邻车辆的场景，并继续保留自车未来自由。生成质量仍需验证，不能靠指定自车闪躲来冒充自由响应。

## 接口、计划和原始审计

[事件请求]({rel}event_request.json) · [实施步骤]({rel}implementation.md) · [全部审计]({rel}analysis.json) · [编译器检查]({rel}compiler_checks.json) · [生成记录]({rel}run.json) · [CSV]({rel}summary.csv)

[条件编译器]({rel}code/timed_event.py) · [推理入口]({rel}code/run_nexus_timed_event.py) · [审计入口]({rel}code/analyze_nexus_timed_event.py)

原始 PT 与轨迹包保存在仓库 `artifacts/nexus_3dgs/timed_event_20261026/raw/seed_000` 至 `seed_002`，每组有 `normal` 和 `event` 两个子目录。
'''
    (report/(STEM+'.md')).write_text(md)
    import markdown
    body=markdown.markdown(md,extensions=['tables','fenced_code'])
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nexus 定时事件控制与交互验证</title><style>body{max-width:1440px;margin:auto;padding:28px;font:18px/1.75 system-ui;color:#24364c}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #c3cfdd;padding:9px;text-align:left}video,img{width:100%;height:auto;margin:16px 0}code{overflow-wrap:anywhere;background:#eef2f7}a{color:#17649b}h1,h2{color:#164f74}</style>'+body+'</html>'
    (report/(STEM+'.html')).write_text(page)
    files=[report/(STEM+'.html'),report/(STEM+'.md')]+[p for p in assets.rglob('*') if p.is_file()]
    manifest={'technical_control_passed':True,'complete_interaction_chain_validated':False,
              'files_sha256':{str(p.relative_to(report)):digest(p) for p in files}}
    (report/(STEM+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    checks={'local':verify(report)}
    if a.deliver_root:
        destination=a.deliver_root;destination.mkdir(parents=True,exist_ok=True)
        shutil.copytree(assets,destination/'assets'/STEM)
        for suffix in ('.html','.md','_manifest.json'):shutil.copy2(report/(STEM+suffix),destination/(STEM+suffix))
        checks['delivered']=verify(destination)
    (report/(STEM+'_delivery_checks.json')).write_text(json.dumps(checks,indent=2)+'\n')
    if a.deliver_root:shutil.copy2(report/(STEM+'_delivery_checks.json'),a.deliver_root/(STEM+'_delivery_checks.json'))
    progress={'steps':{str(i):'complete' for i in range(1,7)},'technical_control_passed':True,
              'complete_interaction_chain_validated':False,'report':str(report/(STEM+'.html')),'delivery_checks':checks}
    (a.root/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
