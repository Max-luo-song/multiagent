"""Publish a scoped recommendation from paired mode and preprocessing controls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[2]
STEM='nexus_input_modes_comparison_20261026'
LABELS={'bp_history':'历史 cut-in → BP','history_full_target':'相同历史＋A 后续固定','future_full_target':'全部历史＋A 未来事件固定',
        'pure_sg_free_ego':'仅固定 A，其余全生成','official_sg_fixed_ego':'官方样式：自车也固定','pure_sg_bad_schedule':'错误历史日程对照'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def span(values,digits=3):
    if not values:return '—'
    lo,hi=min(values),max(values)
    return f'{lo:.{digits}f}' if abs(hi-lo)<10**(-digits) else f'{lo:.{digits}f}～{hi:.{digits}f}'


def table(header,rows):
    return '| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def verify(folder):
    manifest=json.loads((folder/(STEM+'_manifest.json')).read_text())
    for relative,digest in manifest['files_sha256'].items():assert sha(folder/relative)==digest,relative
    page=(folder/(STEM+'.html')).read_text();links=sorted(set(re.findall(r'(?:src|href)="([^"]+)"',page)))
    for link in links:assert (folder/link).is_file(),link
    return {'status':'passed','hashes_verified':len(manifest['files_sha256']),'html_links_verified':len(links)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deliver-root',type=Path);args=p.parse_args()
    roots={'official':ROOT/'artifacts/nexus_3dgs/input_modes_20261026','raw_velocity':ROOT/'artifacts/nexus_3dgs/input_modes_rawvel_20261026'}
    runs={n:json.loads((r/'run.json').read_text()) for n,r in roots.items()}
    audits={n:json.loads((r/'audit.json').read_text()) for n,r in roots.items()}
    for n,r in roots.items():
        assert runs[n]['status']=='complete' and audits[n]['status']=='passed'
        assert json.loads((r/'review/verification.json').read_text())['status']=='passed'
    report=ROOT/'report';assets=report/'assets'/STEM;assets.mkdir(parents=True,exist_ok=False)
    for name,root in roots.items():
        dest=assets/name;dest.mkdir()
        for file in ('run.json','protocol.json','audit.json','summary.csv'):shutil.copy2(root/file,dest/file)
        shutil.copytree(root/'conditions',dest/'conditions')
        shutil.copytree(root/'review',dest/'review')
        if (root/'source_snapshots').exists():shutil.copytree(root/'source_snapshots',dest/'source_snapshots')
    (assets/'code').mkdir()
    files=[ROOT/'nexus_3dgs/input_mode_conditions.py']+[ROOT/'scripts/pipeline'/name for name in
        ('compare_nexus_input_modes.py','check_nexus_input_modes.py','audit_nexus_input_modes.py','review_nexus_input_modes.py','export_nexus_input_modes_report.py')]
    for file in files:shutil.copy2(file,assets/'code'/file.name)
    case=ROOT/'artifacts/nexus_3dgs/multicar_event_033_20261026'
    for file in ('sensor_preflight.json','selection_evidence.json','initial_geometry.png'):shutil.copy2(case/file,assets/file)
    for name in ('multicar_selection_20261026','multicar_selection_all_20261026'):
        shutil.copy2(ROOT/'artifacts/nexus_3dgs'/name/'selection.json',assets/(name+'.json'))
    shutil.copy2(ROOT/'docs/nexus_input_modes_comparison_20261026.md',assets/'comparison_plan.md')
    # Config is the same frozen source used in the earlier accepted experiment.
    config=ROOT/'artifacts/nuplan_evaluation/independent256_v1/config.yaml'
    frozen=json.loads((ROOT/'artifacts/nexus_3dgs/interaction_v9_20261025/frozen_inputs.json').read_text())
    assert sha(config)==frozen['inputs']['source_config']['sha256'];shutil.copy2(config,assets/'config.yaml')
    official=audits['official']['mode_summaries'];raw=audits['raw_velocity']['mode_summaries']
    rows=[]
    for mode,s in official.items():
        drift=[v['quality']['ego_current_drift_from_reference_m'] for v in runs['official']['samples'].values() if v['mode']==mode and v['variant']=='event']
        rows.append([LABELS[mode],s['event_seeds'],span(s['ego_response_m']),span(s['third_response_m']),span(drift,2),
                     '/'.join(map(str,s['event_generated_future_pair_overlaps']))])
    measurements=table(['模式','事件种子数','自车最大位置差/m','C 最大位置差/m','自车当前位置偏移/m','各事件种子重叠对×时刻'],rows)
    noise_bad=runs['official']['samples']['pure_sg_bad_schedule/seed_000/event']['quality']
    noise_good=runs['official']['samples']['pure_sg_free_ego/seed_000/event']['quality']
    velocity_rows=[]
    for name,summary in (('官方 ResetV',official),('原始速度',raw)):
        for mode in ('bp_history','future_full_target'):
            s=summary[mode]
            velocity_rows.append([name,LABELS[mode],span(s['ego_response_m']),span(s['third_response_m']),span(s['event_boundary_acceleration_mps2'],1)])
    velocity_table=table(['预处理','模式','自车位置差/m','C 位置差/m','预测边界加速度/m·s⁻²'],velocity_rows)
    rel=f'assets/{STEM}/'
    baseline=runs['official']['samples']['bp_history/seed_000/event']['quality']
    boundary=audits['official']['samples']['bp_history/seed_000/event']['boundary_kinematics']
    raw_boundary=audits['raw_velocity']['samples']['bp_history/seed_000/event']['boundary_kinematics']
    recommendation='主接口采用“所有车辆历史固定＋危险车完整未来固定＋自车及其他车未来自由”；历史 cut-in 的 BP 作为基线与事件后续测试。当前两者的交互生成均未达到验收，不能宣称已有可靠三车连锁响应。'
    md=f'''# Nexus 两种输入方式：项目适用性与实测对比

**{recommendation}** 该选择依据任务接口需求与对照诊断，不是宣称某模式已取得成功演示。

## 先回答两种思路

| 思路 | 是否可做 | 实际含义 | 项目用途 |
| --- | --- | --- | --- |
| 在前两秒安排 cut-in，再预测 | 可以 | A 的 cut-in 已经成为给定历史，只观察当前以后的响应；A 后续也自由 | BP 基线、事件后的行为续写 |
| 固定 A 整段轨迹，安排未来某时刻 cut-in | 可以设置条件 | A 的动作与时间可控制，其他对象生成；官方 SG 训练始终固定 ego 全轨迹 | 需保留全体历史、放开 ego 未来后，用于本项目未来事件编辑 |
| 只固定 A，其余连历史一起随机生成 | 接口可运行 | 原有自车和其他车当前布局也会改变；自由 ego 不是官方 SG 固定自车设置 | 不宜直接用作真实重建场景的编辑主线 |

第一种里，自车的前两秒历史也是给定的，无法观察模型在这两秒内自行闪躲；能观察的是当前之后的反应。第二种也不是必然产生交互，固定的是危险车的动作，响应质量仍要验证。

## 为什么采用保留历史的第二种

上游给出“哪辆车、何时、什么危险动作”，我们应稳定保持原场景的身份、初始状态和重建资产绑定，同时给未来事件确定的轨迹条件。将所有历史锁定，能避免 SG 重新生成起始交通；只固定 A 的未来，不向 ego 或 C 指定响应。

这属于通过任务掩码实现的条件生成扩展。官方 `generate_task_mask` 的 SG 分支始终设置 ego 全轨迹为已知，不能把释放 ego 后的结果当作已验证的原生 SG 能力。源码参考与本次编译代码均留档。

## 对比怎么做

- 场景 033：自车约 12.45m/s，左前 A 约 10.15m/s，右侧 C 约 11.77m/s；三者同向。原公交场景周围车辆过远，仅保留为接口验证。
- 原始 256 个场景中按历史几何筛出 14 个三车候选；此前传感器端点可用的 53 个场景中，没有命中本次严格的行进三车条件。
- 全部使用官方权重，不训练；主模式种子 0、1、2，正常／事件配对同噪声；展示种子预先固定为 0。
- 完成 27 次模式／噪声控制，再补充 12 次原始速度控制，共 39 次成功推理。
- 历史 cut-in 位于 [-2,0] 秒，未来 cut-in 位于 [0.5,2.5] 秒，向右横移均为 3 米。两者不是相同时刻的任务，不能只按响应差大小排名。
- 加入“历史 cut-in＋A 全未来固定”模式，与 BP 使用逐值一致的历史，单独考察固定 A 后续的作用。
- 若直接把 A 原位置切入历史，当前车框已重叠。因此比较前统一将 A 沿车头前移 2 米，正常／事件及所有模式都相同。这是公开的输入编辑，不修改输出轨迹。
- 官方样式固定 ego 只运行种子 0 作参考；其 ego 的行为不是自由响应，不参与主方案选择。

![三车初始几何示意]({rel}initial_geometry.png)

上图是原始候选，A 的原始前距约 6 米；本次比较公共前移 2 米后的前距约 8 米。该示意没有道路拓扑，车道可行性尚未核验。

## 之前的怪轨迹是否由噪声引起

**确实有明确的历史噪声日程问题，但修复后仍有其他问题。** 默认全序列日程把前 5 个时刻的噪声时间置零；未知历史中的随机状态因此没有按正确时刻去噪。修复为全时域可去噪，再由条件掩码锁定已知状态。

同一场景、同一 seed 0、同一条件与初始噪声，仅修正日程：

| 未知车辆相邻 0.5 秒位移 | 错误日程 | 修正日程 |
| --- | --- | --- |
| P95/m | {noise_bad['unknown_adjacent_step_p95_m']:.2f} | {noise_good['unknown_adjacent_step_p95_m']:.2f} |
| 最大值/m | {noise_bad['unknown_adjacent_step_max_m']:.2f} | {noise_good['unknown_adjacent_step_max_m']:.2f} |

这是连续性诊断，覆盖有效且至少一端未知的历史／未来相邻状态；不是预测误差或安全评分。错误日程样本没有通过轨迹包合法性导出检查，原始 PT 仍保留。

修正后，自由历史 SG 的自车当前点仍偏移约 28.85～43.61 米，三组事件场景仍有 245～786 个涉及生成状态的重叠车辆对×时刻。不能把平滑当成保留原场景或合理多车交互。

**所有车历史固定时，这个历史噪声问题不会触发。** 因为那些前 5 个状态本来就是条件。此前公交／局部条件失败，以及本次历史锚定模式的问题，需要另找原因。

<video controls preload="metadata" src="{rel}official/review/noise_schedule_control.mp4"></video>

## 全部主模式的响应与几何

{measurements}

位置差均为同 seed 事件组相对正常组、未来有效时刻的最大位置差，表中列出三种子的范围。重叠使用 16 个未来半秒采样，真实参考车体尺寸、ego 后轴到中心偏移 1.461 米；只统计至少一方为生成状态的重叠对。它没有覆盖连续时间，也不是渲染安装后的车体审计。

纯 SG 较大的响应伴随初始交通重生成，不能与保持原历史的模式当作公平的同场景反应强弱比较。官方固定 ego 分支也不能用于证明自车自由反应。

## 自车预测的实际异常

正常与事件组的自车、C 差异均不足以支持三车连锁反应。车框未撞到 A 也不代表避让成功：默认预处理下，种子 0 的 BP 自车在预测首个半秒区间的纵向速度约为 {boundary['ego_first_future_longitudinal_speed_mps']:.2f}m/s，而记录中的当前前进速度约 12.45m/s；预测边界的位置差分加速度约 {boundary['ego_prediction_boundary_acceleration_mps2']:.1f}m/s²。视频可看到预测起点的反向突变。

这不是提供的闪躲轨迹，也没有用后处理消除。Nexus 原有行为预测指标主要排除 ego，因此此前非自车指标不能替代本次的自车有效性验收。

## 速度预处理的额外控制

沿用官方 ResetV 的主对比之外，另保留原始速度通道，重新进行 BP 与“全部历史＋A 未来固定”的 12 次配对。未修改官方预处理实现或旧量化结果。

{velocity_table}

原始速度控制中，种子 0 BP 的首个未来区间纵向速度仍约 {raw_boundary['ego_first_future_longitudinal_speed_mps']:.2f}m/s，预测边界加速度约 {raw_boundary['ego_prediction_boundary_acceleration_mps2']:.1f}m/s²。其余种子也有起点突变，并出现部分自车—C 重叠。只改输入方式、只修历史噪声、或只改速度预处理，都没有让本例达到完整交互验收。

## 模式动画

每段包含 2 秒历史＋后续生成，共 100 帧、10fps。左正常、右 cut-in；颜色区分 E／A／C。图示跟随每组自车平移，标题另列当前状态漂移。背景网格不是道路或可行驶区域。此场景缺少真实 RGB／LiDAR，以下均为俯视轨迹动画，不是 3DGS 视频。
'''
    for mode in ('bp_history','history_full_target','future_full_target','pure_sg_free_ego'):
        md+=f'\n### {LABELS[mode]}\n\n<video controls preload="metadata" src="{rel}official/review/{mode}.mp4"></video>\n'
    md+='\n### 原始速度对照\n'
    for mode in ('bp_history','future_full_target'):
        md+=f'\n{LABELS[mode]}：\n\n<video controls preload="metadata" src="{rel}raw_velocity/review/{mode}.mp4"></video>\n'
    md+=f'''
## 接下来应走的路线

1. 接口上采用所有历史锚定＋危险车完整未来条件，自车未来保持自由；BP 历史事件作为对照。
2. 优先解决当前 checkpoint／生成配置下自车预测起点的反向突变与不合理加速度，再评价事件响应；初始点正确与轨迹平滑都只是部分验收。
3. 继续检查道路、车道和无关车辆冲突。事件执行、自车响应、C 响应与完整交互分别记录，不能用给定 ego 轨迹补出成功故事。
4. 选定行为验证通过且真实传感器完整的场景后，再投入 3DGS 重建与视频质量优化。033 的 RGB／LiDAR 端点当前缺失，不复用公交场景资产冒充新地点。

这是一例场景的小样本功能对比，支持当前接口选择与问题定位，不是两种模型任务的泛化性能排名。未验证严格因果链，也未运行独立自车策略闭环。

## 原始证据

[对比前计划]({rel}comparison_plan.md) · [主对比协议]({rel}official/protocol.json) · [主对比独立审计]({rel}official/audit.json) · [主对比 CSV]({rel}official/summary.csv)

[速度控制协议]({rel}raw_velocity/protocol.json) · [速度控制审计]({rel}raw_velocity/audit.json) · [速度控制 CSV]({rel}raw_velocity/summary.csv) · [当前传感器检查]({rel}sensor_preflight.json)

[全部 256 场景筛选]({rel}multicar_selection_all_20261026.json) · [39 次推理的运行入口]({rel}code/compare_nexus_input_modes.py) · [条件编译器]({rel}code/input_mode_conditions.py)

第一轮 27 次使用的原运行源码另存于资产目录 `official/source_snapshots`，其 SHA 与协议一致；速度对照使用显式参数化后的入口。主实验原始 PT 与轨迹包保存在仓库 `artifacts/nexus_3dgs/input_modes_20261026/`，速度对照保存在 `input_modes_rawvel_20261026/`。未覆盖原公交、局部事件或 v9 资产。
'''
    import markdown
    body=markdown.markdown(md,extensions=['tables','fenced_code'])
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nexus 两种输入方式对比</title><style>body{max-width:1440px;margin:auto;padding:28px;font:18px/1.75 system-ui;color:#24364b}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #c3cfdd;padding:9px;text-align:left}video,img{width:100%;height:auto;margin:16px 0}code{overflow-wrap:anywhere;background:#eef2f7}a{color:#17649b}h1,h2{color:#164f74}</style>'+body+'</html>'
    (report/(STEM+'.md')).write_text(md);(report/(STEM+'.html')).write_text(page)
    result_doc=ROOT/'docs/nexus_input_modes_results_20261026.md'
    result_doc.write_text('# 输入方式比较结论\n\n'+recommendation+'\n\n'+measurements+'\n\n'+velocity_table+'\n\n完整报告：[HTML](../report/'+STEM+'.html)。\n')
    resources=[report/(STEM+'.html'),report/(STEM+'.md')]+[p for p in assets.rglob('*') if p.is_file()]
    manifest={'status':'generated','successful_inference_samples':sum(len(r['samples']) for r in runs.values()),
              'complete_interaction_chain_validated':False,'recommended_interface':'all histories + target full future; ego future free',
              'files_sha256':{str(p.relative_to(report)):sha(p) for p in resources}}
    (report/(STEM+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    checks={'local':verify(report)}
    if args.deliver_root:
        destination=args.deliver_root;destination.mkdir(parents=True,exist_ok=True)
        shutil.copytree(assets,destination/'assets'/STEM)
        for suffix in ('.html','.md','_manifest.json'):shutil.copy2(report/(STEM+suffix),destination/(STEM+suffix))
        checks['delivered']=verify(destination)
    (report/(STEM+'_delivery_checks.json')).write_text(json.dumps(checks,indent=2)+'\n')
    if args.deliver_root:shutil.copy2(report/(STEM+'_delivery_checks.json'),args.deliver_root/(STEM+'_delivery_checks.json'))
    (roots['official']/'comparison_completion.json').write_text(json.dumps({'comparison_complete':True,'samples':39,
        'generation_quality_accepted':False,'report':str(report/(STEM+'.html')),'delivery_checks':checks},indent=2)+'\n')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
