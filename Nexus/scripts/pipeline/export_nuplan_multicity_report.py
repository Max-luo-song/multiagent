"""Export completed four-city evaluation results as portable MD/HTML/ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import zipfile

from export_nuplan_metrics_report import render
from run_nuplan_metrics import json_write
from run_repro_demo import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run/'run.json').read_text())
    official = json.loads((run/'official_metrics_by_seed.json').read_text())
    audit = json.loads((run/'audited_displacement_by_city_seed.json').read_text())
    plan = json.loads((run/'frozen_plan.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['evaluated_scenarios'] == 128 and manifest['evaluated_scene_seeds'] == 384
    assert manifest['model_parameter_versions_unchanged'] and manifest['training_updates'] == 0
    assert set(manifest['records']) == {f'{s["index"]}:{seed}' for s in plan['scenarios'] for seed in manifest['seeds']}
    assert file_sha256(run/'frozen_plan.json') == manifest['plan_sha256']
    assert file_sha256(Path(__file__).with_name('run_nuplan_multicity.py')) == manifest['runner_sha256']
    assert all((run/v['prediction_file']).is_file() for v in manifest['records'].values())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'assets').mkdir()
    (output/'data').mkdir()
    for name in ('run.json','config.yaml','frozen_plan.json','official_metrics_by_seed.json','audited_displacement_by_city_seed.json'):
        shutil.copy2(run/name, output/'data'/name)
    builder_check = run.parent/'multicity128_plan_v2_builder_check.json'
    check = json.loads(builder_check.read_text())
    assert check['status'] == 'complete' and check['plan_sha256'] == manifest['plan_sha256']
    shutil.copy2(builder_check, output/'data'/builder_check.name)
    seeds = [str(s) for s in manifest['seeds']]
    def values(group, key, scale=1):
        return [official[s][group][key]*scale for s in seeds]
    def mean_sd(v):
        return {'mean':statistics.mean(v), 'seed_sample_sd':statistics.stdev(v)}
    def display(v):
        return f'{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}'
    cities = [('las_vegas','Las Vegas'),('sg-one-north','Singapore'),('us-ma-boston','Boston'),('us-pa-pittsburgh-hazelwood','Pittsburgh')]
    summary = {'scope':manifest['scope'], 'seeds':manifest['seeds'], 'official':{}, 'audited':{}}
    rows = []
    specifications = [
        ('普通预测整秒误差均值（1–8 秒，米）','agent_behavior_prediction','displacement@avg_8s',1),
        ('给定终点整秒误差均值（1–7 秒，米）','intent_condtioned_agent_behavior_prediction','displacement@avg_7s',1),
        ('普通预测代码碰撞计数比 × 100','collision_rate_bp','collision@avg',100),
        ('给定终点代码碰撞计数比 × 100','collision_rate_gp','collision@avg',100),
        ('普通预测道路距离代理 × 100','traj_metric_bp','offroad_rate',100),
        ('给定终点道路距离代理 × 100','traj_metric_gp','offroad_rate',100),
    ]
    for label, group, key, scale in specifications:
        vals = values(group,key,scale)
        summary['official'][group+'/'+key] = dict(mean_sd(vals), scale=scale)
        rows.append('| '+label+' | '+' | '.join(f'{v:.4f}' for v in vals)+' | '+display(vals)+' |')
    city_rows = []
    for city,label in [('all','全部城市')]+cities:
        summary['audited'][city] = {}
        for task in ('bp','goal'):
            summary['audited'][city][task] = mean_sd([audit[s][city][task]['mean_error_m'] for s in seeds])
            assert len({audit[s][city][task]['valid_points'] for s in seeds}) == 1
        city_rows.append(f'| {label} | {audit[seeds[0]][city]["scenes"]} | '+
                         display([audit[s][city]['bp']['mean_error_m'] for s in seeds])+' | '+
                         display([audit[s][city]['goal']['mean_error_m'] for s in seeds])+' | '+
                         str(audit[seeds[0]][city]['bp']['valid_points'])+' / '+str(audit[seeds[0]][city]['goal']['valid_points'])+' |')
    json_write(output/'data/summary.json',summary)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11.5,4),constrained_layout=True)
    for ax,group,title in zip(axes,('agent_behavior_prediction','intent_condtioned_agent_behavior_prediction'),('History conditioned','History + final valid state conditioned')):
        for seed in seeds:
            ax.plot(range(1,9),[official[seed][group][f'displacement@{t:.6f}s'] for t in range(1,9)],marker='o',label='Seed '+seed,alpha=.8)
        ax.set(title=title,xlabel='Future horizon (s)',ylabel='Official displacement error (m)')
        ax.set_xticks(range(1,9)); ax.grid(alpha=.25); ax.legend()
    axes[1].annotate('8 s state is an input condition',xy=(8,0),xytext=(3.4,.7),arrowprops={'arrowstyle':'->'},fontsize=8)
    fig.savefig(output/'assets/01_horizon_errors.png',dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,4.5),constrained_layout=True)
    for task,shift,color,label in [('bp',-.18,'#3976bc','History conditioned'),('goal',.18,'#21a084','History + final valid state')]:
        vals = [summary['audited'][city][task] for city,_ in cities]
        ax.bar([i+shift for i in range(4)],[v['mean'] for v in vals],width=.34,yerr=[v['seed_sample_sd'] for v in vals],capsize=4,color=color,label=label)
    ax.set_xticks(range(4),[label for _,label in cities]); ax.set_ylabel('Audited future-point mean error (m)')
    ax.set_title('Four-city subset; error bars show seed SD, not a dataset confidence interval')
    ax.grid(axis='y',alpha=.2); ax.legend()
    fig.savefig(output/'assets/02_city_errors.png',dpi=160); plt.close(fig)
    report = '''# Nexus 四城定量复现结果报告

报告整理日期：2026 年 10 月 21 日

## 1. 已完成内容

此前已完成行为预测和场景生成演示，以及 8 个调试场景上的官方指标试跑、速度预处理对照和约束敏感性对照。本次进一步完成独立四城验证子集评测，使用已发布权重进行推理，没有训练或微调。

| 项目 | 完成情况 |
| --- | --- |
| 唯一场景 | 128 个，4 个城市各 32 个 |
| 行程 | 32 段，每城 8 段；排除前期调试所用整段行程 |
| 随机种子 | 0、1、2，共 384 个场景与种子组合 |
| 任务 | 普通行为预测、给定最后有效状态的行为预测，共 768 条场景级任务输出 |
| 批量推理 | 每批 4 个场景；384 份预测文件已保存 |
| 完成状态 | complete；模型参数版本检查通过；训练更新为 0 |

场景从 1,381 个官方验证数据库中按固定哈希顺序选取，先应用官方首尾场景规则，再检查历史至少 2 秒、未来至少 8 秒。每段行程选 4 个间隔至少 15 秒的场景，选样不使用预测结果。128 个 token 均通过真正的 nuPlan 场景构建器检查。它是城市平衡的验证子集，不是全验证集，也不是总体分布的无偏估计。

## 2. 配置与任务条件

使用官方 nuplan.ckpt、正式 Nexus 特征构建器和官方 ResetVAugmentor；map / sma / clip 强度均为 1，32 步 full_sequence 采样。没有为了得到较低误差而替换成前期调试中的原始速度或弱约束设置。普通预测给定全部车辆过去 2 秒及当前状态，生成未来 8 秒；另一项任务额外给定各车辆最后有效时刻的真实完整状态，具有更强输入条件。

每批和每个种子的实际随机数种子由冻结清单哈希、种子标签与批次 token 派生，保存于运行记录中。各官方指标使用独立数据副本，避免掩码和地图被原地修改后影响其他指标。

## 3. 官方代码指标

表内“±”表示三个随机种子的样本标准差，只描述采样随机性，不是场景总体的置信区间。

| 指标 | 种子 0 | 种子 1 | 种子 2 | 均值 ± 种子标准差 |
| --- | --- | --- | --- | --- |
'''+'\n'.join(rows)+'''

![两项任务在不同预测时域的误差](assets/01_horizon_errors.png)

普通预测误差随预测时域增加，8 秒时约为 14 米，说明远期轨迹仍有较大偏差。给定终点任务误差较低，但它已获得未来状态作为条件；8 秒处误差为零来自输入约束，不能作为独立预测终点成功的证据。部分短轨迹的已给定终点早于 8 秒，前七秒官方均值仍可能包含这类条件点。

官方位置误差先在各整秒时刻对有效车辆求均值，再平均各整秒结果；并不等于全部半秒采样点上的 ADE。碰撞计数比按相交车辆对计数后除以有效车辆数，道路距离代理按车辆中心到地图采样点的 2.75 米阈值判断；不能分别直接解释为碰撞车辆占比和可行驶区域多边形越界率。给定终点任务的位置误差较低，但碰撞计数比和道路距离代理均高于普通预测，因此没有出现全部指标同时改善。

## 4. 按城市展示与误差审计

下表对所有有效未来半秒采样点求位置误差，排除自车、历史和无效帧；给定终点任务另外排除每辆车已知的最后有效点。该审计口径与上一表的官方整秒口径不同。

| 城市 | 场景数 | 普通预测误差（米） | 给定终点误差（米） | 每个种子的有效点数：普通 / 终点 |
| --- | --- | --- | --- | --- |
'''+'\n'.join(city_rows)+'''

![四城子集误差对比](assets/02_city_errors.png)

本子集中的 Las Vegas 和 Singapore 误差高于 Boston 和 Pittsburgh；同一城市内三个随机种子的波动远小于城市之间的差异。这说明仅更换采样种子不足以解释本次差异。但城市之间的道路、速度、车辆数量和场景类型并未匹配，不能据此断言某个城市的道路结构就是原因，也不能外推到各城市全部数据。

各城市场景数相同，有效车辆轨迹点数量却不同。总体审计误差按有效轨迹点加权，并非四个城市误差的等权平均。给定终点审计明确排除了较早结束轨迹的条件点，避免将已知答案计为预测精度；两类任务的有效点集合仍有差异，不能将两列简单相除作为严格同口径提升比例。

## 5. 当前复现结论与边界

已完成从单场景演示、小规模对照到四城 128 场景、3 种子双任务定量评测的流程，具备固定数据清单、配置、预测张量、指标和可核查运行记录。当前可以称为“已完成四城验证子集上的推理与定量评测复现”。

仍不能称为“论文全部结果复现完成”：本次没有运行全验证集；速度预处理与发布权重训练配置的对应关系尚未确认；碰撞和道路指标与论文定义仍需进一步对齐。批量、异步的官方时间字段不作为论文单场景延迟。前期 8 场景与本次场景集不同，两阶段数值变化不能用作模型性能提升的证据。

首次 v1 启动曾因选样未包含官方首尾过滤规则而失败，未产生任何预测。本次使用修复后的 v2 清单，保留 v1 记录用于追溯，没有混入正式结果。

## 6. 文件与离线阅读

本报告包包含 Markdown、两张 PNG、图片内嵌的离线 HTML，以及 data 目录中的配置、清单、逐种子指标和完整运行记录。原始预测张量约 196 MB，保留在服务器，本展示包不重复打包。

服务器原始结果目录：`'''+str(run)+'''`

下载完整 ZIP 后解压，保留 Markdown 与 assets 的相对位置；离线 HTML 可单独打开，图片不依赖服务器。飞书导入 Markdown 时，若图片未自动导入，将 assets 的两张 PNG 插入对应位置；单独上传 Markdown 不会携带本地图片。此前的演示 GIF 仍在先前的 Nexus复现成果报告完整包中。
'''
    md = output/'Nexus四城定量复现结果报告.md'
    md.write_text(report,encoding='utf-8')
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Nexus 四城定量复现结果</title><style>body{font:16px/1.8 system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:20px;color:#243247}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd6e0;padding:8px}.table{overflow:auto}h2{margin-top:32px}code{overflow-wrap:anywhere}</style><body>'+render(report,output)+'</body></html>'
    (output/'Nexus四城定量复现结果报告_离线预览.html').write_text(html,encoding='utf-8')
    refs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report)
    assert len(refs) == 2 and all((output/ref).is_file() for ref in refs)
    assert html.count('src="data:image/png;base64,') == 2
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output/'SHA256SUMS.txt').write_text('\n'.join(file_sha256(p)+'  '+p.relative_to(output).as_posix() for p in files)+'\n')
    files.append(output/'SHA256SUMS.txt')
    archive = output.with_name(output.name+'_完整包.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path,output.name+'/'+path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print('REPORT='+str(md),flush=True)
    print('ZIP='+str(archive),flush=True)
    print('PORTABLE_REPORT_CHECK_OK',flush=True)


if __name__ == '__main__':
    main()
