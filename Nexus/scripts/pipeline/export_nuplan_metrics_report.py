"""Export completed pilot diagnostics as a portable Markdown/HTML/ZIP report."""

import argparse
import base64
import hashlib
from html import escape
import json
from pathlib import Path
import re
import shutil
import zipfile


def render(text, output):
    def inline(value):
        value = escape(value)
        value = re.sub(r'`([^`]+)`', r'<code>\1</code>', value)
        return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', value)
    lines = text.splitlines()
    result, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        heading = re.match(r'^(#{1,6}) (.+)$', line)
        image = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if not line:
            i += 1
            continue
        if heading:
            n = len(heading[1])
            result.append(f'<h{n}>{inline(heading[2])}</h{n}>')
        elif image:
            data = base64.b64encode((output / image[2]).read_bytes()).decode('ascii')
            result.append(f'<figure><img src="data:image/png;base64,{data}" alt="{escape(image[1])}"><figcaption>{escape(image[1])}</figcaption></figure>')
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            result.append('<div class="table"><table>')
            for index, row in enumerate(rows):
                if index == 1 and all(re.fullmatch(r':?-+:?', c) for c in row):
                    continue
                tag = 'th' if index == 0 else 'td'
                result.append('<tr>' + ''.join(f'<{tag}>{inline(c)}</{tag}>' for c in row) + '</tr>')
            result.append('</table></div>')
            continue
        else:
            result.append('<p>' + inline(line) + '</p>')
        i += 1
    return '\n'.join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--preprocessing', type=Path, required=True)
    parser.add_argument('--constraints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = {'pilot': args.pilot.resolve(), 'preprocessing': args.preprocessing.resolve(), 'constraints': args.constraints.resolve()}
    manifests = {k: json.loads((p / 'run.json').read_text()) for k, p in runs.items()}
    assert all(m['status'] == 'complete' for m in manifests.values())
    assert all(m['evaluated_scenarios'] == 8 and m['training_updates'] == 0 for m in manifests.values())
    metrics = {k: json.loads((p / 'metrics.json').read_text()) for k, p in runs.items()}
    official = metrics['preprocessing']['official_reset_v']
    raw = metrics['preprocessing']['raw_velocity']
    grid = metrics['constraints']
    assert all(metrics['pilot'][k] == official[k] for k in official if k != 'time')
    repeat_matches = all(raw[k] == grid['map_sma_strong'][k] for k in raw if k != 'time')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'assets').mkdir()
    (output / 'data').mkdir()
    for name, path in runs.items():
        for file in ('run.json', 'metrics.json', 'config.yaml'):
            shutil.copy2(path / file, output / 'data' / (name + '_' + file))
    shutil.copy2(runs['preprocessing'] / 'preprocessing_comparison.png', output / 'assets/01_velocity_control.png')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bp_key, goal_key = 'agent_behavior_prediction', 'intent_condtioned_agent_behavior_prediction'
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    labels = {'map_sma_strong': 'Map/SMA strength 1.0', 'map_sma_weak': 'Map/SMA strength 0.1', 'clip_only': 'Clip only'}
    for ax, key, title in zip(axes, (bp_key, goal_key), ('History conditioned', 'History + goal conditioned')):
        for branch, label in labels.items():
            ax.plot(range(1, 9), [grid[branch][key][f'displacement@{t:.6f}s'] for t in range(1, 9)], marker='o', label=label)
        ax.set(title=title, xlabel='Future horizon (s)', ylabel='Mean displacement error (m)')
        ax.set_xticks(range(1, 9))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[1].annotate('8 s state is provided', xy=(8, 0), xytext=(4.8, 0.4), arrowprops={'arrowstyle': '->'}, fontsize=8)
    fig.savefig(output / 'assets/02_constraint_control.png', dpi=160)
    plt.close(fig)
    def bp(m):
        return m[bp_key]['displacement@avg_8s']
    def goal(m):
        return m[goal_key]['displacement@avg_7s']
    def safety_row(label, m):
        return (f'| {label} | {bp(m):.4f} | {goal(m):.4f} | '
                f'{100*m["collision_rate_bp"]["collision@avg"]:.4f} | '
                f'{100*m["collision_rate_gp"]["collision@avg"]:.4f} | '
                f'{100*m["traj_metric_bp"]["offroad_rate"]:.4f} | '
                f'{100*m["traj_metric_gp"]["offroad_rate"]:.4f} |')
    grid_rows = '\n'.join(safety_row(label, grid[key]) for key, label in [('map_sma_strong', '原始速度 + 强约束'), ('map_sma_weak', '原始速度 + 弱约束'), ('clip_only', '原始速度 + 仅裁剪')])
    record_count = sum(r['velocity_diagnostics']['valid_history_pairs'] for r in manifests['preprocessing']['records'])
    def weighted(key):
        return sum(r['velocity_diagnostics'][key]*r['velocity_diagnostics']['valid_history_pairs'] for r in manifests['preprocessing']['records'])/record_count
    same = '重复的强约束分支与上一轮原始速度分支，除计时外的聚合指标逐值一致。' if repeat_matches else '两轮实验分别保证内部同噪声，不默认认为跨轮次的随机采样完全相同。'
    constraint_finding = ('该批场景中，减弱约束后的两个误差均值均低于强约束，说明约束强度也影响本轮误差。'
                          if bp(grid['map_sma_weak']) < bp(grid['map_sma_strong']) and goal(grid['map_sma_weak']) < goal(grid['map_sma_strong']) else
                          '不同约束设置呈现指标差异，不能用某一个最低误差配置替代正式评测协议。')
    report = f'''# Nexus 定量复现阶段报告

实验日期：2026 年 10 月 20 日；报告整理日期：2026 年 10 月 21 日

## 1. 本阶段完成内容

在先前已完成行为预测和场景生成演示的基础上，本阶段跑通官方验证集上的双任务指标计算，并完成速度预处理和约束强度两组对照。所有实验使用同一官方 nuplan.ckpt，未训练或微调模型，输出通过有限值检查，模型参数版本检查通过。

| 实验 | 完成范围 | 验证目的 |
| --- | --- | --- |
| 官方验证配置试跑 | 8 场景 × 两类任务 | 检查验证分支、预处理、指标聚合 |
| 速度预处理对照 | 同一 8 场景 × 两种预处理 × 两类任务 | 检查 ResetV 对速度通道与结果的影响 |
| 约束敏感性对照 | 同一 8 场景 × 三种约束 × 两类任务 | 检查地图和平滑约束强度的影响 |

共完成 96 次任务级采样调用，其中包含有意重复的对照分支；唯一场景数仍为 8，不能把采样调用数当作场景数。

## 2. 数据与评测口径

共享数据包含与官方划分一致的 1,381 个验证数据库。本次使用前两个验证数据库中的 8 个唯一场景；两个数据库来自同一次 Las Vegas 行程的不同分段，样本不具有全验证集代表性。场景都有至少 2 秒历史和 8 秒未来，轨迹间隔为 0.5 秒。

两种任务分别为：保留全部车辆历史、生成未来的普通预测；以及在历史之外，额外给定各车辆最后有效时刻完整状态的给定终点预测。后者终点误差为零不能当成独立终点预测成绩。

使用官方特征构建、模型验证分支及指标公式。位置误差排除自车，在 1—8 秒整秒时刻计算；普通预测列报告八个整秒均值，给定终点列报告前七个整秒均值。它们不等同于所有半秒采样点上的标准 ADE，部分较早结束的短轨迹也可能在前七秒包含已给定终点。

各指标使用独立数据副本，避免官方代码中有效性掩码和地图坐标的原地修改相互污染。该适配保持各指标内部公式不变。

## 3. 官方配置试跑与速度预处理对照

官方 ResetVAugmentor 实现将速度通道的相邻时间差写回速度通道。对近似匀速车辆，这会把原本非零的速度转成接近零的数值。对照分支保留正式特征构建器输出的原始速度，其他输入和设置保持一致。

| 代码指标 | 官方 ResetV | 保留原始速度 |
| --- | --- | --- |
| 普通预测整秒误差均值（米） | {bp(official):.4f} | {bp(raw):.4f} |
| 给定终点前七秒整秒误差均值（米） | {goal(official):.4f} | {goal(raw):.4f} |
| 普通预测碰撞计数比 × 100 | {100*official['collision_rate_bp']['collision@avg']:.4f} | {100*raw['collision_rate_bp']['collision@avg']:.4f} |
| 给定终点碰撞计数比 × 100 | {100*official['collision_rate_gp']['collision@avg']:.4f} | {100*raw['collision_rate_gp']['collision@avg']:.4f} |

![速度预处理对照](assets/01_velocity_control.png)

两条分支在每个场景上使用相同 CPU/CUDA 随机数状态，并核验消耗后的状态一致。官方分支除计时外的指标与第一轮试跑逐值一致。

在 {record_count} 个有效历史相邻状态对上，原始速度与位置差分参考的平均向量差为 {weighted('raw_velocity_vs_position_difference_mean_l2'):.4f} 米/秒，ResetV 后通道为 {weighted('reset_velocity_vs_position_difference_mean_l2'):.4f} 米/秒。此结果与位移误差下降共同支持速度预处理是误差来源之一。但碰撞计数比没有同步改善，因此不能称为整体质量全面提升；也尚未确认作者发布权重训练时的预处理。

## 4. 约束强度对照

固定原始速度特征，保持场景、权重和随机噪声相同。强约束为 map / sma / clip 强度 [1, 1, 1]，对应官方测试脚本；弱约束为 [0.1, 0.1, 1]，仅用于诊断；仅裁剪对应模型构造器的默认约束。弱约束不声称是论文设置。

| 设置 | 普通误差（米） | 给定终点误差（米） | 普通碰撞比 × 100 | 终点碰撞比 × 100 | 普通越界代理 × 100 | 终点越界代理 × 100 |
| --- | --- | --- | --- | --- | --- | --- |
{grid_rows}

![约束强度对照](assets/02_constraint_control.png)

{same} {constraint_finding}

本次结果显示明确取舍：普通预测越界代理指标从强约束的约 4.79% 上升到弱约束的约 15.90%、仅裁剪的约 17.78%；给定终点任务的误差则分别约为 2.88、1.68、1.63 米。减弱约束改善部分误差，但没有带来所有指标的统一改善。约束会改变轨迹的平滑程度、运动幅度和道路贴合程度，因此必须同时观察误差、安全和道路指标。本报告展示所有分支，不通过挑选最低误差配置宣称达到论文成绩。

## 5. 结果解释与当前边界

代码中的碰撞指标按相交车辆对计数，再除以有效车辆数；越界代理指标按车辆中心到地图采样点的距离阈值判断。它们与统计发生碰撞的不同车辆、直接检查可行驶区域等口径有区别，不能直接混用为论文完全一致的指标。

本阶段验证了定量流程可运行、结果可重复，并找到了速度预处理和约束设置需要进一步核对的证据。尚未完成全验证集评测，未验证训练时的完整预处理配置，也未完成所有指标与论文定义的对齐。

正式扩大评测前，应固定评测任务、数据清单、预处理和约束配置，处理指标定义差异，再在独立明确的场景集上运行。当前 8 场景属于诊断集，不应作为论文性能复现成功的依据。

## 6. 证据与可下载文件

本包 data 目录包含三轮实验的运行记录、完整指标和配置。所有逐场景预测及原始特征仍保存在服务器的 artifacts/nuplan_metrics 对应运行目录中，本展示包未复制这些张量。

运行目录：

原始试跑：`{runs['pilot'].name}`

预处理对照：`{runs['preprocessing'].name}`

约束对照：`{runs['constraints'].name}`

论文任务和指标定义参考：https://arxiv.org/html/2504.10485v1#A4
'''
    (output / 'Nexus定量复现阶段报告.md').write_text(report, encoding='utf-8')
    html = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nexus 定量复现阶段报告</title><style>body{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#243247;background:#f4f6fa;margin:0}main{max-width:1100px;background:white;margin:24px auto;padding:36px}h2{margin-top:32px;border-bottom:1px solid #dde4ee}img{width:100%;height:auto}figure{margin:20px 0}figcaption{color:#64748b;text-align:center}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #dce3ed;padding:9px;text-align:left}th{background:#eef3f8}.table{overflow:auto}code{background:#f0f3f7;padding:2px 4px}@media(max-width:700px){main{padding:18px;margin:0}}</style></head><body><main>' + render(report, output) + '</main></body></html>'
    (output / 'Nexus定量复现阶段报告_离线预览.html').write_text(html, encoding='utf-8')
    guide = '''# 下载与提交

下载同名完整包 ZIP，解压后保留 MD 与 assets 目录的相对位置。打开离线预览 HTML 可直接阅读，图已内嵌，不依赖服务器。

飞书使用“导入为云文档”导入 MD。如图片未自动导入，将 assets 中两张 PNG 拖入正文相应位置。单独上传 MD 不会携带本地图片。原始逐场景 .pt 张量保留在服务器；本包用于图文展示和核对指标。
'''
    (output / '下载与提交说明.md').write_text(guide, encoding='utf-8')
    refs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', report)
    assert len(refs) == 2 and all((output / ref).is_file() for ref in refs)
    assert html.count('src="data:image/png;base64,') == 2
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output / 'SHA256SUMS.txt').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.relative_to(output).as_posix() for p in files) + '\n')
    files.append(output / 'SHA256SUMS.txt')
    archive = output.with_name(output.name + '_完整包.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, output.name + '/' + path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print('REPORT=' + str(output / 'Nexus定量复现阶段报告.md'))
    print('ZIP=' + str(archive))
    print('RAW_BRANCH_REPEATED_EXACTLY_EXCLUDING_TIMING=' + str(repeat_matches))
    print(grid_rows)


if __name__ == '__main__':
    main()
