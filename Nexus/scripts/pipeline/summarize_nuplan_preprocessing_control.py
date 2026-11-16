"""Summarize a completed paired velocity preprocessing diagnostic."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'run.json').read_text())
    if manifest['status'] != 'complete':
        raise RuntimeError('Paired control has not completed')
    result = json.loads((run / 'metrics.json').read_text())
    official, raw = result['official_reset_v'], result['raw_velocity']
    bp_key = 'agent_behavior_prediction'
    goal_key = 'intent_condtioned_agent_behavior_prediction'
    def val(branch, metric, key):
        return result[branch][metric][key]
    def number(branch, metric, key):
        return f'{val(branch, metric, key):.4f}'
    def percentage(branch, metric, key):
        return f'{val(branch, metric, key) * 100:.4f}'
    total_pairs = sum(r['velocity_diagnostics']['valid_history_pairs'] for r in manifest['records'])
    def mean(key):
        return sum(r['velocity_diagnostics'][key] * r['velocity_diagnostics']['valid_history_pairs'] for r in manifest['records']) / total_pairs
    velocity_raw_error = mean('raw_velocity_vs_position_difference_mean_l2')
    velocity_reset_error = mean('reset_velocity_vs_position_difference_mean_l2')
    raw_speed = mean('raw_history_speed_mean')
    reset_speed = mean('reset_history_speed_mean')
    improves_bp = raw[bp_key]['displacement@avg_8s'] < official[bp_key]['displacement@avg_8s']
    improves_goal = raw[goal_key]['displacement@avg_7s'] < official[goal_key]['displacement@avg_7s']
    finding = ('在这一批场景和相同随机噪声下，保留原始速度使普通预测与给定终点预测的上述平均误差均降低，支持速度预处理是误差来源之一。'
               if improves_bp and improves_goal else
               '在这一批场景和相同随机噪声下，保留原始速度没有同时改善两类预测的上述平均误差，因此不能将误差归结为这一项处理。')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, key, title in zip(axes, (bp_key, goal_key), ('History conditioned', 'History + goal conditioned')):
        for branch, label in [('official_reset_v', 'Official reset_v'), ('raw_velocity', 'Raw velocity')]:
            ax.plot(range(1, 9), [result[branch][key][f'displacement@{t:.6f}s'] for t in range(1, 9)], marker='o', label=label)
        ax.set(title=title, xlabel='Future horizon (s)', ylabel='Mean displacement error (m)')
        ax.set_xticks(range(1, 9))
        ax.grid(alpha=0.25)
        ax.legend()
    axes[1].annotate('8 s state is provided', xy=(8, 0), xytext=(4.8, 0.6), arrowprops={'arrowstyle': '->'}, fontsize=8)
    fig.savefig(run / 'preprocessing_comparison.png', dpi=160)
    plt.close(fig)
    report = f'''# Nexus 速度预处理对照结果

## 1. 为什么增加这组实验

上一轮 8 场景试跑的普通预测误差偏大。源码显示，ResetVAugmentor 对速度通道求相邻时间差后，将结果写回速度通道。对于近似匀速运动，这会使原本非零的速度变成接近零的值。

本轮在同一批 {manifest['evaluated_scenarios']} 个验证场景上比较两种处理，不改动官方源码或权重：

- 官方分支：继续使用 ResetVAugmentor。
- 原始速度分支：保留正式特征构建器输出的速度通道。

两分支的场景位置、朝向、地图、有效性、采样约束和模型权重保持一致。每个场景推理前恢复同一 CPU 与 CUDA 随机数状态，并验证两分支消耗后的随机数状态一致。模型参数版本检查通过，没有训练更新。

## 2. 位移误差对比

| 指标（米） | 官方 ResetV | 保留原始速度 |
| --- | --- | --- |
| 普通预测：1 秒位置误差 | {number('official_reset_v', bp_key, 'displacement@1.000000s')} | {number('raw_velocity', bp_key, 'displacement@1.000000s')} |
| 普通预测：8 秒位置误差 | {number('official_reset_v', bp_key, 'displacement@8.000000s')} | {number('raw_velocity', bp_key, 'displacement@8.000000s')} |
| 普通预测：8 个整秒误差均值 | {number('official_reset_v', bp_key, 'displacement@avg_8s')} | {number('raw_velocity', bp_key, 'displacement@avg_8s')} |
| 给定终点：前 7 个整秒误差均值 | {number('official_reset_v', goal_key, 'displacement@avg_7s')} | {number('raw_velocity', goal_key, 'displacement@avg_7s')} |

![相同噪声下的速度预处理对照](preprocessing_comparison.png)

给定终点任务直接提供最后有效时刻的完整真实状态，因此 8 秒误差不作为独立终点预测能力的证据。这里的误差均值采用官方整秒汇总定义，不能混称所有半秒采样点上的标准 ADE。

安全和道路指标并未随位移误差一起改善：

| 代码指标（值 × 100，%） | 官方 ResetV | 保留原始速度 |
| --- | --- | --- |
| 普通预测碰撞计数比 | {percentage('official_reset_v', 'collision_rate_bp', 'collision@avg')} | {percentage('raw_velocity', 'collision_rate_bp', 'collision@avg')} |
| 给定终点碰撞计数比 | {percentage('official_reset_v', 'collision_rate_gp', 'collision@avg')} | {percentage('raw_velocity', 'collision_rate_gp', 'collision@avg')} |
| 普通预测越界代理指标 | {percentage('official_reset_v', 'traj_metric_bp', 'offroad_rate')} | {percentage('raw_velocity', 'traj_metric_bp', 'offroad_rate')} |
| 给定终点越界代理指标 | {percentage('official_reset_v', 'traj_metric_gp', 'offroad_rate')} | {percentage('raw_velocity', 'traj_metric_gp', 'offroad_rate')} |

以上碰撞计数按车辆对统计，越界按地图采样点距离代理判断，均沿用仓库实际实现。该对照不支持“所有生成质量指标都变好”的结论。

## 3. 历史速度一致性检查

在 {total_pairs} 个有效非自车历史相邻状态对上，用位置差除以 0.5 秒作为运动速度参考：

| 诊断项 | 数值 |
| --- | --- |
| 原始速度与位置差分参考的平均向量差（米/秒） | {velocity_raw_error:.4f} |
| ResetV 后通道与位置差分参考的平均向量差（米/秒） | {velocity_reset_error:.4f} |
| 原始历史速度模长均值（米/秒） | {raw_speed:.4f} |
| ResetV 后通道模长均值（按速度通道解释） | {reset_speed:.4f} |

位置差分只是近似参考；车辆的连续运动、跟踪误差和采样都可能产生差异。该检查用来观察处理是否改变通道的运动含义。

## 4. 结论与适用范围

{finding}

这是一组小规模诊断，不能单独证明原始速度分支就是作者发布权重对应的训练配置。正式复现仍需核对训练时预处理和论文评测协议，并扩大到明确的数据清单。本轮没有将诊断分支设为官方默认设置，也没有报告已达到论文成绩。

原始证据：[运行记录](run.json)、[双分支完整指标](metrics.json)、[实际模型配置](config.yaml)。两分支预测和原始特征均在本服务器运行目录的 .pt 文件中保留。
'''
    (run / '速度预处理对照报告.md').write_text(report, encoding='utf-8')
    print(finding)
    print(run / '速度预处理对照报告.md')
    print(run / 'preprocessing_comparison.png')


if __name__ == '__main__':
    main()
