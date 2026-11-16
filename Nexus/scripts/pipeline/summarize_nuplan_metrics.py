"""Write a bounded validation pilot report and displacement-error figure."""

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
        raise RuntimeError('Evaluation has not completed')
    metrics = json.loads((run / 'metrics.json').read_text())
    bp = metrics['agent_behavior_prediction']
    goal = metrics['intent_condtioned_agent_behavior_prediction']

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times = list(range(1, 9))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for label, values in [('History conditioned', bp), ('History + goal conditioned', goal)]:
        ax.plot(times, [values[f'displacement@{t:.6f}s'] for t in times], marker='o', label=label)
    ax.set(xlabel='Future horizon (s)', ylabel='Mean displacement error (m)',
           title=f'nuPlan validation pilot: {manifest["evaluated_scenarios"]} scenes')
    ax.set_xticks(times)
    ax.annotate('8 s state is provided', xy=(8, goal['displacement@8.000000s']),
                xytext=(5.1, 1.5), arrowprops={'arrowstyle': '->'}, fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(run / 'displacement_error.png', dpi=160)
    plt.close(fig)

    def value(group, key, scale=1):
        return f'{metrics[group][key] * scale:.4f}'

    records = manifest['records']
    two_task_average = sum(r['two_task_synchronized_seconds'] for r in records) / len(records)
    rows = '\n'.join(f'| {r["index"] + 1} | {r["token"]} | {r["scenario_type"]} | {r["valid_non_ego_future_tokens"]} |' for r in records)
    report = f'''# Nexus nuPlan 小规模定量评测试跑

本轮已完成官方验证集上的 {manifest['evaluated_scenarios']} 个场景评测，使用官方 nuplan.ckpt，随机种子 {manifest['seed']}，未进行训练更新。运行状态为 complete，参数版本检查通过。

## 1. 数据与任务

从与官方 val 划分一致的 {manifest['available_val_databases']} 个共享数据库中，按文件名排序取前 {len(manifest['databases'])} 个日志，再选择具有至少 2 秒历史和 8 秒未来的唯一场景。这是流程验证用的便利样本，不代表全验证集分布。

使用正式 NexusFeatureBuilder、验证配置的 ResetVAugmentor、官方 forward_validation 和聚合指标。采样采用 full_sequence、32 步，以及 map / sma / clip 约束，各强度为 1.0。

- 普通预测：给定所有车辆过去 2 秒至当前时刻的状态，生成未来 8 秒。
- 给定终点预测：在上述条件外，再给定各车辆最后有效时刻的真实状态。终点属于输入条件。

## 2. 位移误差

| 代码指标 | 普通预测 | 给定终点预测 |
| --- | --- | --- |
| 1 秒位置误差（米） | {value('agent_behavior_prediction', 'displacement@1.000000s')} | {value('intent_condtioned_agent_behavior_prediction', 'displacement@1.000000s')} |
| 4 秒位置误差（米） | {value('agent_behavior_prediction', 'displacement@4.000000s')} | {value('intent_condtioned_agent_behavior_prediction', 'displacement@4.000000s')} |
| 8 秒位置误差（米） | {value('agent_behavior_prediction', 'displacement@8.000000s')} | {value('intent_condtioned_agent_behavior_prediction', 'displacement@8.000000s')} |
| 前 7 个整秒位置误差均值（米） | {value('agent_behavior_prediction', 'displacement@avg_7s')} | {value('intent_condtioned_agent_behavior_prediction', 'displacement@avg_7s')} |
| 8 个整秒位置误差均值（米） | {value('agent_behavior_prediction', 'displacement@avg_8s')} | {value('intent_condtioned_agent_behavior_prediction', 'displacement@avg_8s')} |

![两种预测任务的位移误差](displacement_error.png)

官方实现排除自车，并分别在未来 1—8 秒对有效车辆汇总位置误差，再对整秒误差求均值。不能将其直接当作所有 0.5 秒采样点上的标准 ADE。给定终点任务的 8 秒误差可能因直接提供终点而很小，因此前 7 个整秒的误差更适合观察终点之间的生成效果；较早结束的短轨迹仍可能在前 7 秒包含已给定终点。

## 3. 代码中的安全与道路指标

| 代码指标 | 普通预测 | 给定终点预测 |
| --- | --- | --- |
| 碰撞计数比（代码值 × 100，%） | {value('collision_rate_bp', 'collision@avg', 100)} | {value('collision_rate_gp', 'collision@avg', 100)} |
| 越界代理指标（代码值 × 100，%） | {value('traj_metric_bp', 'offroad_rate', 100)} | {value('traj_metric_gp', 'offroad_rate', 100)} |

碰撞实现统计相交车辆对，再除以有效车辆数，区别于统计发生碰撞的不同车辆数量。越界代理指标依据车辆中心到地图采样点的距离阈值 2.75 米，按源码中的有效车辆分母汇总，区别于直接检查可行驶区域多边形。因此本报告保留“代码指标”标签，不据此宣称复现论文同名数值。

每个指标使用独立数据副本，避免官方实现中的自车有效性原地修改和地图重复反归一化相互污染。该适配不改变各指标内部公式。

同步后的两任务总推理时间均值为 {two_task_average:.3f} 秒/场景，包含两个采样任务、编码和相关调用；不与论文单任务时间比较。官方未完整同步的时间字段另保留在原始记录中。

## 4. 场景清单

| 序号 | token | 场景类型 | 有效非自车未来状态数 |
| --- | --- | --- | --- |
{rows}

日志来源：

''' + '\n'.join('- ' + Path(p).name for p in manifest['databases']) + '''

## 5. 本阶段结论与边界

本次将复现推进到验证集上的双任务推理和官方代码指标聚合，预测逐场景保存，指标均通过有限值检查，模型没有训练更新。它验证了评测流程可以执行，尚不足以形成论文性能结论。

扩大评测前，需要确定具有代表性的场景清单，核对论文与仓库中碰撞计数、位移误差、约束强度等定义及配置差异，再执行独立的正式评测。当前两个日志和少量场景的结果不应与论文表格直接比较。

原始证据：[运行记录](run.json)、[实际配置](config.yaml)、[完整指标](metrics.json)、[指标累积状态](metric_states.json)。预测张量保存为同目录的 prediction_*.pt。

论文任务与指标定义：[附录 D](https://arxiv.org/html/2504.10485v1#A4)。
'''
    (run / '评测结果报告.md').write_text(report, encoding='utf-8')
    print(run / '评测结果报告.md')
    print(run / 'displacement_error.png')


if __name__ == '__main__':
    main()
