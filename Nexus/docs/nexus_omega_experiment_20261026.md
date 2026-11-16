# Nexus：OMEGA 思想接入与首轮效果

参考：[OMEGA 论文 v4](https://arxiv.org/html/2512.07661v4)、[官方项目页](https://opendrivelab.com/OMEGA/)。论文直接将公开 Nexus 权重作为基础模型，主要改动在推理采样层，不要求重新训练 Nexus。本次根据论文独立实现机制，未取得或声称使用 OMEGA 官方实现。

当前结果是**有局部改善、尚未通过完整交互验收的实验候选**。原 Nexus 默认路径、官方权重、已认可的 v9 重建资产均保持原状。

## 实现

- `scenediffusion.py` 增加可选的 `anchor_refiner`：在约束后的干净状态估计上优化，再重新构成下一步反向分布。关闭时不新增随机数；六份原采样输出与既有预测逐值一致。
- `omega_guidance.py` 实现等协方差高斯的精确 KL 距离 `sum(A² Δx² / (2 variance))` 和有界优化。保留全部历史及指定公交条件；无效状态不修改，不跨缺席区间计算运动学残差。
- 优化使用历史观测尺寸的物理车框，保留自车后轴到车框中心 1.461 米偏移；模型不能靠缩小未来车身规避碰撞检查。未来输出尺寸本身尚未加入一致性约束，其漂移单独记录。
- 包含位置—速度一致性、侧滑／朝向、加速度、平滑和车道线段距离代理。与论文的完整自行车模型、路线 DFS／道路区域、双圆安全模型和 SE-IBR 对抗博弈存在明确差异。
- Warm-up／Rolling-Zero 的实验分支保留，但本次明显退步，不推荐采用。它没有连接外部环境反馈，不属于闭环仿真。

## 冻结条件和试验过程

场景 `a89c1931e7b25976`，公交 `441830199c6752a5`。第 3～6 秒公交倒车靠近 4.5 米，同时间窗原地停止为配对对照；其余未来由 Nexus 生成，公交条件窗外也自由。预先固定种子 0、1、2，展示种子固定为 0。

全部方法使用相同历史、地图、未来存在 mask、输入条件及初始噪声。未给定未来数值清零，但未来存在 mask 仍来自记录，不能称部署时纯历史预测。

结果均在 `artifacts/nexus_omega/`：

| 目录 | 内容与结果 |
|---|---|
| `pilot_v1` | 掩码类型错误，预测前失败；地图通道筛选随后修正 |
| `pilot_v2` | 六份 baseline 完成；缺席 token 的 `atan2(0,0)` 导数导致首个引导失败，已修复 |
| `pilot_v3` | 四组完整消融，共 24 份预测：baseline、anchor、schedule、omega |
| `pilot_v4_kl20` | 物理坐标 Adam，KL=20，当前主要候选；已完成 v9 视频 |
| `pilot_v5_kl200` | 扩大 KL 至 200，碰撞采样点没有进一步减少 |
| `pilot_v6_normalized_kl20` | 归一化 Adam 对照，未整体优于当前候选 |
| `pilot_v7_lbfgs` | 带线搜索的白化 L-BFGS；同时增加碰撞权重／迭代次数，未整体优于当前候选，不是求解器单因素消融 |
| `confirmed_candidate_v1` | 整理后的可复用代码六次回归，与 `pilot_v4_kl20` 全部保存张量逐值相同 |
| `candidate_independent_audit.json` | 独立检查条件、KL、六份回归、冻结 v9、实际车体与完整视频 |
| `implementation_v1`、`implementation_v2`、`implementation_v3_lbfgs` | 经原协议哈希核对的历史源代码快照 |

这些是同一个开发场景上的机制、参数和求解器比较，存在开发选择，不能作为独立泛化成绩。全部候选和失败结果保留，不把三个种子算成三个新场景。

## 主要结果与取舍

原生 0.5 秒采样、使用观测尺寸的预测车辆物理框：

| 公交事件 | 原 Nexus | 当前候选 |
|---|---:|---:|
| seed 0：自车—公交重叠点／16 | 7 | 4 |
| seed 1：自车—公交重叠点／16 | 6 | 2 |
| seed 2：自车—公交重叠点／16 | 3 | 2 |
| 合计／48 | 16 | 8 |
| 全预测车辆无重叠的事件种子／3 | 0 | 0 |
| 正常对照中无重叠种子／3 | 2 | 3 |

固定 seed 0、冻结 v9 的 80 个真实曝光时刻：

| 实际渲染车体指标 | 原 Nexus | 当前候选 |
|---|---:|---:|
| 正常对照：自车—公交重叠帧 | 14 | 0 |
| 公交事件：自车—公交重叠帧 | 32 | 18 |
| 公交事件：最小 SAT 间隔／米 | -2.234 | -0.616 |
| 公交事件：全场景重叠对×帧 | 46 | 32 |

每种条件另有 14 个无关背景车重叠对×帧，不能隐去。事件组仍穿车，不能作为合格安全场景。两套评估采样、车辆覆盖和车体安装不同，不能混用分母。

还存在代价：三个事件种子的全车位置推导加速度超 5 m/s² 的采样点从 24 增至 30；自车峰值加速度的最大值约 1.559→1.638 m/s²，jerk 最大值约 4.852→5.845 m/s³。车道线段距离代理未新增超限，但这不等于可行驶区合规。引导采样约 15 秒，原采样约 0.4 秒，未进行严格效率基准。因此不能表述成所有指标全面提升。

独立数值检查覆盖高斯 KL 闭式结果、条件／无效状态保护、信赖域与 Rolling-Zero 冻结；三项通过。视频完整 80 帧、10 FPS、8 秒，源像素在对照拼接编码前精确保留，不叠加轨迹修复，不对无真值新视角计算 PSNR。

## 使用当前候选

在本工作树根目录执行，输出必须为新目录：

```bash
/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/envs/nexus/bin/python -u \
  scripts/pipeline/run_nexus_omega_experiment.py \
  --methods baseline anchor --solver adam_physical --kl-budget 20 \
  --output artifacts/nexus_omega/new_experiment
```

脚本显式从原项目读取数据、身份清单、环境和 checkpoint；模型源码优先使用当前工作树。当前 runner 限定这一例已审计场景，换场景需要先补齐身份来源，不能复用公交 slot/model index。默认仅跑 baseline／anchor；完整失败消融需显式指定 `--methods baseline anchor schedule omega`。

轻量检查：

```bash
/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/envs/nexus/bin/python \
  scripts/pipeline/check_nexus_omega.py
```

报告入口：`report/nexus_omega_20261026/index.html`。下一步应先解决残余穿车及运动学取舍，再冻结配置开展其他场景验证；现阶段不将该分支替换为默认采样，也不把外部优化效果称作 Nexus 通过训练学会了避让。
