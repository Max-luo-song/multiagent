# Nexus 第二轮完善与交接（2026-10-26）

本轮围绕同一开发场景 `a89c1931e7b25976` 修复输入与约束机制，使用预先固定的种子 0、1、2。没有扫描整个数据集，没有训练 Nexus 或重新训练 3DGS。所有正式代码、实验和报告均保存在 `/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/`。

## 当前主候选与证据

主候选为 `nexus_repro/Nexus/artifacts/nexus_omega_v2/order_candidate_v1/`。按碰撞优先选择，未切换项目默认采样器。

相对上一轮 `artifacts/nexus_omega/pilot_v4_kl20`，三个事件种子合计：

| 指标 | 上一轮 | 当前主候选 |
|---|---:|---:|
| 自车与公交重叠，原生 0.5 秒采样点 | 8 | 0 |
| 全车辆重叠，对×时刻，0.1 秒检查 | 74 | 0 |
| 位置差分加速度大于 5 m/s² 的点 | 30 | 4 |
| 真实可行驶区域中心越界点 | 0 | 0 |
| 完整车框部分越界点，0.1 秒检查 | 108 | 0 |

正常对照并非全部通过：种子 0 在 0.1 秒检查中还有 2 个预测车与记录背景车的重叠点、2 个完整车框越界点；另外两个种子该项为 0。实际 v9 渲染采用不同的 80 个摄像机曝光时刻，不能混用两个时基的计数：事件视频全车辆重叠对×帧为 0，正常视频仍为 3（末尾背景车）。三事件种子的加速度超限分别为 1、2、1，主要是历史到未来的边界。

## 修复与独立对照

1. 发布版 `ResetV` 对速度通道进行差分后又写回速度通道，并非从位置恢复速度。新增仅使用历史位置/历史传感器速度的可选恢复分支，保持历史位置、朝向、尺寸和公交条件不变。相同无原生约束、无引导设置的单因素对照为 `released_velocity_control` 对 `causal_velocity_v1/baseline`：三个事件种子的加速度超限点 **21→2**。不能把多因素综合比较当成这个对照。
2. 原生平滑/裁剪会修改速度与尺寸通道。新实验通过显式 `--native-constraints none` 独立消融；默认发布路径保留。
3. 新增逐车精确高斯 KL 预算、密集 OBB/SAT、历史前车顺序约束、真实地图距离场约束。保持给定条件精确，自车未来自由，未来自由输入清零；以观测车体尺寸检查碰撞，避免靠预测尺寸缩小取得改善。
4. 给九辆未预测背景车提供了记录的未来几何，属于额外环境信息。未来存在 mask 也沿用记录。该结果不能称作纯历史预测、官方 OMEGA 完整复现或未见场景泛化。

## 运动可行性实验分支

`motion_probe_v2` 与对应的 `motion_validation_v2` 使用冻结参数分别验证种子 0 和 1、2；`motion_candidate_v2` 为逐文件哈希校验后合并的结果，不再采样。其完整数值以独立 `audit_motion_final/audit.json` 和交付报告为准。

该分支在最后两个去噪步骤、构造反向高斯核之前，对干净锚点 XY 进行有条件的凸约束恢复；失败求解拒绝，给定历史/公交保持精确，逐车 KL 超限拒绝。为在既有 KL 预算内分配更多 XY 调整量，可将自由速度通道更新向模型锚点收缩。不是采样结束后修改输出轨迹。

全部三个种子的正常与事件组加速度超限均降为 0，但三个事件种子的密集背景重叠为 3、0、9，合计 12；正常组合计 14。事件组完整车框部分越界点合计 3。因此保留为实验分支。种子 0 实际 v9 渲染的正常/事件重叠对×帧分别为 5/3。相对主候选还改变了权重和 KL 预算，不能将差异归因于单一 QP 开关。只约束 XY 导出的运动，不保证输出速度、朝向和位置的完整八维自行车模型一致性。报告另列速度通道残差。百次内循环的运行成本远高于原采样，不是实时方案。

## 复现与验收

工作目录：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus`。

```bash
PY=/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/envs/nexus/bin/python
$PY scripts/pipeline/check_nexus_omega_v2.py
$PY scripts/pipeline/run_nexus_omega_v2.py \
  --methods anchor --seeds 0 1 2 --profile structural \
  --history-velocity backward_xy --native-constraints none \
  --iterations 100 --kl-budget 2000 --kl-weight 0.002 \
  --collision-weight 1000 --following-weight 1000 \
  --acceleration-weight 100 --acceleration-limit 4 \
  --road-weight 1000 --clearance 0.5 --lock-predicted-heading \
  --background-replay artifacts/nexus_omega_v2/background_inputs/background_replay.npz \
  --drivable-sdf artifacts/nexus_omega_v2/drivable_sdf_inputs/drivable_sdf.npz \
  --output artifacts/nexus_omega_v2/order_reproduction_new
```

结果目录必须使用新的名称；保留所有失败分支。每次实验保存完整协议、输入及噪声哈希、实际源代码快照、原始 tensor 和条件、逐步 KL 与目标值。报告中的原生检查、0.1 秒密集检查、v9 实际曝光时刻检查分别记录。

数值测试覆盖子帧碰撞、SAT 梯度、精确条件/尺寸/逐车 KL、距离场梯度、历史速度的因果性、前车穿越分支、边界运动连续性、预算分配保持条件。实际交付另校验官方 Nexus 和冻结 v9 权重哈希、种子 0 渲染源 tensor 完全相等、视频源像素及帧数。

报告入口：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/report/nexus_omega_v2_20261026/index.html`。报告的技术完整性通过不等于行为安全全部通过。

## 后续步骤

先联合解决正常组末尾背景车碰撞、边界加速度和八维状态一致性，同时控制计算成本；再冻结配置到 3–5 个独立场景检验泛化。目前没有闭环决策策略、连续时间安全证明或三车因果链验证。
