# 车道、切入制动与侧滑优化候选（2026-10-27）

本轮新增可选的控制空间采样约束，没有覆盖已有升级模块或修改官方权重。
验证仍使用 nuPlan 场景 033（Las Vegas，token `320271cfc9ca5e86`）、seed 0。

## 改动

- `omega_kinematic.py`：在反向扩散的 clean anchor 内优化加速度、曲率，由同一个有界非侧滑运动积分生成 XY、朝向和速度。使用 0.1s 积分、0.5s 原生输出；给定的历史和 A 全部未来逐位保持。
- 跟驰关系在未来每个时刻更新。对于从前方进入当前地图车道的给定车辆，增加让行次序偏好，避免通过加速抢到其前方或横移关闭跟驰损失。
- 地图车道是空间软偏好，不是按时刻给定的 E/C 未来路径。E/C 的速度、加速度和转向仍为自由变量。所有车辆的初始状态仅取当前观测。
- `trajectory_acceptance.py`：输出后只做验收，不修补轨迹；同时检查有向车框、背景、实际车道、道路范围及位置推导的运动状态。正常 E/C 要求完整车框留在原车道，事件中允许合理换道。失败样本只作诊断，不导出通过验收的下游 bundle。
- 新 runner 单独加载可选模块；原默认采样逻辑和原有升级源文件保持不变。

这是新的推理算法，使用模型软先验和控制优化，不声称保持原 OMEGA 的精确 Gaussian KL 保证；也不是包含轮胎力等因素的完整车辆动力学仿真。

## 验证

合成三车检查：

```bash
../envs/nexus/bin/python -B scripts/pipeline/check_nexus_kinematic_guidance.py
```

检查给定条件不变、正向无侧滑、正常车道保持、切入制动以及 E/A、E/C 的正间隔。自检结果见 `artifacts/nexus_kinematic_20261027/control_selfcheck.json`。

真实配对生成（输出目录存在时拒绝覆盖，复现需指定新的 stage）：

```bash
../envs/nexus/bin/python -B scripts/pipeline/run_nexus_kinematic_candidate.py --stage candidate_v1 --seeds 0
artifacts/nexus_3dgs/venv/bin/python -B scripts/pipeline/report_nexus_kinematic_candidate.py --stage candidate_v1
```

该场景的角色为 E=0、A=2、C=12；验收适配器目前采用这一映射。地图准备只使用当前姿态和 SDK 车道拓扑；未来存在掩码和 97 个背景参与者记录回放仍是显式额外信息。A 共用前移 2m 的反事实编辑偏移，未来 0.5–2.5s 向右切入 3m。旧／新输入、掩码和初始噪声相同，报告生成时逐项断言。

|指标|旧方案|新方案|
|---|---:|---:|
|正常 E 车框越出原车道检查点|36|0|
|切入 E/A 重叠检查点|26|0|
|切入 E/C 重叠检查点|0|0|
|切入 E 最大运动／朝向偏差|57.00°|1.30°|
|切入 E 最大位置差分加速度|69.73 m/s²|4.50 m/s²|

检查点为未来 0.1–8.0s、0.1s 间隔，共 80 点。新切入 E 的前 2.5s 区间平均速度最低约 4.93 m/s，原始观测速度约 12.45 m/s。

主三车通过本轮检查。全场景仍未通过：正常有 61 个、事件有 50 个初始道路内车辆的车框越界检查点；正常、事件各有 3 个预测车辆与记录背景重叠检查点。详细诊断集中在预测槽 8/25 的道路边缘和槽 43 与背景 74 的重叠，见 `candidate_v1/global_diagnosis.json`。没有导出获准交付下游的 bundle。

仅验证单场景、单种子，不能推广为 Nexus 整体能力提升。C 的条件轨迹变化也不能证明“E 的避让导致 C 响应”的因果链。本轮视频为真实新预测的俯视动画，包含显示插值，不是 3DGS 渲染；原生预测没有生成后修补。

## 产物

- 实验：`artifacts/nexus_kinematic_20261027/candidate_v1/`
- 新正常／切入视频：`report/nexus_kinematic_20261027_candidate_v1/media/seed_000_pair.mp4`
- 旧／新切入对比：`report/nexus_kinematic_20261027_candidate_v1/media/event_before_after.mp4`
- 完整报告会复制到工作区 `hzy/report/nexus_kinematic_20261027_candidate_v1/`。
