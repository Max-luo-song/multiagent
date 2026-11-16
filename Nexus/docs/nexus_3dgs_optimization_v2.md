# 末尾朝向翻转、全场景交互与环境复核

后续复核又发现巴士轮廓重影、另一辆对向车 G7 的末尾急转，以及示意图难读。最新处理与中文视频入口见 [optimization_v3 说明](nexus_3dgs_optimization_v3.md)；本页保留 v2 的实验记录。

本轮延续用户确认面包车稳定后的反馈：右侧车在末尾仍有撕裂，并要求检查另一终端 dsh 是否改动 Nexus 依赖。旧权重、轨迹和视频均保留，新增结果在 `artifacts/nexus_3dgs/optimization_v2/`。

## 环境检查结论

没有发现已记录的依赖版本被改动。本轮没有安装、升级或卸载依赖。

- Nexus：Torch `2.3.0+cu121`、NumPy `1.23.0`、OmegaConf `2.3.1`、nuplan-devkit `1.2.2` 等记录中的核心版本匹配。
- 3DGS：与既有 `environment.freeze.txt` 的 **76 项本地依赖**逐项一致。其 NumPy 为独立的 `1.26.4`，Torch 继续复用原 Nexus 环境。
- 已实际校验官方 Nexus checkpoint 哈希、严格加载权重并对缓存 nuPlan 场景执行 CUDA 采样，输出有限，模型参数没有更新。
- 已实际执行 PyTorch3D CUDA KNN、gsplat 前后向、nvdiffrast 前后向、有界天空梯度和完整重建模块导入，全部通过。
- 执行期间一度出现较长等待，两个进程均处于共享文件系统 I/O 等待（`cxiWaitEventWait`），没有依赖异常。后续模型推理和实际渲染正常完成。

完整记录在 [environment_audit/verification.json](../artifacts/nexus_3dgs/optimization_v2/environment_audit/verification.json)。Nexus 没有早期完整依赖清单，因此结论覆盖已记录核心版本、实际包路径及运行链，不能扩大成全部文件从未变化的取证证明。

## 右侧车最后一秒撕裂

根因定位到 `2a3e0f8d92fa5791`（Nexus slot 6）在第 **7.5 s** 的孤立朝向异常：相邻采样约为 `45.99° → -133.69° → 45.61°`，其中相邻位置位移最大仅 `0.1703 m`。SLERP 正常地插值了这个错误姿态，导致车辆最后一秒来回转过约 180°，暴露未观测表面。继续增加重建训练并不能纠正输入朝向。

新增 `nexus_3dgs/kinematics.py` 和 `repair_nexus_heading.py`：只有相邻姿态有效、两侧朝向一致、位移很小、中央姿态与两侧均近似反向时，才纠正这一个半转分支。历史、自车、给定条件与终点受保护，不使用记录的未来朝向作为修复目标。

本场景只改了一个旋转样本，将 `-133.69°` 改为 `46.31°`。位置、尺度、有效性、身份及提供的终点逐项完全不变。最后一秒累计朝向变化 **358.99° → 1.008°**，最大未来转向速率 **359.38°/s → 1.392°/s**。同一 checkpoint、同一位置和相机策略渲染后，最后几帧不再发生整车反转。

- [右侧车最后两秒放大慢放](../artifacts/nexus_3dgs/optimization_v2/heading_comparison/right_car_tail_slow.mp4)：原像素放大，无平滑；每帧重复 3 次，标注为 3 倍慢放。
- [完整前视对照](../artifacts/nexus_3dgs/optimization_v2/heading_comparison/same_trajectories_stages.mp4)：80 帧、10 Hz；脚本验证只有审计中的旋转样本不同。
- [轨迹改动审计](../artifacts/nexus_3dgs/optimization_v2/heading_fixed_goal/heading_audit.json)及 [旋转量统计](../artifacts/nexus_3dgs/optimization_v2/heading_motion.json)。

这是明确记录的运动学后处理，原始 Nexus 权重和输出包保留；不宣称模型原生输出已不存在此类错误。

## 两组剩余侧后方重叠

旧方案只调整自车，未绑定的场景车辆按记录轨迹运行，会驶入较慢的自由生成车辆。新增 `SceneTraffic` 按同一 checkpoint 的实际车身尺寸及固定安装标定采样全部 **23 个重建车辆参与者（含自车）**。原场景有 22 辆车、3 个自行车实例和 2 个行人实例；本次车辆资产与碰撞检查不包含未建模的行人和自行车。记录轨迹按完整时间网格有效掩码插值，避免填补缺失时段。

`optimize_nexus_scene.py` 从已有三米自车间距方案继续进行全场景离线投影：

- 只允许发生冲突的两辆自由生成车沿当前朝向调整未来纵向位置；不给它们套用记录未来路线。
- 历史、给定公交车、自车、全部记录上下文车辆、朝向和可见性保持不变，没有通过删车或延迟出现来消除重叠。
- 约束物理车框分离、速度和加速度；优化按 20 Hz，保存后重新加载全部资产和轨迹，以 100 Hz 及实际相机曝光时刻独立检查。

| 条件 | 投影前车辆对重叠样本（100 Hz） | 投影后 | 自车最小 SAT 分离量 | 全场景最小 SAT 分离量 |
| --- | --- | --- | --- | --- |
| 公交车停止 | 143 | 0 | 约 3 m | 0.14334 m |
| 公交车沿记录行驶 | 145 | 0 | 约 3 m | 0.14334 m |

受调整车辆为 `04e7afccd5395007`（slot 9）和 `3be5a2ef339c56a9`（slot 5）。停止组的最大额外纵向偏移为 `4.743 m`、`1.170 m`，行驶组为 `4.877 m`、`1.322 m`，详细逐时刻偏移和求解结果在 `scene_projected/run.json`。它们为后续进入场景的记录车辆留出空间；记录车辆本身没有改道或隐藏。

另直接调用真实渲染器的 `TrajectoryController.edit_frame`，检查实际相机曝光时的全部车辆位姿（包含渲染器对未绑定车辆的逐帧规则），两组各 80 帧均无车辆对重叠，最小 SAT 分离量约为 `0.197 m`，且所有临时修改均完整恢复。记录在 `actual_stop_clearance.json`、`actual_go_clearance.json`，避免仅离线采样通过而漏掉实际渲染时序差异。

- [全部场景参与者的前后对照和相机](../artifacts/nexus_3dgs/optimization_v2/all_actor_comparison/all_actor_interaction.mp4)
- [侧后方冲突区域放大](../artifacts/nexus_3dgs/optimization_v2/conflict_area_comparison/all_actor_interaction.mp4)：`G` 为自由生成车，`L` 为记录上下文车；仅缩放鸟瞰视口，模拟和碰撞检查仍包含所有参与者。

该结果是 **Nexus + 离线场景约束投影**。100 Hz 的二维物理框检查不是连续时间或闭环安全证明，亦不包含全部静态环境障碍。自车前进距离的 `4.8083 m` 观测范围上限继续保留；背景重影、未观测表面和任意新视角质量仍受重建覆盖限制。本轮没有通过生成图像或重绘视频补造这些表面。

## 验证与复现

新增回归覆盖孤立半转、真实移动转向、角度跨越 ±180°、给定条件保护、无效时间段、后车进入场景时的避让，以及固定车辆碰撞时拒绝假修复。9 项扩展回归和原有 18 项回归均通过。真实模型探针、渲染和视频解码也已执行，汇总见 `optimization_v2/verification.json`。

从 Nexus 根目录使用明确的 Python 路径，输出目录须为新目录：

```bash
ROOT="$PWD"
PY3D="$ROOT/artifacts/nexus_3dgs/venv/bin/python"
CKPT="$ROOT/artifacts/nexus_3dgs/followup/reconstruction/nuplan/ownership_3cams_15000/checkpoint_final.pth"
GROUND="$ROOT/artifacts/nexus_3dgs/improvements/ground_3cams.json"

"$PY3D" scripts/pipeline/repair_nexus_heading.py \
  --bundle "$ROOT/artifacts/nexus_3dgs/nuplan_sensor_scene_goal_conditioned" \
  --output "$ROOT/artifacts/nexus_3dgs/optimization_v2/NEW_HEADING_BUNDLE"

"$PY3D" scripts/pipeline/optimize_nexus_scene.py \
  --source-experiment "$ROOT/artifacts/nexus_3dgs/followup/filtered_supported" \
  --checkpoint "$CKPT" --ground-model "$GROUND" \
  --output "$ROOT/artifacts/nexus_3dgs/optimization_v2/NEW_SCENE_PROJECTION"
```

渲染继续使用 `scripts/pipeline/nexus_3dgs.sh render --stabilize-all-rigid`，分别选择新 heading bundle，或新场景投影的 `bus_stop` / `bus_follow_logged`。包内记录每一步后处理；无需改变已有 Python 环境。
