# 面包车漂移、侧车噪声与条件视频穿模修复

这轮对应用户指出的两个具体视频问题。此前结果保留在 `improvements/`，本轮输出独立放在 `artifacts/nexus_3dgs/followup/`。

后续用户指出的末尾右侧车撕裂、剩余两组侧后方重叠，以及 dsh 环境影响检查，见 [新一轮优化和环境复核](nexus_3dgs_optimization_v2.md)。下列结果作为该轮基线保留。

## 已定位的原因

1. 左侧面包车为 track `12b9315c130a552b`，模型实例 1，约 57 m 远，超出原验证平面的覆盖。此前每帧回退到学习的位姿，整个片段的世界高度波动约 1.20 m。右侧近车为 `2a3e0f8d92fa5791`，模型实例 3；后面的车包括 `b4c792c576ff51ef`。
2. 车辆高斯的中心虽然在框内，椭球的长轴仍可能伸出车体。右侧实例的最大标准差约 0.59 m，另一辆约 1.32 m，而车高仅约 1.7～2.1 m。只限制高斯中心不能限制这些伸出车体的高斯尾部。
3. 背景层中也能看到公交车颜色和轮廓。移动后露出的背景重影与轨迹碰撞是两个问题，单独修复其中一个不充分。
4. 两组原始公交车条件实验确实有几何碰撞。旧审计把自车后轴位置当成车框中心，漏算了部分前悬。nuPlan Pacifica 的真实参数为前悬至前保险杠 4.049 m、后轴至后保险杠 1.127 m、宽 2.297 m，后轴到中心偏移为 1.461 m。

定位结果及图像实例标识见 `followup/assets/assets.json` 和 `actors_020.png`、`actors_060.png`、`actors_099.png`。

## 配准修复与分项验证

`--stabilize-all-rigid` 为全部重建车辆使用固定安装标定，包括没有绑定 Nexus 生成轨迹的实例。若某个 track 在片段中超出验证平面，整条 track 使用其重建位姿拟合的固定世界高度和倾斜，并保留目标 XY/yaw；不再逐帧恢复噪声，也不把未验证平面外推到远处。该回退包含局部平坦轨迹假设，不能用于宣称解决任意坡面问题。

对同一 checkpoint、同一轨迹的左侧面包车实际图像，固定初始 ROI `[239,243,302,292]`，前后向一致的纹理追踪结果为：

| 局部图像运动量 | 上一版 | 全车固定配准 |
| --- | --- | --- |
| 垂直范围（含透视运动） | 6.727 px | 1.907 px |
| 逐帧垂直运动 RMS | 0.2490 px | 0.0294 px |
| 垂直二阶差分 RMS | 0.1442 px | 0.0113 px |
| 最少持续追踪点 | 54 | 52 |

这是面包车局部纹理指标，不是全视频抖动或画质分数。相关轨迹 XY/yaw 没有改变。

物理外观约束使用旋转椭球的协方差投影，限制 3σ 支撑位于车框加 0.08 m 范围内，并测试最大标准差 0.12 m。单纯在旧权重上限制尺度能减少溢出，但暴露车体孔洞，故只保留为消融，不直接当成高质量成品。`registration_support_comparison/` 包含这三个阶段的同步视频。

## 受约束重建

在已有三相机真实图像上重新训练，保留原始图像和标注：

- 从真实 LiDAR 和真实 3D 车辆框提取可见返回点；至少 12 个可见点才构建投影凸包，并向内腐蚀 2 像素，得到车辆内部区域。
- 这些区域是保守的观测代理，不是语义真值轮廓；不补造未拍到的车身。303 张监督图及来源清单保存在三相机场景的 `vehicle_interior_masks/`。
- 在这些区域约束车辆层的不透明度和 RGB，避免只有最终合成图正确、车辆颜色却被静态背景吸收。辅助渲染保留训练梯度，主渲染的增密信息不被覆盖。
- 训练初始化将每条车辆轨迹的世界高度设为该轨迹的中位高度，保留真实 XY/yaw；采用局部平坦道路假设。
- 训练期间限制车身高斯支撑和最大标准差 0.16 m，让优化在约束下填充车体，而非最后把大高斯截断。
- 使用 `[0,2,4]` 三相机、64 分辨率有界天空，15000 步；相比单相机 5000 步，每视图平均训练预算相当。它同时改变了数据、约束和预算，不是单因素消融。

配置为 `followup/ownership_3cams_15000.yaml`。旧配置不自动启用上述实验损失。背景体积剔除实验另存于 `background_clean/`，只删除落入已观测车辆内部的静态高斯，保留底部 0.25 m 附近的路面点；该方法仍不能恢复一直被遮挡的背景，不视作完整补全。

## 穿模修复的范围

新增 `repair_nexus_intervention.py` 对保存的 Nexus 输出做明确标注的下游时序优化：

- 只改变自车沿原生成路径前进的时序；历史、给定公交车和其余车辆的位姿完全不变。
- 自车采用真实物理车长/宽及后轴偏移；其余车辆使用重建资产尺寸和实际安装偏移。检查对象包含所有重建车辆，而不限于 Nexus 绑定车辆。
- 约束前进速度、加速度及车身间距；优化在 20 Hz 采样，另用 100 Hz 核验保存轨迹的实际插值。也报告其他车之间的重叠，不将仅自车通过的检查写成全场景保证。
- 若静止自车也不可行、求解失败或独立检查失败，停止输出可用结果，而不是隐藏碰撞。

在旧权重上的 3 m 间距候选，自车的停止组行进由 15.56 m 限制为 3.03 m，行驶组由 16.35 m 限制为 11.71 m。更严格的物理检查在原轨迹分别发现 250 和 168 个 100 Hz 自车/车辆重叠样本；过滤后均为 0。这些数字属于此候选，最终权重必须重新求解和核验。

输出必须标注 **Nexus + 下游时序过滤**，不能称为 Nexus 模型已经学会避碰。车辆矩形检查不保证连续时间安全，也不包含所有静态环境障碍。未拍到的背景、车体表面和新视角仍可能重影、模糊或露底。

## 必要验证

`check_nexus_followup.py` 覆盖后轴偏移导致的漏检、300 组与 Shapely 多边形交叉验证、旋转高斯支撑与有限梯度、未绑定及平面外车辆的固定配准和渲染异常后的完整恢复。原 18 项坐标/时间/身份/投影回归继续通过。CPU 检查点加载时，实例索引、尺寸和有效掩码统一移动到模型设备，避免 CUDA 渲染设备不一致。

最终结果与视频链接见下方，完整校验及未解决项保存在 `followup/verification.json`。

## 最终可审阅结果

- [上一版／本轮画质同步对照](../artifacts/nexus_3dgs/followup/appearance_comparison/same_trajectories_stages.mp4)
- [车辆固定区域像素细节对照](../artifacts/nexus_3dgs/followup/appearance_comparison/actor_details_preview.png)
- [三米间距及移动距离上限的停止／行驶演示](../artifacts/nexus_3dgs/followup/supported_intervention_comparison/bus_intervention_diagnostic.mp4)
- [仅限间距、未限制相机移动的诊断视频](../artifacts/nexus_3dgs/followup/filtered_intervention_comparison/bus_intervention_diagnostic.mp4)

最终重建为 `followup/reconstruction/nuplan/ownership_3cams_15000/checkpoint_final.pth`，保留原 Nexus 轨迹的画质版为 `render_ownership15000/`。对应同一组生成轨迹，最终左侧面包车纹理追踪的逐帧垂直 RMS 为 **0.2490→0.0343 px**，二阶差分 RMS 为 **0.1442→0.00764 px**，持续追踪点为 54→57。垂直总范围 6.727→2.212 px 还包含真实透视运动，不以零位移作为目标。

对同一记录相机、同一 80 帧的重建评价如下，左右车辆区域的固定坐标定义在 `reference_regions.json`。这些是训练视角的误差，不是生成视频的新视角真值：

| 固定区域 | 旧 PSNR | 新 PSNR | 旧 SSIM | 新 SSIM |
| --- | --- | --- | --- | --- |
| 完整前视图像 | 30.4811 | 33.3668 | 0.8918 | 0.9181 |
| 左侧面包车区域 | 27.6139 | 28.4859 | 0.8003 | 0.8074 |
| 右侧车辆区域 | 30.0352 | 30.7794 | 0.8659 | 0.8857 |

面包车漂移明显减少，右侧车的观测视角误差改善，但右侧车辆在新视角中仍有模糊、碎裂和噪声，不能写成已彻底解决。

### 穿模与可视化范围

最终物理过滤重新使用新权重的实际车身尺寸和安装偏移求解。只限制车身间距时，自车停止／行驶组分别前进约 2.972／11.655 m；两组自车在 100 Hz 插值检查中与所有重建车辆的重叠样本均为 0，最小分离约 3 m。然而，行驶组的相机比记录中拍摄到的位置多前进约 7 m，背景会严重扭曲，不能把几何检查通过等同于画面完整。

另保留一个显式保守演示：`--limit-to-observed-travel` 将自车沿生成路径的前进距离上限设为该片段记录相机平台走过的 **4.8083 m**。停止组前进约 **2.972 m**，行驶组前进 **4.8083 m**，两组最小车框分离约 3 m、自车重叠样本均为 0。该限制使用已知重建观测范围，不是给 Nexus 模型增加未来输入，也不是严格的逐像素视角覆盖证明；视频中明确写出移动上限及下游过滤。

这仍不是全场景无碰撞：两组还有两对侧后方生成车／记录车重叠，分别涉及 `04e7afccd5395007` 与 `57c120b0216e5f3e`、`3be5a2ef339c56a9` 与 `c76fba780fca5607`。停止／行驶组分别有 143／145 个 100 Hz 非自车车辆对重叠样本。它们完整保留在 `filtered_supported/run.json`，视频也标注存在这些重叠。下一步需统一新增/未绑定车辆的行为控制，不能继续混用互不响应的记录轨迹。

22 项回归通过；最终视频均为 80 帧、10 Hz，检查结果见 `followup/verification.json`。本轮结论是**局部漂移与自车穿公交车得到处理，侧车新视角质量和全场景交互仍未完成**。

## 使用本轮修复

在 Nexus 根目录执行，原权重和旧结果目录不覆盖：

```bash
NEXUS_ROOT="$PWD"
FOLLOWUP="$NEXUS_ROOT/artifacts/nexus_3dgs/followup"
CKPT="$FOLLOWUP/reconstruction/nuplan/ownership_3cams_15000/checkpoint_final.pth"
GROUND="$NEXUS_ROOT/artifacts/nexus_3dgs/improvements/ground_3cams.json"

# 原生成轨迹不变，只应用本轮重建和全部车辆稳定配准
bash scripts/pipeline/nexus_3dgs.sh render \
  --resume-from "$CKPT" --ground-model "$GROUND" --stabilize-all-rigid \
  --bundle "$NEXUS_ROOT/artifacts/nexus_3dgs/nuplan_sensor_scene_goal_conditioned" \
  --render-cameras 0 --output "$FOLLOWUP/NEW_APPEARANCE_RENDER"

# 明确标注的物理间距与观测移动距离过滤；必须使用新的输出目录
"$NEXUS_ROOT/artifacts/nexus_3dgs/venv/bin/python" scripts/pipeline/repair_nexus_intervention.py \
  --experiment "$NEXUS_ROOT/artifacts/nexus_3dgs/improvements/step05_interaction" \
  --checkpoint "$CKPT" --ground-model "$GROUND" --clearance 3 \
  --limit-to-observed-travel --output "$FOLLOWUP/NEW_FILTERED_EXPERIMENT"
```

渲染过滤结果时，`--bundle` 选择新实验中的 `bus_stop` 或 `bus_follow_logged`，继续使用相同 checkpoint、地面模型及 `--stabilize-all-rigid`。新重建配置的车身范围约束随 `config.yaml` 生效，无须额外传入 `.12 m` 的旧权重消融参数。需要重跑训练时使用 `ownership_3cams_15000.yaml`，其真实 LiDAR 内部监督已保存并记录来源。
