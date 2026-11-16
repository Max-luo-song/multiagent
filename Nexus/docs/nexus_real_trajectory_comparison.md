# 真实轨迹与 Nexus 轨迹的受控渲染

结果目录：`artifacts/nexus_3dgs/real_vs_nexus/`。

主视频：`review/real_vs_nexus.mp4`，依次为真实照片录像、真实标注轨迹渲染、保存的 Nexus 意图条件预测渲染。每段 80 帧、10 fps、8 秒。

**本例真实运动下也出现画质下降，不能把整景重影仅归因于 Nexus。** 同一个 v5 模型的真实标注轨迹，静止／运动段 PSNR 为 32.77／27.35 dB；保留原拟合车辆位姿的更有利控制为 37.30／28.04 dB。后者的保守车辆排除区域从 38.88 降到 28.85 dB。实验尚未区分几何与相机标定分别贡献多少。

真实自车相对首帧平面位移大于 0.1 米，或转动大于 0.2°，归为运动段：43 帧基本静止、37 帧运动／转动。指标只对有相应真实照片的记录相机计算；生成相机不计算与错误视角照片的 PSNR。不是独立未见场景测试。

## 控制与来源

- 使用同一个 v5 场景检查点，哈希 `ccf77a0792d26249bd2e3e84be0b764cc8dc26c0f2d744bb6ad3eda4686e7fed`，没有重新训练模型。
- 原始自由预测来自 `behavior_pred`，包哈希 `fc082cd6892e892d75bb35e52d33ab4d04270294e3b1b6dd2f36404c1f6247ca`。
- 原始条件预测来自 `intent_conditioned_behavior_pred`，包哈希 `0f81b3c503f267ec7f6766ef18ceccca5f818efe3fee968789c06999d0421bcd`。
- 两种预测来自同一个保存的推理文件，未增加本项目后续的朝向、车距或交互修正。原推理配方已有的约束依然属于其来源。
- 之前 `bus_follow_logged` 交互视频含离线约束和给定巴士运动，单独保留在 `review/current_interaction.mp4`，不冒充原始输出。

原始自由预测的自车首尾位移约 0.33 米；条件预测约 4.69 米；真实记录约 4.60 米。不能用自由预测的较小运动幅度证明正常行驶时的渲染能力。

## 比较方法

真实车辆标注在每帧相机曝光时刻插值，不填补无效轨迹缺口。本例相机比对应点云时间晚约 37 毫秒。真实／预测车辆使用相同固定资产挂载和道路高度／倾斜模型，使用标注或预测的 XY／朝向。额外原拟合位姿控制用于识别这种挂载转换造成的影响。

相机位置使用真实传感器姿态与原标定。生成相机仅替换 XY／朝向，保留测量的高度、俯仰与侧倾。主对照中，原模型的逐帧学习相机校正同样右乘在两个分支上；另有不使用学习校正的传感器标定控制。不能把两种标定混用后将差别解释成轨迹效果。

`review/motion_factorial.mp4` 的四格分别是真实／真实、条件相机／真实车辆、真实相机／条件车辆、条件相机／条件车辆。两个混合组合是诊断用反事实，不是 Nexus 原生场景。在固定真实相机、仅替换其他车辆运动时，保守车辆包围盒排除区域外的背景，80 帧像素完全一致。它是不变性检查，不是生成图像的真值质量分数。

全部 11 组各渲染 80 帧，共 880 帧。模型文件、外观和相机参数保持不变，临时实例位姿精确恢复。`raw_predictions_audit.json` 中真实车辆中心的往返转换最大误差约 `8.7e-11` 米，旋转矩阵元素最大误差约 `3.9e-9`。

## 观察范围

80 帧记录前相机到公交车中心距离为 14.25～18.33 米。之前交互方案最小约 11.72 米，原始条件预测最小约 14.45 米。之前交互方案要求了更近的车身观察位置，可能放大资产缺陷。该距离检查只包含前相机，不代表完整训练可见性，也不能证明哪条轨迹必然渲染清楚。

## 复现

输入已保存到实验目录 `inputs/`，包括固定模型、两份原始预测包、既有交互状态和场景元数据。原照片仍通过元数据中的 nuPlan 原始路径读取。

```bash
E="$PWD/artifacts/nexus_3dgs/real_vs_nexus"
R="$PWD/artifacts/nexus_3dgs/real_vs_nexus_reproduce"
bash scripts/pipeline/nexus_3dgs.sh script \
  "$PWD/scripts/pipeline/render_nexus_trajectory_control.py" \
  --checkpoint "$E/inputs/model/checkpoint_final.pth" \
  --source "$E/inputs/existing_interaction.json" \
  --scene "$E/inputs/scene" --output "$R"
bash scripts/pipeline/nexus_3dgs.sh script \
  "$PWD/scripts/pipeline/render_nexus_raw_prediction_control.py" \
  --experiment "$R" --scene "$E/inputs/scene" \
  --registration-source "$E/inputs/existing_interaction.json" \
  --free-bundle "$E/inputs/free" --goal-bundle "$E/inputs/goal"
bash scripts/pipeline/nexus_3dgs.sh script \
  "$PWD/scripts/pipeline/analyze_nexus_trajectory_control.py" --experiment "$R"
```

几何控制行为检查：`scripts/pipeline/check_nexus_trajectory_control.py`，4 项通过。输入哈希见 `inputs/manifest.json`；原始预测来源见 `raw_predictions_audit.json`；完整控制核验见 `audit.json` 和 `review/review.json`；真实相机画质见 `quality.json`；固定相机背景不变性见 `fixed_camera_background.json`。

报告位于用户指定的 `/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/report/nexus_real_trajectory_comparison.html`，包含视频、观看说明、完整结果与限制。
