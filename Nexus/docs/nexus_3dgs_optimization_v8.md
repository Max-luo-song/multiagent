# v8：车辆左右晃动与黑车“四散”修复

本轮针对 v7 候选中反馈的两个问题，保持原有 3DGS 背景、车辆高斯形状、纹理、透明度、相机与曝光参数，修正运动接入和车体位姿。**没有重新训练 Nexus 或 3DGS，也没有对视频像素做时间平滑、融合或锐化。** v7 原始文件仍保留。

## 先看结果

[Nexus 回放：v7 与 v8 完整对照](assets/nexus_3dgs_v8/nexus_before_after.mp4) · [黑车放大对照](assets/nexus_3dgs_v8/black_car_closeup.mp4)

主视频左侧是 v7 候选，右侧是 v8 修复分支，共 8 秒、完整 80 帧。先看 **0～1 秒**的公交车横向动作，再暂停在 **7.5 秒**看右侧黑车。两边相机完全相同。

![右侧黑车第 7.5 秒，左为 v7，右为仅修异常朝向](assets/nexus_3dgs_v8/black_car_closeup_095.jpg)

## 黑车为什么会像散开

黑车 `2a3e0f8d92fa5791` 几乎停车，预测在约 7.5 秒处却出现一次约 180° 的朝向跳变。相邻采样的位移最多约 17 厘米；渲染插值会让车身快速转身，暴露原本重建不足的表面，形成散开的外观。

v7 的原始预测对照没有启用此前的孤立朝向保护。本次在独立派生分支中接回这层保护：只改这一个异常朝向采样，车辆位置、时间、有效性、历史与给定终点都保持原样，不额外读取真实未来运动。

**单独修黑车朝向时，第 20～89 帧共 70 帧与 v7 逐像素相同；最后一秒黑车不再翻转。**

[三阶段拆分：v7／仅修黑车／完整 v8](assets/nexus_3dgs_v8/heading_ablation.mp4)

## 公交车为什么会左右晃

两条回放中有不同来源：

- **记录轨迹回放：**逐帧车体拟合引入了小幅位置与旋转抖动。v8 在三维车体位姿上做有界稳定化，场景坐标下的纵向运动和高度保持，连续有效段的两端锁定；不跨越无效轨迹缺口。实际公交车横向最大修正约 **3.7 厘米**，旋转最大约 **0.63°**。旋转二阶差分的 RMS 指标下降约 **96%**，这是位姿抖动指标，不是“视觉误差降低同等比例”。
- **Nexus 原始预测回放：**公交车第一步存在明显横移。v8 对公交横向位移加入与预测朝向一致的运动约束；在固定参考方向下，第一步横移从 **35.7 厘米降到 3.4 厘米**，所有时刻的平面修正最多 **35 厘米**。公交的纵向前进过程、时间采样、历史、给定终点、朝向和高度都保持原输入；Nexus 自车及其他车辆的预测位置保留，黑车只作前述异常朝向修正。

因此，右栏是**明确标注的修复派生轨迹**，不能再称作未经处理的 Nexus 原始输出。原始预测文件与模型权重没有修改。未被 Nexus 预测的场景车辆使用稳定化后的记录车体位姿。

[记录轨迹：真实照片／v7／v8](assets/nexus_3dgs_v8/recorded_before_after.mp4) · [记录轨迹公交车放大](assets/nexus_3dgs_v8/bus_closeup.mp4)

![记录轨迹第 70 帧，左真实照片、中 v7、右 v8](assets/nexus_3dgs_v8/recorded_before_after_070.jpg)

## 画质和回归检查

| 记录相机整图平均 PSNR，dB | v7 候选 | v8 |
| --- | --- | --- |
| 基本静止 43 帧 | 34.25 | 35.20 |
| 运动 37 帧 | 31.44 | 31.37 |
| 全部 80 帧 | 32.95 | 33.43 |

分数只对应有真实照片的记录相机。运动段差异约 -0.07 dB，整体画质基本保持；静止段与全段平均分数提高。Nexus 新视角没有对应真实照片，不计算其 PSNR。

- Nexus 两组的全部 80 个相机位姿完全相同。
- 记录轨迹两组在共同车辆排除区域之外，背景像素最大差为 **0／255**。
- 高斯外观、几何、透明度、背景、天空、相机、曝光与可见性参数精确保持；只改变车体位姿。
- 相同物理车体位姿下，两个模型渲染最大差为 1／255，符合浮点旋转往返的舍入范围。
- 本段 80 帧检查未新增车辆包围盒重叠。
- 四项稳定化行为测试、三项低速朝向保护回归测试通过，原始数据和预测文件保留。

本轮集中处理运动稳定性，v7 遗留的细节模糊和新视角局部缺陷仍存在；没有把这次结果描述为所有画质问题都已解决。

## 文件位置

- 本轮目录：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nexus_3dgs/optimization_v8`
- 车体稳定化检查点：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nexus_3dgs/optimization_v8/stability_candidates/observed_body_stable/checkpoint_final.pth`
- 明确修正后的预测派生包：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nexus_3dgs/optimization_v8/stability_candidates/heading_and_bus_path`
- 主视频：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nexus_3dgs/optimization_v8/review/nexus_before_after.mp4`
- 保留 v7 对照：[v7 报告](nexus_3dgs_v7.html)

[逐项验证](assets/nexus_3dgs_v8/stability_validation.json) · [车体位姿审计](assets/nexus_3dgs_v8/body_stability_audit.json) · [公交路径修正范围](assets/nexus_3dgs_v8/path_stabilization.json) · [黑车朝向修正](assets/nexus_3dgs_v8/heading_audit.json)

![公交车朝向变化：保留缓慢转向，减少逐帧摆动](assets/nexus_3dgs_v8/bus_yaw_comparison.png)
