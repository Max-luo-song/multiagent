# v7：运动重影的对齐检查与重建实验

**当前状态：尚未通过整体替换门槛：有改善，也有退步；保留为候选实验。** 这份报告同时展示改善和剩余问题。旧 v5 模型仍保留；没有将候选结果冒充已经彻底修好的版本。

[真实运动完整对照视频](assets/nexus_3dgs_v7/real_motion_comparison.mp4)

## 怎么看

左栏是真实照片组成的录像，中栏是原 v5，右栏是本轮候选。共 8 秒，保留第 20～99 帧全部图像。先看开头的清晰度，再在 **5 秒（第 70 帧）**、**7.9 秒（第 99 帧）**暂停，对照路面箭头、白色建筑边缘、树冠、公交车轮廓和右侧车辆。没有补帧、锐化或时间平滑。

![第 70 帧，完整三栏对照](assets/nexus_3dgs_v7/real_motion_comparison_070.jpg)

## 实际查明了什么

1. 先前的控制实验已发现：同一个旧 3DGS，即使使用真实运动轨迹，移动后也会变差，因此不能仅归因于 Nexus。
2. 本轮用真实静态图像特征和真实点云分别检查，发现当前跨帧位姿与静态观测不完全一致。第 20→70 帧，留出的静态特征投影误差中位数由 **10.82 像素降到 1.60 像素**。独立点云配准在第 70 帧得到约 0.94° 的旋转校正。
3. 只替换旧模型的相机，在部分帧改善、部分帧反而退步。旧高斯已经吸收了原先的误差，所以不能直接把新位姿硬套上去作为完整修复。
4. 已建立单独的校正输入，并完成新的静态场景训练及遮挡检查。过宽的车辆排除区域曾导致车旁建筑缺少训练约束，因此又改用固定车辆与背景共同渲染来补全可见区域。

这些检查说明“投影与重建的一致性需要处理”，没有据此把问题直接定性为 nuPlan 原始标定错误，也没有证明 Nexus 轨迹先天无法清晰渲染。

## 完整结果，而非只选改善帧

PSNR 单位 dB，越高表示与对应真实照片越接近。车辆排除区域仅用于分析背景；**整图指标与视频同样重要**。

| 分组 | 旧 v5 整图 | 候选整图 | 旧 v5 车辆排除区域 | 候选车辆排除区域 |
| --- | --- | --- | --- | --- |
| 基本静止 43 帧 | 37.30 | 34.25 | 40.08 | 42.75 |
| 移动／转动 37 帧 | 28.04 | 31.44 | 28.88 | 33.98 |
| 运动段留出验证 3 帧 | 26.61 | 31.36 | 27.18 | 33.38 |
| 全部 80 帧 | 33.02 | 32.95 | 34.90 | 38.70 |

静止／运动分组沿用前次真实运动实验：相对首帧位移超过 0.1 米或转动超过 0.2° 为运动。运动段留出验证帧为 70、80、90。数值门槛要求运动段整图均值提高，且静止段整图均值下降不超过 1 dB；这仍不能替代局部视觉检查。

![第 20 帧，用于检查静止画质有没有下降](assets/nexus_3dgs_v7/real_motion_comparison_020.jpg)

![第 99 帧，用于检查运动后段](assets/nexus_3dgs_v7/real_motion_comparison_099.jpg)

## 训练和验证范围

- 新输入为 131 帧、四路相机，旧 v5 为 101 帧、三路相机。本报告是两条完整流程的比较，不能将全部差异都归功于单个参数修改。
- 本次对照视频和画质分数仅检查前向相机、同一场景的 80 帧；尚未据此验收所有相机或其他场景。
- 新背景初始化与梯度训练排除了每第 10 个时刻的 RGB，验证相机的学习残差如存在则从相邻训练时刻插值。点云配准使用所有时刻的 LiDAR，包含 RGB 验证时刻。因此这是 RGB 留出验证，不是全传感器留出测试。
- 车辆资产继承自旧模型，旧模型见过原始完整场景。整图分数应视为重建保真度检查，不是严格的未见场景泛化成绩。
- 车辆观测位姿另做小幅拟合，训练时平移限制为 25 厘米、旋转限制为 3°；留出时刻直接对相邻有效的已拟合车体位置插值、对旋转做 SLERP；序列端点或缺少双方可见邻帧时保留原候选位姿。这是标注与资产的对齐，不是修改 Nexus 的预测轨迹。最终候选又固定背景、相机与观测位姿，对车辆高斯的形状、透明度和外观进行了单独重训。
- 原始 RGB、点云、内外参和拍摄时间戳没有改写。131 帧中 1,759 组车体与点云相对位姿检查通过，最大矩阵误差约 2.33e-10。
- 本轮没有训练 Nexus。其保存的原始预测不因场景重建而改变。

## 文件在哪里

- 当前候选权重：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nexus_3dgs/optimization_v7/reconstruction/nuplan/assets_absolute_validation/checkpoint_final.pth`
- 本轮工作目录：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nexus_3dgs/optimization_v7`
- 完整真实轨迹评估：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nexus_3dgs/optimization_v7/assets_motion_validation`
- 主视频：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nexus_3dgs/optimization_v7/review_assets/real_motion_comparison.mp4`
- 原有模型、失败分支和新候选分开保留。单独车辆调色、相机残差细调等分支不自动作为最终版本。

本轮分支中，`static_registered_15000` 单独训练时仍有车旁背景缺少约束的问题；`rgb_residuals_7000` 相机残差方案在部分验证帧退步；`vehicles_appearance_4000` 调色收益较小。最终候选采用固定校正相机的背景共同渲染训练，并单独拟合真实车体位姿。中间分支都保留作对照。

## 车辆重训仍有取舍

车辆高斯完整重训后，部分运动帧的整图分数继续提高，但公交车尾部文字更软，预测视角的建筑也仍有局部拉花。因此没有采用“分数提高就替换旧版”的做法。

[保留旧车辆高斯的真实轨迹对照](assets/nexus_3dgs_v7/vehicle_preserved_real.mp4) · [保留旧车辆高斯的 Nexus 对照](assets/nexus_3dgs_v7/vehicle_preserved_nexus.mp4)

这两段对照保留原来的车辆形状与纹理，只使用新的背景与真实车体位姿拟合，便于单独比较车辆重训造成的变化。后续需优先检查动态对象在各相机曝光时刻的位姿同步，以及独立的多视图视觉／点云联合标定；目前尚未完成全部重影的修复。

[逐帧分数与实际相机位姿](assets/nexus_3dgs_v7/validation.json) · [投影检查](assets/nexus_3dgs_v7/projection_audit.json) · [点云配准](assets/nexus_3dgs_v7/lidar_registration.json) · [几何保护检查](assets/nexus_3dgs_v7/geometry_verification.json)

## 同一 Nexus 预测重新回放

[Nexus 预测对照视频](assets/nexus_3dgs_v7/nexus_prediction_comparison.mp4)

中、右两栏读取同一个已保存的原始意图条件预测，未加入后续离线车距或交互修改。使用共同的地面与车辆资产挂载；新版本的高度、侧倾和俯仰来自校正后的记录数据。左侧真实录像仅供理解场景，预测视角与它不同，因此不计算预测视角 PSNR。


---

# v7：位姿一致性与场景重建实验

本轮已完成；候选尚未通过整体替换检查，旧 v5 保留。背景和运动段改善，静止段整图仍退步，车辆重训后巴士尾部文字更软。Nexus 参数和保存预测没有修改。

- 主报告：`report/nexus_3dgs_v7.html`，同时导出到用户指定的 `/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/report/`。
- 最新完整对照：`review_assets/real_motion_comparison.mp4`、`review_assets/nexus_prediction_comparison.mp4`。
- 保留旧车辆高斯的完整对照：`review/real_motion_comparison.mp4`、`review/nexus_prediction_comparison.mp4`。
- 最新逐帧评估：`assets_motion_validation/validation.json`。
- 最终状态和全部分组均值：`decision.json`。

## 输入与保护检查

`registered_data/` 为 131 帧、四路相机的单独校正输入。原始图像、点云、内外参仍复用，未改写。相机按曝光时刻插值校正；点云和标注采用同一个世界校正。1,759 组相对位姿检查通过，最大矩阵误差 2.33e-10。原始多视角数据和语义文件仍复用 v6 的共享输入。

`static_seed/seed_audit.json` 记录新背景初始化。每第 10 个时刻的 RGB 未参与新高斯初始化或梯度训练，但配准使用了这些时刻的 LiDAR，旧车辆资产也见过原始完整场景；这不是全传感器未见数据测试。

## 各分支

所有模型位于 `reconstruction/nuplan/` 下：

| 目录 | 作用与状态 |
| --- | --- |
| `static_registered_15000` | 全新背景和天空；旧车辆资产、相机与曝光固定。车辆排除过宽，车旁背景出现缺少训练约束的问题。 |
| `vehicles_appearance_4000` | 只拟合旧车辆颜色和透明度；收益有限，未采用。 |
| `rgb_residuals_7000` | RGB 小幅相机残差与背景细调；部分验证帧退步，未采用。 |
| `composite_background_5000` | 沿用相机残差方案，用固定车辆共同渲染补全背景；不作为最终分支。 |
| `composite_fixed_camera_5000` | 固定独立点云校正相机，共同渲染车辆与背景，修复车旁缺少约束的问题。 |
| `actual_body_poses_4000` | 只拟合真实车辆的观测位姿；高斯形状、颜色与透明度精确保持。 |
| `body_absolute_validation` | 验证帧直接插值相邻已拟合绝对车体位姿，不读取验证 RGB。 |
| `registered_assets_6000` | 冻结背景、相机、曝光和真实车体位姿，重训车辆高斯的形状与外观。 |
| `assets_absolute_validation` | 车辆重训后的验证位姿插值候选；最新完整评估使用此模型，尚未通过整体替换检查。 |

## 重现入口

在项目根目录使用 `bash scripts/pipeline/nexus_3dgs.sh script <脚本绝对路径> ...`，环境无需安装新依赖。

- `audit_nuplan_motion_projection.py`：图像静态特征与 LiDAR／地面投影独立检查，最终审计为 `../projection_audit_v4/`。
- `audit_nuplan_lidar_registration.py`：静态点云配准，完整结果为 `../lidar_pose_refinement_full/`。
- `prepare_nuplan_registered_scene.py`、`verify_nuplan_registered_scene.py`：构建校正数据和检查实际几何不变量。
- `build_nuplan_static_seed.py`、`train_nuplan_registered_static.py`：新背景初始化及训练。
- `train_nuplan_registered_vehicles.py`：颜色拟合、背景共同渲染、真实车体位姿拟合、车辆高斯重训；具体分支分别由 `--background-only`、`--poses-only`、`--refit-assets` 控制。
- `interpolate_nuplan_validation_bodies.py`：只修改内部验证时刻的车体位姿，其他时刻与所有高斯、相机参数精确保持。
- `evaluate_nuplan_registered_static.py`：前向真实照片逐帧评估。
- `render_nexus_registered_prediction.py`：同一原始 Nexus 条件预测回放，临时位姿精确恢复，不计算不存在真值的新视角 PSNR。
- `review_nuplan_registered_scene.py`、`export_nexus_v7_report.py`、`verify_nexus_v7_report.py`：完整对照、报告与逐帧解码校验。

下一步优先检查动态车体在各相机曝光时刻的同步，以及独立的多视图图像与点云联合校准。不要将本候选描述为全部重影已修复。
