# B车同步多视角渲染（2026-10-28）

本轮按“把视角换成中间那辆车，并提供多视角”的要求，在B车车顶设置朝前、后、左、右的四路虚拟相机，输出同步四宫格和各方向独立视频。视频内容仍为A切入、B减速、C变道绕行。

## 如何观看

打开 `media/event_B_multiview.mp4`。左上是前视，右上是后视，左下是左视，右下是右视。先看前方A切入，再看后方C接近，随后看左侧C变道并超越。后视使用相机实际向后的方向，不进行镜像翻转。单路视频无文字覆盖，四宫格的方向标签在画面外。

`media/event_B_front.mp4` 为B车前视；`media/normal_B_multiview.mp4` 为未切入组的四视角对照。`media/event_external_overview.mp4` 保留上一轮外部观察与俯视示意，方便确认A/B/C身份和位置。

## 相机与运动

逐帧读取上一轮 `artifacts/nexus_3dgs/asset_insert_20261028/render_final/run.json` 的实际渲染三车位姿，不重新生成、插值或修改轨迹，也不重新进行道路适配或落地校准。每个相机的世界变换严格为 `T_world_B × T_B_camera`，随B车平移、转向、俯仰和侧倾。相机不再跟随此前的外部观察路径。

车辆坐标约定为x向前、y向左、z向上；相机坐标为x向右、y向下、z向前。四向朝向分别为0°、180°、90°、−90°，均下俯3°。四路共用车顶中心上方0.15米的光心，约在道路上方1.90米。水平视场105°，960×540，固定内参和曝光，无随时间变焦。初始分散安装在车辆边缘的试渲染出现近处角色中心落入视野缝隙，因此最终改用共光心车顶环视布置；失败试渲染保留。

这是B车周围的虚拟安装相机，不是该MPV车型的实车标定。提供的资产没有车内模型；为避免车身内部遮挡，这四路仅渲染A、C和道路背景，省略自身B车。B车完整保留在源运动、相机宿主位姿和外部观察视频中。

两组相机安装规则完全相同。由于正常组和切入组的B运动不同，两组相机在世界中的路径相应不同，不再要求世界相机矩阵相同。

## 渲染及核验

背景仍为121场景，源运动仍为059场景的Nexus＋行为引导结果，已进行新道路适配。每组81帧、10fps，内容覆盖0—8秒。C原有4.0秒完整变道、5.2秒完全超过B的时序保持不变。

原生3DGS渲染分别计算四个方向的相机射线、深度遮挡和车辆球谐颜色，没有将外部观察视频裁剪或扭曲为车载视角。原先的薄高斯阴影近似继续应用于A、C。原始Nexus预测、资产、背景检查点和旧渲染记录均以SHA256保护。

逐帧检查B车到相机的安装变换，核对源角色位姿未变、相机旋转有效、输出数值有限、视频帧数和帧率，并记录A/C中心在哪些相机视锥内。视锥中心检查不等于无实际遮挡或车辆每个像素完整可见；应结合视频观察。原三车车框审计继续适用于相同角色位姿，但相机视角本身不构成新的安全或因果证明。

## 质量限制

当前背景主要使用前向及侧前向图像训练，后方和侧方的新视角覆盖较弱，因此可能存在背景缺失、模糊、拉伸及浮点。相机向侧方看树木、路缘时尤其明显。新增多视角主要用于观察相同交互事件的不同方向，没有宣称背景已经达到完整环视重建质量。

## 代码与复现

- `scripts/pipeline/render_inserted_threecar_b_views.py`：从已保存位姿原生重渲染四路相机。
- `scripts/pipeline/report_inserted_threecar_b_views.py`：四宫格、独立视频、视野时间表和HTML报告。
- 实验输出：`artifacts/nexus_3dgs/asset_insert_20261028/b_multiview_final/`。
- 展示目录：上级 `hzy/report/nexus_threecar_B_multiview_20261028/`。

使用已有项目环境包装器，以加载本地动态库。输出目录须为未存在的新目录：

```bash
bash scripts/pipeline/nexus_3dgs.sh script \
  /inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/scripts/pipeline/render_inserted_threecar_b_views.py \
  --output /inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nexus_3dgs/asset_insert_20261028/b_multiview_reproduction
```

`run.json` 保存四路安装矩阵、逐帧相机矩阵、视锥检查和源文件哈希。报告 `summary.json` 保存视频清单，`visibility_timeline.json` 保存逐帧各角色中心的视野分布，`verification.json` 和 `copy_verification.json` 保存交付核验。
