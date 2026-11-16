# 三车交互的3DGS资产插入演示（2026-10-28）

## 交付与结果

主视频为 `media/event_with_bev.mp4`：左侧俯视示意标注A/B/C，右侧3DGS画面不叠加车辆标签。`media/comparison.mp4` 左侧未切入、右侧切入，另提供两组无标注的纯3DGS视频。每组81帧、10 fps，运动覆盖0—8秒。

事件组呈现A切入并减速、B制动、C变道绕行。C在4.0秒完整进入相邻车道并保持至少1秒，5.2秒车尾完全超过B车头。正常组C保持原车道。两组A的纵向减速安排相同，主要区别为A是否横向切入；正常组并非A始终匀速。

独立检查实际渲染姿态和资产尺寸构成的三维包围框，将其投影到道路XY：两组各81帧均无三车车框重叠、无驶出选定双车道路域，正常组无错误车道。两组最小车框间距分别0.664米、0.748米。每组三车共243次中心投影均在画面内。两组相机矩阵和内参逐帧一致。以上为离散采样检查，不是连续时间、全部交通法规或完整场景安全证明。

## 背景与资产

按用户允许更换合适道路背景的要求，筛选传感器齐全的候选场景，选择121：token `88ad40d17ae35947`，日志 `2021.10.05.07.38.12_veh-50_00898_01058`，地图 `sg-one-north`。相邻同向车道52395/52553约有118.7米连续长度，70米内横向弯曲约0.1米，适合本次变道演示。

数据准备101帧、404幅真实图像、约978万LiDAR点；实际背景训练使用相机0/1/2的303幅图像，训练8000步。相机和记录车辆姿态冻结并通过逐位不变性检查。仅训练新的3DGS背景，Nexus权重未更新。

资产来自 `/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/3dgs_pro/`：

| 文件 | 使用角色 | 高斯数量 | SHA256 |
| --- | --- | ---: | --- |
| 1.ply，白色SUV | A、C | 14699 | 2dec41a2e36c5cd373c81f6f26fd201cc3b76e7f50f1be6225bb378e5fef1f76 |
| 3.ply，白色MPV | B | 15753 | 5fc8848be25b13adaa641cf0bcc591c2307fdcff49cb37502e06cfccc3118b17 |

按原始包围框中心归中，采用 `0.97 * min(原轨迹长度/资产长度, 原轨迹宽度/资产宽度)` 等比例缩放，保持车身比例。读取标准PLY的channel-major球谐颜色、对数尺度、透明度logit及wxyz四元数；将3σ高斯支撑范围限制于资产边界。初期颜色排列错误的彩色预览仅保留为诊断记录，最终采用正确标准排列。原始PLY不改写。

## 轨迹来源与道路适配

源数据为 `artifacts/nexus_lane_response_v2_20261027/pilot_seed0/case_059/{normal,event}/prediction.pt`，角色A39、B41、C44。它是此前Nexus结合显式车道选择和物理引导的结果，不代表Nexus原生自主学会此高级决策。沿用2秒历史后的8秒未来，插值到0.1秒用于渲染。

此处没有在121背景重新运行Nexus，也不是059原场景重建。仅插入三辆车，不回放原场景52个参与者。A后续停车作为给定危险行为保留；不能将原场景其他前车原因归因于这段只有三车的画面。

直接刚体移植曾产生39次车道违规，首次常量车道间距适配仍有28次；失败脚本及搜索记录均保留。最终将源轨迹分解为沿路进度、横向偏移和相对道路切线朝向，按源/目标成对车道在各位置的实际间距转换横向偏移，恢复到目标道路。两组、三车使用同一规则，保留时间和沿路进度。初始C位于相机前方约16米，统一横向微调−0.4米。适配后重新检查实际资产车框，不直接沿用旧场景的检查结论。

## 落地、相机与渲染

道路高度基于选定车道内的真实LiDAR拟合，剔除标注车辆，平面RMSE约0.038米。对原生背景期望深度进行接地点诊断后，使用PCHIP平滑车辆安装高度/俯仰，最大高度修正约0.137米，XY和原始源轨迹文件保持不变。该修正是与重建表面配准，不是独立真实地形标定。

采用原标定前视相机高度，无额外抬高或横移；观察相机沿事件组B纵向进度的85%平滑前进，两组共用相机路径，固定曝光。焦距在前5秒平滑由1倍变化到1.7倍，属于实际光栅化内参变化。它是展示观察相机，不是源Nexus自车视角。

车辆和背景在原生高斯深度管线中合成，不使用车图贴片、网格替身、补帧或锐化。每车一个薄高斯提供接触阴影近似，不作物理光照仿真。渲染期间临时替换原记录交通的高斯读取，随后恢复，不改写背景权重。背景、源预测和原始资产均有SHA256不变性保护。

## 复现与记录

仓库根目录为 `/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus`。本次实验目录为 `artifacts/nexus_3dgs/asset_insert_20261028/`；新背景检查点为其中 `reconstruction/nuplan/background8000/checkpoint_final.pth`。

主要实现：

- `nexus_3dgs/inserted_vehicle.py`：原生资产读取、缩放、变换与支撑限制。
- `scripts/pipeline/prepare_inserted_threecar_frenet.py`：源/目标成对车道适配。
- `scripts/pipeline/render_inserted_threecar.py`：原生高斯合成和实际逐帧姿态记录。
- `scripts/pipeline/probe_inserted_road_depth.py`：道路深度与落地诊断。
- `scripts/pipeline/report_inserted_threecar.py`：实际车框独立验收、俯视示意和视频报告。
- `scripts/pipeline/prepare_nuplan_3dgs.py`、`scripts/pipeline/train_nexus_3dgs_pilot.py`：复用的数据准备和背景训练入口。

最终渲染命令（输出目录必须是未存在的新目录）：

```bash
artifacts/nexus_3dgs/venv/bin/python scripts/pipeline/render_inserted_threecar.py \
  --checkpoint artifacts/nexus_3dgs/asset_insert_20261028/reconstruction/nuplan/background8000/checkpoint_final.pth \
  --output artifacts/nexus_3dgs/asset_insert_20261028/render_reproduction \
  --camera-lift 0 --camera-side 0 --zoom-end 1.7 \
  --ground-profile artifacts/nexus_3dgs/asset_insert_20261028/vehicle_ground_profile.json
```

`render_final/run.json` 保留最终渲染的相机、内参、三车姿态、源文件哈希和参数；`insertion_protocol.json` 保留映射和源数据信息；`rendered_geometry_audit.json` 保留实际车框审计；`final_verification.json` 保留交付核验；`reproduction/` 保留相关源码快照与哈希清单。报告和展示视频归档在上级 `hzy/report/nexus_threecar_asset_insert_20261028/`。

## 当前限制

车辆运动已可展示，但背景树木、远景和路缘仍有模糊和浮点，未达到此前公交v9画质。新视角质量没有完全解决。此结果验证了三车轨迹到新道路及外部高斯资产视频的呈现链路；未重新证明B→C严格因果链，原B固定诊断没有整体通过。未验证完整原始交通参与者、安全泛化或Nexus在新场景自主生成同样高级行为。
