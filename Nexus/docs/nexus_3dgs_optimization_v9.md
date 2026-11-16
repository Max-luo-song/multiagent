# v9 公交细节恢复候选

v8 保留为你已认可的基线。本轮只把公交车替换为 v7 车辆整体重训之前的高斯资产，保留 v8 的全部运动稳定化、派生轨迹、相机、背景和其他车辆。**没有重新训练 Nexus 或 3DGS，没有视频锐化、补帧、时间融合，也没有把真实视频贴回渲染画面。**公交细节来自此前已有的三维高斯资产；该旧资产的车尾下部曾使用真实观测构建连续曲面，不能据此宣称任意视角均有完整几何。

## 先看哪里

- [完整 Nexus 回放：左 v8，右 v9](assets/nexus_3dgs_v9/nexus_before_after.mp4)
- [公交放大：左 v8，右 v9](assets/nexus_3dgs_v9/bus_closeup.mp4)
- [真实照片／v8／v9 公交放大](assets/nexus_3dgs_v9/recorded_bus_comparison.mp4)
- [仅看 v9 完整渲染](assets/nexus_3dgs_v9/camera_0.mp4)

所有视频都是第 20～99 帧完整 80 帧、10 FPS、8 秒。重点看车尾 “SBS Transit” 字样、车窗底部小字、橙紫车身分界和尾灯。放大视频只对同一区域作等比例显示缩放，不增加原图细节。

![Nexus 公交放大，第 5 秒](assets/nexus_3dgs_v9/bus_closeup_070.jpg)

## 本轮发现与实际改动

公交尾部文字在 v7 的车辆几何与外观整体重训后变软。只恢复公交重训前的资产，在相同 v8 运动与相机下可以恢复可见文字和边缘细节。这说明本例的模糊至少有一部分来自公交资产重训；不能据此判断所有模糊均由同一个原因造成，也不能归结为 Nexus 轨迹必然无法清晰渲染。

本轮恢复的是公交的高斯位置、尺度、旋转、颜色与透明度，共 126,168 个高斯。**全部车体位姿保持 v8 原值**，不会恢复 v7 的晃动或黑车异常翻转。v8 的公交横向轨迹修正也完全保留；这仍是明确标注的派生预测回放。

![记录相机：真实照片、v8、v9，第 5 秒](assets/nexus_3dgs_v9/recorded_bus_comparison_070.jpg)

## 画质有取舍，v8 不覆盖

公交文字更清楚，但与真实照片逐像素的一致性并非每帧提高。公交局部细节的位置、遮挡、反光以及原有几何误差仍会影响分数，因此本版作为细节候选单独交付。下表仅评估有真实照片的记录相机；不对 Nexus 预测视角计算不存在真值的 PSNR。公交区域采用向内收缩的三维框投影，含部分背景，不是精确语义轮廓。分组沿用先前真实运动实验。

| 分组 | v8 整图 PSNR | v9 整图 PSNR | v8 公交区域 PSNR | v9 公交区域 PSNR |
| --- | --- | --- | --- | --- |
| 基本静止 43 帧 | 35.20 | 35.18 | 29.80 | 29.84 |
| 运动 37 帧 | 31.37 | 30.84 | 26.74 | 25.31 |
| 全部 80 帧 | 33.43 | 33.17 | 28.38 | 27.74 |

以上是本场景重建保真度检查，旧公交资产见过原场景，不能作为未见场景泛化成绩。

## 已完成的保护与回归检查

- 重新加载输出权重后逐项核对：除公交高斯资产外，所有参数精确保持；全序列车体位姿、可见性和车辆身份保持。
- 两组完整 80 帧使用完全相同的 Nexus 派生轨迹包和相机矩阵；记录相机及实际车体位姿也精确相同。
- 记录视角共同车辆排除区域外，背景像素最大差为 0／255；记录与预测回放中右侧黑车检查区域最大差均为 0／255。
- 原始模型、v8 权重、原始预测和 v8 派生预测保留。本轮没有改依赖。
- 完整视频逐帧解码与报告资源哈希检查见交付检查文件。

## 文件位置

- 本轮目录：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nexus_3dgs/optimization_v9`
- 公交候选权重：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nexus_3dgs/optimization_v9/bus_asset_restored/checkpoint_final.pth`
- 主视频：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nexus_3dgs/optimization_v9/review/nexus_before_after.mp4`
- 公交放大：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nexus_3dgs/optimization_v9/review/bus_closeup.mp4`
- [参数保护审计](assets/nexus_3dgs_v9/asset_audit.json) · [完整验证](assets/nexus_3dgs_v9/detail_validation.json) · [逐帧质量统计](assets/nexus_3dgs_v9/bus_detail_metrics.json)
- [此前 v8 报告](nexus_3dgs_v8.html)
