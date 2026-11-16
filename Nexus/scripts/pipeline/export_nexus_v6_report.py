"""Export the motion repair experiment and its quality limits for review."""
import argparse,hashlib,html,json,shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];v=root/'artifacts/nexus_3dgs/optimization_v6';out=a.output;assets=out/'assets/nexus_3dgs_motion_v6';assets.mkdir(parents=True,exist_ok=True)
    quality=json.loads((v/'final_reference_quality.json').read_text());comparison=json.loads((v/'final_comparison/comparison.json').read_text());geometry=json.loads((v/'final_geometry_audit.json').read_text())
    assert quality['frames']==80 and comparison['frames']==160
    assert comparison['actual_cameras_identical'] and comparison['actual_actor_poses_identical']
    q=quality['summary']['full'];oldpsnr=q['old']['psnr'];newpsnr=q['new']['psnr']
    passes=all(quality['summary'][region]['new']['psnr']>=quality['summary'][region]['old']['psnr']-.3 and quality['summary'][region]['new']['ssim']>=quality['summary'][region]['old']['ssim']-.005 for region in quality['summary'])
    status='局部约束改善，整体运动重影仍未全部解决' if passes else '几何约束改善，但未通过画质验收；保留为实验候选'
    acceptance={'status':'partial_only' if passes else 'candidate_rejected','full_scene_motion_ghosting_solved':False,'reference_quality_gate_passed':passes,'thresholds':{'maximum_region_mean_psnr_drop_db':.3,'maximum_region_mean_ssim_drop':.005},'regions':list(quality['summary']),'source_views_are_not_independent_holdout':True,'notes':'This gate only detects observed-camera regressions; passing it would not prove generated-view quality.'}
    (v/'final_acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    files={'comparison.mp4':v/'final_comparison/ego_motion_comparison.mp4','preview.jpg':v/'final_comparison/chapter2_070.jpg','motion_controls.jpg':v/'motion_control_explained.jpg','candidate_go.mp4':v/'final_candidate/go/both/camera_0.mp4','candidate_stop.mp4':v/'final_candidate/stop/both/camera_0.mp4','reference_quality.json':v/'final_reference_quality.json','geometry_audit.json':v/'final_geometry_audit.json','same_actual_poses.json':v/'final_comparison/comparison.json','acceptance.json':v/'final_acceptance.json','protected_parameters.json':v/'reconstruction/nuplan/static_radiance_25000/protected_parameters.json','technical.md':root/'docs/nexus_3dgs_optimization_v6.md'}
    manifest=[]
    files['behavior_checks.json']=v/'checks.json'
    for name,source in files.items():
        target=assets/name;shutil.copyfile(source,target);digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sha=digest(target);assert sha==digest(source);manifest.append({'file':str(target.relative_to(out)),'bytes':target.stat().st_size,'sha256':sha})
    A='assets/nexus_3dgs_motion_v6';road0=geometry['road']['old']['normal_distance_p90_m'];road1=geometry['road']['new']['normal_distance_p90_m']
    sky0=sum(x['old_finite_opacity_mean'] for x in geometry['sky'])/len(geometry['sky']);sky1=sum(x['new_finite_opacity_mean'] for x in geometry['sky'])/len(geometry['sky'])
    table=f'| 原记录相机的 80 帧整图 PSNR | {oldpsnr:.2f} → {newpsnr:.2f} dB |\n| 道路支撑高斯距实测平面的误差 P90 | {road0:.3f} → {road1:.3f} 米 |\n| 保守天空区域的有限深度不透明度（两帧抽查） | {sky0:.3f} → {sky1:.3f} |\n| 公交车停车、行进两段实际相机与车辆位姿 | 160 帧一致 |\n| 车辆模型、原 101 帧车辆位姿和相机参数 | 参数核验保留 |'
    md=f'''# 自车运动重影：v6 修复实验

**{status}。整景重影尚未完全解决，原 v5 稳定版没有被替换。**

[播放相同轨迹的完整前后对照]({A}/comparison.mp4) · [公交车行进候选]({A}/candidate_go.mp4) · [公交车停车候选]({A}/candidate_stop.mp4)

![相同相机和实际车辆状态下，左侧原 v5，右侧静态重建候选]({A}/preview.jpg)

## 怎么看

- 左边是原 v5，右边是本轮候选。前 8 秒公交车停车，后 8 秒公交车行进；两段中的自车都按原轨迹运行。
- 在约 4～7 秒、12～15 秒暂停，对照路面箭头、建筑边缘、树冠和公交车车顶。局部几何改善不代表所有区域都更清楚。
- 视频使用相同实际相机与车辆位姿，没有换成原始驾驶视频，也没有补帧、时间平滑或锐化。
- 此候选允许背景曝光重新学习，生成视频统一使用第 20 帧曝光，避免时间变化引起闪烁；车辆在高斯光栅化前转换颜色基，保持原车辆的辐射颜色。原记录视角指标使用对应图像的曝光。

## 本轮实际修复和训练

1. 修复颜色校正网络全零初始化导致逐图像分支无法学习的问题；旧检查点直接载入仍保持原行为。
2. 修复训练时天空射线与优化后的高斯相机方向不一致的问题；采用显式配置，兼容旧模型。
3. 补齐同一 nuPlan 场景四路相机 131 帧、524 张真实图像。先进行 6,000 步静态几何训练，再固定几何进行 4,000 步外观训练。车辆和相机参数冻结。
4. 使用真实点云深度、实测道路平面与带置信度的语义预测。训练排除标注车辆及附近阴影，减少交通物体写入背景。
5. 完成四组运动对照、旋转/平移拆分、相同状态重放核验和 45 项相关行为检查。

## 验证结果

| 检查 | 结果 |
| --- | --- |
{table}

道路约束和天空分层有实际改善，但记录视角的画质检查仍有代价；建筑、树木、车辆表面的重影并未全部消失。该候选暂不替换原版。记录相机的 PSNR/SSIM 是退化检查，不能作为生成新视角的真实质量证明；基础模型已经见过这些图像，微调排除部分帧也不构成独立测试集。

## 问题主要在哪里

![相机与车辆运动分别控制，均展示末帧]({A}/motion_controls.jpg)

上左相机和车辆都固定，上右仅相机运动，下左仅车辆运动，下右两者都运动。仅相机运动时，背景就出现重影。旋转对照基本稳定，平移暴露问题，支持继续处理深度、表面厚度和遮挡。不能据此排除车辆自身资产的问题。

本轮也尝试了直接压平道路、道路高斯层、局部点云背景以及全局天空替换。其中部分区域更清楚，但出现接缝或记录视角退化，没有合入最终候选。统一改用传感器相机标定的对照也没有改善，未采用。

[记录视角质量]({A}/reference_quality.json) · [道路与天空检查]({A}/geometry_audit.json) · [实际相机和车辆一致性]({A}/same_actual_poses.json) · [45 项行为检查]({A}/behavior_checks.json) · [验收判定]({A}/acceptance.json) · [技术说明与复现入口]({A}/technical.md)
'''
    (out/'nexus_3dgs_motion_v6.md').write_text(md)
    tr=''.join(f'<tr><td>{html.escape(region)}</td><td>{values["old"]["psnr"]:.2f} → {values["new"]["psnr"]:.2f} dB</td><td>{values["old"]["ssim"]:.4f} → {values["new"]["ssim"]:.4f}</td></tr>' for region,values in quality['summary'].items())
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>自车运动重影：v6 修复实验</title><style>body{{font:17px/1.8 system-ui,sans-serif;max-width:1250px;margin:30px auto;padding:0 20px;background:#f4f6fa;color:#172238}}section{{background:white;padding:22px;margin:22px 0;border-radius:12px}}.scope{{padding:18px;background:#ffedcf;border-radius:10px}}video,img{{width:100%;border-radius:8px}}button{{padding:9px;margin:4px;cursor:pointer}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}a{{margin-right:14px}}</style><h1>自车运动后的重影：v6 修复实验</h1><p class="scope"><b>{status}。</b><br>已修复训练与天空射线一致性问题，整体运动重影尚未全部解决；原 v5 没有被替换。</p><section><h2>先看相同轨迹下的完整对照</h2><p>左侧原 v5，右侧重建候选。前 8 秒公交车停车，后 8 秒公交车行进；自车继续沿原轨迹运动。暂停在约 5 秒和 13 秒，看道路箭头、建筑边缘、树冠、车顶；请同时留意细节是否退化。</p><video id="v" controls preload="metadata" poster="{A}/preview.jpg"><source src="{A}/comparison.mp4" type="video/mp4"></video><p><button onclick="jump(5)">停车段 5 秒</button><button onclick="jump(13)">行进段 13 秒</button><button onclick="document.getElementById('v').playbackRate=.5">半速</button><button onclick="document.getElementById('v').playbackRate=1">原速</button></p><p>160 帧实际相机和车辆状态核验一致。没有替换成真实驾驶视频，没有时间平滑、补帧或锐化。生成视频采用固定观测曝光，车辆颜色在原生高斯渲染中保留。</p></section><section><h2>这轮做了哪些实际修复</h2><p>修复颜色校正网络的失效初始化，以及训练时天空射线没有随优化相机同步的问题。补齐 524 张真实四路相机图像，先训练静态几何 6,000 步，再固定几何优化外观 4,000 步。车辆模型、车辆位姿和相机参数冻结，结束后逐参数核验。45 项相关行为检查通过。</p><p>道路支撑高斯距离实测平面的 P90 误差：<b>{road0:.3f} → {road1:.3f} 米</b>；两帧保守天空抽查的有限深度不透明度：<b>{sky0:.3f} → {sky1:.3f}</b>。这些证明局部约束改善，不能代替整幅画面质量检查。</p></section><section><h2>画质检查及尚未解决的问题</h2><table><tr><th>固定检查区域</th><th>80 帧平均 PSNR</th><th>80 帧平均 SSIM</th></tr>{tr}</table><p>本轮候选仍有画质代价，建筑、树木和车辆的重影尚未全部消除，暂不替换原版。这里使用已经观测过的记录相机图像检查退化，不是生成视角的真实质量证明，也不是独立测试集。</p></section><section><h2>为什么重点检查三维场景</h2><img src="{A}/motion_controls.jpg" alt="四组运动对照"><p>上左全部固定，上右仅相机运动，下左仅车辆运动，下右全部运动。仅相机运动时，背景仍出现重影。旋转组基本稳定，平移组更明显，因此优先检查几何深度与遮挡；车辆资产仍需独立处理。</p></section><section><h2>文件与细节</h2><p><a href="{A}/candidate_go.mp4">行进候选视频</a><a href="{A}/candidate_stop.mp4">停车候选视频</a><a href="nexus_3dgs_motion_v6.md">文字报告</a><a href="{A}/technical.md">技术说明</a><a href="{A}/reference_quality.json">画质检查</a><a href="{A}/geometry_audit.json">几何检查</a><a href="{A}/acceptance.json">验收判定</a></p></section><script>function jump(t){{let v=document.getElementById('v');v.currentTime=t;v.play();}}</script></html>'''
    (out/'nexus_3dgs_motion_v6.html').write_text(page);(out/'nexus_3dgs_motion_v6_manifest.json').write_text(json.dumps({'status':'complete','result':acceptance,'assets':manifest},indent=2)+'\n');print('V6_REPORT_OK '+str(out.resolve()),flush=True)


if __name__=='__main__':main()
