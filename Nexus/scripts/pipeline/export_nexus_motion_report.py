"""Export the verified local motion improvement with offline videos and scope."""
import argparse,hashlib,html,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=ROOT/'artifacts/nexus_3dgs/optimization_v5';out=a.output;assets=out/'assets/nexus_3dgs_motion';assets.mkdir(parents=True,exist_ok=True)
 panel=json.loads((v/'coherent_panel_audit.json').read_text());comparison=json.loads((v/'coherent_full_review/comparison.json').read_text());q=json.loads((v/'coherent_reference_quality.json').read_text())['summary'];assert panel['status']=='passed' and comparison['actual_cameras_identical'] and comparison['actual_actor_poses_identical']
 for tag in ('stop','go'):assert json.loads((v/f'coherent_{tag}_clearance.json').read_text())['status']=='passed'
 files={'comparison.mp4':v/'coherent_rear_review/bus_ghost_comparison.mp4','preview.png':v/'coherent_rear_review/bus_follow_logged_050.png','full_comparison.mp4':v/'coherent_full_review/ego_motion_comparison.mp4','bus_stop.mp4':v/'coherent_rear_stop/camera_0.mp4','bus_go.mp4':v/'coherent_rear_go/camera_0.mp4','construction_audit.json':v/'coherent_panel_audit.json','same_poses.json':v/'coherent_full_review/comparison.json','reference_quality.json':v/'coherent_reference_quality.json'}
 manifest=[]
 for name,source in files.items():
  target=assets/name;shutil.copyfile(source,target);sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();assert sha(source)==sha(target);manifest.append({'file':str(target.relative_to(out)),'sha256':sha(target),'bytes':target.stat().st_size})
 previous=[out/'nexus_3dgs_progress.html',*sorted(out.glob('20??-??-??/nexus_3dgs_progress.html'),reverse=True)];previous=next((p for p in previous if p.is_file()),None)
 previous_link=f'<a href="{html.escape(str(previous.relative_to(out)))}">之前的多车交互报告</a>' if previous else ''
 A='assets/nexus_3dgs_motion';before,after=[q['full'][k]['psnr'] for k in ('old','new')]
 md=f'''# 运动后的重影：本轮局部结果

本轮已改善公交车后面紫色面板上的文字叠层。**地面、建筑、天空、车窗和车顶的重影仍未全部解决。** 原来的 v3 总览保留，下面是单独的局部改进版本。

![相同相机和轨迹下，左侧原版、右侧车尾表面改进]({A}/preview.png)

[播放前后对照，含车尾放大]({A}/comparison.mp4) · [全幅对照]({A}/full_comparison.mp4) · [仅停车]({A}/bus_stop.mp4) · [仅行进]({A}/bus_go.mp4)

## 怎么看

- 左边是原版，右边是这次局部优化。上排看整幅画面，下排看同一个橙框区域。
- 总视频前 8 秒是公交车停车，后 8 秒是公交车行进。建议看 **总视频约 12～14 秒**：自车接近时，注意 “SBS Transit” 是否出现多套错位边缘。
- 两侧使用相同的 Nexus 轨迹包、实际车辆位置和相机。放大区域只有裁切和缩放，没有加锐化、补帧或画面平滑。
- 地面、建筑和车顶仍有残影；本轮结果只支持“车尾面板局部改善”。

## 做了什么

原模型用许多有厚度的小高斯拼成车尾，离开训练时的观察位置后，文字容易散开。先尝试压薄高斯、限制视角颜色变化，改善均不足。直接把独立点云小平面拼起来又出现了空洞。

最终在真实点云支持下，拟合连续的局部车尾表面，再从一张真实 1920×1080 原始照片提取颜色，生成 78,934 个原生 3D 高斯；将这块面板原来的 12,248 个高斯透明度压低，减少叠层。新面板处于车辆自身的三维坐标中，会随车辆运动并接受正常的透视投影和遮挡，不是贴在视频上的固定图片。这是当前观测到的局部表面重建，不能代表任意角度都正确。

## 检查结果和代价

| 检查 | 结果 |
| --- | --- |
| 两段完整对照 | 160 帧，16 秒，相机与实际车辆位姿一致 |
| 新面板支撑 | 新增中心全部在原车辆物理支撑范围内 |
| 实际渲染车辆包围盒检查 | 两段均通过，无重叠采样 |
| 80 帧记录相机整图 PSNR | {before:.2f} → {after:.2f} dB，轻微下降 |
| 左侧面包车、右侧车辆固定检查区域 | 指标保持一致 |
| 背景、天空、其他车辆、相机参数 | 本次车尾版本中逐项保持一致 |

PSNR 检查使用已经观测过的记录相机，只用于监控原有画质是否退化；生成的新视角没有对应真实图像，不能把该指标当成新视角质量证明。局部面板更稳定与记录视角轻微退化之间存在取舍。

额外真实观测的背景实验已完成初步检查，但损伤了记录视角细节，因此未合入上述车尾视频。道路标线补片同样出现错位箭头，因此也未采用。

[构建检查]({A}/construction_audit.json) · [相机与车辆一致性]({A}/same_poses.json) · [记录视角质量]({A}/reference_quality.json)
'''
 (out/'nexus_3dgs_motion_update.md').write_text(md)
 page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>运动重影：车尾局部改进</title><style>body{{font:17px/1.8 system-ui,sans-serif;max-width:1200px;margin:28px auto;padding:0 22px;background:#f5f7fb;color:#182337}}video,img{{width:100%;border-radius:10px}}section{{background:white;padding:22px;margin:20px 0;border-radius:12px}}button,a{{margin-right:12px}}button{{padding:9px;cursor:pointer}}.scope{{background:#fff0cf;padding:16px}}table{{border-collapse:collapse}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:10px}}</style><h1>运动后的重影：车尾局部改进</h1><p class="scope">本轮改善的是公交车后面紫色面板上的文字叠层。地面、建筑、天空、车窗和车顶的重影仍未全部解决。</p><section><h2>先看相同轨迹下的前后对照</h2><p>左侧原版，右侧局部优化；上排看整幅画面，下排看同一个橙框。先看总视频 12～14 秒的车尾文字。</p><video id="comparison" controls preload="metadata" poster="{A}/preview.png"><source src="{A}/comparison.mp4" type="video/mp4"></video><p><button onclick="jump(0)">停车段 0 秒</button><button onclick="jump(8)">行进段 8 秒</button><button onclick="jump(13)">接近时 13 秒</button><button onclick="document.getElementById('comparison').playbackRate=.5">半速</button><button onclick="document.getElementById('comparison').playbackRate=1">原速</button></p><p>两侧相机、实际车辆位置和 Nexus 轨迹包一致。只有裁切和缩放，没有锐化、补帧或平滑。</p></section><section><h2>这一帧有什么变化</h2><img src="{A}/preview.png" alt="同一相机与车辆位姿下，车尾文字局部改进前后"><p>看 “SBS Transit” 的错位边缘。周围建筑、地面、车顶仍有残影，这些区域尚未修复。</p></section><section><h2>改动与检查</h2><p>用真实点云拟合连续的车尾表面，再从真实原始照片提取颜色，生成车辆自身三维坐标中的原生高斯。它随车辆运动并接受透视和遮挡。当前只覆盖有观测支持的局部面板。</p><table><tr><th>检查</th><th>结果</th></tr><tr><td>完整对照</td><td>160 帧、16 秒，实际相机和车辆位姿一致</td></tr><tr><td>两段车辆包围盒</td><td>检查通过，无重叠采样</td></tr><tr><td>80 帧记录视角整图 PSNR</td><td>{before:.2f} → {after:.2f} dB，有轻微代价</td></tr><tr><td>左右车辆固定区域</td><td>指标一致</td></tr></table><p>记录视角指标用于检查原画质，不是生成新视角的真实质量证明。背景的额外观测实验有明显画质退化，未合入；道路补片也有错位箭头，未采用。</p></section><section><h2>其他文件</h2><p><a href="{A}/full_comparison.mp4">全幅对照</a><a href="{A}/bus_stop.mp4">仅停车</a><a href="{A}/bus_go.mp4">仅行进</a><a href="nexus_3dgs_motion_update.md">文字文档</a>{previous_link}</p></section><script>function jump(t){{let v=document.getElementById('comparison');v.currentTime=t;v.play();}}</script></html>'''
 (out/'nexus_3dgs_motion_update.html').write_text(page)
 (out/'nexus_3dgs_motion_manifest.json').write_text(json.dumps({'status':'passed','scope':'verified local rear-panel improvement; full-scene motion ghosting unresolved','assets':manifest},indent=2)+'\n');print('MOTION_REPORT_OK '+str(out),flush=True)
if __name__=='__main__':main()
