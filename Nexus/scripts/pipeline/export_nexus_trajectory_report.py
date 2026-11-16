"""Export the completed measured-vs-generated trajectory experiment."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=a.experiment;out=a.output;assets=out/'assets/nexus_real_trajectory';assets.mkdir(parents=True,exist_ok=True)
    audit=json.loads((v/'audit.json').read_text());raw=json.loads((v/'raw_predictions_audit.json').read_text());review=json.loads((v/'review/review.json').read_text());q=json.loads((v/'quality.json').read_text());invariance=json.loads((v/'fixed_camera_background.json').read_text());support=json.loads((v/'observation_support.json').read_text())
    assert audit['status']==raw['status']=='passed' and review['status']=='complete' and review['frames']==80
    assert audit['checkpoint_sha256']==raw['checkpoint_sha256'] and review['raw_factorial_same_cameras_and_generated_actor_states_verified']
    files={name+'.mp4':v/'review'/f'{name}.mp4' for name in ['real_vs_nexus','free_prediction','current_interaction','factory_real_vs_nexus','fitted_reference','motion_factorial']}
    files.update({'preview.jpg':v/'review/real_vs_nexus_070.jpg','factorial.jpg':v/'review/motion_factorial_070.jpg','all_sources.jpg':v/'raw_prediction_review.jpg','quality.json':v/'quality.json','fixed_camera_background.json':v/'fixed_camera_background.json','observation_support.json':v/'observation_support.json','control_audit.json':v/'audit.json','raw_predictions_audit.json':v/'raw_predictions_audit.json','review.json':v/'review/review.json'})
    manifest=[]
    for name,source in files.items():
        target=assets/name;shutil.copyfile(source,target);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();digest=sha(target);assert digest==sha(source);manifest.append({'file':str(target.relative_to(out)),'bytes':target.stat().st_size,'sha256':digest})
    A='assets/nexus_real_trajectory';summary=q['summary'];rr=summary['native_rr'];fit=summary['native_fitted'];factory=summary['factory_rr']
    def table_row(label,item):return f'| {label} | {item["stationary"]["psnr"]:.2f} | {item["moving"]["psnr"]:.2f} | {item["all"]["psnr"]:.2f} |'
    table='\n'.join([table_row('真实标注轨迹＋共同的学习相机校正',rr),table_row('记录视角＋模型原拟合车辆位姿',fit),table_row('真实标注轨迹＋传感器原标定',factory)])
    real0=np.array(audit['motion'][0]['measured_ego_world']);real1=np.array(audit['motion'][-1]['measured_ego_world']);travel_real=float(np.linalg.norm(real1[:2,3]-real0[:2,3]));travel_free=raw['motion']['free'][-1]['travel_m'];travel_goal=raw['motion']['goal'][-1]['travel_m']
    bus_min=lambda name:min(x['bus_center_distance_m'] for x in support['conditions'][name] if 'bus_center_distance_m' in x)
    low,high=support['observed_front_bus_center_distance_range_m'];existing_min=bus_min('native_nn');goal_min=bus_min('raw_goal_nn')
    conclusion={'status':'complete','experiment_only_no_scene_retraining':True,'measured_trajectory_also_shows_motion_rendering_error':True,'fixed_camera_background_mean_abs_8bit':invariance['mean_abs_8bit'],'existing_interaction_is_not_unedited_nexus_output':True,'cannot_conclude_nexus_trajectories_inherently_render_blurry':True,'limitations':['single scene and one fixed v5 model','measured actor XY/yaw use the common road and asset mounting model','native camera corrections remain part of the rendering/adapter baseline','generated cameras have no paired real RGB; no generated-view PSNR assigned','view-support distances cover 80 front-camera observations only']}
    md=f'''# 真实轨迹与 Nexus 轨迹：同一 3DGS 的对照

**真实轨迹下，相机开始运动后也出现明显画质下降。因此，当前重影不能只归因于 Nexus 轨迹。** 本次是定位实验，没有训练或修改场景模型，也不是新的画质修复版本。

[先看三栏主对照]({A}/real_vs_nexus.mp4) · [四格拆分对照]({A}/motion_factorial.mp4) · [自由预测原始输出]({A}/free_prediction.mp4)

![左侧原始真实录像，中间真实轨迹渲染，右侧 Nexus 条件预测原始输出渲染]({A}/preview.jpg)

## 怎么看

- 主视频共 8 秒。左侧是同一时刻的真实照片组成的录像，中间和右侧都由同一个 v5 3DGS 生成。
- 建议先暂停在 **5 秒**，对照建筑轮廓、树冠、路面箭头和车顶；再看约 **7.5 秒**。轨迹不同会造成视角和车辆位置不同，不能要求中间和右侧像素对齐。
- 右侧使用保存的 Nexus 意图条件预测输出，未加本项目后续的朝向、车距或交互轨迹修正。原始推理配方本身的内置约束仍然存在。
- 两个渲染分支使用相同的场景模型、内参、曝光和同一时刻的学习相机校正；另有传感器原标定的独立控制。没有时间平滑、锐化或补帧。

## 结果

| 真实视角检查（平均 PSNR，dB） | 相机基本静止 | 相机移动／转动 | 全部 80 帧 |
| --- | --- | --- | --- |
{table}

静止段为 43 帧，运动段为 37 帧。分组按真实自车相对首帧位移超过 0.1 米，或旋转超过 0.2°，不是按画质挑选帧。更有利的“原拟合车辆位姿”控制同样从 {fit['stationary']['psnr']:.2f} 降到 {fit['moving']['psnr']:.2f} dB；其保守车辆排除区域指标从 {fit['stationary']['background_mask_psnr']:.2f} 降到 {fit['moving']['background_mask_psnr']:.2f} dB。当前渲染／相机适配基准本身存在运动后的误差，实验尚未分清几何和标定各占多少。

这些分数只针对**有对应真实照片的记录相机**。生成相机没有对应真实照片，因此没有为 Nexus 新视角计算 PSNR。图像来自原模型观测范围，不是独立未见场景测试。

## 分开看相机运动与其他车辆运动

![横向替换相机运动，纵向替换其他车辆运动]({A}/factorial.jpg)

四格中上左全部采用真实运动，右上只换相机运动，左下只换其他车辆运动，右下使用完整 Nexus 条件预测。两个混合组合是诊断用反事实，不是 Nexus 原生生成场景。

固定真实相机、只替换其他车辆运动时，在两组车辆包围盒之外的保守背景区域，80 帧平均绝对像素差为 **{invariance['mean_abs_8bit']:.6f}／255**。本例中该区域像素完全一致，说明更换其他车辆运动本身不会让整幅背景变模糊；相机视点变化与新视角下的重建／适配误差需要优先处理。这不是对生成轨迹安全性或车辆表面质量的证明。

## 原始预测与之前的交互视频要分开

- 本段真实自车首尾平面位移约 **{travel_real:.2f} 米**，Nexus 条件预测约 **{travel_goal:.2f} 米**；两者具有可比较的运动幅度。
- 自由预测的自车只有约 **{travel_free:.2f} 米**位移。因此不能因为这段较稳定，就证明模型在正常自车运动时也能清晰渲染。
- 之前交互视频沿用的是经过离线约束的派生轨迹，并包含给定巴士运动。它不是本次回放的原始条件预测。
- 80 帧记录前向相机到公交车中心的距离为 **{low:.2f}～{high:.2f} 米**；之前交互方案最小约 **{existing_min:.2f} 米**，原始条件预测最小约 **{goal_min:.2f} 米**。现有交互方案确实要求更近的公交车观察位置，可能放大资产问题。这里仅比较前向相机与车中心距离，不代表完整训练视角覆盖。

[原始自由预测对照]({A}/free_prediction.mp4) · [之前交互方案对照]({A}/current_interaction.mp4) · [原拟合位姿控制]({A}/fitted_reference.mp4) · [传感器标定控制]({A}/factory_real_vs_nexus.mp4)

## 实验如何保证可比较

同一个 v5 检查点用于全部 11 组、每组 80 帧，共 880 张渲染图。模型文件哈希、静态与动态外观参数保持不变，临时车辆位姿在结束后精确恢复。真实车辆标注在相机曝光时刻插值，未填补无效轨迹缺口；本例相机比对应点云时刻晚约 37 毫秒。

真实车辆运动使用标注的 XY／朝向，并采用与预测分支相同的地面和固定资产挂载；不是把原始标注框不加对齐地套到高斯资产。自车高度、侧倾和俯仰使用真实记录，两边只替换平面 XY／朝向。额外的原拟合位姿控制用于检查这种固定资产挂载本身的影响。

自由预测来源任务为 `{raw['raw_prediction_sources']['free']['task']}`，条件预测为 `{raw['raw_prediction_sources']['goal']['task']}`；两者来自同一个已保存推理文件，没有重新训练或重新采样。原有模型训练了 101 帧、三路相机；本次没有使用补充数据再次训练模型。

**目前支持的判断：渲染／适配基准已有问题，生成观察位置也会影响效果；本例不能证明 Nexus 轨迹先天无法清晰渲染。**

[真实视角画质数据]({A}/quality.json) · [固定相机背景检查]({A}/fixed_camera_background.json) · [原始预测来源与保护检查]({A}/raw_predictions_audit.json) · [完整控制检查]({A}/control_audit.json) · [前向观测距离]({A}/observation_support.json)
'''
    (out/'nexus_real_trajectory_comparison.md').write_text(md)
    rows=''.join(f'<tr><td>{label}</td><td>{item["stationary"]["psnr"]:.2f}</td><td>{item["moving"]["psnr"]:.2f}</td><td>{item["all"]["psnr"]:.2f}</td></tr>' for label,item in [('真实轨迹＋共同学习校正',rr),('模型原拟合车辆位姿',fit),('真实轨迹＋传感器标定',factory)])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实轨迹与 Nexus 轨迹对照</title><style>body{{font:17px/1.85 system-ui,sans-serif;max-width:1400px;margin:28px auto;padding:0 20px;background:#f3f6fa;color:#142136}}section{{padding:24px;background:white;margin:22px 0;border-radius:12px}}.callout{{padding:18px;background:#fff0d2;border-radius:10px}}video,img{{width:100%;border-radius:8px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}a{{margin-right:13px}}button{{padding:9px;margin:4px;cursor:pointer}}</style><h1>真实轨迹与 Nexus 轨迹：同一 3DGS 的对照</h1><p class="callout"><b>真实轨迹下，运动后的画质也明显下降。</b> 当前重影不能只归因于 Nexus 轨迹。本次只定位问题，模型没有重训或修改。</p><section><h2>先看这个 8 秒视频</h2><p>左：真实录像。中：真实轨迹输入 3DGS。右：Nexus 条件预测原始输出输入同一个 3DGS。建议全屏，在 5 秒、7.5 秒暂停，看树冠、建筑、路面箭头和车顶。轨迹不同，视角和车辆位置也会不同。</p><video id="main" controls preload="metadata" poster="{A}/preview.jpg"><source src="{A}/real_vs_nexus.mp4" type="video/mp4"></video><p><button onclick="jump(0)">开始</button><button onclick="jump(5)">5 秒</button><button onclick="jump(7.5)">7.5 秒</button><button onclick="document.getElementById('main').playbackRate=.5">半速</button><button onclick="document.getElementById('main').playbackRate=1">原速</button></p><p>右侧未加本项目后续的朝向、车距和交互轨迹修正；保存的原始推理配方仍有其内置约束。两边采用相同相机校正和曝光，没有时间平滑、补帧或锐化。</p></section><section><h2>真实运动本身能否重建清楚</h2><table><tr><th>记录相机平均 PSNR（dB）</th><th>基本静止 43 帧</th><th>移动／转动 37 帧</th><th>全部 80 帧</th></tr>{rows}</table><p>即使用模型原本拟合的车辆位姿，运动段也明显下降；其车辆排除区域从 {fit['stationary']['background_mask_psnr']:.2f} 降到 {fit['moving']['background_mask_psnr']:.2f} dB。当前重建／相机适配本身有问题，尚未确定几何与标定各自占比。</p><p>分组依据真实自车位移大于 0.1 米或转动大于 0.2°。只对有真实照片的记录相机计分；Nexus 新视角没有配对真值，不计算该分数。这也不是独立未见场景测试。</p></section><section><h2>相机与其他车辆，哪个在影响背景</h2><video controls preload="metadata" poster="{A}/factorial.jpg"><source src="{A}/motion_factorial.mp4" type="video/mp4"></video><p>上左全部真实运动，右上仅换相机，左下仅换其他车辆，右下完整条件预测。两个混合组合仅用于诊断。</p><p>固定真实相机只换其他车辆时，车辆保守排除区域外的背景，80 帧平均差为 <b>{invariance['mean_abs_8bit']:.6f}／255</b>。本例这些像素完全一致。整幅背景的变化主要随观察位置改变，不能直接解释为车辆预测让背景变模糊。</p></section><section><h2>为什么另外保留原始预测与现有方案</h2><img src="{A}/all_sources.jpg" alt="真实录像、真实轨迹、原始条件预测、原始自由预测、现有交互方案"><p>真实自车首尾位移约 {travel_real:.2f} 米，原始条件预测约 {travel_goal:.2f} 米。原始自由预测只有约 {travel_free:.2f} 米，画面较稳定不能代表运动渲染已经正确。</p><p>之前的交互视频含离线约束与给定巴士运动，不能当成 Nexus 原始输出。前向观测中相机到巴士中心的距离为 {low:.2f}～{high:.2f} 米，现有交互最小 {existing_min:.2f} 米，原始条件预测最小 {goal_min:.2f} 米；现有方案要求更近的观察位置，可能放大资产缺陷。这仅是前向距离检查，不是完整视角覆盖证明。</p><p><a href="{A}/free_prediction.mp4">自由预测原始输出</a><a href="{A}/current_interaction.mp4">现有交互方案</a><a href="{A}/fitted_reference.mp4">模型原拟合位姿控制</a><a href="{A}/factory_real_vs_nexus.mp4">传感器标定控制</a></p></section><section><h2>检查与文件</h2><p>同一个 v5 模型、11 组各 80 帧，共 880 张渲染。模型哈希与外观参数不变，临时位姿精确恢复。真实标注对齐到相机拍摄时刻，使用双方共同的地面与资产挂载。4 项几何控制行为检查通过，逐帧相机与车辆控制也通过核验。</p><p><a href="nexus_real_trajectory_comparison.md">完整文字说明</a><a href="{A}/quality.json">真实视角分数</a><a href="{A}/fixed_camera_background.json">背景不变性检查</a><a href="{A}/raw_predictions_audit.json">原始预测来源</a><a href="{A}/control_audit.json">模型与控制检查</a></p><p>这些结果支持优先处理重建／相机适配及新视角覆盖；没有证据证明 Nexus 轨迹先天无法清晰渲染。</p></section><script>function jump(t){{let v=document.getElementById('main');v.currentTime=t;v.play();}}</script></html>'''
    (out/'nexus_real_trajectory_comparison.html').write_text(page)
    (out/'nexus_real_trajectory_manifest.json').write_text(json.dumps({'status':'complete','conclusions':conclusion,'checkpoint_sha256':audit['checkpoint_sha256'],'assets':manifest},indent=2)+'\n');print('REAL_TRAJECTORY_REPORT_OK '+str(out.resolve()),flush=True)


if __name__=='__main__':main()
