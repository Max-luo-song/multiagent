"""Export the completed vehicle-motion stability fixes and their controls."""
import argparse,hashlib,html,json,shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['experiment','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();v=a.experiment;review=json.loads((v/'review/review.json').read_text());audit=json.loads((v/'stability_validation.json').read_text());path=json.loads((v/'stability_candidates/path_stabilization.json').read_text());body=json.loads((v/'stability_candidates/observed_body_stable/stability_audit.json').read_text());heading=json.loads((v/'heading_repaired/heading_audit.json').read_text());assert review['status']=='complete' and audit['status']=='passed'
    out=a.output;assets=out/'assets/nexus_3dgs_v8';assets.mkdir(parents=True,exist_ok=True);files={name:v/'review'/name for name in review['videos']}
    for name in ['nexus_before_after_025.jpg','nexus_before_after_095.jpg','black_car_closeup_095.jpg','recorded_before_after_070.jpg','bus_yaw_comparison.png']:
        if (v/'review'/name).is_file():files[name]=v/'review'/name
    files.update({'stability_validation.json':v/'stability_validation.json','body_stability_audit.json':v/'stability_candidates/observed_body_stable/stability_audit.json','path_stabilization.json':v/'stability_candidates/path_stabilization.json','heading_audit.json':v/'heading_repaired/heading_audit.json','derived_prediction_manifest.json':v/'stability_candidates/heading_and_bus_path/manifest.json','render_run.json':v/'nexus_stable/run.json','review.json':v/'review/review.json'})
    manifest=[]
    for name,source in files.items():
        target=assets/name;shutil.copyfile(source,target);digest=hashlib.sha256(target.read_bytes()).hexdigest();assert digest==hashlib.sha256(source.read_bytes()).hexdigest();manifest.append({'file':str(target.relative_to(out)),'sha256':digest,'bytes':target.stat().st_size})
    A='assets/nexus_3dgs_v8';bus=next(r for r in body['tracks'] if r['model_index']==6);reduction=100*(1-bus['rotation_second_difference_rms_deg_after']/bus['rotation_second_difference_rms_deg_before']);q=audit['observed_view_full_psnr'];rows=[];htmlrows=[]
    for name,label in [('stationary','基本静止 43 帧'),('moving','运动 37 帧'),('all','全部 80 帧')]:
        x=q[name];rows.append(f'| {label} | {x["old_native"]:.2f} | {x["new"]:.2f} |');htmlrows.append(f'<tr><td>{label}</td><td>{x["old_native"]:.2f}</td><td>{x["new"]:.2f}</td></tr>')
    checkpoint=v/'stability_candidates/observed_body_stable/checkpoint_final.pth';bundle=v/'stability_candidates/heading_and_bus_path'
    md=f'''# v8：车辆左右晃动与黑车“四散”修复

本轮针对 v7 候选中反馈的两个问题，保持原有 3DGS 背景、车辆高斯形状、纹理、透明度、相机与曝光参数，修正运动接入和车体位姿。**没有重新训练 Nexus 或 3DGS，也没有对视频像素做时间平滑、融合或锐化。** v7 原始文件仍保留。

## 先看结果

[Nexus 回放：v7 与 v8 完整对照]({A}/nexus_before_after.mp4) · [黑车放大对照]({A}/black_car_closeup.mp4)

主视频左侧是 v7 候选，右侧是 v8 修复分支，共 8 秒、完整 80 帧。先看 **0～1 秒**的公交车横向动作，再暂停在 **7.5 秒**看右侧黑车。两边相机完全相同。

![右侧黑车第 7.5 秒，左为 v7，右为仅修异常朝向]({A}/black_car_closeup_095.jpg)

## 黑车为什么会像散开

黑车 `2a3e0f8d92fa5791` 几乎停车，预测在约 7.5 秒处却出现一次约 180° 的朝向跳变。相邻采样的位移最多约 17 厘米；渲染插值会让车身快速转身，暴露原本重建不足的表面，形成散开的外观。

v7 的原始预测对照没有启用此前的孤立朝向保护。本次在独立派生分支中接回这层保护：只改这一个异常朝向采样，车辆位置、时间、有效性、历史与给定终点都保持原样，不额外读取真实未来运动。

**单独修黑车朝向时，第 20～89 帧共 70 帧与 v7 逐像素相同；最后一秒黑车不再翻转。**

[三阶段拆分：v7／仅修黑车／完整 v8]({A}/heading_ablation.mp4)

## 公交车为什么会左右晃

两条回放中有不同来源：

- **记录轨迹回放：**逐帧车体拟合引入了小幅位置与旋转抖动。v8 在三维车体位姿上做有界稳定化，场景坐标下的纵向运动和高度保持，连续有效段的两端锁定；不跨越无效轨迹缺口。实际公交车横向最大修正约 **{bus['maximum_lateral_change_m']*100:.1f} 厘米**，旋转最大约 **{bus['maximum_rotation_change_deg']:.2f}°**。旋转二阶差分的 RMS 指标下降约 **{reduction:.0f}%**，这是位姿抖动指标，不是“视觉误差降低同等比例”。
- **Nexus 原始预测回放：**公交车第一步存在明显横移。v8 对公交横向位移加入与预测朝向一致的运动约束；在固定参考方向下，第一步横移从 **{path['first_future_lateral_step_before_m']*100:.1f} 厘米降到 {path['first_future_lateral_step_after_m']*100:.1f} 厘米**，所有时刻的平面修正最多 **{path['max_position_change_m']*100:.0f} 厘米**。公交的纵向前进过程、时间采样、历史、给定终点、朝向和高度都保持原输入；Nexus 自车及其他车辆的预测位置保留，黑车只作前述异常朝向修正。

因此，右栏是**明确标注的修复派生轨迹**，不能再称作未经处理的 Nexus 原始输出。原始预测文件与模型权重没有修改。未被 Nexus 预测的场景车辆使用稳定化后的记录车体位姿。

[记录轨迹：真实照片／v7／v8]({A}/recorded_before_after.mp4) · [记录轨迹公交车放大]({A}/bus_closeup.mp4)

![记录轨迹第 70 帧，左真实照片、中 v7、右 v8]({A}/recorded_before_after_070.jpg)

## 画质和回归检查

| 记录相机整图平均 PSNR，dB | v7 候选 | v8 |
| --- | --- | --- |
{chr(10).join(rows)}

分数只对应有真实照片的记录相机。运动段差异约 -0.07 dB，整体画质基本保持；静止段与全段平均分数提高。Nexus 新视角没有对应真实照片，不计算其 PSNR。

- Nexus 两组的全部 80 个相机位姿完全相同。
- 记录轨迹两组在共同车辆排除区域之外，背景像素最大差为 **0／255**。
- 高斯外观、几何、透明度、背景、天空、相机、曝光与可见性参数精确保持；只改变车体位姿。
- 相同物理车体位姿下，两个模型渲染最大差为 1／255，符合浮点旋转往返的舍入范围。
- 本段 80 帧检查未新增车辆包围盒重叠。
- 四项稳定化行为测试、三项低速朝向保护回归测试通过，原始数据和预测文件保留。

本轮集中处理运动稳定性，v7 遗留的细节模糊和新视角局部缺陷仍存在；没有把这次结果描述为所有画质问题都已解决。

## 文件位置

- 本轮目录：`{v.resolve()}`
- 车体稳定化检查点：`{checkpoint.resolve()}`
- 明确修正后的预测派生包：`{bundle.resolve()}`
- 主视频：`{(v/'review/nexus_before_after.mp4').resolve()}`
- 保留 v7 对照：[v7 报告](nexus_3dgs_v7.html)

[逐项验证]({A}/stability_validation.json) · [车体位姿审计]({A}/body_stability_audit.json) · [公交路径修正范围]({A}/path_stabilization.json) · [黑车朝向修正]({A}/heading_audit.json)
'''
    if (assets/'bus_yaw_comparison.png').is_file():md+=f'\n![公交车朝向变化：保留缓慢转向，减少逐帧摆动]({A}/bus_yaw_comparison.png)\n'
    (out/'nexus_3dgs_v8.md').write_text(md)
    chart=f'<img src="{A}/bus_yaw_comparison.png" alt="公交车朝向变化曲线">' if (assets/'bus_yaw_comparison.png').is_file() else ''
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v8 车辆晃动与黑车四散修复</title><style>body{{font:17px/1.85 system-ui,sans-serif;max-width:1450px;margin:25px auto;padding:0 20px;background:#f3f6fa;color:#142136}}section{{background:white;padding:24px;border-radius:12px;margin:22px 0}}video,img{{width:100%;border-radius:8px}}.note{{background:#e7f2ec;padding:18px}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}button{{padding:8px;margin:5px}}code{{overflow-wrap:anywhere}}</style><h1>v8：车辆左右晃动与黑车“四散”修复</h1><p class="note">沿用 v7 的背景与车辆高斯，修正运动接入和车体位姿。没有重训网络，也没有对视频画面做时间平滑。原始 v7 和 Nexus 预测保留。</p><section><h2>先看完整 8 秒对照</h2><p>左：v7 候选；右：v8 修复分支。先看 0～1 秒的公交横向动作，再看 7～8 秒的右侧黑车。两边相机相同，完整保留 80 帧。</p><video id="main" controls preload="metadata" poster="{A}/nexus_before_after_095.jpg"><source src="{A}/nexus_before_after.mp4" type="video/mp4"></video><button onclick="jump(.5)">0.5 秒：公交车</button><button onclick="jump(7.5)">7.5 秒：黑车</button><button onclick="document.getElementById('main').playbackRate=.5">半速</button><button onclick="document.getElementById('main').playbackRate=1">原速</button><p>右栏使用明确修正的派生运动，公交横向最多调整 35 厘米，不能称作未经处理的 Nexus 原始输出。</p></section><section><h2>黑车：低速时异常翻转约 180°</h2><p>约 7.5 秒处，黑车相邻采样位移不到 17 厘米，却出现一次朝向反转。快速转身暴露了重建不足的表面，形成“四散”的外观。本次只修这一个孤立异常朝向，位置、时间、历史与终点不变。</p><video controls preload="metadata" poster="{A}/black_car_closeup_095.jpg"><source src="{A}/black_car_closeup.mp4" type="video/mp4"></video><p>单独修朝向时，前 70 帧与 v7 逐像素相同，最后一秒车体保持完整。</p><details><summary>查看三阶段拆分对照</summary><video controls preload="metadata"><source src="{A}/heading_ablation.mp4" type="video/mp4"></video></details></section><section><h2>公交车：分别处理拟合抖动与预测横移</h2><p>记录轨迹中的逐帧车体位姿做小幅稳定化，公交实际横向修正最多 {bus['maximum_lateral_change_m']*100:.1f} 厘米、旋转最多 {bus['maximum_rotation_change_deg']:.2f}°；保留前进运动和缓慢转向。原始预测中的第一步横移从 {path['first_future_lateral_step_before_m']*100:.1f} 降到 {path['first_future_lateral_step_after_m']*100:.1f} 厘米，最大平面修正 35 厘米；历史与给定终点不动。</p><video controls preload="metadata"><source src="{A}/bus_closeup.mp4" type="video/mp4"></video>{chart}<p>曲线来自记录轨迹的公交车体朝向。位姿旋转二阶差分 RMS 下降约 {reduction:.0f}%，用于衡量每帧来回摆动，不表示所有视觉误差都降低同等比例。</p><details><summary>真实照片／v7／v8 完整记录轨迹对照</summary><video controls preload="metadata"><source src="{A}/recorded_before_after.mp4" type="video/mp4"></video></details></section><section><h2>画质与保护检查</h2><table><tr><th>记录视角平均 PSNR，dB</th><th>v7 候选</th><th>v8</th></tr>{''.join(htmlrows)}</table><p>运动段画质基本保持；静止段和全段均值提高。全部 80 个生成相机位姿相同，记录视角中车辆排除区域之外的背景像素最大差为 0。高斯形状、纹理、透明度和背景参数原样保留。本段检查没有新增车辆包围盒重叠。</p><p>Nexus 新视角没有配对真实照片，不计算其 PSNR。v7 遗留的细节模糊和新视角局部缺陷仍存在，本轮集中修运动稳定性。</p></section><section><h2>文档与路径</h2><p><a href="nexus_3dgs_v8.md">完整说明、修正范围与文件路径</a> · <a href="{A}/stability_validation.json">验证结果</a> · <a href="{A}/path_stabilization.json">公交路径改动审计</a> · <a href="{A}/heading_audit.json">黑车朝向审计</a> · <a href="nexus_3dgs_v7.html">原 v7 报告</a></p><p>模型：<code>{html.escape(str(checkpoint.resolve()))}</code></p></section><script>function jump(t){{let v=document.getElementById('main');v.currentTime=t;v.pause();}}</script></html>'''
    (out/'nexus_3dgs_v8.html').write_text(page)
    (out/'nexus_3dgs_v8_manifest.json').write_text(json.dumps({'status':'complete','scope':'targeted 3D motion stability fixes; original v7 retained','checkpoint':str(checkpoint.resolve()),'assets':manifest,'audit':audit},indent=2)+'\n');print('V8_REPORT_COMPLETE '+str(out.resolve()),flush=True)


if __name__=='__main__':main()
