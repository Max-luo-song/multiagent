"""Export a portable Chinese results report with existing verified media."""
import argparse
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'report');a=p.parse_args()
    out=a.output.resolve();assets=out/'assets/nexus_3dgs';assets.mkdir(parents=True,exist_ok=True)
    source=ROOT/'artifacts/nexus_3dgs/optimization_v3';selected=source/'original_roof'
    verification=json.loads((source/'verification.json').read_text())
    g7=json.loads((selected/'g7_audit/verification.json').read_text())
    stop=json.loads((selected/'actual_stop_clearance.json').read_text());go=json.loads((selected/'actual_go_clearance.json').read_text())
    files={
        'interaction.mp4':source/'final/interaction_explained.mp4',
        'bus_stop.mp4':source/'final/bus_stop.mp4',
        'bus_go.mp4':source/'final/bus_follow_logged.mp4',
        'bus_comparison.mp4':source/'final/bus_ghost_comparison.mp4',
        'interaction_preview.png':source/'final/preview.png',
        'bus_comparison_preview.png':selected/'bus_comparison/bus_stop_035.png',
        'g7_heading.png':source/'final/g7_heading_audit.png',
        'v3_verification.json':source/'verification.json',
        'stop_clearance.json':selected/'actual_stop_clearance.json',
        'go_clearance.json':selected/'actual_go_clearance.json',
        'g7_verification.json':selected/'g7_audit/verification.json',
    }
    manifest=[]
    for name,path in files.items():
        path=path.resolve(strict=True);destination=assets/name;shutil.copy2(path,destination)
        digest=hashlib.sha256(path.read_bytes()).hexdigest();assert hashlib.sha256(destination.read_bytes()).hexdigest()==digest
        manifest.append({'file':str(destination.relative_to(out)),'source':str(path),'bytes':destination.stat().st_size,'sha256':digest})
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    A='assets/nexus_3dgs';quality=verification['reference_quality']['full']
    pairs=[('巴士停车',g7['variants']['bus_stop']),('巴士沿记录轨迹行进',g7['variants']['bus_follow_logged'])]
    rows='\n'.join(f"| {name} | {r['before']['max_heading_step_deg']:.2f}° | {r['after']['max_heading_step_deg']:.2f}° |" for name,r in pairs)
    intro='已跑通 Nexus 多车轨迹到 3DGS 场景渲染的流程，并完成局部车顶残影、模型车 7 异常急转和示意图可读性改进。自车移动后的全场景模糊与重影仍在优化，本报告不把它列为已解决。'
    legend=[('红色','自车','看它与巴士及周围车的距离变化。'),('橙色','给定巴士','停车或沿记录轨迹行进，是本例的干预条件。'),('蓝色','模型预测车','由 Nexus 预测，再经过离线运动与碰撞约束。'),('灰色','记录轨迹车辆','提供场景上下文，使用记录轨迹。'),('紫色','对向车 7（原 G7）','重点观察末尾是否突然反向；车头朝下本来就是正常的对向行驶方向。')]
    legend_md='\n'.join(f'| {c} | {role} | {meaning} |' for c,role,meaning in legend)
    md=f'''# Nexus × 3DGS：已有结果与观看说明

生成时间：{stamp}。主要展示当前保留的 v3 结果；正在验证的 v4 模型未替换该版本。

{intro}

## 1. 先看多车交互总览

![多车交互：左侧俯视示意，右侧同一时刻的3DGS画面]({A}/interaction_preview.png)

[播放完整交互视频：约16秒]({A}/interaction.mp4) · [只看巴士停车]({A}/bus_stop.mp4) · [只看巴士行进]({A}/bus_go.mp4)

**建议先看左图的相对位置，再看右图的渲染质量。** 左图与右图取自同一组实际渲染位姿。左图适合看车的运动关系；右图适合检查重影、漂移和纹理。

- 总视频前约8秒为“巴士停车”，后约8秒为“巴士沿记录轨迹行进”；每一段都从自己的0秒重新计时。
- 看橙色巴士的运动条件，再看红色自车和蓝色预测车的相对位置。当前场景含停车和低速车辆，不需要期待所有车都同时大幅移动。
- 当前场景共建模22辆车，加自车为23个车辆参与者；画面和俯视窗口只会显示其中一部分，且并非每辆车全程都有有效预测。

| 颜色 | 角色 | 应该怎么看 |
| --- | --- | --- |
{legend_md}

**左图的几个符号：** 车内白箭头指车头；细线只画过去2秒；浅蓝区域表示前相机视野方向。图上方是起始自车的前方，坐标单位是米。这是车辆包围盒示意，不能当作精确车道地图。左图中的车可能在相机画面外，也可能被巴士遮挡。

**轨迹来源：** 橙色巴士是给定条件；蓝色车辆是 Nexus 预测加明确的离线约束；灰色车辆是记录轨迹。此次工作没有更新 Nexus 模型权重。对向车7的有效预测在4秒结束，之后不外推、不补造轨迹。

## 2. 看巴士车顶的局部前后对照

![巴士车顶局部前后对照]({A}/bus_comparison_preview.png)

[播放巴士车顶前后对照]({A}/bus_comparison.mp4)

**左边是旧模型，右边是车顶清理后的保留版本。** 上排看整幅画面，下排看同一车顶区域的放大图。两边使用相同轨迹与相机，应重点看车顶上方的残留纹理、轮廓外的薄雾，以及车身文字是否保留。放大图仅裁切和缩放原像素。

已确认的局部结果：移除159个缺少观测支撑的车顶高斯，收紧1,249个车顶边缘高斯；三个检查时刻的车顶轮廓外平均透明度下降约84.9%。这只描述三个时刻的局部代理指标，不代表全视频重影下降84.9%。

记录相机下整图 PSNR 为 {quality['old']['psnr']:.2f} → {quality['new']['psnr']:.2f} dB，左侧面包车和右侧车辆检查区域指标保持一致。该检查说明局部清理没有明显损伤已观测视角，不能证明生成新视角已经清楚。

## 3. 看对向车7的异常急转修复

![对向车7朝向变化：红色修复前，蓝色修复后]({A}/g7_heading.png)

**这张图中红线是修复前、蓝线是修复后。** 重点看3.5～4秒处是否突然跳变，不必把角度数值直接对照俯视图的上下方向。对向车的车头朝下是正常的；本次修复针对低速时不合理的急转与侧移。

| 给定条件 | 修复前最大相邻朝向变化 | 修复后 |
| --- | ---: | ---: |
{rows}

表中是相邻预测样本之间的朝向变化，用于衡量“突跳”，并非全程累计转角。修复只调整自由预测中的异常尾段，保留历史、给定巴士和轨迹有效范围。

## 4. 已经检查了什么

| 检查项 | 已有结果 | 正确理解 |
| --- | --- | --- |
| 多车交互与3DGS渲染 | 两种巴士条件已生成完整视频 | 右图由3DGS按车辆和相机位姿渲染 |
| 实际渲染时刻的车辆车框 | 每种条件80帧，均为0组重叠 | 检查覆盖全场景23个车辆参与者 |
| 最小车框分离指标 | 停车约{stop['minimum_sat_separation_m']:.3f}米；行进约{go['minimum_sat_separation_m']:.3f}米 | 是二维车框分离指标，不宜直接当作真实保险杠最短距离 |
| 适配与轨迹回归 | v3的18项基础检查、13项扩展检查通过 | 验证身份、时间、坐标和临时位姿恢复等流程 |

车框检查只说明已采样时刻的二维车辆包围盒没有重叠，不能替代连续时间或闭环安全验证，也没有覆盖全部行人、自行车及静态障碍。画面中的重影、阴影与纹理仍需要单独观察。

## 5. 当前仍要重点看的问题

自车静止时相对清楚，自车移动后道路、建筑、车辆和天空仍可能模糊或重影。这是当前保留版本的已知限制。相机移动后会暴露原观测中不明显的几何深度、遮挡和高斯混合问题，不能凭一张静止截图判断已修复。

正在进行的 v4 优化只在独立目录验证，不把失败或未验收的模型替换进上述视频。请优先用本报告理解已有功能和局部改善，再用移动阶段检查全场景画质。

## 文件与证据

本报告的视频、截图和审计文件都已复制到 `assets/nexus_3dgs/`，连同整个 `report/` 目录保存即可离线查看。

- [保留版本的验证记录]({A}/v3_verification.json)
- [停车条件车框检查]({A}/stop_clearance.json) / [行进条件车框检查]({A}/go_clearance.json)
- [对向车7检查]({A}/g7_verification.json)
- [报告素材来源与哈希](nexus_3dgs_report_manifest.json)
'''
    (out/'nexus_3dgs_progress.md').write_text(md)
    legend_html=''.join(f'<tr><td>{html.escape(c)}</td><td>{html.escape(role)}</td><td>{html.escape(text)}</td></tr>' for c,role,text in legend)
    g7_html=''.join(f"<tr><td>{name}</td><td>{r['before']['max_heading_step_deg']:.2f}°</td><td>{r['after']['max_heading_step_deg']:.2f}°</td></tr>" for name,r in pairs)
    style='''*{box-sizing:border-box}body{margin:0;background:#f5f7fb;color:#25324a;font-family:system-ui,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;line-height:1.8}main{max-width:1160px;margin:auto;padding:32px 24px 64px}header{padding:28px 0}h1{font-size:32px;line-height:1.4}h2{font-size:25px;margin:0 0 16px}h3{font-size:19px}p{margin:12px 0}section{background:white;border:1px solid #e2e8f0;border-radius:16px;padding:26px;margin:22px 0}a{color:#1861b8;text-underline-offset:3px}img,video{display:block;width:100%;height:auto;border-radius:10px;background:#edf1f7}figure{margin:18px 0}figcaption,.muted{color:#60718a;font-size:14px}table{border-collapse:collapse;width:100%;margin:16px 0;font-size:15px}td,th{padding:10px 12px;border-bottom:1px solid #e4e9f1;text-align:left}th{background:#f0f4fa}.note{border-left:4px solid #e7a436;padding:12px 16px;background:#fff7e7}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}.card{background:#eaf1fb;padding:16px;border-radius:12px}.card strong{display:block;font-size:26px}.steps{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}button,.pill{display:inline-block;border:1px solid #b9cee8;background:#edf4fe;border-radius:8px;padding:8px 12px;color:#1e548e;cursor:pointer;font:inherit}.pill{cursor:default;font-size:14px}.links{display:flex;gap:18px;flex-wrap:wrap}ul,ol{padding-left:25px}@media(max-width:640px){main{padding:16px 12px}section{padding:18px}h1{font-size:25px}.cards{grid-template-columns:1fr}table{font-size:13px;display:block;overflow:auto}td,th{padding:8px}}@media print{video,button{display:none}section{break-inside:avoid}body{background:white}main{max-width:none;padding:0}}'''
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nexus × 3DGS：已有结果与观看说明</title><style>{style}</style></head><body><main>
<header><span class="pill">已有成果 · v3 保留版本</span><h1>Nexus × 3DGS<br>已有结果与观看说明</h1><p>{intro}</p><p class="muted">快照时间：{stamp}。v4 全场景运动优化正在独立验证，尚未替换本页视频。</p><div class="cards"><div class="card"><strong>2 种</strong>给定巴士运动条件</div><div class="card"><strong>23 个</strong>场景车辆参与者（含自车）</div><div class="card"><strong>80 帧 / 条件</strong>实际渲染车框检查，0组重叠</div></div><p class="links"><a href="#interaction">先看交互</a><a href="#bus">看车顶对照</a><a href="#g7">看异常急转修复</a><a href="#limits">看当前限制</a><a href="nexus_3dgs_progress.md">Markdown版本</a></p></header>
<section id="interaction"><h2>1. 先看左图的运动关系，再看右图的画质</h2><video id="interactionVideo" controls preload="metadata" poster="{A}/interaction_preview.png"><source src="{A}/interaction.mp4" type="video/mp4"></video><div class="steps"><button data-seek="0">从“巴士停车”开始</button><button data-seek="8">从“巴士行进”开始</button><button id="slow">0.5倍速 / 正常速度</button></div><p>完整视频约16秒：前约8秒为巴士停车，后约8秒为巴士沿记录轨迹行进。每一段从自己的0秒重新计时。两侧来自同一组实际渲染位姿。</p><div class="links"><a href="{A}/bus_stop.mp4">单独播放停车条件</a><a href="{A}/bus_go.mp4">单独播放行进条件</a><a href="{A}/interaction_preview.png">查看示意图大图</a></div><table><thead><tr><th>颜色</th><th>角色</th><th>应该怎么看</th></tr></thead><tbody>{legend_html}</tbody></table><ol><li>先看橙色巴士的条件，再看红色自车、蓝色预测车的相对位置。场景中含停车和低速车辆，不需要期待每辆车都大幅移动。</li><li>车内白箭头指车头；细线只表示过去2秒；浅蓝区域表示前相机视野方向。</li><li>图上方是起始自车的前方，单位为米。它是车框示意，不能当作精确车道地图。图中的车可能在相机画面外，或被巴士遮挡。</li></ol><p class="note">蓝色车辆来自 Nexus 预测并经过离线运动/碰撞约束；橙色巴士是给定条件，灰色车辆使用记录轨迹。没有更新 Nexus 模型权重。对向车7在4秒后没有有效预测，不外推、不补造轨迹。</p></section>
<section id="bus"><h2>2. 巴士车顶：看局部轮廓外的残留</h2><video controls preload="metadata" poster="{A}/bus_comparison_preview.png"><source src="{A}/bus_comparison.mp4" type="video/mp4"></video><p><strong>左侧旧模型，右侧车顶清理后。</strong>上排看完整画面，下排看同一个车顶区域的放大图。两边使用相同轨迹与相机，重点看车顶上方多余纹理、轮廓外薄雾及车身文字。放大图仅裁切和缩放原像素。</p><p>已移除159个缺少观测支撑的车顶高斯，收紧1,249个车顶边缘高斯。三个检查时刻的局部轮廓外平均透明度下降约84.9%；该数值不代表全视频重影改善比例。</p><p>记录相机整图 PSNR：{quality['old']['psnr']:.2f} → {quality['new']['psnr']:.2f} dB，面包车和右侧车辆检查区域指标保持一致。这说明局部清理保留了已观测视角的外观，生成新视角仍需单独检查。</p><a href="{A}/bus_comparison_preview.png">查看前后对照大图</a></section>
<section id="g7"><h2>3. 对向车7：看末尾有没有突然反向</h2><figure><a href="{A}/g7_heading.png"><img src="{A}/g7_heading.png" alt="对向车7朝向曲线，红色修复前，蓝色修复后"></a><figcaption>在这张曲线图中，红线为修复前，蓝线为修复后。重点看3.5～4秒的突跳。</figcaption></figure><p>对向车车头朝下是正常的；修复针对低速时不合理的急转和侧移。角度曲线的数值不用直接对照俯视图的上下方向。</p><table><thead><tr><th>给定条件</th><th>修复前最大相邻朝向变化</th><th>修复后</th></tr></thead><tbody>{g7_html}</tbody></table><p>表中衡量的是相邻样本之间的突跳，并非全程累计转角。只调整异常自由预测尾段，历史、给定巴士与有效时长均保留。</p></section>
<section><h2>4. 已有检查结果应该怎样理解</h2><p>两种条件各检查80个实际渲染时刻，覆盖全场景23个车辆参与者，均为<strong>0组二维车框重叠</strong>。停车/行进条件的最小车框分离指标约为{stop['minimum_sat_separation_m']:.3f}/{go['minimum_sat_separation_m']:.3f}米；它不能直接当作真实保险杠最短距离。v3的18项基础检查与13项扩展检查通过。</p><p>这些检查用于确认身份、时间、坐标和采样车框关系。它们不替代连续时间或闭环安全验证，也没有覆盖全部行人、自行车及静态障碍。画面重影、阴影和纹理仍需单独观察。</p><p class="links"><a href="{A}/v3_verification.json">版本验证记录</a><a href="{A}/stop_clearance.json">停车车框检查</a><a href="{A}/go_clearance.json">行进车框检查</a><a href="{A}/g7_verification.json">车7检查</a></p></section>
<section id="limits"><h2>5. 当前仍未解决：自车移动后的全场景重影</h2><p class="note">自车静止时相对清楚，移动后道路、建筑、车辆和天空仍可能模糊或重影。上述视频展示已跑通的功能和局部改善，不能作为全场景画质已经完成的证明。</p><p>相机移动后会暴露原观测中不明显的几何深度、遮挡和高斯混合问题。因此应检查整个移动过程，尤其是建筑边缘、车道线和车身轮廓。正在优化的 v4 方案保留在独立目录，失败或未验收模型不会替换这里的保留版本。</p></section>
<footer class="muted"><p>视频、截图和审计文件均位于本报告的 assets/nexus_3dgs 目录。保留整个 report 文件夹即可离线查看。<a href="nexus_3dgs_report_manifest.json">素材来源及文件哈希</a></p></footer>
</main><script>const v=document.getElementById('interactionVideo');document.querySelectorAll('[data-seek]').forEach(b=>b.addEventListener('click',()=>{{v.currentTime=Number(b.dataset.seek);v.play().catch(()=>{{}})}}));document.getElementById('slow').addEventListener('click',()=>{{v.playbackRate=v.playbackRate===1?.5:1}});</script></body></html>'''
    (out/'nexus_3dgs_progress.html').write_text(page)
    (out/'README.md').write_text('# 已有成果报告\n\n优先用浏览器打开 [Nexus × 3DGS 成果与观看说明](nexus_3dgs_progress.html)，可以直接播放视频。\n\n也可以阅读 [Markdown 版本](nexus_3dgs_progress.md)。视频、截图和审计文件在 assets/nexus_3dgs 中。\n')
    result={'status':'complete','snapshot_utc':stamp,'primary_version':'optimization_v3/original_roof/roof_m65','html':'nexus_3dgs_progress.html','markdown':'nexus_3dgs_progress.md','all_media_copied_not_symlinked':True,'artifacts':manifest,'scope':'existing verified functionality and local improvements; global moving-ego rendering issue remains explicitly disclosed'}
    (out/'nexus_3dgs_report_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'complete','html':str(out/'nexus_3dgs_progress.html'),'markdown':str(out/'nexus_3dgs_progress.md'),'assets':len(manifest),'copied_media_mb':round(sum(x['bytes'] for x in manifest)/1024**2,2)},ensure_ascii=False))


if __name__=='__main__':main()
