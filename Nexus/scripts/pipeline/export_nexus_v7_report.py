"""Export completed sensor-alignment and reconstruction review artifacts."""
import argparse, hashlib, html, json, shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['experiment','validation','review','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();v=a.experiment;root=Path(__file__).resolve().parents[2]
    review=json.loads((a.review/'review.json').read_text());quality=json.loads((a.validation/'validation.json').read_text())
    projection=json.loads((root/'artifacts/nexus_3dgs/projection_audit_v4/projection_audit.json').read_text())
    registration=root/'artifacts/nexus_3dgs/lidar_pose_refinement_full/lidar_registration.json'
    selected=Path(review['checkpoint']);scene=next((v/'registered_data').iterdir());assert review['status']=='complete'
    pose_interpolated=(selected.parent/'heldout_body_pose_interpolation.json').is_file()
    pose_policy='留出时刻直接对相邻有效的已拟合车体位置插值、对旋转做 SLERP；序列端点或缺少双方可见邻帧时保留原候选位姿' if pose_interpolated else '留出时刻只插值邻近训练时刻的位姿残差'
    asset_policy='最终候选又固定背景、相机与观测位姿，对车辆高斯的形状、透明度和外观进行了单独重训' if 'assets' in str(selected.parent.name) else '本候选的车辆形状、纹理和透明度保持固定'
    assets=a.output/'assets/nexus_3dgs_v7';assets.mkdir(parents=True,exist_ok=True);files={}
    for name in ['real_motion_comparison.mp4','nexus_prediction_comparison.mp4','real_motion_comparison_020.jpg','real_motion_comparison_070.jpg','real_motion_comparison_099.jpg','nexus_prediction_comparison_070.jpg','review.json']:
        if (a.review/name).is_file():files[name]=a.review/name
    files.update({'validation.json':a.validation/'validation.json','projection_audit.json':root/'artifacts/nexus_3dgs/projection_audit_v4/projection_audit.json','lidar_registration.json':registration,'geometry_verification.json':scene/'geometry_verification.json','seed_audit.json':v/'static_seed/seed_audit.json','selected_config.yaml':selected.parent/'config.yaml'})
    if (v/'vehicle_retrain_comparison.json').is_file():files['vehicle_retrain_comparison.json']=v/'vehicle_retrain_comparison.json'
    for dest,source in [('vehicle_preserved_real.mp4','real_motion_comparison.mp4'),('vehicle_preserved_nexus.mp4','nexus_prediction_comparison.mp4'),('vehicle_preserved_070.jpg','nexus_prediction_comparison_070.jpg')]:
        if (v/'review'/source).is_file():files[dest]=v/'review'/source
    for name in ['registered_static_audit.json','rgb_refinement_audit.json','vehicle_refit_audit.json','heldout_body_pose_interpolation.json','source_training_audit.json']:
        if (selected.parent/name).is_file():files[name]=selected.parent/name
    manifest=[]
    for name,source in files.items():
        target=assets/name;shutil.copyfile(source,target);digest=hashlib.sha256(target.read_bytes()).hexdigest();assert digest==hashlib.sha256(source.read_bytes()).hexdigest();manifest.append({'file':str(target.relative_to(a.output)),'sha256':digest,'bytes':target.stat().st_size})
    A='assets/nexus_3dgs_v7';summary=review['summary'];passed=review['candidate_passed_numeric_review_gate']
    status='通过当前数值门槛，仍需结合视频检查局部缺陷' if passed else '尚未通过整体替换门槛：有改善，也有退步；保留为候选实验'
    table=[];htmlrows=[]
    for key,label in [('stationary','基本静止 43 帧'),('moving','移动／转动 37 帧'),('moving_rgb_validation','运动段留出验证 3 帧'),('all','全部 80 帧')]:
        x=summary[key];old=x['old_native'];new=x['new'];values=[old['full_psnr'],new['full_psnr'],old['static_psnr'],new['static_psnr']]
        table.append('| '+label+' | '+' | '.join(f'{z:.2f}' for z in values)+' |');htmlrows.append('<tr><td>'+label+'</td>'+''.join(f'<td>{z:.2f}</td>' for z in values)+'</tr>')
    pair=next(x for x in projection['rows'] if x['frames']==[20,70]);rawerror=pair['errors']['factory']['evaluation']['median_px'];corrected=pair['errors']['independent_lidar_registration']['evaluation']['median_px']
    md=f'''# v7：运动重影的对齐检查与重建实验

**当前状态：{status}。** 这份报告同时展示改善和剩余问题。旧 v5 模型仍保留；没有将候选结果冒充已经彻底修好的版本。

[真实运动完整对照视频]({A}/real_motion_comparison.mp4)

## 怎么看

左栏是真实照片组成的录像，中栏是原 v5，右栏是本轮候选。共 8 秒，保留第 20～99 帧全部图像。先看开头的清晰度，再在 **5 秒（第 70 帧）**、**7.9 秒（第 99 帧）**暂停，对照路面箭头、白色建筑边缘、树冠、公交车轮廓和右侧车辆。没有补帧、锐化或时间平滑。

![第 70 帧，完整三栏对照]({A}/real_motion_comparison_070.jpg)

## 实际查明了什么

1. 先前的控制实验已发现：同一个旧 3DGS，即使使用真实运动轨迹，移动后也会变差，因此不能仅归因于 Nexus。
2. 本轮用真实静态图像特征和真实点云分别检查，发现当前跨帧位姿与静态观测不完全一致。第 20→70 帧，留出的静态特征投影误差中位数由 **{rawerror:.2f} 像素降到 {corrected:.2f} 像素**。独立点云配准在第 70 帧得到约 0.94° 的旋转校正。
3. 只替换旧模型的相机，在部分帧改善、部分帧反而退步。旧高斯已经吸收了原先的误差，所以不能直接把新位姿硬套上去作为完整修复。
4. 已建立单独的校正输入，并完成新的静态场景训练及遮挡检查。过宽的车辆排除区域曾导致车旁建筑缺少训练约束，因此又改用固定车辆与背景共同渲染来补全可见区域。

这些检查说明“投影与重建的一致性需要处理”，没有据此把问题直接定性为 nuPlan 原始标定错误，也没有证明 Nexus 轨迹先天无法清晰渲染。

## 完整结果，而非只选改善帧

PSNR 单位 dB，越高表示与对应真实照片越接近。车辆排除区域仅用于分析背景；**整图指标与视频同样重要**。

| 分组 | 旧 v5 整图 | 候选整图 | 旧 v5 车辆排除区域 | 候选车辆排除区域 |
| --- | --- | --- | --- | --- |
{chr(10).join(table)}

静止／运动分组沿用前次真实运动实验：相对首帧位移超过 0.1 米或转动超过 0.2° 为运动。运动段留出验证帧为 70、80、90。数值门槛要求运动段整图均值提高，且静止段整图均值下降不超过 1 dB；这仍不能替代局部视觉检查。

![第 20 帧，用于检查静止画质有没有下降]({A}/real_motion_comparison_020.jpg)

![第 99 帧，用于检查运动后段]({A}/real_motion_comparison_099.jpg)

## 训练和验证范围

- 新输入为 131 帧、四路相机，旧 v5 为 101 帧、三路相机。本报告是两条完整流程的比较，不能将全部差异都归功于单个参数修改。
- 本次对照视频和画质分数仅检查前向相机、同一场景的 80 帧；尚未据此验收所有相机或其他场景。
- 新背景初始化与梯度训练排除了每第 10 个时刻的 RGB，验证相机的学习残差如存在则从相邻训练时刻插值。点云配准使用所有时刻的 LiDAR，包含 RGB 验证时刻。因此这是 RGB 留出验证，不是全传感器留出测试。
- 车辆资产继承自旧模型，旧模型见过原始完整场景。整图分数应视为重建保真度检查，不是严格的未见场景泛化成绩。
- 车辆观测位姿另做小幅拟合，训练时平移限制为 25 厘米、旋转限制为 3°；{pose_policy}。这是标注与资产的对齐，不是修改 Nexus 的预测轨迹。{asset_policy}。
- 原始 RGB、点云、内外参和拍摄时间戳没有改写。131 帧中 1,759 组车体与点云相对位姿检查通过，最大矩阵误差约 2.33e-10。
- 本轮没有训练 Nexus。其保存的原始预测不因场景重建而改变。

## 文件在哪里

- 当前候选权重：`{selected}`
- 本轮工作目录：`{v.resolve()}`
- 完整真实轨迹评估：`{a.validation.resolve()}`
- 主视频：`{(a.review/'real_motion_comparison.mp4').resolve()}`
- 原有模型、失败分支和新候选分开保留。单独车辆调色、相机残差细调等分支不自动作为最终版本。

本轮分支中，`static_registered_15000` 单独训练时仍有车旁背景缺少约束的问题；`rgb_residuals_7000` 相机残差方案在部分验证帧退步；`vehicles_appearance_4000` 调色收益较小。最终候选采用固定校正相机的背景共同渲染训练，并单独拟合真实车体位姿。中间分支都保留作对照。

## 车辆重训仍有取舍

车辆高斯完整重训后，部分运动帧的整图分数继续提高，但公交车尾部文字更软，预测视角的建筑也仍有局部拉花。因此没有采用“分数提高就替换旧版”的做法。

[保留旧车辆高斯的真实轨迹对照]({A}/vehicle_preserved_real.mp4) · [保留旧车辆高斯的 Nexus 对照]({A}/vehicle_preserved_nexus.mp4)

这两段对照保留原来的车辆形状与纹理，只使用新的背景与真实车体位姿拟合，便于单独比较车辆重训造成的变化。后续需优先检查动态对象在各相机曝光时刻的位姿同步，以及独立的多视图视觉／点云联合标定；目前尚未完成全部重影的修复。

[逐帧分数与实际相机位姿]({A}/validation.json) · [投影检查]({A}/projection_audit.json) · [点云配准]({A}/lidar_registration.json) · [几何保护检查]({A}/geometry_verification.json)
'''
    if (a.review/'nexus_prediction_comparison.mp4').is_file():
        md+=f'''\n## 同一 Nexus 预测重新回放\n\n[Nexus 预测对照视频]({A}/nexus_prediction_comparison.mp4)\n\n中、右两栏读取同一个已保存的原始意图条件预测，未加入后续离线车距或交互修改。使用共同的地面与车辆资产挂载；新版本的高度、侧倾和俯仰来自校正后的记录数据。左侧真实录像仅供理解场景，预测视角与它不同，因此不计算预测视角 PSNR。\n'''
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'nexus_3dgs_v7.md').write_text(md)
    nexus_section=f'<section><h2>同一 Nexus 原始预测</h2><video controls preload="metadata"><source src="{A}/nexus_prediction_comparison.mp4" type="video/mp4"></video><p>中、右读取同一份原始预测。左侧真实录像仅供参考；预测视角没有配对真值，不计算 PSNR。</p><h3>车辆重训仍有取舍</h3><p>重训车辆后，一些运动帧的整图分数提高，但公交车尾部文字更软，预测视角的建筑也仍有局部拉花。因此没有替换旧版，不能宣布全部重影已经修复。</p><p>下面保留旧车辆高斯的对照，便于比较车辆重训造成的变化：</p><video controls preload="metadata"><source src="{A}/vehicle_preserved_nexus.mp4" type="video/mp4"></video><p><a href="{A}/vehicle_preserved_real.mp4">保留旧车辆高斯的真实轨迹对照</a></p><p>后续需优先检查动态对象的曝光时刻同步及独立多视图视觉／点云联合标定。</p></section>' if (a.review/'nexus_prediction_comparison.mp4').is_file() else ''
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v7 运动重影检查与重建</title><style>body{{font:17px/1.8 system-ui,sans-serif;max-width:1450px;margin:25px auto;padding:0 20px;background:#f3f6fa;color:#142136}}section{{background:white;border-radius:12px;margin:20px 0;padding:24px}}video,img{{width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.status{{background:#fff0d2;padding:18px}}button{{padding:8px;margin:6px}}</style><h1>v7：运动重影检查与重建</h1><p class="status"><b>{status}。</b>旧 v5 保留；请同时检查静止段与运动段。</p><section><h2>先看完整 8 秒对照</h2><p>左：真实录像；中：旧 v5；右：本轮候选。先看开头，再看 5 秒和 7.9 秒，比较建筑、树冠、箭头、公交车和右侧车辆。保留全部 80 帧，没有锐化、补帧或时间平滑。</p><video id="main" controls preload="metadata" poster="{A}/real_motion_comparison_070.jpg"><source src="{A}/real_motion_comparison.mp4" type="video/mp4"></video><button onclick="jump(0)">开头</button><button onclick="jump(5)">5 秒</button><button onclick="jump(7.9)">最后一帧</button><button onclick="document.getElementById('main').playbackRate=.5">半速</button></section><section><h2>查明的原因与边界</h2><p>真实轨迹下也有重影；静态图像特征与点云独立检查均发现位姿一致性误差。第 20→70 帧的静态特征投影误差由 {rawerror:.2f} 降到 {corrected:.2f} 像素。仅改旧模型相机会在后段退步，因此又建立独立校正输入重新重建。</p><p>过宽的车辆排除区域还会让车旁建筑缺少训练约束；新实验改用固定车辆与背景共同渲染。整图与局部仍需一起验收，不能仅凭背景分数宣称修复。</p></section><section><h2>全部帧的量化检查</h2><table><tr><th>分组</th><th>旧版整图</th><th>候选整图</th><th>旧版车辆排除区域</th><th>候选车辆排除区域</th></tr>{''.join(htmlrows)}</table><p>单位 dB，越高越接近记录照片。新背景使用 RGB 留出验证，LiDAR 配准使用了验证时刻的点云；车辆继承旧模型，因此整图是重建保真度检查，不能当成严格未见场景泛化分数。</p><p>新流程为 131 帧四路相机，旧模型为 101 帧三路相机；完整流程差异不能都归因于某一个修复。</p></section>{nexus_section}<section><h2>静止与运动后段也要看</h2><img src="{A}/real_motion_comparison_020.jpg" alt="第20帧"><img src="{A}/real_motion_comparison_099.jpg" alt="第99帧"></section><section><h2>文件与详细解释</h2><p><a href="nexus_3dgs_v7.md">完整文字报告与路径</a> · <a href="{A}/validation.json">逐帧分数</a> · <a href="{A}/projection_audit.json">独立投影检查</a> · <a href="{A}/geometry_verification.json">原数据与相对位姿保护检查</a></p><p>原始数据与 Nexus 没有重训或改写。候选权重：<code>{html.escape(str(selected))}</code></p></section><script>function jump(t){{let v=document.getElementById('main');v.currentTime=t;v.pause();}}</script></html>'''
    (a.output/'nexus_3dgs_v7.html').write_text(page)
    (a.output/'nexus_3dgs_v7_manifest.json').write_text(json.dumps({'status':'complete','candidate_status':status,'selected_checkpoint':str(selected),'selected_checkpoint_sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),'review':review,'assets':manifest},indent=2)+'\n');print('V7_REPORT_COMPLETE '+str(a.output.resolve()),flush=True)


if __name__=='__main__':main()
