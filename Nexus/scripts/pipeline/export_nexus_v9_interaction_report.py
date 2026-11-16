"""Build a self-contained local delivery after all five authorized steps finish."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.pipeline.update_nexus_v9_presentation import EXPLANATION


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--review',type=Path,help='Presentation directory; defaults to the original review')
    a=p.parse_args();review_dir=a.review or a.experiment/'review'
    frozen=json.loads((a.experiment/'frozen_inputs.json').read_text());analysis=json.loads((a.experiment/'paired_analysis.json').read_text())
    paired=json.loads((a.experiment/'raw_pair/run.json').read_text());review=json.loads((review_dir/'verification.json').read_text())
    input_audit=json.loads((a.experiment/'raw_pair_input_audit.json').read_text());assert input_audit['status']=='passed'
    pilot=a.experiment/'new_scene_pilot';plan=json.loads((pilot/'frozen_scenes.json').read_text());run=json.loads((pilot/'run.json').read_text())
    assert paired['status']=='complete' and review['status']=='passed' and run['status']=='complete'
    assert len(run['scenes'])==3 and all(x['status']=='complete' for x in run['scenes'].values())
    original_checks={}
    for key in ('visual_checkpoint','nexus_checkpoint','baseline_manifest','baseline_trajectories','asset_registration','registered_scene','original_scene','source_raw_features'):
        record=frozen['inputs'][key];original_checks[key]=sha(Path(record['path']))==record['sha256'];assert original_checks[key],key
    a.report.mkdir(parents=True,exist_ok=True);stem='nexus_3dgs_interaction_v9';assets=a.report/'assets'/stem;assets.mkdir(parents=True,exist_ok=False)
    sources={
        'paired_interaction.mp4':review_dir/'paired_interaction.mp4',
        'bus_stop_explained.mp4':review_dir/'bus_stop_explained.mp4',
        'bus_follow_logged_explained.mp4':review_dir/'bus_follow_logged_explained.mp4',
        'bus_stop_raw.mp4':a.experiment/'renders/bus_stop/camera_0.mp4',
        'bus_follow_logged_raw.mp4':a.experiment/'renders/bus_follow_logged/camera_0.mp4',
        'frozen_inputs.json':a.experiment/'frozen_inputs.json','paired_analysis.json':a.experiment/'paired_analysis.json',
        'paired_run.json':a.experiment/'raw_pair/run.json','raw_pair_input_audit.json':a.experiment/'raw_pair_input_audit.json',
        'frozen_scenes.json':pilot/'frozen_scenes.json','pilot_run.json':pilot/'run.json'}
    if (pilot/'qualitative_review.json').exists():sources['qualitative_review.json']=pilot/'qualitative_review.json'
    if (pilot/'execution_interruption.json').exists():sources['execution_interruption.json']=pilot/'execution_interruption.json'
    for image in review_dir.glob('pair_*.jpg'):sources[image.name]=image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname=str(ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'))
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    times=np.arange(1,17)*.5
    for name,label,color in [('bus_stop','公交停车','#de8c17'),('bus_follow_logged','公交按记录行驶','#287bcd')]:
        axes[0].plot(times,analysis['variants'][name]['ego_speed_mps'],label=label,color=color)
    axes[0].set_ylabel('自车位置差分速度 (m/s)',fontproperties=font);axes[0].legend(prop=font)
    with np.load(a.experiment/'raw_pair/bus_stop/trajectories.npz') as data:left=data['predicted_poses_world'][5:,0,:2,3]
    with np.load(a.experiment/'raw_pair/bus_follow_logged/trajectories.npz') as data:right=data['predicted_poses_world'][5:,0,:2,3]
    axes[1].plot(times,np.linalg.norm(left-right,axis=-1),color='#dc4545');axes[1].set_ylabel('两条件自车位置响应差异 (m)',fontproperties=font)
    for ax in axes:ax.set_xlabel('未来时间 (s)',fontproperties=font);ax.grid(alpha=.25)
    plot=review_dir/'response_metrics.png';fig.savefig(plot,dpi=180);plt.close(fig);sources[plot.name]=plot
    summaries=[]
    for row in plan['scenarios']:
        key=str(row['index']);entry=run['scenes'][key];evaluation=Path(entry['evaluation']);validation=json.loads((evaluation/'validation.json').read_text())
        protection=json.loads((Path(entry['checkpoint']).parent/'pose_protection.json').read_text())
        assert protection['status']=='passed' and validation['checkpoint_unchanged'] and validation['temporary_actor_poses_restored']
        assert sha(Path(entry['checkpoint']))==entry['checkpoint_sha256']
        coverage=Path(entry['work'])/'coverage.json';assert coverage.exists()
        coverage_record=json.loads(coverage.read_text())
        for name in ('recorded.mp4','nexus.mp4','comparison.mp4','validation.json','comparison_020.jpg','comparison_070.jpg','comparison_099.jpg'):
            sources[f'{key}_{name}']=evaluation/name
        sources[f'{key}_coverage.json']=coverage
        sources[f'{key}_pose_protection.json']=Path(entry['checkpoint']).parent/'pose_protection.json'
        summaries.append({'index':row['index'],'category':row['pilot_category'],'city':row['city'],'token':row['token'],
                          'mean_recorded_psnr_db':validation['metrics']['all']['mean_recorded_psnr_db'],
                          'rgb_gradient_holdout_psnr_db':validation['metrics']['rgb_gradient_holdout']['mean_recorded_psnr_db'],
                          'rendered_assets':validation['rendered_asset_count'],'bound_prediction_actors':validation['bound_prediction_actors'],
                          'unbound_prediction_actors':len(validation['unbound_prediction_tokens']),
                          'tracks_with_zero_geometric_camera_center_observations':[t['track_token'] for t in coverage_record['tracks'] if t['geometric_camera_observations']==0]})
    hashes={}
    for name,source in sources.items():
        destination=assets/name;shutil.copy2(source,destination);hashes[name]=sha(destination);assert hashes[name]==sha(source)
    summary={'status':'complete','steps_1_to_5_complete':True,'original_inputs_unchanged':original_checks,
             'paired_response_ego':next(row for row in paired['actors'] if row['slot']==0),
             'raw_pair_geometry':analysis['variants'],'actual_render_geometry':analysis['render_geometry'],'new_scenes':summaries,
             'main_conclusion':'Conditioning the bus changes the model response, but both raw variants overlap the bus. This does not demonstrate safe yielding.',
             'pilot_scope':'Three independently reconstructed scenes from existing evaluated data; engineering validation, no unseen-trip or v9-level visual acceptance claim.'}
    summary['execution_code_sha256']={name:sha(ROOT/name) for name in (
        'scripts/pipeline/diagnose_nexus_interaction.py','scripts/pipeline/render_nexus_registered_prediction.py',
        'scripts/pipeline/analyze_nexus_v9_pair.py','scripts/pipeline/review_nexus_v9_pair.py',
        'scripts/pipeline/select_nexus_3dgs_pilot.py','scripts/pipeline/prepare_nuplan_3dgs.py',
        'scripts/pipeline/train_nexus_3dgs_pilot.py','scripts/pipeline/evaluate_nexus_3dgs_pilot.py',
        'scripts/pipeline/audit_nexus_pilot_coverage.py','nexus_3dgs/runtime.py','nexus_3dgs/replay.py')}
    summary['preserved_failed_attempts']=[str(path) for path in pilot.glob('*/initial_5000.log') if 'NaN detected' in path.read_text(errors='replace')]
    (assets/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');hashes['summary.json']=sha(assets/'summary.json')
    rel='assets/'+stem+'/'
    link=lambda name,label:f'<a href="{rel+name}">{html.escape(label)}</a>'
    video=lambda name:f'<video controls preload="metadata" src="{rel+name}" style="width:100%;max-height:750px"></video>'
    stop=analysis['variants']['bus_stop'];go=analysis['variants']['bus_follow_logged'];ego=summary['paired_response_ego']
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nexus + 3DGS：v9 配对交互与三场景验证</title>',
           '<style>body{max-width:1500px;margin:auto;padding:24px;font:18px/1.7 system-ui;color:#243248;background:#f8fafc}table{border-collapse:collapse;width:100%}th,td{padding:10px;border:1px solid #ccd5df}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%}a{color:#175b9d}.finding{background:#fff2dc;padding:20px}</style>',
           '<h1>Nexus + 3DGS：v9 配对交互与三场景验证</h1>',
           '<p>已按顺序完成输入冻结、配对推理、原始响应分析、完整可视化、三个新场景独立重建与检查。没有训练 Nexus，也没有重跑已完成的定量批次。</p>',
           '<p class="finding"><b>核心结果：</b>改变公交未来条件会改变自车预测，但两组原始输出均出现自车与公交车框重叠。当前结果支持“存在条件响应”，不能据此称模型学会安全让行。</p>',
           '<section><h2>公交停车／行驶：同历史、同噪声的原始响应</h2>',
           '<p>橙色 B1 公交未来是给定条件；红色 E 自车及蓝色 P 车辆是模型预测；灰色 R 车辆使用记录位姿。两组均清空其他车辆的未来输入，保留相同的记录有效性 mask。没有车距过滤、轨迹稳定化或异常朝向修复。</p>',
           f'<p>自车末端响应差异 {ego["last_valid_response_distance_m"]:.3f} m；最大位置响应差异 {ego["max_response_distance_m"]:.3f} m。</p>',
           '<table><tr><th>原始预测指标</th><th>公交停车</th><th>公交按记录行驶</th></tr>',
           f'<tr><td>自车 8 秒路径长度</td><td>{stop["ego_future_path_length_m"]:.3f} m</td><td>{go["ego_future_path_length_m"]:.3f} m</td></tr>',
           f'<tr><td>自车与公交重叠的半秒未来采样</td><td>{stop["ego_bus_overlapping_future_samples"]}/16</td><td>{go["ego_bus_overlapping_future_samples"]}/16</td></tr>',
           f'<tr><td>自车—公交最小二维多边形距离</td><td>{stop["minimum_ego_bus_polygon_gap_m"]:.3f} m</td><td>{go["minimum_ego_bus_polygon_gap_m"]:.3f} m</td></tr></table>',
           '<p>以上使用记录车辆尺寸和自车后轴到中心 1.461 m 偏移。车框重叠是离散几何诊断，不是现实事故率，也不是闭环驾驶因果证明。</p>',
           f'<img src="{rel}response_metrics.png" alt="自车速度和两条件的位置响应差异">',
           video('paired_interaction.mp4'),
           '<p>全部 80 帧、10 FPS、8 秒；俯视图来自实际渲染临时位姿。车辆编号仅在左侧俯视图显示，右侧高斯画面不叠加标记。后段预测自车逼近／进入公交车身，会暴露未重建表面；原始视频保留这一结果。</p>',
           '<p>'+link('bus_stop_raw.mp4','公交停车：无图示原始渲染')+' · '+link('bus_follow_logged_raw.mp4','公交行驶：无图示原始渲染')+'</p>',
           '<p>'+link('bus_stop_explained.mp4','停车条件：单栏说明视频')+' · '+link('bus_follow_logged_explained.mp4','行驶条件：单栏说明视频')+'</p>',
           f'<img src="{rel}pair_050.jpg" alt="两条件在约五秒处的俯视与前视对应图">',
           EXPLANATION,
           '<h3>实际渲染车身检查</h3><table><tr><th>条件</th><th>首次自车—公交重叠</th><th>有任意车辆重叠的帧数</th><th>车辆对×帧重叠计数</th></tr>']
    for name,title in [('bus_stop','停车'),('bus_follow_logged','按记录行驶')]:
        value=analysis['render_geometry'][name];parts.append(f'<tr><td>{title}</td><td>{value["first_ego_bus_overlap_s"]:.2f} s</td><td>{value["frames_with_any_overlap"]}/80</td><td>{value["overlapping_pair_frame_samples"]}</td></tr>')
    parts.extend(['</table><p>此表包括全部重建车辆及使用记录位姿的车辆，与上表 16 个原始半秒时刻的口径不同。所有重叠均保留，未通过隐藏车辆或改轨迹消除。</p></section>',
                  '<section><h2>三个新场景：分别重建与回放</h2><p>三个日志、两个城市，每场景 101 帧、四路相机、505 个真实传感器文件已完整核对；每个场景独立训练 5000 步 3DGS。保留原始标定，固定相机和车辆观测位姿，只优化场景资产。没有把原场景 v9 权重移植到新地点。</p>',
                  '<p>这些场景来自已完成的独立 256 场景批次。本次是接入流程的工程验证，不计为新的正式定量样本或未见行程泛化成绩。每第十个 RGB 时刻留出梯度训练，但 LiDAR 和初始化仍使用该场景，不能视为完整独立留出。</p>',
                  '<p>抽查的新场景画面仍有道路、建筑边缘、车辆表面和新视角伪影；本批没有将任何新场景提升为 v9 画质基线。完整视频和局部缺陷均保留。</p>',
                  '<table><tr><th>场景</th><th>城市</th><th>记录运动平均 PSNR</th><th>已绑定／缺失预测车辆资产</th><th>中心未进入所选相机的预测车辆</th></tr>'])
    labels={'turning':'转向路口','following':'跟车片段','intersection':'信号灯路口'}
    for s in summaries:parts.append(f'<tr><td>{s["index"]}：{labels[s["category"]]}</td><td>{html.escape(s["city"])}</td><td>{s["mean_recorded_psnr_db"]:.2f} dB</td><td>{s["bound_prediction_actors"]}／{s["unbound_prediction_actors"]}</td><td>{len(s["tracks_with_zero_geometric_camera_center_observations"])}</td></tr>')
    parts.append('</table><p>PSNR 只比较记录相机与真实照片。新视角没有配对照片，不计算预测视角 PSNR；初始重建完成不等于达到原场景 v9 的视觉验收水平。缺失的车辆资产明确列出，没有伪造外观补齐。部分车辆中心未投影进所选相机或只获得少量观测，即使能绑定 LiDAR 初始化的资产，也不能视为有完整 RGB 外观覆盖；相机中心投影本身亦不证明未遮挡。</p>')
    for s in summaries:
        k=str(s['index']);parts.extend([f'<h3>{k}：{labels[s["category"]]} · {s["token"]}</h3>',video(k+'_comparison.mp4'),'<p>'+link(k+'_coverage.json','LiDAR 与相机覆盖审计')+' · '+link(k+'_validation.json','完整逐帧验证')+'</p>'])
    parts.extend(['</section><section><h2>可复现输入与边界</h2><p>v9、v8、原始 Nexus 输出及原报告均保留。新的配对输出与 v8 历史派生展示轨迹分开保存。当前未加入后处理分支；所有安全诊断均针对实际保存的原始结果。</p>',
                  '<p>'+link('frozen_inputs.json','冻结输入与哈希')+' · '+link('paired_run.json','配对噪声与条件核对')+' · '+link('raw_pair_input_audit.json','保存输入的独立复核')+' · '+link('paired_analysis.json','原始响应与渲染几何')+' · '+link('frozen_scenes.json','训练前冻结的新场景清单')+' · '+link('summary.json','汇总 JSON')+'</p>',
                  '<p>所有交付视频均完整解码检查。行人／自行车未单独动画化；新视角、遮挡后显露面和反光仍有限制。后续若研究安全交互，应设计机制或约束对照，并继续区分模型响应与下游过滤效果。</p></section></html>'])
    page='\n'.join(parts);(a.report/(stem+'.html')).write_text(page)
    md=['# Nexus + 3DGS：v9 配对交互与三场景验证','',f'完整离线报告：[{stem}.html]({stem}.html)','',
        '步骤 1—5 已完成。改变公交未来条件会改变自车预测，但两组原始输出均与公交发生车框重叠，不能据此称模型学会安全让行。','',
        f'自车停车／行驶条件路径长度为 {stop["ego_future_path_length_m"]:.3f}／{go["ego_future_path_length_m"]:.3f} m，末端响应差异 {ego["last_valid_response_distance_m"]:.3f} m。',
        f'原始半秒未来采样的自车—公交重叠为 {stop["ego_bus_overlapping_future_samples"]}/16 和 {go["ego_bus_overlapping_future_samples"]}/16。','',
        f'[配对完整视频]({rel}paired_interaction.mp4)','',
        '三个新场景分别完成独立 5000 步 3DGS 重建，以及真实照片／记录运动／保存的 Nexus 目标条件预测的完整 80 帧对照。此为工程验证，不是新增未见行程泛化成绩，也未宣称达到原场景 v9 的视觉质量。','']
    for s in summaries:md.append(f'- {s["index"]} {labels[s["category"]]}：记录运动平均 PSNR {s["mean_recorded_psnr_db"]:.2f} dB；已绑定预测车辆 {s["bound_prediction_actors"]}，缺失资产 {s["unbound_prediction_actors"]}。')
    md.extend(['',f'实验根目录：`{a.experiment}`','',f'[完整数据汇总]({rel}summary.json)'])
    (a.report/(stem+'.md')).write_text('\n'.join(md)+'\n')
    (a.report/(stem+'_manifest.json')).write_text(json.dumps({'status':'complete','assets':hashes,'source_experiment':str(a.experiment)},indent=2)+'\n')
    progress={f'{i}_{name}':'complete' for i,name in enumerate(['freeze','pair','analysis','visualization','new_scenes'],1)}
    (a.experiment/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    print('V9_INTERACTION_REPORT_COMPLETE',a.report/(stem+'.html'),flush=True)


if __name__=='__main__':main()
