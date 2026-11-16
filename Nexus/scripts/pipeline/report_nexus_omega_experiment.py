"""Build an auditable offline research report and fixed-seed v9 comparison."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = Path('/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/nexus_omega/matplotlib_cache'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded(path):
    import torch
    value = torch.load(path, map_location='cpu', weights_only=False)
    state = value['sampled_tensor'][0].numpy()*np.array([104,104,1,1,4,4,5,1.6])+np.array([0,0,0,0,0,0,4.5,2])
    gt = value['scene_tensor_gt'][0].numpy()*np.array([104,104,1,1,4,4,5,1.6])+np.array([0,0,0,0,0,0,4.5,2])
    return state, gt


def bus_separation(state, gt):
    # Independent two-box SAT on the same historical physical dimensions.
    yaw = np.arctan2(state[:2, :, 3], state[:2, :, 2])
    h = np.stack([np.cos(yaw), np.sin(yaw)], -1)
    center = state[:2, :, :2].copy()
    center[0] += 1.461*h[0]
    basis = np.stack([h,np.stack([-h[...,1],h[...,0]],-1)],-2)
    axes = np.concatenate([basis[0],basis[1]],-2)
    ri = (np.abs((basis[0,:,None]*axes[:,:,None]).sum(-1))*gt[0,4,6:8]/2).sum(-1)
    rj = (np.abs((basis[1,:,None]*axes[:,:,None]).sum(-1))*gt[1,4,6:8]/2).sum(-1)
    return (np.abs(((center[1]-center[0])[:,None]*axes).sum(-1))-ri-rj).max(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiments', type=Path, nargs='+', required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    assets = args.output/'assets'
    assets.mkdir()
    rows, entries, protocols = [], [], []
    initial_guidance_hash = None
    for experiment in args.experiments:
        run = json.loads((experiment/'run.json').read_text())
        protocol = json.loads((experiment/'protocol.json').read_text())
        assert run['status']=='complete' and run['model_parameters_unchanged']
        protocols.append(protocol)
        guidance_key = 'nuplan_extent/planning/training/modeling/models/diffusion/omega_guidance.py'
        if initial_guidance_hash is None:
            initial_guidance_hash = protocol['source_sha256'][guidance_key]
        for row in run['records']:
            row = dict(row)
            key = row['method']
            if key=='anchor':
                if protocol['config'].get('solver') == 'lbfgs':
                    revision = '_LBFGS'
                else:
                    revision = '_normAdam' if protocol['source_sha256'][guidance_key]!=initial_guidance_hash else ''
                key = f'anchor{revision}_KL{protocol["config"]["kl_budget"]:g}'
            row['label'] = key
            row['experiment'] = str(experiment.resolve())
            rows.append(row)
            entries.append((key,experiment,row))
    methods = list(dict.fromkeys(row['label'] for row in rows))
    aggregates = []
    for label in methods:
        group = [r for r in rows if r['label']==label and r['variant']=='event']
        aggregates.append({'label':label,
            'ego_bus_overlapping_samples':sum(r['metrics']['ego_bus_overlap_frames'] for r in group),
            'collision_free_event_seeds':sum(r['metrics']['physical_box_pair_frames_overlapping']==0 for r in group),
            'min_bus_separation_m':min(r['metrics']['minimum_ego_bus_sat_separation_m'] for r in group),
            'max_ego_acceleration_mps2':max(r['metrics']['ego_peak_acceleration_mps2'] for r in group),
            'all_vehicle_acceleration_violation_samples':sum(r['metrics']['all_agent_acceleration_over_5mps2_samples'] for r in group),
            'lane_proxy_violation_samples':sum(r['metrics']['lane_segment_proxy_over_2p5m_frames'] for r in group),
            'mean_sampling_seconds':float(np.mean([r['seconds'] for r in group]))})
    # Every method and condition uses the exact same starting noise per seed.
    for seed in {r['seed'] for r in rows}:
        assert len({r['noise_sha256'] for r in rows if r['seed']==seed}) == 1
    assert all(r['conditions_exact'] and r['ego_future_free'] for r in rows)
    baseline = [r for r in rows if r['method']=='baseline']
    assert all(r['saved_baseline_max_abs_difference']==0 for r in baseline)
    write_json = lambda path,value:path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    write_json(assets/'results.json',{'aggregates':aggregates,'records':rows,'protocols':protocols})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(15,8),sharex=True)
    colors = ['#334155','#3b82f6','#ea580c','#dc2626','#059669','#9333ea','#a16207','#0d9488']
    time = (np.arange(21)-4)*.5
    for index,label in enumerate(methods):
        if label in ('schedule','omega'):
            continue  # their failures remain in the full numeric table
        for seed in (0,1,2):
            _,experiment,row = next(e for e in entries if e[0]==label and e[2]['seed']==seed and e[2]['variant']=='event')
            state,gt = decoded(experiment/row['path']/'prediction.pt')
            direction = gt[0,4,2:4]
            progress = (state[0,:,:2]-gt[0,4,:2])@direction
            separation = bus_separation(state,gt)
            assert np.isclose(separation[5:].min(),row['metrics']['minimum_ego_bus_sat_separation_m'])
            axes[0,seed].plot(time[4:],progress[4:],label=label,color=colors[index%len(colors)],linewidth=2)
            axes[1,seed].plot(time[4:],separation[4:],label=label,color=colors[index%len(colors)],linewidth=2)
    for seed in (0,1,2):
        axes[0,seed].set_title(f'Fixed seed {seed}: event condition')
        axes[1,seed].axhline(0,color='#dc2626',linestyle='--')
        axes[1,seed].set_xlabel('Future time (s)')
        for axis in axes[:,seed]:
            axis.axvspan(3,6,color='#f59e0b',alpha=.12)
            axis.grid(alpha=.2)
        axes[0,seed].legend(fontsize=8)
    axes[0,0].set_ylabel('Ego longitudinal displacement (m)')
    axes[1,0].set_ylabel('Signed ego-bus box separation (m)')
    fig.suptitle('Nexus + independent OMEGA-inspired guidance | 1 development scene | fixed dimensions\nOrange interval: prescribed bus reverse approach. Negative separation: physical boxes overlap.')
    fig.tight_layout()
    fig.savefig(assets/'trajectory_comparison.png',dpi=160)
    plt.close(fig)

    candidate = args.candidate.resolve()
    sources = {'baseline':BRIDGE/'artifacts/nexus_3dgs/timed_event_20261026/renders',
               'candidate':candidate/'renders'}
    video_checks,render_metrics = {},{}
    if all((sources['candidate']/f'anchor_{n}'/'run.json').exists() for n in ('normal','event')):
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw,ImageFont
        fontpath = BRIDGE/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'
        font = ImageFont.truetype(str(fontpath),25)
        small = ImageFont.truetype(str(fontpath),21)
        cand_protocol = json.loads((candidate/'protocol.json').read_text())
        budget = cand_protocol['config']['kl_budget']
        for variant in ('normal','event'):
            paths = [sources['baseline']/variant,sources['candidate']/f'anchor_{variant}']
            runs = [json.loads((path/'run.json').read_text()) for path in paths]
            assert all(len(run['frames'])==80 for run in runs)
            checks = []
            for label,run in zip(('baseline','candidate'),runs):
                meta = [f['scene_metadata'] for f in run['frames']]
                render_metrics[label+'_'+variant] = {
                    'ego_bus_overlap_frames':sum(f['ego_bus_sat_separation_m']<0 for f in meta),
                    'min_ego_bus_separation_m':min(f['ego_bus_sat_separation_m'] for f in meta),
                    'all_actor_overlap_pair_frames':sum(len(f['overlapping_actor_pairs']) for f in meta),
                    'scope':'actual v9 body installation, 23 actors, 80 exposure samples; not continuous-time or full obstacle safety'}
            outfile = assets/f'{variant}_before_after.mp4'
            with imageio.get_writer(str(outfile),fps=10,macro_block_size=2,
                                   ffmpeg_params=['-threads','2','-crf','18']) as writer:
                for index in range(80):
                    canvas = Image.new('RGB',(1920,650),'#f8fafc')
                    draw = ImageDraw.Draw(canvas)
                    for col,(path,run) in enumerate(zip(paths,runs)):
                        frame = run['frames'][index]
                        source = np.asarray(Image.open(path/f'{frame["frame"]:03d}_0.png').convert('RGB'))
                        canvas.paste(Image.fromarray(source),(960*col,55))
                        label = '原 Nexus' if col==0 else f'采样中优化引导 KL={budget:g}'
                        draw.text((960*col+15,9),label,font=font,fill='#20344b')
                        gap = frame['scene_metadata']['ego_bus_sat_separation_m']
                        draw.text((960*col+15,603),f'自车—公交间隔 {gap:.2f} m；负值表示重叠',font=small,
                                  fill='#bb3344' if gap<0 else '#20344b')
                        assert np.array_equal(np.asarray(canvas)[55:595,960*col:960*(col+1)],source)
                    writer.append_data(np.asarray(canvas))
                    if index in (0,30,50,79):
                        canvas.save(assets/f'{variant}_{index:03d}.jpg',quality=93)
            with imageio.get_reader(outfile) as reader:
                info=reader.get_meta_data()
                count=sum(1 for _ in reader)
            assert count==80 and info['fps']==10
            video_checks[outfile.name] = {'decoded_frames':count,'fps':10,'source_pixels_exact_before_encoding':True}
        write_json(assets/'render_metrics.json',render_metrics)

    def table(items,cols):
        head=''.join('<th>'+html.escape(title)+'</th>' for _,title in cols)
        body=''
        for row in items:
            cells=[]
            for key,_ in cols:
                value=row[key]
                if isinstance(value,float):value=f'{value:.3f}'
                cells.append('<td>'+html.escape(str(value))+'</td>')
            body+='<tr>'+''.join(cells)+'</tr>'
        return '<table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table>'
    columns=[('label','方案'),('ego_bus_overlapping_samples','公交重叠采样点／48'),
        ('collision_free_event_seeds','无预测车重叠种子／3'),('min_bus_separation_m','最小公交间隔 m'),
        ('max_ego_acceleration_mps2','自车峰值加速度 m/s²'),
        ('all_vehicle_acceleration_violation_samples','全车加速度超限点'),
        ('lane_proxy_violation_samples','车道线段代理超限点'),('mean_sampling_seconds','平均采样秒')]
    data=[]
    for r in rows:
        data.append(dict(label=r['label'],seed=r['seed'],variant=r['variant'],**r['metrics']))
    detailcols=[('label','方案'),('seed','种子'),('variant','条件'),('ego_bus_overlap_frames','公交重叠／16'),
        ('physical_box_pair_frames_overlapping','全预测车重叠对×时刻'),
        ('minimum_ego_bus_sat_separation_m','公交最小间隔 m'),('ego_peak_acceleration_mps2','自车加速度 m/s²'),
        ('ego_peak_jerk_mps3','自车 jerk m/s³'),('lane_segment_proxy_over_2p5m_frames','车道代理超限点')]
    video_html=''.join(f'<h2>{"公交停止对照" if variant=="normal" else "公交倒车事件"}：固定种子 0</h2><video controls preload="metadata" src="assets/{variant}_before_after.mp4"></video>' for variant in ('normal','event') if f'{variant}_before_after.mp4' in video_checks)
    text='''这是一组基于 OMEGA 思想的独立工程试验。官方 Nexus 权重未更新；v9 资产保持原值。
先完成 baseline／anchor／schedule／omega 四组消融，再在同一个开发场景上增加 KL=20、200 的敏感性试验，验证归一化 Adam 和带线搜索的 L-BFGS 求解器。后者同时提高碰撞惩罚和迭代上限，因此该比较属于工程候选比较，不是单一因素消融。
这组开发结果用于判断机制与参数，不能当作独立泛化验证，也不是 OMEGA 官方指标复现。'''
    limitations='''公交只在第 3～6 秒受完整状态条件约束，其他时段、自车及其余车辆未来自由。全部方法共享历史、地图、存在 mask 与相同初始噪声。
存在 mask 沿用记录中的未来信息；整段条件生成可能提前响应，不能视为在线因果闭环。原 Nexus 六次输出与已有预测逐值相同。
原生统计只覆盖预测车辆和 16 个未来采样点；渲染统计另按 23 辆车与 80 个真实曝光时刻检查。
车道线段距离是代理，不能代替可行驶区检查。完整自行车模型、路线 DFS、SE-IBR、行人与自行车动画均未实现。
原生 map→sma→clip 保留，因此当前试验也检验了新优化与既有约束的组合；两阶段分支出现退步，不能据此否定原论文。
采样耗时包含 Python 工程开销，未做严格效率基准。给定公交条件精确保留，输出后没有轨迹修复；优化发生在采样内部。'''
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Nexus 优化引导试验</title>
<style>body{{font:17px/1.7 system-ui,sans-serif;max-width:1500px;margin:36px auto;padding:0 24px;color:#20344b}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}}td,th{{padding:9px;border-bottom:1px solid #dce3eb;text-align:left}}th{{background:#eef3f8}}video,img{{width:100%;border-radius:8px}}.note{{background:#fff4dc;padding:16px;border-left:5px solid #dd9222}}p{{white-space:pre-line}}</style>
<h1>Nexus：OMEGA 思想接入与效果验证</h1><p>{html.escape(text)}</p>
<p class="note">数值变好与完整验收分开判断。请同时查看剩余碰撞、加速度、道路代理及固定种子视频；所有失败样本均保留。</p>
<p>参考：<a href="https://arxiv.org/html/2512.07661v4">OMEGA 论文 v4</a> · <a href="https://opendrivelab.com/OMEGA/">官方项目页</a> · <a href="assets/results.json">全部数值与协议</a></p>
<h2>三个事件种子的汇总</h2>{table(aggregates,columns)}
<img src="assets/trajectory_comparison.png" alt="所有固定种子的自车位移与公交间隔">{video_html}
<h2>逐种子结果（含正常对照和失败调度）</h2>{table(data,detailcols)}
<h2>评估范围与实现差异</h2><p>{html.escape(limitations)}</p>
<h2>渲染中的实际车体检查</h2><pre>{html.escape(json.dumps(render_metrics,indent=2,ensure_ascii=False))}</pre></html>'''
    (args.output/'index.html').write_text(document)
    md='# Nexus：OMEGA 思想接入与效果验证\n\n'+text+'\n\n'
    md+='| 方案 | 公交重叠采样点／48 | 最小间隔 m | 自车峰值加速度 m/s² | 全车加速度超限点 |\n|---|---:|---:|---:|---:|\n'
    for a in aggregates:
        md+=f'| {a["label"]} | {a["ego_bus_overlapping_samples"]} | {a["min_bus_separation_m"]:.3f} | {a["max_ego_acceleration_mps2"]:.3f} | {a["all_vehicle_acceleration_violation_samples"]} |\n'
    md+='\n'+limitations+'\n\n[打开离线报告](index.html)\n'
    (args.output/'README.md').write_text(md)
    write_json(args.output/'verification.json',{'status':'passed','baseline_exact_match_count':len(baseline),
        'conditions_preserved_count':len(rows),'same_initial_noise_across_methods':True,
        'video_checks':video_checks,'files':{str(f.relative_to(args.output)):sha(f) for f in args.output.rglob('*') if f.is_file()}})
    print('OMEGA_REPORT_COMPLETE',args.output,flush=True)


if __name__=='__main__':
    main()
