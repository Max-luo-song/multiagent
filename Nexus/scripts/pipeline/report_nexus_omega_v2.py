"""Deliver second-stage evidence, all ablations, and raw v9 comparisons."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/nexus_omega_v2/matplotlib_cache'))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--audits',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experimental-render',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assets=a.output/'assets';assets.mkdir()
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from audit_nexus_omega_v2 import sample_positions,box_sat,SCALE,MEAN
    torch.set_num_threads(2)
    rows=[]
    for audit in a.audits:
        value=json.loads((audit/'audit.json').read_text())
        assert value['status']=='complete'
        rows+=value['records']
    candidate=a.candidate.resolve();name=candidate.name
    candidate_run=json.loads((candidate/'run.json').read_text())
    protocol=json.loads((candidate/'protocol.json').read_text())
    assert candidate_run['status']=='complete' and candidate_run['model_parameters_unchanged']
    # Both old/new pairs share the same explicit starting latent for each seed.
    for seed in {r['seed'] for r in rows}:
        assert len({r['noise_sha256'] for r in rows if r['seed']==seed})==1
    aggregates=[]
    for key in dict.fromkeys((r['run'],r['method']) for r in rows):
        group=[r for r in rows if (r['run'],r['method'])==key and r['variant']=='event']
        aggregates.append({'run':key[0],'method':key[1],'seeds':len(group),
            'native_bus_overlap':sum(r['metrics']['native']['ego_bus_overlapping_samples'] for r in group),
            'dense_all_overlap':sum(r['metrics']['dense']['all_vehicle_pair_samples_overlapping'] for r in group),
            'dense_bus_overlap':sum(r['metrics']['dense']['ego_bus_overlapping_samples'] for r in group),
            'acceleration_violations':sum(r['metrics']['kinematics']['all_agent_acceleration_over_5_samples'] for r in group),
            'road_center_off':sum(r['metrics']['dense']['initially_onroad_center_off_samples'] for r in group),
            'road_full_box_off':sum(r['metrics']['dense']['drivable_full_box_off_samples'] for r in group)})
    old=next(r for r in aggregates if r['run']=='pilot_v4_kl20' and r['method']=='anchor')
    new=next(r for r in aggregates if r['run']==name and r['method']=='anchor')
    # State-head consistency and cost are separate from position-derived safety.
    state_checks=[]
    for row in candidate_run['records']:
        payload=torch.load(candidate/row['path']/'prediction.pt',map_location='cpu',weights_only=False)
        state=payload['sampled_tensor'][0].numpy()*np.asarray(SCALE)+np.asarray(MEAN)
        valid=payload['valid_mask'][0].bool().numpy()
        mask=(valid[:,1:]&valid[:,:-1])[:,4:]
        residual=np.linalg.norm(np.diff(state[:,:,:2],axis=1)[:,4:]/.5-state[:,5:,4:6],axis=-1)[mask]
        state_checks.append({'seed':row['seed'],'variant':row['variant'],'inference_seconds':row['seconds'],
            'velocity_head_vs_backward_xy_rms_mps':float(np.sqrt(np.mean(residual**2))),
            'velocity_head_vs_backward_xy_max_mps':float(residual.max())})
    decision={'status':'experimental_candidate','promoted_to_default':False,'candidate':str(candidate),
        'previous':old,'current':new,'profile':protocol,'scope':'single development scene; all failures preserved',
        'extra_future_information':'recorded future geometry of nine unpredicted replay vehicles is supplied to constraint optimization; future presence masks are logged',
        'selection_reason':'prioritize collision reduction; retain acceleration-restoration branch as experimental because it reintroduces background overlaps',
        'state_consistency_and_runtime':state_checks,
        'not_implemented':['closed-loop policy feedback','Nexus weight training','strict continuous-time guarantee','pedestrian/bicycle animation','formal unseen-scene validation']}
    (assets/'decision.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False)+'\n')
    (assets/'all_results.json').write_text(json.dumps({'aggregates':aggregates,'records':rows},indent=2)+'\n')
    for filename in ('request.json','normal_condition.json','event_condition.json','protocol.json'):
        shutil.copy2(candidate/filename,assets/filename)
    # Plot raw predictions; neither figure nor video edits a trajectory.
    comparisons=[('released',ROOT/'artifacts/nexus_omega/pilot_v3/baseline','#64748b'),
                 ('previous',ROOT/'artifacts/nexus_omega/pilot_v4_kl20/anchor','#dc2626'),
                 ('current',candidate/'anchor','#059669')]
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
    times=np.arange(1,81)*.1
    for label,root,color in comparisons:
        for seed in (0,1,2):
            d=torch.load(root/f'seed_{seed:03d}/event/prediction.pt',map_location='cpu',weights_only=False)
            valid=d['valid_mask'][0].bool().numpy();active=np.flatnonzero(valid.any(-1))
            state=d['sampled_tensor'][0].numpy()[active]*np.array(SCALE)+np.array(MEAN)
            gt=d['scene_tensor_gt'][0].numpy()[active]*np.array(SCALE)+np.array(MEAN)
            pos,yaw,seen=sample_positions(state,valid[active],times,np)
            sep,i,j,_,_=box_sat(pos,yaw,gt[:,4,6:8],np)
            axes[0,seed].plot(times,sep[(i==0)&(j==1)][0],label=label,color=color,lw=2)
            speed=np.linalg.norm(np.diff(state[0,:,:2],axis=0)/.5,axis=-1)
            axes[1,seed].plot(np.arange(1,17)*.5,speed[4:],label=label,color=color,lw=2)
    for seed in (0,1,2):
        axes[0,seed].set_title(f'Fixed seed {seed}')
        axes[0,seed].axhline(0,color='#dc2626',ls='--')
        for ax in axes[:,seed]:ax.axvspan(3,6,color='#f59e0b',alpha=.12);ax.grid(alpha=.2)
        axes[0,seed].legend();axes[1,seed].set_xlabel('Future time (s)')
    axes[0,0].set_ylabel('Signed ego-bus box separation (m)')
    axes[1,0].set_ylabel('Position-derived ego speed (m/s)')
    fig.suptitle('Second-stage Nexus constraint pilot | 1 development scene | raw generated trajectories')
    fig.tight_layout();fig.savefig(assets/'trajectory_comparison.png',dpi=160);plt.close(fig)
    render_stats={};video_checks={}
    previous=ROOT/'artifacts/nexus_omega/pilot_v4_kl20/renders'
    font=ImageFont.truetype(str(ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'),25)
    for variant in ('normal','event'):
        paths=[previous/f'anchor_{variant}',candidate/f'renders/anchor_{variant}']
        runs=[json.loads((path/'run.json').read_text()) for path in paths]
        assert all(len(run['frames'])==80 and run['status']=='passed' for run in runs)
        for label,run in zip(('previous','current'),runs):
            frames=run['frames'];metas=[f['scene_metadata'] for f in frames]
            render_stats[label+'_'+variant]={'ego_bus_overlap_frames':sum(m['ego_bus_sat_separation_m']<0 for m in metas),
                'min_ego_bus_signed_sat_m':min(m['ego_bus_sat_separation_m'] for m in metas),
                'all_vehicle_overlap_pair_frames':sum(len(m['overlapping_actor_pairs']) for m in metas)}
        path=assets/f'{variant}_before_after.mp4'
        with imageio.get_writer(str(path),fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18']) as writer:
            for idx in range(80):
                canvas=Image.new('RGB',(1920,650),'#f8fafc');draw=ImageDraw.Draw(canvas)
                for col,(folder,run) in enumerate(zip(paths,runs)):
                    frame=run['frames'][idx];rgb=np.asarray(Image.open(folder/f'{frame["frame"]:03d}_0.png').convert('RGB'))
                    canvas.paste(Image.fromarray(rgb),(col*960,55))
                    draw.text((col*960+15,10),'上一轮优化' if col==0 else '本轮结构约束候选',font=font,fill='#20344b')
                    gap=frame['scene_metadata']['ego_bus_sat_separation_m']
                    draw.text((col*960+15,604),f'自车—公交间隔 {gap:.2f} m；负值表示重叠',font=font,fill='#bb3344' if gap<0 else '#20344b')
                    assert np.array_equal(np.asarray(canvas)[55:595,col*960:(col+1)*960],rgb)
                writer.append_data(np.asarray(canvas))
                if idx in (0,30,50,79):canvas.save(assets/f'{variant}_{idx:03d}.jpg',quality=93)
        with imageio.get_reader(str(path)) as reader:
            fps=reader.get_meta_data()['fps'];count=sum(1 for _ in reader)
        assert fps==10 and count==80
        video_checks[path.name]={'fps':fps,'decoded_frames':count,'source_pixels_exact_before_encoding':True}
    (assets/'render_metrics.json').write_text(json.dumps(render_stats,indent=2)+'\n')
    experimental_html=''
    if a.experimental_render:
        experimental_stats={}
        for variant in ('normal','event'):
            folder=a.experimental_render/f'renders/anchor_{variant}'
            meta=json.loads((folder/'run.json').read_text())
            assert meta['status']=='passed' and len(meta['frames'])==80
            assert meta['source_prediction_sha256']==digest(a.experimental_render/f'anchor/seed_000/{variant}/prediction.pt')
            name_video=f'experimental_motion_{variant}.mp4'
            shutil.copy2(folder/'camera_0.mp4',assets/name_video)
            assert digest(assets/name_video)==digest(folder/'camera_0.mp4')
            experimental_stats[variant]={'all_vehicle_overlap_pair_frames':sum(len(f['scene_metadata']['overlapping_actor_pairs']) for f in meta['frames']),
                'ego_bus_overlap_frames':sum(f['scene_metadata']['ego_bus_sat_separation_m']<0 for f in meta['frames'])}
        (assets/'experimental_render_metrics.json').write_text(json.dumps(experimental_stats,indent=2)+'\n')
        experimental_html='<h2>加速度可行性实验分支（未选为主候选）</h2><p>同时调整了优化权重与逐车 KL 预算，不能作为单独 QP 的因果消融。最后两个去噪步骤对 XY 求解约束恢复；给定状态保留，失败求解拒绝。部分车辆仍有背景重叠，完整八维状态和自行车模型一致性尚未保证。</p><video controls preload="metadata" src="assets/experimental_motion_event.mp4"></video><pre>'+html.escape(json.dumps(experimental_stats,indent=2,ensure_ascii=False))+'</pre>'
    def table(values,columns):
        head=''.join('<th>'+html.escape(title)+'</th>' for key,title in columns)
        body=''.join('<tr>'+''.join('<td>'+html.escape(str(row[key]))+'</td>' for key,title in columns)+'</tr>' for row in values)
        return '<table><tr>'+head+'</tr>'+body+'</table>'
    columns=[('run','实验'),('method','方法'),('seeds','种子数'),('native_bus_overlap','公交重叠／0.5秒'),
        ('dense_bus_overlap','公交重叠／0.1秒'),('dense_all_overlap','全车辆重叠对×时刻／0.1秒'),
        ('acceleration_violations','加速度超5的点'),('road_center_off','已在路内车辆的中心越界点'),('road_full_box_off','完整车框部分越界点')]
    summary=f'三个固定事件种子：公交原生重叠点 {old["native_bus_overlap"]}→{new["native_bus_overlap"]}；0.1秒全车辆重叠对×时刻 {old["dense_all_overlap"]}→{new["dense_all_overlap"]}；加速度超限点 {old["acceleration_violations"]}→{new["acceleration_violations"]}。'
    scope='''本轮依据源码与数值核对，单独验证了历史速度恢复：发布 ResetV 对速度通道差分后写回速度通道，实际不是位置导出的速度。新分支只使用历史位置和历史传感器状态；历史位置、地图、槽位及公交未来条件不变。原始默认分支保留。
结构候选在去噪内部优化，给定公交条件精确保留，自车未来自由，没有输出后的轨迹修复。KL预算为逐车预算，不能与上一轮全场景预算20混为一谈。
历史速度的同条件、单因素对照：三事件种子的加速度超限点从21降至2。这与最终候选对上一轮的综合比较不是同一个实验。
加速度按 XY 差分计算，不代表速度输出通道已经与位置完全一致；候选每个种子的速度通道残差和运行耗时列在 decision.json。当前计算开销不适合宣称实时。
新候选使用真实地图可行驶区域距离场，并显式提供九辆未预测车辆的记录未来作为回放环境约束；它比纯历史预测拥有更多环境信息。未来存在mask仍沿用记录。不能把这一比较当成纯历史预测或官方OMEGA复现。
本例只有一个开发场景、三个固定种子；0.1秒采样不构成连续时间证明。真实地图包含道路、路口、通用可行驶区域和停车区，不能代替交通规则、路线或完整障碍物合规。
默认采样、Nexus权重和v9外观资产均未更新。没有完成决策模型闭环或三车因果链验证。下表保留全部失败分支和完整车框越界检查。'''
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Nexus 第二轮完善</title>
<style>body{{font:17px/1.7 system-ui;max-width:1500px;margin:35px auto;padding:0 24px;color:#20344b}}table{{border-collapse:collapse;font-size:14px;width:100%}}td,th{{padding:9px;border-bottom:1px solid #d5dce5;text-align:left}}th{{background:#edf3f9}}img,video{{width:100%;border-radius:8px}}p{{white-space:pre-line}}.note{{background:#fff4dc;padding:16px}}</style>
<h1>Nexus 第二轮完善：输入速度、密集约束与真实道路</h1><p class="note">{summary}</p>
<p>按碰撞优先保留主候选，仍有加速度超限；正常对照视频还有 {render_stats['current_normal']['all_vehicle_overlap_pair_frames']} 个背景车辆重叠对×帧。候选未切换为默认。数值改善与完整验收分别记录。</p>
<h2>公交事件：左上一轮，右本轮，固定种子0</h2><video controls preload="metadata" src="assets/event_before_after.mp4"></video>
<h2>正常对照：固定种子0</h2><video controls preload="metadata" src="assets/normal_before_after.mp4"></video>
<h2>完整种子轨迹</h2><img src="assets/trajectory_comparison.png">
<h2>全部消融与失败结果</h2>{table(aggregates,columns)}
<h2>v9实际车体、80个真实曝光时刻</h2><pre>{html.escape(json.dumps(render_stats,indent=2,ensure_ascii=False))}</pre>
{experimental_html}
<h2>速度通道一致性与计算成本</h2><pre>{html.escape(json.dumps(state_checks,indent=2,ensure_ascii=False))}</pre>
<h2>输入和边界</h2><p>{html.escape(scope)}</p>
<p><a href="assets/decision.json">候选与完整协议</a> · <a href="assets/all_results.json">全部独立审计</a> · <a href="assets/protocol.json">运行协议</a></p></html>'''
    (a.output/'index.html').write_text(doc)
    (a.output/'README.md').write_text('# Nexus 第二轮完善\n\n'+summary+'\n\n'+scope+'\n\n[打开报告](index.html)\n')
    verification={'status':'passed','videos':video_checks,'candidate_source':str(candidate),
        'model_parameters_unchanged':True,'files':{str(f.relative_to(a.output)):digest(f) for f in a.output.rglob('*') if f.is_file()}}
    (a.output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print('OMEGA_V2_REPORT_COMPLETE',a.output,flush=True)


if __name__=='__main__':main()
