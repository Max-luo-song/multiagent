"""Deliver third-stage behavior evidence, including rejected experiments."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/nexus_omega_v3/matplotlib_cache'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--geometry-audits',type=Path,nargs='+',required=True)
    p.add_argument('--behavior-audits',type=Path,nargs='+',required=True)
    p.add_argument('--review-run',type=Path,required=True)
    p.add_argument('--decision',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assets=a.output/'assets';assets.mkdir()
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(2)
    geometry=[];behavior=[]
    for folder in a.geometry_audits:
        d=json.loads((folder/'audit.json').read_text());assert d['status']=='complete'
        geometry+=d['records']
    for folder in a.behavior_audits:
        d=json.loads((folder/'behavior.json').read_text());assert d['status']=='complete'
        behavior+=d['records']
    decision=json.loads(a.decision.read_text())
    by_geometry={(r['run'],r['method'],r['seed'],r['variant']):r for r in geometry}
    by_behavior={(r['run'],r['method'],r['seed'],r['variant']):r for r in behavior}
    assert sha(ROOT.parent/'checkpoints/nuplan.ckpt')=='679f6ccfbd5563f696218311566b76bb62f73c7fbfb04b39f2714fb80aefacfc'
    rows=[];integrity=[]
    for root in a.runs:
        run=json.loads((root/'run.json').read_text());assert run['status']=='complete' and run['model_parameters_unchanged']
        proto=json.loads((root/'protocol.json').read_text())
        shutil.copy2(root/'protocol.json',assets/(root.name+'_protocol.json'))
        if 'nexus_omega_v3' in root.parts:
            for relative,digest in proto['source_sha256'].items():assert sha(root/'sources'/relative)==digest
        for record in run['records']:
            key=(root.name,record['method'],record['seed'],record['variant'])
            if key not in by_geometry or key not in by_behavior:raise ValueError('Missing audit: '+str(key))
            g=by_geometry[key];b=by_behavior[key];path=root/record['path']/'prediction.pt'
            assert sha(path)==g['prediction_sha256']==b['prediction_sha256']
            if 'nexus_omega_v3' in root.parts:
                d=torch.load(path,map_location='cpu',weights_only=False)
                assert torch.isfinite(d['sampled_tensor'][d['valid_mask'].bool()]).all()
                known=d['task_mask'].bool();torch.testing.assert_close(d['sampled_tensor'][known],d['conditioned_scene_tensor'][known],rtol=0,atol=0)
                assert (d['conditioned_scene_tensor'][~known]==0).all() and not known[:,0,5:].any()
                guidance=json.loads((root/record['path']/'guidance.json').read_text())
                assert all(x['kl_max_per_agent']<=proto['config']['kl_budget']*1.0001 for x in guidance)
                integrity.append({'run':root.name,'seed':record['seed'],'variant':record['variant'],
                    'conditions_exact':True,'ego_future_free':True,'per_agent_kl_within_budget':True,
                    'source_snapshots_exact':True,'prediction_sha256':g['prediction_sha256']})
            e=b['metrics']['ego'];dense=g['metrics']['dense']
            rows.append({'run':root.name,'method':record['method'],'seed':record['seed'],'variant':record['variant'],
                'dense_overlap':dense['all_vehicle_pair_samples_overlapping'],
                'road_center_off':dense['initially_onroad_center_off_samples'],'road_box_off':dense['drivable_full_box_off_samples'],
                'acc_over5':g['metrics']['kinematics']['all_agent_acceleration_over_5_samples'],
                'progress_m':round(e['final_forward_progress_m'],3),'left_m':round(e['final_left_displacement_m'],3),
                'yaw_deg':round(e['max_abs_yaw_delta_deg'],2),'reverse_m':round(e['reverse_distance_body_axis_m'],3),
                'lateral_rms_mps':round(e['xy_lateral_velocity_rms_mps'],3),
                'velocity_residual_mps':round(e['xy_velocity_head_rms_mps'],3),
                'curvature_excess_rps':round(e.get('curvature_bound_excess_max_rps',0.),3),
                'yaw_rate_rps':round(record['metrics']['ego_peak_yaw_rate_rps'],3),'seconds':round(record['seconds'],1)})
    for seed in {r['seed'] for r in geometry}:
        assert len({r['noise_sha256'] for r in geometry if r['seed']==seed})==1
    (assets/'evidence.json').write_text(json.dumps({'rows':rows,'geometry':geometry,'behavior':behavior},indent=2)+'\n')
    (assets/'integrity.json').write_text(json.dumps({'status':'passed','scope':'technical integrity, not behavioral acceptance','records':integrity},indent=2)+'\n')
    shutil.copy2(a.decision,assets/'decision.json')
    shutil.copy2(ROOT/'docs/nexus_omega_v3_plan_20261026.md',assets/'plan.md')
    shutil.copy2(ROOT/'docs/nexus_omega_v3_findings_20261026.md',assets/'findings.md')
    if (a.decision.parent/'decisions').is_dir():
        shutil.copytree(a.decision.parent/'decisions',assets/'branch_decisions')
    scale=np.array([104,104,1,1,4,4,5,1.6]);mean=np.array([0,0,0,0,0,0,4.5,2])
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    roots=[ROOT/'artifacts/nexus_omega/pilot_v4_kl20',ROOT/'artifacts/nexus_omega_v2/order_candidate_v1',a.review_run]
    for root,label,color in zip(roots,['Earlier left response','V2 collision-priority','V3 reviewed branch'],['#dc2626','#64748b','#059669']):
        payload=torch.load(root/'anchor/seed_000/event/prediction.pt',map_location='cpu',weights_only=False)
        state=payload['sampled_tensor'][0,0].numpy()*scale+mean
        start=np.arctan2(state[4,3],state[4,2]);head=np.array([np.cos(start),np.sin(start)]);normal=np.array([-head[1],head[0]])
        xy=state[4:,:2]-state[4,:2];yaw=np.arctan2(state[4:,3],state[4:,2]);yaw=np.arctan2(np.sin(yaw-start),np.cos(yaw-start))
        axes[0].plot(xy@head,xy@normal,label=label,color=color)
        axes[1].plot(np.arange(17)*.5,np.degrees(yaw),label=label,color=color)
        speed=np.diff(state[4:,:2],axis=0)/.5
        h=state[5:,2:4];h=h/np.maximum(np.linalg.norm(h,axis=-1,keepdims=True),1e-8)
        axes[2].plot(np.arange(1,17)*.5,(speed*h).sum(-1),label=label,color=color)
    axes[0].set(xlabel='Forward displacement (m)',ylabel='Left displacement (m)',title='Raw ego path; fixed seed 0')
    axes[1].set(xlabel='Future time (s)',ylabel='Heading change (degrees)',title='Heading response')
    axes[2].set(xlabel='Future time (s)',ylabel='Body-axis XY speed (m/s)',title='Negative values indicate reverse')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    for ax in axes[1:]:ax.axvspan(3,6,color='#f59e0b',alpha=.15)
    fig.tight_layout();fig.savefig(assets/'behavior.png',dpi=160);plt.close(fig)
    render_stats={};video_checks={}
    font=ImageFont.truetype(str(ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'),25)
    baseline=ROOT/'artifacts/nexus_omega_v2/order_candidate_v1'
    frozen_sha=None
    for variant in ('normal','event'):
        folders=[root/f'renders/anchor_{variant}' for root in (baseline,a.review_run)]
        metas=[json.loads((folder/'run.json').read_text()) for folder in folders]
        for root,meta in zip((baseline,a.review_run),metas):
            assert meta['status']=='passed' and len(meta['frames'])==80
            assert meta['checkpoint_sha256']=='1debd88b7c7f9172a64c65b00df69257007bb1a6595001ff38ef9305a94a188b'
            if frozen_sha is None:frozen_sha=sha(Path(meta['checkpoint']))
            assert frozen_sha==meta['checkpoint_sha256']
            assert meta['appearance_and_checkpoint_unchanged'] and meta['temporary_actor_poses_restored_exactly']
            assert meta['source_prediction_sha256']==sha(root/f'anchor/seed_000/{variant}/prediction.pt')
            render_stats[root.name+'_'+variant]={'ego_bus_overlap_frames':sum(f['scene_metadata']['ego_bus_sat_separation_m']<0 for f in meta['frames']),
                'all_vehicle_overlap_pair_frames':sum(len(f['scene_metadata']['overlapping_actor_pairs']) for f in meta['frames'])}
        path=assets/f'{variant}_v2_vs_v3.mp4'
        with imageio.get_writer(str(path),fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18']) as writer:
            for k in range(80):
                canvas=Image.new('RGB',(1920,650),'#f8fafc');draw=ImageDraw.Draw(canvas)
                for col,(folder,meta) in enumerate(zip(folders,metas)):
                    frame=meta['frames'][k];rgb=np.asarray(Image.open(folder/f'{frame["frame"]:03d}_0.png').convert('RGB'))
                    canvas.paste(Image.fromarray(rgb),(col*960,55))
                    draw.text((col*960+15,10),'V2 碰撞优先基线' if col==0 else 'V3 本轮审查分支',font=font,fill='#20344b')
                    gap=frame['scene_metadata']['ego_bus_sat_separation_m']
                    draw.text((col*960+15,604),f'自车—公交间隔 {gap:.2f} m；负值表示重叠',font=font,fill='#20344b')
                    assert np.array_equal(np.asarray(canvas)[55:595,col*960:(col+1)*960],rgb)
                writer.append_data(np.asarray(canvas))
                if k in (30,50,79):canvas.save(assets/f'{variant}_{k:03d}.jpg',quality=93)
        with imageio.get_reader(str(path)) as reader:
            fps=reader.get_meta_data()['fps'];count=sum(1 for _ in reader)
        assert fps==10 and count==80
        video_checks[path.name]={'decoded_frames':count,'fps':fps,'source_pixels_exact_before_encoding':True}
    (assets/'render_metrics.json').write_text(json.dumps(render_stats,indent=2)+'\n')
    columns=list(rows[0]);head=''.join('<th>'+html.escape(k)+'</th>' for k in columns)
    body=''.join('<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in columns)+'</tr>' for row in rows)
    scope='同一开发场景；正常/事件配对；使用九辆背景车的记录未来和记录存在 mask。公交第 3–6 秒倒车 4.5 米为固定条件，自车未来自由。0.1 秒检查和真实摄像机曝光时刻分别统计。软运动约束不是严格自行车模型、交通规则合规或连续时间安全证明。没有训练 Nexus、没有修改 v9 资产、没有采样后轨迹修复。未满足本轮验收时不扩展场景、不提升为默认。'
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Nexus 第三轮：恢复合理避让</title><style>body{{font:17px/1.7 system-ui;max-width:1500px;margin:35px auto;padding:0 24px;color:#20344b}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #d5dce5}}img,video{{width:100%}}pre{{white-space:pre-wrap}}.scroll{{overflow:auto}}.note{{background:#fff4dc;padding:16px}}</style>
<h1>Nexus 第三轮：转向、倒退与运动一致性</h1><pre class="note">{html.escape(json.dumps(decision,indent=2,ensure_ascii=False))}</pre>
<p><a href="assets/plan.md">计划与阶段补充</a> · <a href="assets/findings.md">结果和交接</a> · <a href="assets/decision.json">本轮结论</a> · <a href="assets/evidence.json">全部数值与失败结果</a> · <a href="assets/integrity.json">技术完整性审计</a></p>
<h2>事件组：左 V2 基线，右本轮审查分支，固定种子 0</h2><video controls preload="metadata" src="assets/event_v2_vs_v3.mp4"></video>
<h2>正常对照</h2><video controls preload="metadata" src="assets/normal_v2_vs_v3.mp4"></video>
<h2>三版行为比较</h2><img src="assets/behavior.png">
<h2>全部分支与种子</h2><p>dense_overlap 为 0.1 秒车辆重叠对×时刻；reverse_m 为自车沿车体方向累计后退距离；速度残差及侧滑单位 m/s；acc_over5 为全体预测车辆的位置差分加速度超限点。背景未来覆盖外不外推。</p><div class="scroll"><table><tr>{head}</tr>{body}</table></div>
<h2>实际 v9 渲染检查</h2><pre>{html.escape(json.dumps(render_stats,indent=2,ensure_ascii=False))}</pre><p>{html.escape(scope)}</p></html>'''
    (a.output/'index.html').write_text(doc)
    (a.output/'README.md').write_text('# Nexus 第三轮\n\n'+json.dumps(decision,indent=2,ensure_ascii=False)+'\n\n'+scope+'\n')
    verification={'status':'passed','meaning':'technical integrity only','videos':video_checks,
        'files':{str(path.relative_to(a.output)):sha(path) for path in a.output.rglob('*') if path.is_file()}}
    (a.output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print('NEXUS_V3_REPORT_COMPLETE',a.output,flush=True)


if __name__=='__main__':main()
