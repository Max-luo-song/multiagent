"""Publish all independent frozen-case results, including rejected samples."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_multiscene_20261027';STAGE=OUT/'frozen_v4_seed0'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    global OUT,STAGE
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--artifact-name',default=OUT.name);args=parser.parse_args()
    if Path(args.artifact_name).name!=args.artifact_name:raise ValueError('artifact-name must be a directory name')
    OUT=ROOT/'artifacts'/args.artifact_name;STAGE=OUT/'frozen_v4_seed0'
    import numpy as np,torch,imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from shapely import wkb
    from nexus_3dgs.bundle import decode
    from audit_nexus_omega_v2 import sample_positions
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=json.loads((STAGE/'run.json').read_text());assert run['status']=='complete'
    for p,h in run['source_sha256'].items():assert digest(p)==h,p
    report=ROOT/'report'/OUT.name;report.mkdir(exist_ok=False);media=report/'media';media.mkdir()
    font=ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf';fonts={i:ImageFont.truetype(str(font),i) for i in (16,18,20,22,26)}
    times=np.arange(81)*.1;videos={};images=[];sections=[];table=[]
    initial_audit=json.loads((OUT/'input_audit.json').read_text())
    initial_cases={r['index']:r for r in initial_audit['cases']}
    fixed_audit=json.loads((OUT/'fixed_A_background_audit.json').read_text())
    new_samples=sum(not r.get('reused_prediction',False) for r in run['records'])
    event_description=run['protocol'].get('event_description_zh','A在0.5–2.5秒向自车侧横移3米；E/C未来自由。')
    for pair in run['pairs']:
        idx=pair['index'];case=OUT/f'case_{idx:03d}';info=json.loads((case/'preparation.json').read_text());A=info['roles']['A'];C=info['roles']['C'];W=np.array(json.loads((case/'reference/protocol.json').read_text())['world_from_nexus'])
        data={};metrics={};records={r['variant']:r for r in run['records'] if r['index']==idx}
        for variant in ('normal','event'):
            row=records[variant];p=STAGE/row['path']/'prediction.pt';assert digest(p)==row['prediction_sha256'];payload=torch.load(p,map_location='cpu',weights_only=False)
            states=decode(payload['sampled_tensor'][0].numpy());valid=payload['valid_mask'][0].bool().numpy();active=np.flatnonzero(valid.any(-1));states=states[active];valid=valid[active]
            xy,yaw,seen=sample_positions(states,valid,times,np);gt=decode(payload['scene_tensor_gt'][0].numpy())[active]
            data[variant]=(xy.transpose(1,0,2),yaw.T,seen.T,gt[:,4,6:8],list(active).index(A),list(active).index(C))
            metrics[variant]=row['acceptance']
            if variant=='normal':
                initial_heading=gt[0,4,2:4];theta=np.arctan2(initial_heading[1],initial_heading[0]);R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        with np.load(case/'reference/inputs/background_replay.npz') as z:bg={k:z[k] for k in z.files}
        with np.load(case/'reference/inputs/drivable_sdf.npz') as z:sdf={k:z[k] for k in z.files}
        road=Image.fromarray(np.where(sdf['sdf']>=0,248,215).astype(np.uint8)).convert('RGB');d=ImageDraw.Draw(road)
        for slot,col in [(A,'#fff0d9'),(0,'#edf2ff'),(C,'#e5f5f5')]:
            corridor=wkb.loads((case/'map/lanes'/f'{slot}.wkb').read_bytes())
            for poly in (corridor.geoms if hasattr(corridor,'geoms') else [corridor]):
                local=(np.array(poly.exterior.coords)-W[:2,3])@W[:2,:2];pixels=(local-sdf['origin'])/float(sdf['resolution']);d.polygon([tuple(q) for q in pixels],fill=col,outline='#9cabbc')
                for ring in poly.interiors:
                    local=(np.array(ring.coords)-W[:2,3])@W[:2,:2];pixels=(local-sdf['origin'])/float(sdf['resolution']);d.polygon([tuple(q) for q in pixels],fill='#d7d7d7')
        def panel(variant,k,origin,scale):
            xy,yaw,seen,sizes,ai,ci=data[variant];im=Image.new('RGB',(800,850),'#f5f7fa');d=ImageDraw.Draw(im)
            def text(q,s,size=22,color='#24364b'):d.text(q,s,font=fonts[size],fill=color)
            text((20,12),f'{idx:03d}  '+('正常' if variant=='normal' else ('切入后减速' if 'braking' in run['protocol'] else 'A 切入'))+f'    t={times[k]:.1f}s',26)
            speeds=metrics[variant]['motion']['E']['xy_interval_speeds_mps'];speed=info['observed_speeds_mps']['E'] if k==0 else speeds[min(int(np.ceil(times[k]/.5))-1,15)]
            text((20,55),f'E 自车实际速度 {speed:.1f} m/s  ({speed*3.6:.0f} km/h)',22)
            text((20,95),'红 E 自车；橙 A 给定事件车；蓝 C 自由车辆',20)
            text((20,125),'相同视点与比例；淡色区域为各车原车道',18)
            res=float(sdf['resolution']);base=sdf['origin'];cx=380;cy=310
            coeff=(-R[0,1]/scale/res,-R[0,0]/scale/res,(origin[0]+cy/scale*R[0,0]+cx/scale*R[0,1]-base[0])/res,-R[1,1]/scale/res,-R[1,0]/scale/res,(origin[1]+cy/scale*R[1,0]+cx/scale*R[1,1]-base[1])/res)
            tile=road.transform((760,600),Image.Transform.AFFINE,coeff,resample=Image.Resampling.NEAREST,fillcolor=(215,215,215));im.paste(tile,(20,165));canvas=im.copy();draw=ImageDraw.Draw(canvas)
            def project(points):
                local=(np.asarray(points)-origin)@R;return np.stack([400-local[...,1]*scale,475-local[...,0]*scale],-1)
            def body(center,ang,size,color,label=None):
                q=project(center)
                if not (10<q[0]<790 and 150<q[1]<780):return
                rot=np.array([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]]);corn=project(np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*size/2@rot+center)
                draw.polygon([tuple(v) for v in corn],fill=color,outline='#64748b',width=1)
                if label:draw.text((q[0]+10,q[1]-12),label,font=fonts[26],fill=color)
            for i in np.flatnonzero(bg['valid'][k]):body(bg['centers'][k,i],np.arctan2(bg['heading'][k,i,1],bg['heading'][k,i,0]),bg['sizes'][i],'#a9b2bd')
            for i in [j for j in range(len(sizes)) if j not in (0,ai,ci)]+[ai,ci,0]:
                if not seen[k,i]:continue
                center=xy[k,i].copy()
                if i==0:center+=1.461*np.array([np.cos(yaw[k,i]),np.sin(yaw[k,i])])
                role='E' if i==0 else 'A' if i==ai else 'C' if i==ci else None
                body(center,yaw[k,i],sizes[i],{'E':'#d94a4a','A':'#dd921c','C':'#287bcd'}.get(role,'#c5cdd6'),role)
            im.paste(canvas.crop((20,165,780,765)),(20,165));d=ImageDraw.Draw(im)
            text((20,785),'主三车验收：'+('通过' if metrics[variant]['primary_three_accepted'] else '未通过')+'；全场景：'+('通过' if metrics[variant]['full_scene_accepted'] else '未通过'),20)
            text((20,817),'真实生成轨迹；E/C 未来自由；0 秒为历史结束',18)
            return np.asarray(im)
        filename=f'case_{idx:03d}_pair.mp4';path=media/filename
        with imageio.get_writer(path,fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','19']) as writer:
            for k in range(81):
                anchors=[]
                for values in data.values():xy,_,_,_,ai,ci=values;anchors.extend(xy[k,[0,ai,ci]])
                origin=np.mean(anchors,axis=0);local=(np.array(anchors)-origin)@R;scale=min(10.,500/(np.ptp(local[:,0])+24),690/(np.ptp(local[:,1])+20))
                frame=np.concatenate([panel(v,k,origin,scale) for v in ('normal','event')],axis=1);writer.append_data(frame)
                if k in (0,25,80):imageio.imwrite(media/f'case_{idx:03d}_{k:03d}.jpg',frame)
                if k==25:images.append(Image.fromarray(frame).resize((1200,638)))
        with imageio.get_reader(path) as reader:metadata=reader.get_meta_data();count=sum(1 for _ in reader)
        assert count==81 and metadata['fps']==10;videos[str(idx)]={'file':filename,'frames':count,'fps':10,'sha256':digest(path)}
        fig,ax=plt.subplots(figsize=(9,3.5));tt=np.arange(16)*.5+.25
        for v,col in [('normal','#1f77b4'),('event','#d62728')]:ax.plot(tt,metrics[v]['motion']['E']['xy_interval_speeds_mps'],marker='.',label=v,color=col)
        ax.axhline(info['observed_speeds_mps']['E'],ls=':',color='grey',label='observed current E');ax.set(xlabel='Future time (s)',ylabel='E speed (m/s)',title=f'Scene {idx:03d}: native displacement speed');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(media/f'case_{idx:03d}_speed.png',dpi=140);plt.close(fig)
        rows=''
        for v in ('normal','event'):
            m=metrics[v];mo=m['motion']['E'];vv=mo['xy_interval_speeds_mps'];lane=m['original_lane']['E']['box_outside_original_lane_samples']
            rows+=f'<tr><td>{"正常" if v=="normal" else "切入"}</td><td>{m["roles"]["EA"]["overlap_samples"]}</td><td>{m["roles"]["EC"]["overlap_samples"]}</td><td>{m["role_any_overlap_samples"]["C"]}</td><td>{lane}</td><td>{mo["max_xy_acceleration_mps2"]:.2f}</td><td>{mo["max_heading_error_deg"]:.2f}°</td><td>{min(vv):.2f} → {vv[-1]:.2f}</td><td>{"通过" if m["primary_three_accepted"] else "未过"} / {"通过" if m["full_scene_accepted"] else "未过"}</td></tr>'
        initial_note='；'.join(f'{r}在历史终点已部分越出选定车道' for r,m in initial_cases[idx]['initial_body'].items() if not m['initial_body_inside_selected_lane'])
        if initial_note:initial_note='初始观测注意：'+initial_note+'。绝对车道验收保持原阈值，因此短时越线不能直接归因于生成。'
        fixed_conflicts={r['variant']:r['fixed_A_vs_recorded_background_overlap_samples'] for r in fixed_audit['records'] if r['index']==idx}
        condition_note='给定A与固定背景的碰撞采样：正常 '+str(fixed_conflicts.get('normal',0))+'，事件 '+str(fixed_conflicts.get('event',0))+'。'
        if any(fixed_conflicts.values()):condition_note+='这部分冲突在生成E/C之前已经确定，自由车辆无法修复；保留失败标志，不能直接归因于Nexus。'
        sections.append(f'<section id="case{idx}"><h2>场景 {idx:03d}：从{ "右" if info["cut_in_side"]=="left" else "左"}侧切入</h2><p>{initial_note}</p><p>{condition_note}</p><p>{event_description}</p><p>E/A/C 当前速度分别为 {info["observed_speeds_mps"]["E"]:.2f} / {info["observed_speeds_mps"]["A"]:.2f} / {info["observed_speeds_mps"]["C"]:.2f} m/s。左正常，右切入。</p><video controls preload="metadata" src="media/{filename}"></video><p><a href="media/{filename}">直接打开运动视频</a></p><img src="media/case_{idx:03d}_speed.png"><table><tr><th>条件</th><th>E/A重叠</th><th>E/C重叠</th><th>C任意重叠</th><th>E出原车道</th><th>E最大加速度</th><th>E方向偏差</th><th>E最低→末速</th><th>三车/全场景</th></tr>{rows}</table><p>将正常预测 E 与事件 A 组合的离线威胁检查：重叠 {pair["normal_E_plus_event_A_overlap_samples"]} 个采样点，最小间隔 {pair["normal_E_plus_event_A_min_gap_m"]:.2f}m。此组合只用于诊断，未用来条件化 E。E/C 最大轨迹差异 {pair["E_max_position_change_m"]:.2f} / {pair["C_max_position_change_m"]:.2f}m；不证明 E 导致 C 的变化。</p><p>数据日志：{info["scenario"]["log_name"]}，token {info["scenario"]["token"]}。</p></section>')
        print('VIDEO_COMPLETE',idx,flush=True)
    summary={'status':'complete','tested_new_scenes':len(run['pairs']),'real_samples':new_samples,'reused_normal_predictions':len(run['records'])-new_samples,'primary_pairs_passed':sum(p['primary_pair_accepted'] for p in run['pairs']),'full_pairs_passed':sum(p['full_scene_pair_accepted'] for p in run['pairs']),'fixed_A_background_audit':fixed_audit,'initial_geometry_audit':initial_audit,'skipped':run['skipped'],'pairs':run['pairs'],'videos':videos,'frozen_method_unchanged':True,'offline_trajectory_edits':False,'strict_causal_chain_proven':False}
    write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
    write(report/'results.json',run);write(report/'summary.json',summary);write(report/'protocol.json',run['protocol']);write(OUT/'completion.json',summary)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Nexus 跨场景三车验证</title><style>body{font-family:system-ui;max-width:1200px;margin:24px auto;padding:0 16px;background:#f5f7fa;color:#24364b}section{padding:20px;background:white;margin:18px 0;border-radius:12px}video,img{width:100%}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:left}p{line-height:1.8}</style><h1>Nexus：跨场景三车验证</h1>'''
    page+=f'<section><b>展示 {len(run["pairs"])} 个场景对照、{new_samples} 次新增真实采样。正常与切入都通过主三车物理检查：{summary["primary_pairs_passed"]}/{summary["tested_new_scenes"]}；全场景：{summary["full_pairs_passed"]}/{summary["tested_new_scenes"]}。</b><p>固定当前 V4 权重、控制优化参数和 seed 0，不针对新结果调参。每例同一前2秒历史、同一噪声、同一地图初始化；A 未来给定，E/C 未来自由生成8秒。A 在0.5–2.5秒向自车侧横移3米；新场景没有原033的向前2米编辑。</p><p>这些是按历史几何挑选的少量案例，并非随机总体评估。050 的 A 明显更快，可用于观察弱威胁下是否过度反应。速度图报告真实行为，不把033的巡航阈值当成所有路口场景的规则。背景回放和存在掩码仍来自记录。验收失败结果全部保留。</p></section>'
    if 'braking' in run['protocol']:page=page.replace('050 的 A 明显更快，可用于观察弱威胁下是否过度反应。','本补充只包含015和018；事件A在切入后继续减速，以提高碰撞威胁。')
    if run['protocol'].get('normal_prediction_reuse'):page+='<section><b>强化威胁补充实验</b><p>'+event_description+'本页仅新增事件采样，正常对照复用原批次已核对相同条件的输出；不算新的场景，也没有针对结果修改优化器参数。</p></section>'
    page+=''.join(sections)
    for skipped in run['skipped']:page+=f'<section>输入预检未过，未运行：{skipped}</section>'
    page+='<section><p>重叠、道路和原车道：0.1–8.0秒每0.1秒检查有向车身，使用观测尺寸及自车后轴偏移；事件可合理换道。运动一致性按原生0.5秒轨迹检查。离散检查不证明连续时间安全。无采样后轨迹修补；该报告是诊断俯视视频，未重建高斯资产，不作为通过验收的下游交付。</p><a href="results.json">全部原始指标</a> · <a href="protocol.json">冻结实验协议</a></section></html>'
    (report/'index.html').write_text(page)
    if images:
        contact=Image.new('RGB',(1200,638*len(images)),'white')
        for i,im in enumerate(images):contact.paste(im,(0,638*i))
        contact.save(report/'contact_sheet.jpg')
    print('REPORT_COMPLETE',summary,flush=True)
if __name__=='__main__':main()
