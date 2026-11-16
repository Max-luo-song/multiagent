"""Render saved real candidates; display interpolation never edits native output."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'artifacts/nexus_multicar_upgrade_check_20261027'
OUT=ROOT/'artifacts/nexus_kinematic_20261027'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from shapely import wkb
    from nexus_3dgs.bundle import decode
    from nexus_3dgs.scene_projection import pair_separations
    from nexus_3dgs.trajectory_acceptance import evaluate
    from audit_nexus_omega_v2 import sample_positions
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='candidate_v4_cruise_initialized');args=parser.parse_args()
    stage=OUT/args.stage;run=json.loads((stage/'run.json').read_text());assert run['status']=='complete'
    protocol=json.loads((OLD/'protocol.json').read_text());W=np.array(protocol['world_from_nexus'])
    for file,h in run['protocol']['source_sha256'].items():assert digest(file)==h,file
    report=ROOT/'report'/f'nexus_cruise_20261027_{args.stage}';report.mkdir(exist_ok=True);(report/'media').mkdir(exist_ok=True)
    samples={};metrics={};records={};labels={}
    for row in run['records']:
        name=f'new_{row["seed"]}_{row["variant"]}';path=stage/row['path']/'prediction.pt'
        assert digest(path)==row['prediction_sha256'];samples[name]=path;metrics[name]=row['acceptance'];records[name]=row
        labels[name]='本版巡航恢复：'+('正常' if row['variant']=='normal' else 'A 切入')
    oldrun=json.loads((OUT/'candidate_v1/run.json').read_text())
    for row in oldrun['records']:
        name='old_'+row['variant'];path=OUT/'candidate_v1'/row['path']/'prediction.pt'
        assert digest(path)==row['prediction_sha256'];samples[name]=path;records[name]=row
        labels[name]='上一版偏慢：'+('正常' if row['variant']=='normal' else 'A 切入')
    times=np.arange(81)*.1
    with np.load(OLD/'inputs/background_replay.npz') as data:bg={k:data[k] for k in data.files}
    with np.load(OLD/'inputs/drivable_sdf.npz') as data:sdf={k:data[k] for k in data.files}
    with np.load(OUT/'routes.npz') as data:routes={slot:data[f'route_{slot}'] for slot in (0,2,12)}
    road=Image.fromarray(np.where(sdf['sdf']>=0,250,218).astype(np.uint8)).convert('RGB')
    painter=ImageDraw.Draw(road)
    for slot,color in [(2,'#fff4df'),(0,'#edf3ff'),(12,'#eaf7f5')]:
        polygon=wkb.loads((OUT/'lanes'/f'{slot}.wkb').read_bytes())
        for p in (polygon.geoms if hasattr(polygon,'geoms') else [polygon]):
            local=(np.array(p.exterior.coords)-W[:2,3])@W[:2,:2]
            pixels=(local-sdf['origin'])/float(sdf['resolution'])
            painter.polygon([tuple(v) for v in pixels],fill=color,outline='#b5c5d5',width=1)
    fontpath=ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'
    fonts={s:ImageFont.truetype(str(fontpath),s) for s in (16,18,20,22,26)}
    sampled={};payloads={};observed_speed={};R=None
    for name,path in samples.items():
        payload=torch.load(path,map_location='cpu',weights_only=False);payloads[name]=payload
        valid=payload['valid_mask'][0].bool().numpy();active=np.flatnonzero(valid.any(-1))
        states=decode(payload['sampled_tensor'][0].numpy())[active];gt=decode(payload['scene_tensor_gt'][0].numpy())[active]
        xy,yaw,seen=sample_positions(states,valid[active],times,np);xy=xy.transpose(1,0,2);yaw=yaw.T;seen=seen.T
        sizes=gt[:,4,6:8];A=list(active).index(2);C=list(active).index(12)
        if R is None:
            angle=np.arctan2(gt[0,4,3],gt[0,4,2]);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        boxes=np.concatenate([xy.copy(),yaw[...,None]],-1);boxes[:,0,:2]+=1.461*np.c_[np.cos(yaw[:,0]),np.sin(yaw[:,0])]
        sep,_=pair_separations(boxes[:,[0,A,C]],seen[:,[0,A,C]],sizes[[0,A,C]])
        speed=np.linalg.norm(np.diff(states[0,4:,:2],axis=0)/.5,axis=-1)
        observed_speed[name]=float(np.linalg.norm((states[0,4,:2]-states[0,3,:2])/.5))
        sampled[name]=(xy,yaw,seen,sizes,A,C,sep,speed)
        if name.startswith('old_'):metrics[name]=evaluate(payload,reference_root=OLD,lane_root=OUT/'lanes',variant=name[4:])
    for variant in ('normal','event'):
        key=f'new_0_{variant}'
        if key in payloads:
            for field in ('conditioned_scene_tensor','task_mask','valid_mask'):
                assert torch.equal(payloads[key][field],payloads['old_'+variant][field]),field
            assert records[key]['noise_sha256']==records['old_'+variant]['noise_sha256']
    # Apply the same predeclared speed benchmark to the previous version.
    # The new run already evaluated it before deciding downstream eligibility.
    speed_policy=run['protocol']['speed_behavior_gate']
    final_fraction=speed_policy['normal_E_final_speed_min_fraction_current']
    progress_fraction=speed_policy['normal_E_progress_min_fraction_current_speed_times_horizon']
    minimum_fraction=speed_policy.get('normal_E_any_interval_speed_min_fraction_current',0.)
    for variant in ('normal','event'):
        name='old_'+variant
        speed=sampled[name][7];v0=observed_speed[name]
        check={'observed_reference_speed_mps':v0,'final_interval_speed_mps':float(speed[-1]),
               'forecast_progress_m':float(speed.sum()*.5)}
        if variant=='normal':
            check.update(final_speed_threshold_mps=final_fraction*v0,
                         minimum_interval_speed_mps=float(speed.min()),
                         minimum_interval_speed_threshold_mps=minimum_fraction*v0,
                         progress_threshold_m=progress_fraction*v0*8.)
            check['accepted']=bool(speed[-1]>=final_fraction*v0 and speed.min()>=minimum_fraction*v0
                                   and speed.sum()*.5>=progress_fraction*v0*8.)
        else:
            states=decode(payloads[name]['conditioned_scene_tensor'][0].numpy())
            leader_speed=float(np.linalg.norm(states[2,-1,4:6]))
            drop=float(sampled['old_normal'][7][:5].min()-speed[:5].min())
            check.update(final_speed_threshold_mps=.8*min(v0,leader_speed),
                         early_additional_slowdown_mps=drop,
                         early_additional_slowdown_threshold_mps=1.)
            check['accepted']=bool(speed[-1]>=check['final_speed_threshold_mps'] and drop>=1.)
        metrics[name]['speed_behavior']=check
    pair_metrics={}
    for seed in sorted({r['seed'] for r in run['records']}):
        normal=f'new_{seed}_normal';event=f'new_{seed}_event'
        if normal not in payloads or event not in payloads:continue
        ns=decode(payloads[normal]['sampled_tensor'][0].numpy())
        es=decode(payloads[event]['sampled_tensor'][0].numpy())
        mixed=ns[[0,2]].copy();mixed[1]=es[2]
        vv=payloads[normal]['valid_mask'][0].bool().numpy()[[0,2]]
        xy,yaw,seen=sample_positions(mixed,vv,times[1:],np)
        boxes=np.concatenate([xy.transpose(1,0,2),yaw.T[...,None]],-1)
        boxes[:,0,:2]+=1.461*np.c_[np.cos(boxes[:,0,2]),np.sin(boxes[:,0,2])]
        gt=decode(payloads[normal]['scene_tensor_gt'][0].numpy())
        sep,validpair=pair_separations(boxes,seen.T,gt[[0,2],4,6:8])
        pair_metrics[str(seed)]={'ego_max_position_change_m':float(np.linalg.norm(es[0,5:,:2]-ns[0,5:,:2],axis=-1).max()),
                                 'C_max_position_change_m':float(np.linalg.norm(es[12,5:,:2]-ns[12,5:,:2],axis=-1).max()),
                                 'normal_ego_plus_event_A_min_gap_m':float(sep[:,0,1].min()),
                                 'normal_ego_plus_event_A_overlap_samples':int(((sep[:,0,1]<0)&validpair[:,0,1]).sum()),
                                 'strict_causal_chain_proven':False}

    def camera(names,index):
        anchors=[]
        for name in names:
            xy,_,_,_,A,C,_,_=sampled[name];anchors.extend(xy[index,[0,A,C]])
        origin=np.mean(anchors,axis=0);forward=(np.array(anchors)-origin)@R[:,0]
        low=float(forward.min())-10;high=float(forward.max())+16
        scale=min(12.,600/(high-low));origin=origin+R[:,0]*(low+high)/2
        return origin,scale

    def panel(name,index,origin,scale):
        xy,yaw,seen,sizes,A,C,sep,speed=sampled[name]
        im=Image.new('RGB',(1000,940),'#f7fafc');draw=ImageDraw.Draw(im)
        def text(pos,value,size=22,fill='#24364b'):draw.text(pos,value,font=fonts[size],fill=fill)
        text((20,12),f'{labels[name]}  t={times[index]:.1f}s',26)
        text((20,54),'红 E 自车；橙 A 事件车；蓝 C 第三车。',22)
        speed_index=min(max(int(np.ceil(times[index]/.5))-1,0),15)
        current_speed=observed_speed[name] if index==0 else speed[speed_index]
        text((20,94),f'E 实际位移速度：{current_speed:.1f} m/s  （{current_speed*3.6:.0f} km/h）',26)
        text((20,139),'两侧相同视点；淡色区域及边界为各车原车道。',18,'#64748b')
        res=float(sdf['resolution']);base=sdf['origin']
        coeff=(-R[0,1]/scale/res,-R[0,0]/scale/res,(origin[0]+310/scale*R[0,0]+480/scale*R[0,1]-base[0])/res,
               -R[1,1]/scale/res,-R[1,0]/scale/res,(origin[1]+310/scale*R[1,0]+480/scale*R[1,1]-base[1])/res)
        tile=road.transform((960,620),Image.Transform.AFFINE,coeff,resample=Image.Resampling.NEAREST,fillcolor=(218,218,218))
        im.paste(tile,(20,180))
        # Clip map actors/labels to the map viewport, including long buses.
        map_canvas=im.copy();draw=ImageDraw.Draw(map_canvas)
        def project(points):
            local=(np.asarray(points)-origin)@R
            return np.stack([500-local[...,1]*scale,490-local[...,0]*scale],-1)
        for slot,color in [(0,'#c39191'),(2,'#c6a675'),(12,'#7eaeb9')]:
            line=project(routes[slot])
            for k in range(0,len(line)-1,3):
                segment=line[k:k+2]
                if ((segment[:,0]>22)&(segment[:,0]<978)&(segment[:,1]>182)&(segment[:,1]<798)).all():draw.line([tuple(v) for v in segment],fill=color,width=2)
        def body(center,angle,size,color,label=None,collision=False):
            q=project(center)
            if not (35<q[0]<965 and 195<q[1]<785):return
            c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,s],[-s,c]])
            corners=project(np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*size/2@rot+center)
            draw.polygon([tuple(v) for v in corners],fill=color,outline='#b82d45' if collision else '#64748b',width=3 if collision else 1)
            if label:
                arrow=project(np.array([[-.2,0],[.35,0]])*size[0]@rot+center)
                draw.line([tuple(v) for v in arrow],fill='white',width=2);text((q[0]+10,q[1]-14),label,26,color)
        for i in np.flatnonzero(bg['valid'][index]):
            h=np.arctan2(bg['heading'][index,i,1],bg['heading'][index,i,0]);body(bg['centers'][index,i],h,bg['sizes'][i],'#b7c0ca')
        for i in [j for j in range(len(sizes)) if j not in (0,A,C)]+[A,C,0]:
            if not seen[index,i]:continue
            center=xy[index,i].copy()
            if i==0:center+=1.461*np.array([np.cos(yaw[index,i]),np.sin(yaw[index,i])])
            role='E' if i==0 else 'A' if i==A else 'C' if i==C else None
            ri={0:0,A:1,C:2}.get(i)
            collision=ri is not None and any(sep[index,ri,j]<0 for j in range(3) if j!=ri)
            body(center,yaw[index,i],sizes[i],{'E':'#d94a4a','A':'#dd921c','C':'#287bcd'}.get(role,'#c9d3dd'),role,collision)
        im.paste(map_canvas.crop((20,180,980,800)),(20,180));draw=ImageDraw.Draw(im)
        text((25,815),f'E/A 间隔 {sep[index,0,1]:.2f}m    E/C 间隔 {sep[index,0,2]:.2f}m',22)
        text((25,855),'负间隔表示车框重叠。E/C 未来未指定；A 未来给定。',20)
        text((25,897),'真实场景 033；这是轨迹俯视视频。',18,'#64748b')
        return np.asarray(im)

    videos={}
    for seed in sorted({r['seed'] for r in run['records']}):
        pairs=[(f'seed_{seed:03d}_pair',[f'new_{seed}_normal',f'new_{seed}_event'])]
        if seed==0:pairs += [('normal_before_after',['old_normal','new_0_normal']),('event_before_after',['old_event','new_0_event'])]
        for filename,names in pairs:
            if not all(name in sampled for name in names):continue
            path=report/'media'/f'{filename}.mp4'
            with imageio.get_writer(path,fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','19']) as writer:
                for i in range(81):
                    origin,scale=camera(names,i)
                    frame=np.concatenate([panel(name,i,origin,scale) for name in names],1);writer.append_data(frame)
                    if i in (0,15,25,50,80):imageio.imwrite(report/'media'/f'{filename}_{i:03d}.jpg',frame)
            with imageio.get_reader(path) as reader:info=reader.get_meta_data();count=sum(1 for _ in reader)
            assert count==81 and info['fps']==10;videos[filename]={'frames':81,'fps':10};print('KINEMATIC_VIDEO_COMPLETE',filename,flush=True)
    primary=all(r['acceptance']['primary_three_accepted'] for r in run['records'])
    speed_ok=all(r['acceptance']['speed_behavior']['accepted'] for r in run['records'])
    full=all(r['acceptance']['approved_for_downstream'] for r in run['records'])
    if full:verdict='主三车运动、速度行为与全场景检查通过；仍限当前场景及种子'
    elif primary and speed_ok:verdict='主三车运动与速度行为检查通过；其余参与者仍有问题，整场景未通过'
    elif primary:verdict='主三车几何及运动检查通过，速度行为仍有未通过项'
    else:verdict='新方案仍有未通过项，保留真实结果供诊断'
    speed_table='<table><tr><th>方案</th><th>E 起始区间速度</th><th>前2.5s最低速度</th><th>全程最低速度</th><th>末区间速度</th><th>8s通行距离</th><th>速度行为检查</th></tr>'
    for name in ['old_normal','new_0_normal','old_event','new_0_event']:
        vv=sampled[name][7];r=metrics[name]['speed_behavior']
        speed_table+=f'<tr><td>{labels[name]}</td><td>{vv[0]:.2f} m/s</td><td>{vv[:5].min():.2f} m/s</td><td>{vv.min():.2f} m/s</td><td>{vv[-1]:.2f} m/s</td><td>{r["forecast_progress_m"]:.1f} m</td><td>{"通过" if r["accepted"] else "未过"}</td></tr>'
    speed_table+='</table>'
    table='<table><tr><th>方案</th><th>E/A重叠</th><th>E/C重叠</th><th>正常E越出原车道</th><th>E最大加速度</th><th>E方向偏差</th><th>主三车／全场景</th></tr>'
    for name in ['old_normal','new_0_normal','old_event','new_0_event']:
        r=metrics[name];motion=r['motion']['E'];lane=r['original_lane']['E']['box_outside_original_lane_samples'] if r['variant']=='normal' else '允许合理换道'
        table+=f'<tr><td>{labels[name]}</td><td>{r["roles"]["EA"]["overlap_samples"]}</td><td>{r["roles"]["EC"]["overlap_samples"]}</td><td>{lane}</td><td>{motion["max_xy_acceleration_mps2"]:.2f} m/s²</td><td>{motion["max_heading_error_deg"]:.2f}°</td><td>{"通过" if r["primary_three_accepted"] else "未过"}／{"通过" if r["full_scene_accepted"] else "未过"}</td></tr>'
    table+='</table>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Nexus 运动约束优化</title><style>body{font-family:system-ui,sans-serif;max-width:1250px;margin:24px auto;padding:0 18px;color:#233247;background:#f5f7fa}section{background:white;border-radius:12px;padding:20px;margin:18px 0}video{width:100%}p{line-height:1.8}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #dde5ef}</style><h1>Nexus：正常巡航与风险后速度恢复</h1>'''
    page+=f'<section><b>{verdict}</b><p>本页是重新生成的真实数据结果。模型权重未变，改进作用在推理采样内部：保留运动一致性、跟驰与切入让行约束，增加来自当前观测的巡航速度软偏好及速度恢复目标，并使用当前车道的可行控制初始化和完整车身边界偏好。初始化只提供优化起点，E/C 的未来位置、速度与转向仍为自由变量。</p></section>'
    page+='<section><h2>新结果：左正常，右切入</h2><video controls playsinline preload="metadata" src="media/seed_000_pair.mp4"></video><p>0 秒是前两秒历史结束。A 在后续 0.5–2.5 秒向右切入；先看正常 E 是否仍无故持续减速，再看切入 E 的额外减速及后续恢复；仍检查后半段是否有侧滑。两侧相同视点与比例；车道边界来自地图，车身按观测尺寸绘制。</p><a href="media/seed_000_pair.mp4">直接打开 MP4</a></section>'
    page+='<section><h2>巡航改进前／后：左上一版，右本版</h2><video controls playsinline preload="metadata" src="media/event_before_after.mp4"></video><p><a href="media/normal_before_after.mp4">正常场景旧／新对比视频</a></p></section>'
    subprocess.run(['/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/envs/nexus/bin/python',
                    '-B',str(ROOT/'scripts/pipeline/plot_nexus_cruise_speed.py'),
                    '--stage',args.stage,'--output',str(report/'media/speed_comparison.png')],check=True,cwd=ROOT)
    page+='<section><h2>速度与通行效率</h2>'+speed_table+'<img src="media/speed_comparison.png" alt="上一版和本版的正常E、切入E及正常前车速度" style="width:100%">'
    page+=f'<p>速度从原生 0.5s 位置变化计算，是区间平均值。采样前声明的正常速度检查为：末速至少达到观测基准的 {final_fraction*100:.0f}%，全程最低速度至少 {minimum_fraction*100:.0f}%，通行距离至少 {progress_fraction*100:.0f}%；事件应有额外减速，末段速度至少恢复到观测自车／给定前车参考较低者的80%。这些仅是场景033的诊断门槛，不适用于需要停车等所有道路场景。</p></section>'
    page+='<section><h2>几何与运动独立检查</h2>'+table+'<p>重叠与车道检查：未来 0.1–8.0 秒，0.1 秒间隔，共 80 点；使用有向车框、原观测尺寸及自车后轴偏移。正常 E/C 必须完整留在原车道；事件中允许合理换道，但必须满足运动一致性、道路范围及间隔检查。加速度与方向偏差按实际位置变化计算。离散检查不是连续时间证明。</p></section>'
    if '0' in pair_metrics:
        p=pair_metrics['0']
        page+=f'<section><h2>响应范围</h2><p>改变 A 后，E 最大位置变化 {p["ego_max_position_change_m"]:.2f}m，C 最大位置变化 {p["C_max_position_change_m"]:.2f}m。正常预测 E 与给定事件 A 组合时，最小间隔 {p["normal_ego_plus_event_A_min_gap_m"]:.2f}m。该组合只用于离线核查威胁，没有作为 E 的未来条件输入。</p><p>这些是联合条件生成的差异，不证明 E 的变化造成 C 的变化，也不强制 E 一定向 C 所在车道闪躲。</p></section>'
    page+='<section><h2>实验范围与限制</h2><p>同一 nuPlan 场景 033，Las Vegas，token 320271cfc9ca5e86。旧／新使用相同历史、给定 A、有效性掩码和初始噪声。A 共用向前 2m 的编辑偏移；背景 97 个参与者显式记录回放，未来存在掩码来自记录。地图参考只用当前观测与路网，不用 E/C 真实未来。未训练模型，未修改原有升级文件，未在采样完成后修补输出轨迹。</p><p>主三车通过不代表其余车辆均通过，也不能直接证明 E 的响应导致 C 的响应。失败样本保留作诊断，不导出为通过验收的下游轨迹。新算法使用运动控制量优化与模型软先验，不声称具有原 OMEGA 的精确 KL 保证。</p><a href="results.json">完整结果</a> · <a href="protocol.json">实验协议</a></section></html>'
    (report/'index.html').write_text(page)
    result={'status':'complete','primary_pair_accepted':primary,'speed_behavior_pair_accepted':speed_ok,
            'primary_and_speed_pair_accepted':bool(primary and speed_ok),'full_scene_pair_accepted':full,
            'strict_causal_chain_proven':False,'metrics':metrics,'new_run_records':run['records'],
            'response':pair_metrics,
            'videos':videos,'existing_source_unchanged':True,'model_weights_unchanged':True}
    (report/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');shutil.copy2(stage/'protocol.json',report/'protocol.json')
    for ref in re.findall(r'(?:src|href)="([^"]+)"',page):assert (report/ref).is_file(),ref
    for file,h in run['protocol']['source_sha256'].items():assert digest(file)==h,file
    hashes={str(f.relative_to(report)):digest(f) for f in report.rglob('*') if f.is_file()}
    (report/'verification.json').write_text(json.dumps({'video_frames_fps_checked':True,'local_links_checked':True,'given_histories_inputs_noise_match_old':True,'asset_sha256':hashes},indent=2)+'\n')
    (stage/'report_completion.json').write_text(json.dumps({'status':'complete','report':str(report),'primary_pair_accepted':primary,
                                                         'speed_behavior_pair_accepted':speed_ok,'full_scene_pair_accepted':full,'videos':videos},indent=2)+'\n')
    print('KINEMATIC_REPORT_COMPLETE',report,flush=True)


if __name__=='__main__':main()
