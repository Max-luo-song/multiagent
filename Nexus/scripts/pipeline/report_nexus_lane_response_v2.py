"""Motion videos for free C lane-choice experiments and B-transfer probes."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/nexus_lane_response_v2_20261027'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 import numpy as np,torch,imageio.v2 as imageio
 from PIL import Image,ImageDraw,ImageFont
 from shapely import wkb
 from audit_nexus_omega_v2 import sample_positions
 from nexus_3dgs.bundle import decode
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 report=ROOT/'report/nexus_lane_response_v2_20261027';report.mkdir(exist_ok=False);media=report/'media';media.mkdir()
 fontpath=ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf';fonts={s:ImageFont.truetype(str(fontpath),s) for s in (16,18,20,22,26)}
 times=np.arange(81)*.1;sections=[];summary=[];contacts=[];videos={};new_count=0
 for name,source in [('choice',BASE)]:
  stage=source/'pilot_seed0';run=json.loads((stage/'run.json').read_text());assert run['status']=='complete'
  for p,h in run['source_sha256'].items():assert digest(p)==h,p
  new_count+=sum(not r.get('reused_prediction',False) for r in run['records'])
  for result in run['cases']:
   idx=result['index'];case=source/f'case_{idx:03d}';info=json.loads((case/'preparation.json').read_text());A,B,C=[info['roles'][r] for r in ('A','B','C')];W=np.array(json.loads((case/'reference/protocol.json').read_text())['world_from_nexus'])
   sampled={};metrics={};payloads={};modes={}
   for row in [r for r in run['records'] if r['index']==idx]:
    variant=row['variant'];modes[variant]=json.loads((stage/row['path']/'lane_choice.json').read_text())[0]['selected'];path=stage/row['path']/'prediction.pt';assert digest(path)==row['prediction_sha256'];p=torch.load(path,map_location='cpu',weights_only=False);payloads[variant]=p
    states=decode(p['sampled_tensor'][0].numpy());valid=p['valid_mask'][0].bool().numpy();active=np.flatnonzero(valid.any(-1));s=states[active];v=valid[active];xy,yaw,seen=sample_positions(s,v,times,np);gt=decode(p['scene_tensor_gt'][0].numpy())[active]
    speed={r:np.linalg.norm(np.diff(states[i,4:,:2],axis=0)/.5,axis=-1) for r,i in [('B',B),('C',C)]};sampled[variant]=(xy.transpose(1,0,2),yaw.T,seen.T,gt[:,4,6:8],{r:list(active).index(i) for r,i in [('A',A),('B',B),('C',C)]},active,speed);metrics[variant]=row['acceptance']
   with np.load(case/'reference/inputs/background_replay.npz') as z:bg={k:z[k] for k in z.files}
   with np.load(case/'reference/inputs/drivable_sdf.npz') as z:sdf={k:z[k] for k in z.files}
   h=decode(payloads['normal']['sampled_tensor'][0].numpy())[B,4,2:4];ang=np.arctan2(h[1],h[0]);R=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
   road=Image.fromarray(np.where(sdf['sdf']>=0,249,215).astype(np.uint8)).convert('RGB');draw=ImageDraw.Draw(road)
   for i,col in [(A,'#fff0d9'),(B,'#fff0f0'),(C,'#e8f1ff')]:
    corridor=wkb.loads((case/'map/lanes'/f'{i}.wkb').read_bytes())
    for p in (corridor.geoms if hasattr(corridor,'geoms') else [corridor]):
     q=((np.array(p.exterior.coords)-W[:2,3])@W[:2,:2]-sdf['origin'])/float(sdf['resolution']);draw.polygon([tuple(v) for v in q],fill=col,outline='#b7c4cf')
   def panel(variant,k,origin,scale):
    xy,yaw,seen,sizes,roles,active,speed=sampled[variant];im=Image.new('RGB',(800,920),'#f6f8fb');d=ImageDraw.Draw(im)
    def text(pos,value,size=22,color='#24364b'):d.text(pos,value,font=fonts[size],fill=color)
    labels={'normal':'正常：B/C自由生成','event':'A切入，B/C自由选择响应','transfer_control':'诊断：固定B的正常轨迹','transfer_B_response':'诊断：固定B的反应轨迹'}
    text((20,12),labels[variant]+f'  {times[k]:.1f}s',26)
    for r,y,col in [('B',53,'#cf4141'),('C',88,'#287bcd')]:
     value=info['observed_speeds_mps'][r] if k==0 else speed[r][min(int(np.ceil(times[k]/.5))-1,15)];text((20,y),f'{r} 速度 {value:.1f} m/s ({value*3.6:.0f} km/h)',22,col)
    text((20,125),'橙A；红B；蓝C | C候选：'+('保持原车道' if modes[variant]=='keep' else '进入相邻车道'),20)
    text((20,155),'A不切入；两侧仅B给定轨迹不同，C未来自由' if variant.startswith('transfer') else ('A保持原车道；0秒为前2秒历史结束' if variant=='normal' else 'A在0.5–3秒切入；0秒为前2秒历史结束'),18)
    res=float(sdf['resolution']);base=sdf['origin'];cx=380;cy=310
    coeff=(-R[0,1]/scale/res,-R[0,0]/scale/res,(origin[0]+cy/scale*R[0,0]+cx/scale*R[0,1]-base[0])/res,-R[1,1]/scale/res,-R[1,0]/scale/res,(origin[1]+cy/scale*R[1,0]+cx/scale*R[1,1]-base[1])/res)
    tile=road.transform((760,600),Image.Transform.AFFINE,coeff,resample=Image.Resampling.NEAREST,fillcolor=(215,215,215));im.paste(tile,(20,195));canvas=im.copy();dmap=ImageDraw.Draw(canvas)
    def project(p):
     q=(np.asarray(p)-origin)@R;return np.stack([400-q[...,1]*scale,505-q[...,0]*scale],-1)
    def body(center,angle,size,color,label=None):
     q=project(center)
     if not(5<q[0]<795 and 180<q[1]<810):return
     rot=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]);poly=project(np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*size/2@rot+center);dmap.polygon([tuple(v) for v in poly],fill=color,outline='#6d7b8c')
     if label:dmap.text((q[0]+10,q[1]-14),label,font=fonts[26],fill=color)
    for j in np.flatnonzero(bg['valid'][k]):body(bg['centers'][k,j],np.arctan2(bg['heading'][k,j,1],bg['heading'][k,j,0]),bg['sizes'][j],'#b3bcc7')
    ri=list(roles.values())
    for j in [j for j in range(len(active)) if j not in ri]+ri:
     if not seen[k,j]:continue
     center=xy[k,j].copy()
     if active[j]==0:center+=1.461*np.array([np.cos(yaw[k,j]),np.sin(yaw[k,j])])
     label=next((r for r,i in roles.items() if i==j),None);body(center,yaw[k,j],sizes[j],{'A':'#dd921c','B':'#d94a4a','C':'#287bcd'}.get(label,'#c6cfd8'),label)
    im.paste(canvas.crop((20,195,780,795)),(20,195));d=ImageDraw.Draw(im)
    accepted=metrics[variant];text((20,815),'主三车物理检查：'+('通过' if accepted['primary_three_accepted'] else '未通过'),20)
    text((20,847),'全场景检查：'+('通过' if accepted['full_scene_accepted'] else '未通过')+'；其他自由车辆为灰色',18)
    text((20,881),'同一视点与比例；无采样后轨迹修补',18)
    return np.asarray(im)
   for tag,variants in [('main',['normal','event']),('B_transfer',['transfer_control','transfer_B_response'])]:
    filename=f'{name}_case_{idx:03d}_{tag}.mp4';path=media/filename
    with imageio.get_writer(path,fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','19']) as writer:
     for k in range(81):
      anchors=[]
      for v in variants:xy,_,_,_,roles,_,_=sampled[v];anchors.extend(xy[k,list(roles.values())])
      origin=np.mean(anchors,axis=0);q=(np.array(anchors)-origin)@R;scale=min(10.,520/(np.ptp(q[:,0])+22),690/(np.ptp(q[:,1])+18));frame=np.concatenate([panel(v,k,origin,scale) for v in variants],1);writer.append_data(frame)
      if k in (0,20,40,60,80):imageio.imwrite(media/f'{name}_{idx:03d}_{tag}_{k:03d}.jpg',frame)
      if tag=='main' and k==40:contacts.append(Image.fromarray(frame).resize((1200,690)))
    with imageio.get_reader(path) as reader:meta=reader.get_meta_data();frames=sum(1 for _ in reader)
    assert frames==81 and meta['fps']==10;videos[filename]={'frames':frames,'fps':10,'sha256':digest(path)};print('CHAIN_VIDEO_COMPLETE',filename,flush=True)
   fig,axes=plt.subplots(2,2,figsize=(12,6),sharex=True);tt=np.arange(16)*.5+.25
   for row,r,col in [(0,'B','#d94a4a'),(1,'C','#287bcd')]:
    for colidx,(v0,v1,lab0,lab1) in enumerate([('normal','event','Normal A','A cut-in'),('transfer_control','transfer_B_response','Normal B replay','Reacting B replay')]):
     ax=axes[row,colidx];ax.plot(tt,sampled[v0][-1][r],ls='--',color='#7b8794',label=lab0);ax.plot(tt,sampled[v1][-1][r],color=col,marker='.',label=lab1);ax.set_ylabel(f'{r} speed (m/s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
   for ax in axes[-1]:ax.set_xlabel('Future time (s)')
   axes[0,0].set_title('Joint generation: B and C free');axes[0,1].set_title('B-transfer diagnostic: A held normal, C free');fig.tight_layout();fig.savefig(media/f'{name}_{idx:03d}_speeds.png',dpi=150);plt.close(fig)
   audit=json.loads((BASE/'lane_response_audit.json').read_text());lane=next(x for x in audit['cases'] if x['index']==idx)
   fig,axes=plt.subplots(1,2,figsize=(12,3.6),sharey=True)
   for ax,(v0,v1) in zip(axes,[('normal','event'),('transfer_control','transfer_B_response')]):
    for v,color in [(v0,'#7b8794'),(v1,'#287bcd')]:ax.plot(times,lane['variants'][v]['C_lateral_from_source_m'],label=v,color=color)
    ax.set_xlabel('Future time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
   axes[0].set_ylabel('C lateral distance from source route (m)');axes[0].set_title('Main: normal vs cut-in');axes[1].set_title('B-transfer diagnostic');fig.tight_layout();fig.savefig(media/f'{name}_{idx:03d}_C_lateral.png',dpi=150);plt.close(fig)
   def yn(x):return '通过' if x else '未通过'
   table='<table><tr><th>检查</th><th>结果</th></tr>'
   rows=[('主实验预先声明的行为判据',yn(lane['main_behavior_criteria_passed'])),('B最大额外减速',f"{lane['B_max_additional_speed_reduction_mps']:.2f} m/s"),('正常组C完整车身出原车道样本',lane['variants']['normal']['C_source_lane_body_outside_samples']),('事件组C在相邻车道持续至少1秒',yn(lane['variants']['event']['C_target_lane_held_at_least_1s'])),('事件组C首次满足1秒保持的起点',str(lane['variants']['event']['C_target_lane_1s_hold_start_s'])+'s'),('事件组C终点在相邻车道',yn(lane['variants']['event']['C_final_in_target_lane'])),('事件组C超出两车道许可区域样本',lane['variants']['event']['C_union_body_outside_samples']),('事件组跨越车道时位于路口/初始车道对之外样本',lane['variants']['event']['transition_samples_in_connector_or_outside_initial_lane_pair']),('B轨迹替换的变道传播判据',yn(lane.get('B_transfer_lane_choice_criteria_passed',False)))]
   for label,value in rows:table+=f'<tr><td>{label}</td><td>{value}</td></tr>'
   table+='</table>'
   sections.append(f'<section><h2>真实场景 {idx:03d}：A切入，B/C自由响应</h2><p>先看主视频：左正常、右切入。两侧只指定A未来，B/C未来自由。橙色A、红色B、蓝色C；C可以留在原车道，也可以选择同向相邻车道。A两侧使用相同的纵向减速停车安排：从当前观测到的前方停车车辆估算停止距离，两侧约4.14秒开始减速。事件侧另外在0.5–3秒切入。0秒为2秒历史结束，之后展示8秒生成。所有失败均保留。</p><video controls preload="metadata" src="media/{name}_case_{idx:03d}_main.mp4"></video><p><a href="media/{name}_case_{idx:03d}_main.mp4">直接打开主视频</a></p>{table}<img src="media/{name}_{idx:03d}_C_lateral.png"><details><summary>B轨迹对照：A始终正常，仅改变给定B轨迹，C仍自由</summary><p>左侧B为主实验正常轨迹，右侧B为主实验反应轨迹。B固定只用于诊断。C两侧有相同地图选择、初始噪声与候选控制；相邻车道若被A或其他车占用，C可能合理选择继续跟随。</p><video controls preload="metadata" src="media/{name}_case_{idx:03d}_B_transfer.mp4"></video></details><img src="media/{name}_{idx:03d}_speeds.png"></section>')
   summary.append({'index':idx,'lane_response_audit':lane,'chain_metrics':result})
  write(report/'results.json',run)
 page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Nexus：B减速、C选择变道</title><style>body{font-family:system-ui;max-width:1250px;margin:24px auto;padding:0 18px;background:#f5f7fa;color:#24364b}section{padding:22px;margin:20px 0;background:white;border-radius:12px}video,img{width:100%}p{line-height:1.8}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}summary{font-size:20px;cursor:pointer}</style><h1>A切入 → B减速 → C选择跟随或变道</h1>'
 page+=''.join(sections)
 page+='<section><h2>结果范围</h2><p>本次新增显式的地图变道候选搜索：每组都有原车道和相邻车道的相同候选初值，先形成共享的其他车辆响应，再分别优化C的保持/完成变道空间模式，按共同的碰撞、运动、巡航及变道代价选择；选中的模式作为软空间先验，在后续扩散采样内继续优化自由控制量。C的空间模式是新增引导的一部分；没有给C固定未来航点、速度或轨迹条件，没有采样后修补。该能力属于Nexus与新增战术搜索/运动引导的组合，不能称为裸Nexus原生学会变道。</p><p>完整车身、道路及碰撞检查为0.1秒离散检查。另检验变道跨线阶段是否仍位于初始连续车道对，不能将路口偏移当成合格变道。尚未核实边界标线的可跨越属性，也不构成完整交通法规验证。全场景其他参与者的失败仍保留。权重不更新；已知A完整未来可能引发提前响应，不能解释为在线观察后的反应时间。</p><a href="summary.json">汇总</a> · <a href="implementation.md">实现与结果说明</a> · <a href="lane_response_audit.json">独立变道判据</a></section></html>'
 (report/'index.html').write_text(page);write(report/'summary.json',{'cases':summary,'new_inference_samples':new_count,'unique_scene_count':len({r['index'] for r in summary}),'videos':videos,'strict_causal_chain_proven':False,'native_learned_lane_choice_proven':False})
 if contacts:
  im=Image.new('RGB',(1200,690*len(contacts)),'white')
  for i,c in enumerate(contacts):im.paste(c,(0,690*i))
  im.save(report/'contact_sheet.jpg')
 print('LANE_REPORT_COMPLETE','new_samples',new_count,'videos',len(videos),flush=True)
if __name__=='__main__':main()
