"""Readable motion videos and matched B-transfer evidence for the chain pilot."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/nexus_chain_20261027';PURE=ROOT/'artifacts/nexus_chain_cutin_only_20261027'
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
 report=ROOT/'report/nexus_chain_20261027';report.mkdir(exist_ok=False);media=report/'media';media.mkdir()
 fontpath=ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf';fonts={s:ImageFont.truetype(str(fontpath),s) for s in (16,18,20,22,26)}
 times=np.arange(81)*.1;sections=[];summary=[];contacts=[];videos={};new_count=0
 for name,source in [('pure',PURE),('strong',BASE)]:
  stage=source/'pilot_seed0';run=json.loads((stage/'run.json').read_text());assert run['status']=='complete'
  for p,h in run['source_sha256'].items():assert digest(p)==h,p
  new_count+=sum(not r.get('reused_prediction',False) for r in run['records'])
  for result in run['cases']:
   idx=result['index'];case=source/f'case_{idx:03d}';info=json.loads((case/'preparation.json').read_text());A,B,C=[info['roles'][r] for r in ('A','B','C')];W=np.array(json.loads((case/'reference/protocol.json').read_text())['world_from_nexus'])
   sampled={};metrics={};payloads={}
   for row in [r for r in run['records'] if r['index']==idx]:
    variant=row['variant'];path=stage/row['path']/'prediction.pt';assert digest(path)==row['prediction_sha256'];p=torch.load(path,map_location='cpu',weights_only=False);payloads[variant]=p
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
    labels={'normal':'正常：B/C自由生成','event':'A只切入，B/C自由' if name=='pure' else 'A切入并减速，B/C自由','transfer_control':'诊断：固定B的正常轨迹','transfer_B_response':'诊断：固定B的反应轨迹'}
    text((20,12),labels[variant]+f'  {times[k]:.1f}s',26)
    for r,y,col in [('B',53,'#cf4141'),('C',88,'#287bcd')]:
     value=info['observed_speeds_mps'][r] if k==0 else speed[r][min(int(np.ceil(times[k]/.5))-1,15)];text((20,y),f'{r} 速度 {value:.1f} m/s ({value*3.6:.0f} km/h)',22,col)
    text((20,125),'橙A切入车；红B前车；蓝C紧跟后车',20)
    text((20,155),'A不切入；两侧仅B给定轨迹不同，C未来自由' if variant.startswith('transfer') else 'A在0.5–3秒切入；0秒为前2秒历史结束',18)
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
   b=result['B_response'];c=result['C_response'];t=result['C_response_to_B_only_transfer'];pressure=result['event_B_plus_normal_C'];ab=result['event_A_plus_normal_B'];main_ok=result['main_primary_pair_accepted'];diag_ok=result['transfer_primary_pair_accepted'];transfer_support=bool(main_ok and diag_ok and b['maximum_speed_reduction_mps']>=.75 and c['maximum_speed_reduction_mps']>=.75 and t['maximum_speed_reduction_mps']>=.5 and b['maximum_position_difference_m']>=1 and c['maximum_position_difference_m']>=1 and t['maximum_position_difference_m']>=1)
   support=bool(transfer_support and pressure['overlap_samples']>0)
   result={**result,'following_brake_transfer_supported':transfer_support,'following_brake_chain_strong_supported':support}
   title='主例：A只切入，B、C自由响应' if name=='pure' else '加强测试：A切入后继续减速'
   verdict=('观测到跟驰减速传导，并通过更保守的碰撞反事实检查' if support else '观测到跟驰减速，B轨迹对照支持传导；未触发碰撞反事实' if transfer_support else '当前样本尚未满足完整传播判据，保留真实结果')
   table='<table><tr><th>检查</th><th>结果</th></tr>'
   for label,value in [('主实验正常/事件三车物理检查','通过' if main_ok else '未过'),('实际事件视频的B/C车框重叠',str(metrics['event']['roles']['BC']['overlap_samples'])+'点'),('B最大额外减速',f'{b["maximum_speed_reduction_mps"]:.2f} m/s'),('C最大额外减速',f'{c["maximum_speed_reduction_mps"]:.2f} m/s'),('仅换B轨迹时C最大额外减速',f'{t["maximum_speed_reduction_mps"]:.2f} m/s'),('假设C不响应的离线拼接重叠（非生成视频）',f'{pressure["overlap_samples"]}/80点'),('C保持正常轨迹时最小B/C间隔',f'{pressure["minimum_signed_gap_m"]:.2f}m'),('实际事件生成的最小B/C间隔',f'{metrics["event"]["roles"]["BC"]["minimum_signed_gap_m"]:.2f}m'),('B固定诊断两组的三车物理检查','通过' if diag_ok else '未过')]:table+=f'<tr><td>{label}</td><td>{value}</td></tr>'
   table+='</table>'
   sections.append(f'<section><h2>{title}</h2><b>{verdict}</b><p>左正常，右事件。此处B/C都是普通车辆，原自车仍自由生成。B/C观测速度 {info["observed_speeds_mps"]["B"]:.2f}/{info["observed_speeds_mps"]["C"]:.2f} m/s，C紧跟在B后方。主实验只给A未来，不给B/C未来。</p><video controls preload="metadata" src="media/{name}_case_{idx:03d}_main.mp4"></video><p><a href="media/{name}_case_{idx:03d}_main.mp4">直接打开主运动视频</a></p>{table}<details open><summary>验证B→C：让A在两侧都正常行驶，只换B的给定轨迹</summary><p>左侧固定本次Nexus生成的B正常轨迹，右侧固定本次生成的B事件反应轨迹；两侧A完全相同、C未来始终自由，条件掩码和初始噪声相同。C若仍出现明显变化，说明B的轨迹变化本身足以在生成器中引发C响应。B固定仅用于诊断，不属于主实验。</p><video controls preload="metadata" src="media/{name}_case_{idx:03d}_B_transfer.mp4"></video></details><img src="media/{name}_{idx:03d}_speeds.png"><p>离线压力检查：事件A+正常B重叠{ab["overlap_samples"]}点；事件B+正常C重叠{pressure["overlap_samples"]}点。这些拼接只用于核查不响应的后果，重叠点数不属于实际生成视频，没有作为B/C的主实验未来条件。</p></section>')
   summary.append({'kind':name,'index':idx,**result})
  write(report/f'{name}_results.json',run)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Nexus 多车传播 A→B→C</title><style>body{font-family:system-ui;max-width:1250px;margin:24px auto;padding:0 18px;background:#f5f7fa;color:#24364b}section{padding:22px;margin:20px 0;background:white;border-radius:12px}video,img{width:100%}p{line-height:1.8}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}.flow{display:flex;align-items:center;justify-content:center;gap:18px;font-size:24px;padding:24px}.flow span{padding:12px;background:#eef3f8;border-radius:8px}summary{font-size:20px;cursor:pointer}</style><h1>研究 A 切入 → B 减速 → C 跟随减速</h1><section><p>先看下面主视频：橙色A从旁边切入，红色B在前，蓝色C同车道紧跟在后。B/C未来由Nexus与现有运动交互约束联合生成。只切入的主例和额外减速的加强测试分开展示。</p><div class="flow"><span>A 切入</span>→<span>B 减速</span>→<span>C 跟随减速</span></div><p>这是一条待验证的传播关系；不能仅凭三车同时运动就下结论，所以增加保持A正常的B轨迹对照。</p></section>'''
 page+=''.join(sections)
 page+='<section><h2>结果范围</h2><p>本轮选用原033场景中的另一组三辆真实车辆：A槽11、B槽4、C槽14；与此前自车两车示例的角色不同。权重及V4优化参数相同，seed0，历史2秒、预测8秒。原自车及其他已观测车辆也自由生成。背景与存在掩码来自记录，地图与观测用于输入；没有把B/C真实未来作为条件。</p><p>B固定对照只说明在该生成器和条件下，B反应轨迹是否足以引发C变化；不能证明唯一传播路径，也不能当成真实世界在线反应延迟。模型从开始就知道给定A未来，可能提前减速。所有物理检查为离散采样，全场景失败仍保留，不导出为通过验收的下游场景。</p><p>其余筛选候选及地图/固定背景输入失败原因记录在实验协议中。没有修改旧升级实现，没有采样后修补轨迹。</p><a href="summary.json">汇总指标</a> · <a href="implementation.md">实验说明</a></section></html>'
 (report/'index.html').write_text(page);write(report/'summary.json',{'cases':summary,'new_inference_samples':new_count,'unique_scene_count':len({r['index'] for r in summary}),'videos':videos,'strict_causal_chain_proven':False})
 if contacts:
  im=Image.new('RGB',(1200,690*len(contacts)),'white')
  for i,c in enumerate(contacts):im.paste(c,(0,690*i))
  im.save(report/'contact_sheet.jpg')
 print('CHAIN_REPORT_COMPLETE','new_samples',new_count,'videos',len(videos),flush=True)
if __name__=='__main__':main()
