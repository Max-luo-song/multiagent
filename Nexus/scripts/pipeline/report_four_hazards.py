"""Motion videos from sampled Nexus trajectories, with an explicit schematic camera."""
import json,os,sys,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_four_hazards_20261110';REPORT=ROOT/'report/nexus_four_hazards_20261110'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
os.environ['MPLBACKEND']='Agg'

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['brake','reveal','rear','oncoming']);parser.add_argument('--finalize-only',action='store_true');args=parser.parse_args()
 import numpy as np,torch,imageio.v2 as imageio
 from PIL import Image,ImageDraw,ImageFont
 from shapely import wkb
 from shapely.geometry import Polygon,box
 from nexus_3dgs.bundle import decode
 from audit_nexus_omega_v2 import sample_positions
 run=json.loads((OUT/'frozen_v4_seed0/run.json').read_text());assert run['status']=='complete' or args.kind and any(p['kind']==args.kind for p in run['pairs'])
 REPORT.mkdir(parents=True,exist_ok=True);fontpath=ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'
 fonts={s:ImageFont.truetype(str(fontpath),s) for s in [16,18,20,22,24,28,32]}
 colors={'E':(60,208,255),'A':(255,166,65),'S':(196,143,255)}
 times=np.arange(81)*.1;records=[]
 def text(draw,xy,label,size=22,fill='#e5edf8'):draw.text(xy,label,font=fonts[size],fill=fill)
 def parts(geometry):
  if geometry.is_empty:return []
  return [geometry] if geometry.geom_type=='Polygon' else [p for g in geometry.geoms for p in parts(g)] if hasattr(geometry,'geoms') else []
 def lines(geometry):
  if geometry.is_empty:return []
  if geometry.geom_type in ['LineString','LinearRing']:return [np.array(geometry.coords)]
  if geometry.geom_type=='Polygon':return [np.array(geometry.exterior.coords)]+[np.array(r.coords) for r in geometry.interiors]
  return [p for g in geometry.geoms for p in lines(g)] if hasattr(geometry,'geoms') else []
 def corners(center,yaw,size):
  c,s=np.cos(yaw),np.sin(yaw);return np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*size/2@np.array([[c,s],[-s,c]])+center
 def writer(path):return imageio.get_writer(str(path),fps=10,codec='libx264',quality=8,macro_block_size=2,ffmpeg_log_level='error',ffmpeg_params=['-preset','veryfast','-threads','2'])
 for info in run['protocol']['cases']:
  if args.kind and info['kind']!=args.kind:continue
  if args.finalize_only:
   records.append(json.loads((REPORT/f"{info['kind']}_summary.json").read_text()));continue
  kind=info['kind'];case=OUT/info['case'];stage=OUT/'frozen_v4_seed0'/kind;active=info['active_slots'];roles=info['roles'];role_indices={r:active.index(s) for r,s in roles.items()}
  samples={};audits={};states={}
  for variant in ['normal','event']:
   p=torch.load(stage/variant/'prediction.pt',map_location='cpu',weights_only=False);state=decode(p['sampled_tensor'][0].numpy())[active];valid=p['valid_mask'][0].numpy().astype(bool)[active]
   xy,yaw,seen=sample_positions(state,valid,times,np);center=xy.copy();center[0]+=1.461*np.c_[np.cos(yaw[0]),np.sin(yaw[0])]
   speed=np.linalg.norm(np.diff(state[:,:,:2],axis=1)/.5,axis=-1);speed=np.stack([np.interp(times,np.arange(17)*.5,np.r_[info['observed_speeds_mps']['E'] if i==0 else np.linalg.norm(state[i,4,4:6]),speed[i,4:]]) for i in range(len(active))])
   samples[variant]={'xy':xy,'center':center,'yaw':yaw,'speed':speed};states[variant]=state;audits[variant]=json.loads((stage/variant/'acceptance.json').read_text())
  sizes=states['event'][:,4,6:8];W=np.array(info['world_from_nexus']);R=W[:2,:2];offset=W[:2,3]
  road=wkb.loads((case/'reference/inputs/drivable.wkb').read_bytes())
  points=np.concatenate([s['center'].reshape(-1,2) for s in samples.values()]);low=points.min(0)-35;high=points.max(0)+35
  clip=Polygon(np.array([[low[0],low[1]],[low[0],high[1]],[high[0],high[1]],[high[0],low[1]]])@R.T+offset)
  roadpolys=[(np.array(p.exterior.coords)-offset)@R for p in parts(road.intersection(clip))]
  lanes=[]
  for f in ['initial_E_lane.wkb','initial_A_lane.wkb','target_A_lane.wkb']:
   if (case/'map'/f).exists():lanes.extend([(p-offset)@R for p in lines(wkb.loads((case/'map'/f).read_bytes()))])
  h=states['event'][0,4,2:4];h=h/np.linalg.norm(h);side=np.array([-h[1],h[0]])
  # Use the same camera and metric scale for both branches at each instant.
  cams=[];ranges=[]
  for t in range(81):
   pts=np.concatenate([samples[v]['center'][:,t] for v in ['normal','event']]);p=np.c_[pts@side,pts@h]
   mid=(p.min(0)+p.max(0))/2;cams.append(mid[0]*side+mid[1]*h);ranges.append(p.max(0)-p.min(0))
  scale=min(11.,(720-80)/max(35,max(r[0] for r in ranges)+14),(650-80)/max(42,max(r[1] for r in ranges)+15))
  def bev(variant,t,width=760,height=650):
   im=Image.new('RGB',(width,height),'#14202a');d=ImageDraw.Draw(im);s=samples[variant];cam=cams[t]
   def project(p):
    z=p-cam;return np.c_[width/2-z@side*scale,height/2-z@h*scale]
   for poly in roadpolys:d.polygon([tuple(q) for q in project(poly)],fill='#394450')
   for line in lanes:d.line([tuple(q) for q in project(line)],fill='#9babb7',width=2)
   for r,i in role_indices.items():
    # Only the observed past trail is shown; no future path is overlaid.
    trail=project(s['center'][i,max(0,t-20):t+1]);
    if len(trail)>1:d.line([tuple(q) for q in trail],fill=colors[r],width=2)
    poly=project(corners(s['center'][i,t],s['yaw'][i,t],sizes[i]));d.polygon([tuple(q) for q in poly],fill=colors[r],outline='#ffffff')
    cen=project(s['center'][i,t:t+1])[0];hh=np.array([np.cos(s['yaw'][i,t]),np.sin(s['yaw'][i,t])]);nose=project((s['center'][i,t]+hh*sizes[i,0]*.35)[None])[0];d.line([tuple(cen),tuple(nose)],fill='#14202a',width=3)
    text(d,(int(cen[0]+12),int(cen[1]-12)),r,20)
   d.rectangle([14,height-44,174,height-14],fill='#14202a');d.line([(28,height-25),(28+10*scale,height-25)],fill='white',width=3);text(d,(35+10*scale,height-40),'10m',16)
   text(d,(16,12),'俯视 · 上方为初始行驶方向',18)
   return im
  def camera(t,width=760,height=650):
   im=Image.new('RGB',(width,height),'#8cabbf');d=ImageDraw.Draw(im);cy=height*.38;d.rectangle([0,cy,width,height],fill='#3d4248');s=samples['event'];rear=kind=='rear';eh=np.array([np.cos(s['yaw'][0,t]),np.sin(s['yaw'][0,t])]);forward=-eh if rear else eh;left=np.array([-forward[1],forward[0]])
   eye=s['xy'][0,t]+((-1.0 if rear else 1.461+sizes[0,0]/2-.5)*eh);fx=width*.49;cameraheight=1.45
   def coords(q):
    delta=q[:,:2]-eye;return np.c_[-delta@left,delta@forward,q[:,2]]
   def screen(q):return np.c_[width/2+fx*q[:,0]/q[:,1],cy+fx*(cameraheight-q[:,2])/q[:,1]]
   def clipnear(q):
    output=[]
    for a,b in zip(q,np.roll(q,-1,axis=0)):
     insidea=a[1]>=.35;insideb=b[1]>=.35
     if insidea:output.append(a)
     if insidea!=insideb:output.append(a+(b-a)*(.35-a[1])/(b[1]-a[1]))
    return np.array(output)
   for poly in roadpolys:
    q=clipnear(coords(np.c_[poly,np.zeros(len(poly))]))
    if len(q)>=3:d.polygon([tuple(v) for v in screen(q)],fill='#515b63')
   for segment in [np.array([a,b]) for line in lanes for a,b in zip(line[:-1],line[1:])]:
    q=coords(np.c_[segment,np.zeros(2)]);a,b=q
    if max(q[:,1])<.35:continue
    if a[1]<.35:a=a+(b-a)*(.35-a[1])/(b[1]-a[1])
    if b[1]<.35:b=b+(a-b)*(.35-b[1])/(a[1]-b[1])
    d.line([tuple(v) for v in screen(np.array([a,b]))],fill='#c1c7c6',width=2)
   faces=[]
   for r,i in role_indices.items():
    if r=='E':continue
    base=corners(s['center'][i,t],s['yaw'][i,t],sizes[i]);heightcar=1.65 if r=='A' else 1.5;verts=np.r_[np.c_[base,np.zeros(4)],np.c_[base,np.full(4,heightcar)]]
    for j,idx in enumerate([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[4,5,6,7]]):
     q=clipnear(coords(verts[idx]));
     if len(q)<3:continue
     color=tuple(int(c*(.66+.07*j)) for c in colors[r]);faces.append((q[:,1].mean(),q,color))
    braking=t>0 and s['speed'][i,t]<s['speed'][i,t-1]-.03 or s['speed'][i,t]<.15
    if r=='A' and braking:
     ah=np.array([np.cos(s['yaw'][i,t]),np.sin(s['yaw'][i,t])]);aside=np.array([-ah[1],ah[0]]);rearface=s['center'][i,t]-ah*(sizes[i,0]/2+.025)
     for u in [-.34,.34]:
      lamp=rearface+u*sizes[i,1]*aside;points=np.array([np.r_[lamp-.14*aside,.45],np.r_[lamp+.14*aside,.45],np.r_[lamp+.14*aside,.65],np.r_[lamp-.14*aside,.65]]);q=clipnear(coords(points))
      if len(q)>=3:faces.append((q[:,1].mean(),q,(255,45,40)))
   for _,q,color in sorted(faces,key=lambda x:-x[0]):d.polygon([tuple(v) for v in screen(q)],fill=color,outline='#202b33')
   # A simple dashboard communicates viewpoint without drawing labels on the scene.
   d.polygon([(0,height),(0,height-55),(width*.3,height-80),(width*.7,height-80),(width,height-55),(width,height)],fill='#14202a')
   text(d,(18,18),'E车后视示意' if rear else 'E车前视示意',24,fill='#10202c');text(d,(18,52),'几何示意，非3DGS渲染',18,fill='#10202c')
   return im
  def status(t):
   time=times[t]
   if kind=='brake':return '前车正常行驶' if time<info['parameters']['brake_start_s'] else '前车紧急制动 → 观察E车制动'
   if kind=='reveal':
    f=audits['event']['hazard']['static_S_blocked_fraction_0p1s'][t]
    return f'静止车被遮挡 {f*100:.0f}% · 前车'+('尚未变道' if time<.5 else '正在变道' if time<2.5 else '已完成变道')
   if kind=='rear':return '后车高速逼近'+(' · 仍保持高速' if time<info['parameters']['event_A_brake_start_s'] else ' · 后车开始制动')
   return '对向车'+('正常行驶' if time<.5 else '正在越过中心线' if time<2 else '侵入本车道' if time<info['parameters']['return_start_s'] else '正在返回' if time<info['parameters']['return_end_s'] else '已返回对向车道')
  eventfile=REPORT/f'{kind}_motion.mp4';comparefile=REPORT/f'{kind}_comparison.mp4'
  with writer(eventfile) as we,writer(comparefile) as wc:
   for t in range(81):
    frame=Image.new('RGB',(1600,900),'#0c1420');d=ImageDraw.Draw(frame)
    text(d,(30,15),info['title'],32);text(d,(1250,24),f't = {times[t]:.1f}s  |  0.5×速度',22)
    text(d,(30,65),status(t),24,fill='#ffd490');frame.paste(bev('event',t),(25,110));frame.paste(camera(t),(815,110))
    legend='    '.join(f'{r}：'+('自车' if r=='E' else '事件车' if r=='A' else '静止车')+f" {samples['event']['speed'][i,t]*3.6:.1f} km/h" for r,i in role_indices.items());text(d,(30,780),legend,24)
    text(d,(30,825),'E未来由Nexus＋物理引导生成；A未来为指定事件；仅保留角色车辆。',20)
    text(d,(30,857),'右侧为轨迹的几何投影。模型已知指定事件全过程；本片不验证感知延迟或在线闭环反应。',18,fill='#9daec0')
    comp=Image.new('RGB',(1600,900),'#0c1420');cd=ImageDraw.Draw(comp);text(cd,(30,15),info['title']+' · 同初始条件对照',28);text(cd,(1250,24),f't = {times[t]:.1f}s',22)
    text(cd,(30,70),'正常对照',24);text(cd,(820,70),'危险事件',24,fill='#ffd490');comp.paste(bev('normal',t),(25,110));comp.paste(bev('event',t),(815,110))
    for variant,x in [('normal',30),('event',820)]:text(cd,(x,780),f"E车速度 {samples[variant]['speed'][0,t]*3.6:.1f} km/h",24)
    text(cd,(30,830),'同历史、同随机种子、同物理引导；区别是A车的指定未来。0.5×播放。',20)
    for _ in range(2):we.append_data(np.asarray(frame));wc.append_data(np.asarray(comp))
    if t in [0,15,25,40,60]:frame.save(REPORT/f'{kind}_t{t:02d}.jpg',quality=90)
  pair=next(p for p in run['pairs'] if p['kind']==kind)
  row={'kind':kind,'title':info['title'],'experiment_root':str(OUT),'event_parameters':info['parameters'],'scenario_index':info['scenario']['index'],'map':info['scenario']['map_name'],'token':info['scenario']['token'],'event_video':eventfile.name,'comparison_video':comparefile.name,'event_accepted':audits['event']['accepted'],'normal_accepted':audits['normal']['accepted'],'event_audit':audits['event'],'normal_audit':audits['normal'],'comparison':pair};records.append(row)
  (REPORT/f'{kind}_summary.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');print('VIDEO_COMPLETE',kind,'event_accepted',row['event_accepted'],flush=True)
 if args.kind:return
 # One synchronized overview makes all four requested clips easy to find.
 order=['brake','reveal','rear','oncoming'];readers=[imageio.get_reader(str(REPORT/f'{k}_motion.mp4'),input_params=['-threads','1']) for k in order]
 with writer(REPORT/'four_hazards_overview.mp4') as w:
  for frames in zip(*readers):
   canvas=Image.new('RGB',(1600,900))
   for i,f in enumerate(frames):canvas.paste(Image.fromarray(f).resize((800,450)),((i%2)*800,(i//2)*450))
   w.append_data(np.asarray(canvas))
 for r in readers:r.close()
 verification=json.loads((REPORT/'verification.json').read_text())
 for r in records:
  r['final_verification']=next(v for v in verification['selected_cases'] if v['kind']==r['kind'])
 summary={'base_run':str(OUT/'frozen_v4_seed0/run.json'),'selected_experiments':{r['kind']:r['experiment_root'] for r in records},'video_type':'Nexus motion + schematic projection, not 3DGS','playback_speed':.5,'scenarios':records,'verification':verification};(REPORT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 body=[]
 for k in order:
  r=next(r for r in records if r['kind']==k);a=r['event_audit'];hit=sum(p['overlap_samples'] for p in a['pairs'].values());m=a['E_motion']
  adapter=r['final_verification']['runtime_refiner_adapter'];method='额外使用相对停车距离物理引导' if k in ['brake','reveal'] else '额外使用对向侵入让行物理引导' if k=='oncoming' else '沿用原冻结V4物理引导'
  note={'brake':'前车于5.0秒开始急刹。E已提前知道事件，主要表现为预先减速，并非突发感知后的急刹。','reveal':'静止车是插入的S车；画面初始被遮挡，模型仍已知其位置。正常对照中A在静止车前停车，事件片中A变道绕过。','rear':'高速后车已明显贴近，E仅小幅提速，不能据此声称已生成显著的主动避让。','oncoming':'按历史速度估计交会时刻后延长侵入至4.70秒，再于6.50秒返回。首版过早返回和第二版碰撞结果均保留在实验目录。'}[k]
  margin=r['final_verification']['endpoint_stationary_lead_stopping_margins'];margin_note=''
  if margin:margin_note='片尾静止前车停车余量：'+', '.join(f"{x['lead_role']} {x['margin_m']:.2f}米" for x in margin)+'；这是按4.5m/s²制动与0.3秒缓冲计算的诊断，不保证预测范围外安全。'
  body.append(f'''<section><h2>{html.escape(r['title'])}</h2><p>数据：nuPlan 场景 {r['scenario_index']:03d}，{html.escape(r['map'])}。事件车按指定轨迹运动，自车未来自由生成。{method}，模型权重未更新。</p><video controls preload="metadata" poster="{k}_t25.jpg" src="{k}_motion.mp4"></video><p><a href="{k}_motion.mp4">直接打开运动视频</a> · <a href="{k}_comparison.mp4">正常/事件对照视频</a></p><p>{note}</p><p>事件片检查：重叠 {hit} 个0.1秒采样；自车出可行驶区域 {a['body_offroad_samples']['E']} 个采样；最大加速度 {m['max_xy_acceleration_mps2']:.2f} m/s²；{'通过本次几何和运动检查' if a['accepted'] else '存在未通过的检查，请查看summary.json'}。{margin_note}</p><details><summary>详细指标与限制</summary><pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre></details></section>''')
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>四类危险场景运动视频</title><style>body{background:#0c1420;color:#e5edf8;font:18px/1.7 sans-serif;max-width:1300px;margin:auto;padding:30px}video{width:100%;border-radius:10px;background:black}section{padding:25px 0;border-top:1px solid #405060}a{color:#68d0ff}pre{white-space:pre-wrap;font-size:14px}p{color:#bbc9d9}</style><h1>四类危险场景运动视频</h1><p>先看四宫格，再逐个播放。左边是俯视运动，右边是自车前视／后视几何示意；右侧不是3DGS真实场景。视频按0.5倍速度播放，画面时间是仿真时间。</p><video controls src="four_hazards_overview.mp4"></video><p>真实nuPlan道路和车辆历史，编辑事件车未来，Nexus＋冻结V4物理引导生成自车未来。仅保留E/A以及第二场景插入的静止S车，未回放其余原始交通参与者。指定未来会提前提供给模型，第二场景静止车也已提供给模型；本实验不证明感知触发、在线闭环反应或纯Nexus原始能力。模型权重未更新，生成后未修补轨迹。</p>'''+''.join(body)+'</html>'
 page=page.replace('Nexus＋冻结V4物理引导生成自车未来','Nexus＋物理引导生成自车未来（第一、二类补充相对停车距离，第四类补充对向让行约束）')
 (REPORT/'index.html').write_text(page);print('REPORT_COMPLETE',REPORT,flush=True)
if __name__=='__main__':main()
