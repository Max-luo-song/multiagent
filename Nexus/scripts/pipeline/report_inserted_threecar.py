"""Verify actual rendered bodies and deliver readable 3DGS/BEV motion videos."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_3dgs/asset_insert_20261028'
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 import numpy as np,imageio.v2 as imageio
 from PIL import Image,ImageDraw,ImageFont
 from shapely import wkb
 from shapely.geometry import Polygon
 p=json.loads((OUT/'insertion_protocol.json').read_text());render=OUT/'render_final';run=json.loads((render/'run.json').read_text());assert run['status']=='complete' and not run['probe_only']
 assert run['renderer_sha256']==sha(ROOT/'scripts/pipeline/render_inserted_threecar.py');assert run['asset_loader_sha256']==sha(ROOT/'nexus_3dgs/inserted_vehicle.py')
 for file,h in p['source_prediction_sha256'].items():assert sha(file)==h
 for role,asset in p['asset_assignment'].items():assert sha(asset['source'])==asset['source_sha256']==run['asset_info'][role]['source_sha256']
 assert all(run[k] for k in ['background_and_original_models_unchanged','asset_sources_unchanged','Nexus_predictions_unchanged'])
 report=ROOT/'report/nexus_threecar_asset_insert_20261028';report.mkdir(exist_ok=False);media=report/'media';media.mkdir()
 left=wkb.loads((OUT/'AC_lane.wkb').read_bytes());right=wkb.loads((OUT/'B_lane.wkb').read_bytes());road=left.union(right);S=np.linalg.inv(json.loads((Path(p['background_scene'])/'nexus_scene.json').read_text())['frames'][0]['ego_to_world']);W=np.linalg.inv(S);E0=np.array(p['background_camera_anchor_ego']);h=E0[:2,0];h=h/np.linalg.norm(h);H=np.c_[h,[-h[1],h[0]]];roles=['A','B','C'];colors={'A':'#dd921c','B':'#d94a4a','C':'#287bcd'}
 def body(actor):
  T=np.array(actor['body_world']);sz=np.array(actor['size_lwh']);sign=np.array([[a,b,c] for a in [-1,1] for b in [-1,1] for c in [-1,1]]);corn=sign*sz/2@T[:3,:3].T+T[:3,3];return Polygon(corn[:,:2]).convex_hull,corn
 audit={};state={}
 for variant in ['normal','event']:
  records=run['frames'][variant];assert len(records)==81;rows=[];target_inside=[];full_pass=[];minimum=1e9;hits=[];off=[];wrong=[];visible=0
  for k,row in enumerate(records):
   assert row['frame']==k and abs(row['time_s']-k*.1)<1e-6
   polygons={};corners={};coords={}
   for actor in row['actors']:
    role=actor['role'];poly,corn=body(actor);polygons[role]=poly;corners[role]=corn;coords[role]=np.array(actor['center_xy_world']);assert np.allclose(np.array(actor['body_world'])[:2,3],coords[role],atol=1e-9)
    if not road.buffer(.02).covers(poly):off.append([k,role])
    if variant=='normal' and not (left if role=='A' else right).buffer(.02).covers(poly):wrong.append([k,role])
    C=np.array(row['camera_scene']);K=np.array(row['intrinsics']);q=np.linalg.inv(C)@S@np.r_[np.array(actor['body_world'])[:3,3],1.];uv=K@q[:3];uv=uv[:2]/uv[2]
    visible+=int(q[2]>.1 and 0<=uv[0]<960 and 0<=uv[1]<540)
   for i,r in enumerate(roles):
    for other in roles[i+1:]:
     intersection=polygons[r].intersection(polygons[other]).area
     if intersection>1e-6:hits.append([k,r,other])
     minimum=min(minimum,polygons[r].distance(polygons[other]))
   target=left.buffer(.02).covers(polygons['C']);target_inside.append(target)
   actorB=next(a for a in row['actors'] if a['role']=='B');axis=np.array(actorB['body_world'])[:2,0];axis/=np.linalg.norm(axis);full_pass.append(bool(target and np.min(corners['C'][:,:2]@axis)>np.max(corners['B'][:,:2]@axis)))
   rows.append({'polygons':polygons,'centers':coords,'C_in_target':target,'row':row})
  hold=next((i*.1 for i in range(71) if all(target_inside[i:i+11])),None);passed=next((i*.1 for i,v in enumerate(full_pass) if v),None)
  audit[variant]={'accepted':not hits and not off and not wrong,'actual_3D_body_overlap_samples':len(hits),'actual_3D_body_offroad_samples':len(off),'normal_wrong_lane_samples':len(wrong),'minimum_3D_box_XY_polygon_gap_m':float(minimum),'C_complete_target_lane_1s_hold_start_s':hold,'C_final_in_target_lane':bool(target_inside[-1]),'C_rear_fully_passes_B_front_s':passed,'projected_actor_centers_inside_image':visible,'expected_visible_center_checks':243,'scope':'actual rendered 3D bounding boxes projected to road XY, 81 samples; no continuous-time or full traffic-law guarantee'}
  assert audit[variant]['accepted'],audit[variant];assert visible==243,(variant,visible)
  state[variant]=rows
 assert audit['normal']['C_complete_target_lane_1s_hold_start_s'] is None
 assert audit['event']['C_complete_target_lane_1s_hold_start_s'] is not None and audit['event']['C_rear_fully_passes_B_front_s'] is not None
 for a,b in zip(run['frames']['normal'],run['frames']['event']):assert a['camera_scene']==b['camera_scene'] and a['intrinsics']==b['intrinsics']
 write(OUT/'rendered_geometry_audit.json',audit);write(report/'geometry_audit.json',audit)
 fontpath=ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf';fonts={s:ImageFont.truetype(str(fontpath),s) for s in [16,18,20,22,26]}
 trajectories=np.load(OUT/'placed_trajectories.npz');speeds={v:np.linalg.norm(np.diff(trajectories[f'{v}_xy_world'][:,::5],axis=1)/.5,axis=-1) for v in ['normal','event']}
 def panel(variant,k,rgb,origin,scale):
  canvas=Image.new('RGB',(1320,640),'#f5f7fa');d=ImageDraw.Draw(canvas)
  title='未切入：A在左侧减速，B/C继续跟随' if variant=='normal' else '切入减速：B制动，C变道绕行'
  d.text((18,7),title+f'  {k*.1:.1f}s',font=fonts[26],fill='#24364b')
  d.text((18,40),'资产合成演示｜背景121 · 轨迹源059 · 左侧俯视示意，右侧3DGS',font=fonts[18],fill='#52647a')
  tile=Image.new('RGB',(360,540),'#e3e8ee');draw=ImageDraw.Draw(tile)
  def project(xy):
   q=(np.array(xy)-origin)@H;return np.stack([180-q[...,1]*scale,285-q[...,0]*scale],-1)
  for poly,color in [(left,'#f0f5e8'),(right,'#e9eff8')]:
   for q in (poly.geoms if hasattr(poly,'geoms') else [poly]):draw.polygon([tuple(a) for a in project(np.array(q.exterior.coords))],fill=color,outline='#aeb8c5')
  for role in roles:
   poly=state[variant][k]['polygons'][role];points=project(np.array(poly.exterior.coords));draw.polygon([tuple(q) for q in points],fill=colors[role],outline='#55657a');c=project(state[variant][k]['centers'][role]);draw.text((c[0]+14,c[1]-13),role,font=fonts[22],fill=colors[role])
  draw.rectangle((0,0,359,80),fill='#f5f7fa')
  for i,role in enumerate(roles):
   speed=speeds[variant][i,min(max(int(np.ceil(k*.1/.5))-1,0),15)];draw.text((10,5+i*24),f'{role}  {speed:.1f} m/s',font=fonts[18],fill=colors[role])
  draw.rectangle((0,500,359,539),fill='#f5f7fa');label='C：已进入相邻车道' if state[variant][k]['C_in_target'] else ('C：保持原车道' if variant=='normal' else 'C：观察/准备变道')
  draw.text((10,508),label,font=fonts[18],fill=colors['C']);canvas.paste(tile,(0,70));canvas.paste(Image.fromarray(rgb),(360,70));d=ImageDraw.Draw(canvas);d.text((18,615),'三车运动与实际资产尺寸已核对 · 两组使用相同观察相机和焦距',font=fonts[16],fill='#52647a');return np.array(canvas)
 raw_readers={v:imageio.get_reader(render/v/'camera_0.mp4') for v in ['normal','event']};writers={v:imageio.get_writer(media/f'{v}_with_bev.mp4',fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18']) for v in ['normal','event']};pair=imageio.get_writer(media/'comparison.mp4',fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18'])
 try:
  for k in range(81):
   centers=np.array([state[v][k]['centers'][r] for v in ['normal','event'] for r in roles]);origin=centers.mean(0);q=(centers-origin)@H;scale=min(13.,430/(np.ptp(q[:,0])+12),300/(np.ptp(q[:,1])+12));panels={}
   for v in ['normal','event']:
    rgb=raw_readers[v].get_next_data();panels[v]=panel(v,k,rgb,origin,scale);writers[v].append_data(panels[v])
    if k in [0,20,40,60,80]:Image.fromarray(panels[v]).save(media/f'{v}_{k:03d}.jpg')
   comparison=np.concatenate([panels['normal'],panels['event']],axis=1);pair.append_data(comparison)
   if k in [0,40,60,80]:Image.fromarray(comparison).resize((1980,480)).save(media/f'comparison_{k:03d}.jpg')
 finally:
  for reader in raw_readers.values():reader.close()
  for writer in writers.values():writer.close()
  pair.close()
 videos={}
 for v in ['normal','event']:shutil.copy2(render/v/'camera_0.mp4',media/f'{v}_clean_3dgs.mp4')
 for file in media.glob('*.mp4'):
  with imageio.get_reader(file) as reader:count=sum(1 for _ in reader);meta_video=reader.get_meta_data()
  assert count==81 and meta_video['fps']==10;videos[file.name]={'frames':count,'fps':10,'size':list(meta_video['size']),'sha256':sha(file)}
 for name in ['insertion_protocol.json','vehicle_ground_profile.json','native_road_depth_audit.json','asset_inventory.json','chosen_background.json']:shutil.copy2(OUT/name,report/name)
 write(report/'render_run.json',run)
 verdict=f"C约{audit['event']['C_complete_target_lane_1s_hold_start_s']:.1f}秒完整进入相邻车道，约{audit['event']['C_rear_fully_passes_B_front_s']:.1f}秒完全超过B；两组实际三车车框无重叠、无出道路。"
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>三车交互：3DGS资产合成</title><style>body{font-family:system-ui;max-width:1300px;margin:24px auto;padding:0 18px;background:#f3f6fa;color:#24364b}section{background:white;padding:22px;border-radius:12px;margin:18px 0}p{line-height:1.8}video,img{width:100%}a{color:#2364a8}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}</style><h1>A切入 → B减速 → C变道绕行</h1>'''
 page+=f'<section><p>{verdict}</p><p>先看事件视频：左侧俯视示意用橙A、红B、蓝C标注；右侧原生3DGS画面不叠加车辆标签。</p><video controls preload="metadata" src="media/event_with_bev.mp4"></video><p><a href="media/event_with_bev.mp4">直接打开事件视频</a> · <a href="media/event_clean_3dgs.mp4">只看干净3DGS画面</a></p></section>'
 page+='<section><h2>左未切入，右切入</h2><p>两组A使用相同纵向减速安排，主要改变A横向切入；两组观察相机、光学变焦及资产相同。正常组C保持原车道，事件组C选择变道绕行。这里使用之前生成的轨迹做道路适配，没有在此背景重新运行Nexus。</p><video controls preload="metadata" src="media/comparison.mp4"></video><p><a href="media/comparison.mp4">直接打开对照视频</a> · <a href="media/normal_with_bev.mp4">未切入视频</a></p></section>'
 page+='<section><h2>场景与资产</h2><p>背景使用传感器齐全的121路段，具有约119米连续同向相邻车道，使用真实图像/点云新训练8000步。提供的1.ply白色SUV用于A和C，3.ply白色MPV用于B，均保留车身比例并缩放到原轨迹车框内。高斯按标准PLY的颜色通道排列、对数尺度和透明度logit读取，所有原始资产不改写。</p><p>轨迹源为059场景的Nexus＋行为引导结果。原道路后段弯曲，直接刚体移植未通过车道检查，因此用源/目标成对车道的曲线坐标适配，保留事件时间和沿路进度，按实际车道间距转换横向偏移并重新验收。此视频属于新的三车资产合成演示，不是059真实场景重建，也不包含原52个参与者的完整回放。</p><p>观察相机沿B事件组进度平滑前进，使用同一相机路径和1→1.7倍原生焦距变化，固定曝光。车辆高度基于真实点云平面，并用原生道路期望深度做幅度不超过14厘米的平滑高度/俯仰校准。接触阴影为原生薄高斯近似，未做物理光照仿真。</p></section>'
 page+='<section><h2>检查与限制</h2><p>检查了完整81帧、10fps视频、实际渲染三维车框的道路与碰撞、正常车道、C完整变道和超越B。原始Nexus预测、提供的PLY和背景推理权重均通过不变性检查。未使用真实照片贴图覆盖当前画面、视频补帧或锐化。</p><p>车辆资产可以清楚展示运动，但新背景的树木、远景和路缘仍有模糊/浮点，不宣称已达到此前公交v9的画质。小幅落地校准依据重建深度，不是独立真实地形标定。三车离散几何通过不等于全场景安全、交通法规验证或B→C严格因果证明；原B固定诊断未整对通过。</p><a href="geometry_audit.json">实际几何检查</a> · <a href="summary.json">视频校验</a> · <a href="implementation.md">实现说明</a></section></html>'
 (report/'index.html').write_text(page)
 summary={'status':'complete','video_scope':'synthetic three-car asset insertion; derived road-adapted trajectories; not original059 reconstruction','background_scene_index':121,'source_trajectory_scene_index':59,'new_background_training_iterations':8000,'Nexus_training_updates':0,'geometry_audit':audit,'videos':videos,'background_novel_view_quality_fully_solved':False,'strict_causal_chain_proven':False,'full_scene_safety_proven':False}
 write(report/'summary.json',summary);write(OUT/'delivery_summary.json',summary);print('INSERT_REPORT_COMPLETE',report,verdict,flush=True)
if __name__=='__main__':main()
