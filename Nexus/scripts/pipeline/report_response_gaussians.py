"""Package clean native 3DGS clips beside a synchronized metric BEV."""
import json,sys,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_response_gaussians_20261110';REPORT=ROOT/'report/nexus_response_gaussians_20261110'
def main():
 import numpy as np,imageio.v2 as imageio
 from PIL import Image,ImageDraw,ImageFont
 p=json.loads((OUT/'protocol.json').read_text());run=json.loads((OUT/'render_run.json').read_text());assert run['status']=='complete';REPORT.mkdir(exist_ok=True,parents=True)
 fontpath=ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf';font=ImageFont.truetype(str(fontpath),24);small=ImageFont.truetype(str(fontpath),18);colors={'E':'#48d6ff','A':'#ffb44b','S':'#bf95ff'};checks=[];sections=[]
 def writer(path):return imageio.get_writer(path,fps=10,macro_block_size=2,codec='libx264',ffmpeg_params=['-threads','2','-crf','18','-preset','veryfast'])
 for c in p['cases']:
  kind=c['kind'];z=np.load(OUT/f'{kind}.npz');roles=c['roles'];folder=OUT/'render'/kind;read=imageio.get_reader(str(folder/'observer.mp4'),input_params=['-threads','1']);assert read.count_frames()==81
  for view in ['observer','driver']:
   file=folder/f'{view}.mp4';shutil.copy2(file,REPORT/f'{kind}_{view}.mp4');r=imageio.get_reader(str(file),input_params=['-threads','1']);assert r.count_frames()==81 and r.get_data(80).shape==(540,960,3);r.close()
  initial=z['xy'][0,0];angle=z['yaw'][0,0];h=np.array([np.cos(angle),np.sin(angle)]);side=np.array([-h[1],h[0]]);local=np.stack([(z['xy']-initial)@h,(z['xy']-initial)@side],-1);span=max(45,float(np.max(np.ptp(local[:,:,0],axis=0)))+20);scale=min(10.,450/span)
  with writer(REPORT/f'{kind}_gaussian.mp4') as w:
   for k in range(81):
    frame=Image.new('RGB',(1440,650),'#0d1622');d=ImageDraw.Draw(frame);d.text((20,12),c['title']+' · 高斯背景重演',font=font,fill='white');d.text((1230,16),f'{k/10:.1f}s | 0.5×',font=small,fill='white');frame.paste(Image.fromarray(read.get_data(k)),(480,65))
    mid=(local[:,k,0].min()+local[:,k,0].max())/2
    def project(x):return (240-float(x[1])*scale,335-(float(x[0])-mid)*scale)
    d.rectangle([0,65,479,604],fill='#182833')
    # Metric lane guides, not an assertion of native background lane labels.
    for lat in [-5.1,-1.7,1.7,5.1]:
     xx=240-lat*scale;d.line([(xx,65),(xx,604)],fill='#596b75',width=2)
    for i,role in enumerate(roles):
     xy=project(local[i,k]);yaw=z['yaw'][i,k]-angle;hh=np.array([np.cos(yaw),np.sin(yaw)]);ss=np.array([-hh[1],hh[0]]);corners=local[i,k]+np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*z['sizes'][i]/2@np.array([hh,ss]);d.polygon([project(q) for q in corners],fill=colors[role]);d.text((xy[0]+10,xy[1]-10),role,font=small,fill='white')
    d.text((14,76),'俯视运动（米制）',font=small,fill='white');d.text((18,615),'原生3DGS背景＋车辆高斯资产；保留源轨迹时序，背景道路已迁移。',font=small,fill='#b9c9d7')
    if kind=='oncoming':d.text((570,615),'同向背景中的逆向侵入示意，非原双向道路重建。',font=small,fill='#ffd38a')
    for _ in range(2):w.append_data(np.asarray(frame))
    if k in [0,40,80]:frame.save(REPORT/f'{kind}_{k:03d}.jpg',quality=94)
  read.close();file=REPORT/f'{kind}_gaussian.mp4';checks.append({'kind':kind,'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()});note='背景拓扑与原双向道路不同，此片仅作逆向车辆侵入示意。' if kind=='oncoming' else '源事件轨迹迁移到已重建背景；不是原数据场景的照片级重建。'
  sections.append(f'<section><h2>{c["title"]}</h2><video controls src="{kind}_gaussian.mp4" poster="{kind}_040.jpg"></video><p>{note}</p><a href="{kind}_observer.mp4">纯高斯外部视角（1×）</a> · <a href="{kind}_driver.mp4">自车'+('后视' if kind=='rear' else '前视')+'（1×）</a></section>');print('GAUSSIAN_VIDEO_PACKAGED',kind,flush=True)
 readers=[imageio.get_reader(str(REPORT/f'{c["kind"]}_gaussian.mp4'),input_params=['-threads','1']) for c in p['cases']]
 with writer(REPORT/'gaussian_overview.mp4') as w:
  for frames in zip(*readers):
   board=Image.new('RGB',(1440,650))
   for i,rgb in enumerate(frames):board.paste(Image.fromarray(rgb).resize((720,325)),((i%2)*720,(i//2)*325))
   w.append_data(np.asarray(board))
 for r in readers:r.close()
 (REPORT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>四类事件高斯渲染</title><style>body{max-width:1440px;margin:auto;padding:20px;background:#0d1622;color:white;font:18px/1.7 sans-serif}video{width:100%}a{color:#7edaff}section{margin-top:30px;border-top:1px solid #456}</style><h1>四类事件：原生高斯背景重演</h1><p>背景与车辆均采用高斯点进行原生深度合成，未在高斯画面内叠加车辆标签。背景借用scene121，轨迹保持原时间与纵向进度，并沿背景道路迁移。未重新生成Nexus轨迹或训练模型。第四类背景不是双向道路，只能作为逆向侵入视觉示意；新视角可能出现背景模糊和缺失。</p><video controls src="gaussian_overview.mp4"></video>'+''.join(sections))
 (REPORT/'delivery.json').write_text(json.dumps({'status':'complete','native_gaussians':True,'render_run':str(OUT/'render_run.json'),'clips':checks,'scope':p['scope']},ensure_ascii=False,indent=2)+'\n');print('GAUSSIAN_REPORT_READY',REPORT,flush=True)
if __name__=='__main__':main()
