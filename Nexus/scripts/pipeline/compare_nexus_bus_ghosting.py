"""Same-bundle old/new camera and tracked bus-roof pixel comparisons."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-stop',type=Path,required=True);p.add_argument('--new-stop',type=Path,required=True)
    p.add_argument('--old-go',type=Path,required=True);p.add_argument('--new-go',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-same-camera',action='store_true')
    p.add_argument('--focus',choices=('roof','rear'),default='roof')
    p.add_argument('--label',default='同一轨迹包，前景分离与轮廓约束优化')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[2];font=repo/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'
    fonts={k:ImageFont.truetype(str(font),k) for k in (22,26,30)}
    summary={'status':'started','same_bundle':True,'rgb_postprocessing':'labelled crops/resizing only; no smoothing or inpainting','variants':{}}
    with imageio.get_writer(str(a.output/'bus_ghost_comparison.mp4'),fps=10,macro_block_size=2,
                            ffmpeg_params=['-threads','4','-crf','17']) as writer:
        for name,dirs in [('bus_stop',(a.old_stop,a.new_stop)),('bus_follow_logged',(a.old_go,a.new_go))]:
            runs=[json.loads((d/'run.json').read_text()) for d in dirs]
            if any(r['status']!='complete' for r in runs):raise ValueError('Render not complete')
            if runs[0]['bundle_sha256']!=runs[1]['bundle_sha256']:raise ValueError('Different trajectory bundles')
            records=[[im for f in r['frames'] for im in f['images'] if im['camera_id']==0] for r in runs]
            if [r['timestamp_us'] for r in records[0]]!=[r['timestamp_us'] for r in records[1]]:raise ValueError('Different camera times')
            for index,frames in enumerate(zip(*records)):
                if a.require_same_camera:
                    for key in ('camera_world','intrinsics'):
                        np.testing.assert_array_equal(frames[0]['scene_metadata'][key],frames[1]['scene_metadata'][key])
                # Track the projected roof instead of a fixed pixel margin:
                # the departing bus becomes much smaller near the clip end.
                meta=frames[0]['scene_metadata']
                bus=next(actor for actor in meta['actors'] if actor['nexus_slot']==1)
                signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
                yaw=bus['yaw_world_rad'];c,s=np.cos(yaw),np.sin(yaw)
                rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                corners=(signs*np.array(bus['size_lwh'])/2)@rotation.T+bus['center_world']
                cam=np.asarray(meta['camera_world']);local=(corners-cam[:3,3])@cam[:3,:3]
                if local[:,2].min()<=0:raise ValueError('Bus box crosses camera plane')
                pixels=local@np.asarray(meta['intrinsics']).T;uv=pixels[:,:2]/pixels[:,2,None]
                half=int(np.clip((uv[:,0].max()-uv[:,0].min())*.85,80,210))
                cx=int(round((uv[:,0].min()+uv[:,0].max())/2))
                top=int(round(uv[:,1].min()-.35*half))
                before_image=Image.open(dirs[0]/frames[0]['file']).convert('RGB')
                before_pixels=np.asarray(before_image)
                hsv=cv2.cvtColor(before_pixels,cv2.COLOR_RGB2HSV)
                yy,xx=np.indices(hsv.shape[:2])
                search=(xx>=uv[:,0].min())&(xx<=uv[:,0].max())&(yy>=uv[:,1].min())&(yy<=uv[:,1].max())
                orange=search&(hsv[...,0]>=4)&(hsv[...,0]<=30)&(hsv[...,1]>65)&(hsv[...,2]>35)
                ys,xs=np.nonzero(orange)
                if len(xs)>100:
                    # Physical boxes include empty space above the roof. The
                    # observed orange bus panels locate the visible body edge
                    # for this diagnostic crop; both columns use the same box.
                    x0,x1=np.quantile(xs,[.01,.99]);half=int(np.clip((x1-x0)*.9,80,210))
                    cx=int(round((x0+x1)/2));top=int(round(np.quantile(ys,.03)-.45*half))
                cx=int(np.clip(cx,half,meta['width']-half))
                top=int(np.clip(top,0,meta['height']-half))
                if a.focus=='rear':
                    purple=search&(hsv[...,0]>110)&(hsv[...,0]<175)&(hsv[...,1]>40)&(hsv[...,2]>25)&(hsv[...,2]<180)
                    count,labels,stats,_=cv2.connectedComponentsWithStats(purple.astype(np.uint8),8)
                    if count>1:
                        best=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);ys,xs=np.nonzero(labels==best)
                        if len(xs)>50:
                            half=int(np.clip((np.quantile(xs,.99)-np.quantile(xs,.01))*.7,80,210));cx=int(round(np.median(xs)));top=int(round(np.median(ys)-half/2))
                    cx=int(np.clip(cx,half,meta['width']-half));top=int(np.clip(top,0,meta['height']-half))
                box=(cx-half,top,cx+half,top+half)
                canvas=Image.new('RGB',(1920,1080),'#f7f9fc');d=ImageDraw.Draw(canvas)
                condition='巴士停车' if name=='bus_stop' else '巴士沿记录轨迹行进'
                d.text((30,12),condition+'：'+a.label,font=fonts[30],fill='#25314a')
                for col,(directory,frame) in enumerate(zip(dirs,frames)):
                    x=col*960;d.text((x+30,57),'旧模型' if col==0 else '优化后',font=fonts[26],fill='#25314a')
                    image=before_image if col==0 else Image.open(directory/frame['file']).convert('RGB')
                    canvas.paste(image,(x,100))
                    crop=image.crop(box).resize((840,420),Image.Resampling.LANCZOS)
                    canvas.paste(crop,(x+60,650));d=ImageDraw.Draw(canvas)
                    d.rectangle((x+box[0],100+box[1],x+box[2],100+box[3]),outline='#df8b14',width=2)
                    d.text((x+70,657),('车尾局部放大' if a.focus=='rear' else '车顶局部放大'),font=fonts[22],fill='#e0a138',stroke_width=1,stroke_fill='#1d2635')
                pixels=np.asarray(canvas);writer.append_data(pixels)
                if index in (35,50,79):canvas.save(a.output/f'{name}_{index:03d}.png')
            summary['variants'][name]={'frames':len(records[0]),'bundle_sha256':runs[0]['bundle_sha256']}
    with imageio.get_reader(str(a.output/'bus_ghost_comparison.mp4'),input_params=['-threads','2'],output_params=['-threads','2']) as reader:
        summary['decoded_frames']=sum(1 for _ in reader)
    if summary['decoded_frames']!=sum(r['frames'] for r in summary['variants'].values()):raise ValueError('Decode frame count differs')
    summary['camera_pose_and_intrinsics_identical']=a.require_same_camera
    summary['focus']=a.focus
    summary['status']='passed';(a.output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('BUS_COMPARISON_OK '+str(summary['decoded_frames']),flush=True)


if __name__=='__main__':main()
