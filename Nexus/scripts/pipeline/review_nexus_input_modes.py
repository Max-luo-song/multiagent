"""Paired BEV videos for BP/SG comparisons; no unavailable 3DGS is implied."""
import json
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from nexus_3dgs.bundle import decode
    from nexus_3dgs.geometry import resample_poses
    from nexus_3dgs.clearance import rear_axle_to_box
    from nexus_3dgs.scene_projection import pair_separations
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'artifacts/nexus_3dgs/input_modes_20261026')
    root=parser.parse_args().root
    run=json.loads((root/'run.json').read_text());assert run['status']=='complete'
    out=root/'review';out.mkdir(exist_ok=False)
    modes=['bp_history','history_full_target','future_full_target','pure_sg_free_ego']
    modes=[m for m in modes if f'{m}/seed_000/event' in run['samples']]
    noise_control='pure_sg_bad_schedule/seed_000/event' in run['samples']
    names={'bp_history':'历史 cut-in → BP 续写','history_full_target':'相同历史＋固定 A 后续','future_full_target':'所有历史＋固定 A 未来 cut-in',
           'pure_sg_free_ego':'仅固定 A 全轨迹；其余全生成','pure_sg_bad_schedule':'未修正：未知历史未去噪'}
    base=torch.load(root/'bp_history/seed_000/normal/prediction.pt',map_location='cpu',weights_only=False)
    gt=decode(base['scene_tensor_gt'][0].numpy());sizes=gt[:,4,6:8].copy();sizes[0]=[5.176,2.297]
    anchor=gt[0,4,:2];heading=np.arctan2(gt[0,4,3],gt[0,4,2]);R=np.array([[np.cos(heading),-np.sin(heading)],[np.sin(heading),np.cos(heading)]])
    actor=run['protocol']['actor_slot'];third=run['protocol']['third_slot']
    times=np.arange(-2000000,8000000,100000,dtype=np.int64)
    sampled={}
    for mode in modes+(['pure_sg_bad_schedule'] if noise_control else []):
        for variant in (('event',) if mode=='pure_sg_bad_schedule' else ('normal','event')):
            key=f'{mode}/seed_000/{variant}';p=torch.load(root/key/'prediction.pt',map_location='cpu',weights_only=False)
            s=decode(p['sampled_tensor'][0].numpy());valid=p['valid_mask'][0].numpy().astype(bool)
            poses=np.broadcast_to(np.eye(4),(s.shape[1],s.shape[0],4,4)).copy()
            yaw=np.arctan2(s[...,3],s[...,2]).T;c,ss=np.cos(yaw),np.sin(yaw)
            poses[:,:,0,0]=c;poses[:,:,0,1]=-ss;poses[:,:,1,0]=ss;poses[:,:,1,1]=c
            poses[:,:,:2,3]=s[...,:2].transpose(1,0,2)
            ps,vv=resample_poses(np.arange(-2000000,8000001,500000,dtype=np.int64),poses,valid.T,times)
            xy=ps[...,:2,3];yy=np.arctan2(ps[...,1,0],ps[...,0,0]);boxes=np.concatenate([xy,yy[...,None]],axis=-1)
            boxes[:,0]=rear_axle_to_box(boxes[:,0],1.461)
            sep,pairs=pair_separations(boxes[:,[0,actor,third]],vv[:,[0,actor,third]],sizes[[0,actor,third]])
            sampled[(mode,variant)]=(ps,vv,sep)
    font=ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf';fonts={s:ImageFont.truetype(str(font),s) for s in (18,20,22,26)}
    def pane(mode,variant,index):
        poses,valid,sep=sampled[(mode,variant)];item=run['samples'][f'{mode}/seed_000/{variant}'];elapsed=times[index]/1e6
        im=Image.new('RGB',(1000,980),'#f7fafc');d=ImageDraw.Draw(im)
        def text(xy,value,size=22,fill='#24364b'):d.text(xy,value,font=fonts[size],fill=fill)
        text((20,8),names[mode],26)
        text((20,47),f'{"正常对照" if variant=="normal" else "切入条件"}   t={elapsed:.1f}s   {"历史片段" if elapsed<=0 else "后续生成"}',22)
        text((20,84),'轨迹俯视动画；不含道路边界、不是 3DGS 画面。',20,'#a04d31')
        text((20,119),f'生成当前自车位置偏移：{item["quality"]["ego_current_drift_from_reference_m"]:.2f}m',20)
        d.rectangle((10,160,990,810),fill='#edf3f7',outline='#b9c8d5')
        ego=poses[index,0,:2,3]
        def project(xy):
            local=(np.asarray(xy)[...,:2]-ego)@R
            return np.stack([500-local[...,1]*12,660-local[...,0]*10],axis=-1)
        for x in range(-10,41,10):
            y=660-x*10;d.line((10,y,990,y),fill='#d1dce7');text((15,y+2),f'{x}m',18)
        for y in range(-40,41,10):
            x=500-y*12;d.line((x,160,x,810),fill='#d1dce7')
        text((25,171),'视图随本组自车平移，方向保持初始车头朝向。',18)
        for ai,color in ((0,'#d94a4a'),(actor,'#d48c1a'),(third,'#287bcd')):
            trail=poses[max(0,index-20):index+1,ai,:2,3]
            points=project(trail)
            points=points[(points[:,0]>20)&(points[:,0]<980)&(points[:,1]>210)&(points[:,1]<790)]
            if len(points)>1:d.line([tuple(p) for p in points],fill=color,width=2)
        outside=0
        for ai,T in enumerate(poses[index]):
            if not valid[index,ai]:continue
            yaw=np.arctan2(T[1,0],T[0,0]);center=T[:2,3].copy()
            if ai==0:center+=1.461*np.array([np.cos(yaw),np.sin(yaw)])
            q=project(center)
            if not (35<q[0]<965 and 215<q[1]<790):outside+=1;continue
            role='E' if ai==0 else 'A' if ai==actor else 'C' if ai==third else None
            color={'E':'#d94a4a','A':'#d48c1a','C':'#287bcd'}.get(role,'#bac5cf')
            c,s=np.cos(yaw),np.sin(yaw);rot=np.array([[c,s],[-s,c]])
            corners=project(np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*sizes[ai]/2@rot+center)
            d.polygon([tuple(v) for v in corners],fill=color,outline='#384c62',width=2)
            if role:text((q[0]+12,q[1]-12),role,26,color)
        gap=sep[index,0,1]
        text((20,829),f'E—A 有符号车框间隔：{gap:.2f}m；视窗外车辆：{outside}',22,'#b23c3c' if gap<=0 else '#24364b')
        text((20,869),'红 E 自车    橙 A 事件车    蓝 C 第三车    灰 其他车辆',20)
        text((20,908),'未来 ego 始终自由；差异不等于合理避让或严格因果链。',20)
        text((20,946),'种子 0 在运行前指定用于展示；全部种子见审计表。',18)
        return np.asarray(im)
    outputs={}
    for mode in modes:
        path=out/f'{mode}.mp4'
        with imageio.get_writer(path,fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18']) as writer:
            for i in range(100):
                frame=np.concatenate([pane(mode,'normal',i),pane(mode,'event',i)],axis=1);writer.append_data(frame)
                if i in (0,20,40,99):imageio.imwrite(out/f'{mode}_{i:03d}.jpg',frame)
        outputs[path.name]=100
    if noise_control:
        path=out/'noise_schedule_control.mp4'
        with imageio.get_writer(path,fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18']) as writer:
            for i in range(100):
                frame=np.concatenate([pane('pure_sg_bad_schedule','event',i),pane('pure_sg_free_ego','event',i)],axis=1)
                writer.append_data(frame)
                if i==20:imageio.imwrite(out/'noise_schedule_current.jpg',frame)
        outputs[path.name]=100
    checks={}
    for name,expected in outputs.items():
        with imageio.get_reader(out/name) as reader:info=reader.get_meta_data();count=sum(1 for _ in reader)
        assert count==expected and info['fps']==10;checks[name]={'decoded_frames':count,'fps':info['fps']}
    (out/'verification.json').write_text(json.dumps({'status':'passed','videos':checks,'raw_predictions_modified':False,
        'photorealistic_render':False,'seed_predeclared':0,'includes_two_second_history':True},indent=2)+'\n')
    print('INPUT_MODE_REVIEW_COMPLETE',flush=True)


if __name__=='__main__':main()
