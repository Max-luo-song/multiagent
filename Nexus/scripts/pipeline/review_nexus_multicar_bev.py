"""Visualize a new multi-car candidate without claiming missing 3DGS assets."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from nexus_3dgs.bundle import load_bundle
    from nexus_3dgs.geometry import resample_poses
    from nexus_3dgs import nuplan
    root=ROOT/'artifacts/nexus_3dgs/multicar_event_033_20261026'
    request=json.loads((root/'event_request.json').read_text());run=json.loads((root/'raw/run.json').read_text())
    out=root/'review';out.mkdir(exist_ok=False)
    seed=request['render_seed'];bundles={n:load_bundle(root/'raw'/f'seed_{seed:03d}'/n) for n in ('normal','event')}
    arr,meta=bundles['event'];ci=meta['current_index'];initial=arr['reference_poses_world'][ci,0]
    yaw=np.arctan2(initial[1,0],initial[0,0]);R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    times=arr['timestamps_us'][ci]+np.arange(1,81,dtype=np.int64)*100000
    sampled={n:resample_poses(ar['timestamps_us'],ar['predicted_poses_world'],ar['valid'],times) for n,(ar,_) in bundles.items()}
    a_index=list(arr['track_tokens']).index(request['actor_track_token'])
    c_index=list(arr['track_tokens']).index(request['third_actor_track_token_for_audit'])
    with nuplan.connect(meta['database']) as db:
        anchor=nuplan.anchor_frame(db,meta['scenario_token']);objects={o['track_token']:o for o in nuplan.boxes(db,anchor['lidar_token'])}
    sizes=np.array([[5.176,2.297]]+[[objects[str(t)]['length'],objects[str(t)]['width']] for t in arr['track_tokens'][1:]])
    font=ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf';fonts={s:ImageFont.truetype(str(font),s) for s in (18,20,22,26,30)}
    def pane(name,index):
        poses,valid=sampled[name];im=Image.new('RGB',(1000,960),'#f7fafc');d=ImageDraw.Draw(im)
        def text(xy,value,size=22,fill='#24364b'):d.text(xy,value,font=fonts[size],fill=fill)
        elapsed=(times[index]-arr['timestamps_us'][ci])/1e6
        text((25,12),('正常对照' if name=='normal' else '左前车切入条件')+f'  t={elapsed:.1f}s',30)
        text((25,58),'仅轨迹俯视动画；该场景尚无 3DGS 重建资产。',22,'#a04d31')
        text((25,96),'自车 E、右侧 C 与其他车辆未来均自由生成。',22)
        text((25,132),'A 仅在第 0.5～2.5 秒受约束；时段外自由。',22)
        ego=poses[index,0,:2,3]
        def project(xy):
            local=(np.asarray(xy)[...,:2]-ego)@R
            return np.stack([500-local[...,1]*14,695-local[...,0]*10],axis=-1)
        d.rectangle((20,180,980,825),fill='#eef3f7',outline='#b7c6d4')
        for x in range(-10,51,10):
            y=695-x*10;d.line((20,y,980,y),fill='#d3dee8');text((25,y+2),f'{x}m',18)
        for y in range(-30,31,10):
            x=500-y*14;d.line((x,180,x,825),fill='#d3dee8')
        text((35,188),'视图随自车平移，方向保持起始自车朝向。',18)
        for actor in range(len(arr['track_tokens'])):
            if not valid[index,actor]:continue
            T=poses[index,actor];center=T[:2,3].copy();angle=np.arctan2(T[1,0],T[0,0])
            if actor==0:center+=1.461*np.array([np.cos(angle),np.sin(angle)])
            q=project(center)
            if not (40<q[0]<960 and 220<q[1]<805):continue
            role='E' if actor==0 else 'A' if actor==a_index else 'C' if actor==c_index else None
            color={'E':'#d94a4a','A':'#dc9220','C':'#287bcd'}.get(role,'#b6c1cd')
            c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,s],[-s,c]])
            corners=project(np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*sizes[actor]/2@rot+center)
            d.polygon([tuple(v) for v in corners],fill=color,outline='#3e5267',width=2)
            arrow=project(np.array([[-.25,0],[.35,0]])*sizes[actor,0]@rot+center)
            d.line([tuple(v) for v in arrow],fill='white',width=2)
            if role:text((q[0]+12,q[1]-12),role,26,color)
        text((25,845),'红 E 自车生成    橙 A 事件车    蓝 C 第三车    灰 其他生成车',20)
        text((25,885),'是否合理避让与完整交互，以审计为准；不把轨迹差异当因果证明。',20)
        text((25,925),'背景网格只表示相对距离，不代表道路边界或可行驶区域。',18)
        return np.asarray(im)
    path=out/'paired_bev.mp4'
    with imageio.get_writer(path,fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18']) as writer:
        for i in range(80):
            combined=np.concatenate([pane('normal',i),pane('event',i)],axis=1);writer.append_data(combined)
            if i in (0,15,25,79):imageio.imwrite(out/f'pair_{i:03d}.jpg',combined)
    with imageio.get_reader(path) as reader:
        info=reader.get_meta_data();count=sum(1 for _ in reader)
    assert count==80 and info['fps']==10
    (out/'verification.json').write_text(json.dumps({'status':'passed','frames':count,'fps':info['fps'],'photorealistic':False,
        'positions_from_unmodified_raw_predictions':True,'render_seed_predeclared':seed},indent=2)+'\n')
    print('MULTICAR_BEV_REVIEW_COMPLETE',flush=True)


if __name__=='__main__':main()
