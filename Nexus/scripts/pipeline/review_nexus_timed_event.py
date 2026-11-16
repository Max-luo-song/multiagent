"""Compose local-event videos with labels confined to the BEV panel."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from nexus_3dgs.bundle import load_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=a.root/'review';out.mkdir(parents=True,exist_ok=False)
    request=json.loads((a.root/'event_request.json').read_text());slot=1
    runs={n:json.loads((a.root/'renders'/n/'run.json').read_text()) for n in ('normal','event')}
    arr,meta=load_bundle(a.root/'raw'/f'seed_{request["render_seed"]:03d}'/'event')
    run=json.loads((a.root/'raw/run.json').read_text());slot=run['actor_slot']
    anchor=arr['reference_poses_world'][meta['current_index'],0];angle=np.arctan2(anchor[1,0],anchor[0,0])
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    font=ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'
    fonts={s:ImageFont.truetype(str(font),s) for s in (16,18,20,22,24,28)}
    titles={'normal':'正常对照：3～6 秒公交停止','event':'事件条件：3～6 秒公交倒车靠近 4.5 米'}
    def project(xy):
        local=(np.asarray(xy)[...,:2]-anchor[:2,3])@R
        return np.stack([240-local[...,1]*8.5,810-local[...,0]*8.5],axis=-1)
    def panel(name,index):
        frame=runs[name]['frames'][index];actors=frame['scene_metadata']['actors']
        elapsed=(frame['timestamp_us']-int(arr['timestamps_us'][meta['current_index']]))/1e6
        canvas=Image.new('RGB',(1440,940),'#f8fafc');d=ImageDraw.Draw(canvas)
        def text(xy,value,size=22,fill='#23354b'):d.text(xy,value,font=fonts[size],fill=fill)
        text((20,10),f'{titles[name]}    t={elapsed:.2f}s',28)
        text((20,52),'自车 E 与其他车辆未来自由生成；B1 仅指定时段受约束，时段外也自由生成。',22)
        text((20,91),'事件条件执行通过；完整避让与第三车交互尚未通过验证。',24,'#b23c3c')
        d.rectangle((10,145,470,870),fill='#edf2f7',outline='#bbc7d3',width=2)
        for y in range(0,71,10):
            q=810-y*8.5;d.line((10,q,470,q),fill='#d0dbe5');text((12,q+2),str(y),16)
        for x in range(-20,21,10):
            q=240+x*8.5;d.line((q,145,q,870),fill='#d0dbe5')
        text((25,153),'俯视图：上方为起始自车朝向',18)
        overlapping={i for pair in frame['scene_metadata']['overlapping_actor_pairs'] for i in pair}
        for ai,actor in enumerate(actors):
            if not actor['valid']:continue
            q=project(actor['center_world'])
            if not (25<q[0]<450 and 180<q[1]<850):continue
            ego=actor['nexus_slot']==0;target=actor['nexus_slot']==slot
            generated=actor['nexus_slot'] is not None
            color='#d94a4a' if ego else '#d58c19' if target else '#287bcd' if generated else '#8995a3'
            label='E' if ego else 'B1' if target else f'P{actor["nexus_slot"]}' if generated else f'R{actor["model_index"]}'
            yaw=actor['yaw_world_rad'];c,s=np.cos(yaw),np.sin(yaw);rot=np.array([[c,s],[-s,c]])
            length,width=actor['size_lwh'][:2]
            corners=project(np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*[length/2,width/2]@rot+np.array(actor['center_world'])[:2])
            d.polygon([tuple(v) for v in corners],fill=color,outline='#b91e42' if ai in overlapping else '#344155',width=3 if ai in overlapping else 1)
            text((q[0]+6,q[1]-10),label,18,color)
        rgb=Image.open(a.root/'renders'/name/f'{frame["frame"]:03d}_0.png').convert('RGB')
        canvas.paste(rgb,(480,145))
        text((490,707),'高斯画面不叠加车辆标记；编号仅显示在左侧俯视图。',20,'#596a7c')
        text((490,745),f'自车—公交有符号间隔：{frame["scene_metadata"]["ego_bus_sat_separation_m"]:.2f} m',24)
        text((490,785),f'本帧全场景重叠车辆对：{len(frame["scene_metadata"]["overlapping_actor_pairs"])}',24)
        text((490,828),'原始预测回放；无轨迹修复。负间隔表示物理车框重叠。',20,'#596a7c')
        text((20,894),'红 E 自车生成    橙 B1 事件车辆    蓝 P 其他生成车    灰 R 记录车辆',22)
        assert np.array_equal(np.asarray(canvas)[145:685,480:1440],np.asarray(rgb))
        return np.asarray(canvas)
    writers={n:imageio.get_writer(str(out/f'{n}_explained.mp4'),fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','19']) for n in runs}
    combined=imageio.get_writer(str(out/'paired_event.mp4'),fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','19'])
    assert all(len(r['frames'])==80 for r in runs.values())
    try:
        for index in range(80):
            panels=[panel(n,index) for n in runs]
            for n,value in zip(runs,panels):writers[n].append_data(value)
            joined=np.concatenate(panels,axis=1);combined.append_data(joined)
            if index in (0,30,50,79):imageio.imwrite(out/f'pair_{index:03d}.jpg',joined)
    finally:
        for writer in writers.values():writer.close()
        combined.close()
    checks={}
    for path in out.glob('*.mp4'):
        with imageio.get_reader(path) as reader:
            info=reader.get_meta_data();count=sum(1 for _ in reader)
        assert count==80 and info['fps']==10
        checks[path.name]={'decoded_frames':count,'fps':info['fps']}
    (out/'verification.json').write_text(json.dumps({'status':'passed','videos':checks,
        'labels_only_in_bev':True,'source_pixels_exact_before_encoding_all_160_panels':True,'render_seed_predeclared':request['render_seed']},indent=2)+'\n')
    print('TIMED_EVENT_REVIEW_COMPLETE',flush=True)


if __name__=='__main__':main()
