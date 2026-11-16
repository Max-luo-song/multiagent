"""Explain raw paired predictions with BEV derived from actual rendering poses."""
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,help='New presentation directory; existing review and raw renders are retained')
    a=p.parse_args();out=a.output or a.root/'review';out.mkdir(parents=True,exist_ok=False)
    names=['bus_stop','bus_follow_logged'];titles=['公交停车条件','公交按记录行驶条件']
    runs={n:json.loads((a.root/'renders'/n/'run.json').read_text()) for n in names}
    arrays,manifest=load_bundle(a.root/'raw_pair'/names[0])
    assert all(len(r['frames'])==80 for r in runs.values())
    anchor=arrays['reference_poses_world'][manifest['current_index'],0];yaw=np.arctan2(anchor[1,0],anchor[0,0])
    R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    font=ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'
    fonts={s:ImageFont.truetype(str(font),s) for s in (16,18,20,22,24,28,32)}
    colors={'ego':'#dc4545','bus':'#de8c17','generated':'#287bcd','context':'#8b96a5'}
    def role(actor):
        return 'ego' if actor['model_index']==-1 else 'bus' if actor['nexus_slot']==1 else 'generated' if actor['nexus_slot'] is not None else 'context'
    def label(actor):
        return 'E' if role(actor)=='ego' else 'B1' if role(actor)=='bus' else f'P{actor["nexus_slot"]}' if role(actor)=='generated' else f'R{actor["model_index"]}'
    def project(xy):
        local=(np.asarray(xy)[...,:2]-anchor[:2,3])@R
        return np.stack((240-local[...,1]*8.5,840-local[...,0]*8.5),axis=-1)
    def pane(name,index):
        frame=runs[name]['frames'][index];meta=frame['scene_metadata'];canvas=Image.new('RGB',(1440,980),'#f8fafc');d=ImageDraw.Draw(canvas)
        text=lambda xy,value,size=22,fill='#243248':d.text(xy,value,font=fonts[size],fill=fill)
        ni=names.index(name);elapsed=(frame['timestamp_us']-int(arrays['timestamps_us'][manifest['current_index']]))/1e6
        text((25,15),f'{titles[ni]}   t = {elapsed:.2f} s',32)
        text((25,65),'B1 公交未来已给定；E 自车与 P 编号车辆为原始 Nexus 预测',24)
        text((25,103),'灰色 R 为记录车辆；无车距过滤、无轨迹稳定化、无异常朝向修复',22,'#58677c')
        d.rectangle((10,155,470,905),fill='#eaf0f6',outline='#bbc7d3',width=2)
        for x in range(-20,31,10):
            px=240+x*8.5;d.line((px,155,px,905),fill='#d2dce7');text((px+2,882),str(x),16)
        for y in range(0,81,10):
            py=840-y*8.5;d.line((10,py,470,py),fill='#d2dce7');text((12,py+2),str(y),16)
        text((30,165),'俯视图：上方为起始车头方向',18)
        overlapping={i for pair in meta['overlapping_actor_pairs'] for i in pair}
        for ai,actor in enumerate(meta['actors']):
            if not actor['valid']:continue
            center=np.array(actor['center_world'])[:2];q=project(center)
            if not (25<q[0]<450 and 195<q[1]<870):continue
            angle=actor['yaw_world_rad'];c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,s],[-s,c]])
            length,width=actor['size_lwh'][:2];corners=project((np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*[length/2,width/2])@rot+center)
            color=colors[role(actor)];d.polygon([tuple(v) for v in corners],fill=color,outline='#c51642' if ai in overlapping else '#344155',width=3 if ai in overlapping else 1)
            arrow=project(np.array([[-.3,0],[.4,0]])*length@rot+center);d.line([tuple(v) for v in arrow],fill='white',width=2)
            text((q[0]+8,q[1]-8),label(actor),18,color)
        rgb=Image.open(a.root/'renders'/name/f'{frame["frame"]:03d}_0.png').convert('RGB')
        canvas.paste(rgb,(480,155));d=ImageDraw.Draw(canvas)
        text((490,717),'车辆编号仅在左侧俯视图显示；右侧高斯画面不叠加标记。',20,'#58677c')
        text((490,757),f'自车—公交 SAT 有符号间隔：{meta["ego_bus_sat_separation_m"]:.2f} m',24)
        text((490,800),f'全场景本帧重叠车辆对：{len(meta["overlapping_actor_pairs"])}',24,'#c51642' if overlapping else '#243248')
        text((490,849),'负间隔表示框重叠；图中包含画面外车辆。',20,'#58677c')
        text((490,885),'仅离散时刻的二维车框诊断，不是闭环规划或连续时间安全证明。',20,'#58677c')
        text((25,938),'红 E 自车预测     橙 B1 给定公交     蓝 P 模型预测     灰 R 记录车辆',22)
        # Verify the entire image region after composing the explanatory panel.
        # Equality is checked before the normal lossy video/JPEG encoding.
        assert np.array_equal(np.asarray(canvas)[155:695,480:1440],np.asarray(rgb))
        return np.array(canvas)
    writers={n:imageio.get_writer(str(out/f'{n}_explained.mp4'),fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','19']) for n in names}
    combined=imageio.get_writer(str(out/'paired_interaction.mp4'),fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','19'])
    try:
        for i in range(80):
            panels=[pane(n,i) for n in names]
            for n,panel in zip(names,panels):writers[n].append_data(panel)
            joined=np.concatenate(panels,axis=1);combined.append_data(joined)
            if i in (0,30,50,79):imageio.imwrite(out/f'pair_{i:03d}.jpg',joined)
    finally:
        for w in writers.values():w.close()
        combined.close()
    checks={}
    for path in out.glob('*.mp4'):
        with imageio.get_reader(path) as reader:
            meta=reader.get_meta_data();count=sum(1 for _ in reader)
        assert count==80 and meta['fps']==10
        checks[path.name]={'decoded_frames':count,'fps':meta['fps']}
    (out/'verification.json').write_text(json.dumps({'status':'passed','videos':checks,'bev_uses_actual_render_poses':True,'source_render_pixels_unchanged':True,
        'render_overlay_labels':False,'vehicle_labels_only_in_bev':True,'unencoded_render_region_exact_source_pixels_all_160_panels':True,
        'raw_model_predictions_modified':False,'gaussian_render_rerun':False},indent=2)+'\n')
    print('PAIRED_REVIEW_COMPLETE',out,flush=True)


if __name__=='__main__':main()
