"""Chinese, final-only traffic diagram aligned with actual renderer poses."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from nexus_3dgs.bundle import load_bundle

WHITE = (248, 250, 253)
INK = (27, 39, 58)
MUTED = (92, 107, 128)
COLORS = {'ego': '#e34a4a', 'bus': '#df8b14', 'generated': '#3284d8',
          'context': '#939da9', 'g7': '#9251ca'}
W, H = 1920, 1080
MAP = (70, 195, 720, 975)
XR, YR = (-22., 43.), (-8., 70.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--stop-render', type=Path, required=True)
    p.add_argument('--go-render', type=Path, required=True)
    p.add_argument('--font', type=Path, default=REPO/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=('both','bus_stop','bus_follow_logged'), default='both')
    a = p.parse_args()
    report = json.loads((a.experiment/'run.json').read_text())
    if report['status'] != 'complete':
        raise ValueError('Incomplete trajectory experiment')
    a.output.mkdir(parents=True, exist_ok=True)
    fonts = {size: ImageFont.truetype(str(a.font), size) for size in (18, 20, 22, 24, 26, 28, 32, 38)}
    summary = {'status': 'started', 'font': str(a.font.resolve()), 'variants': {},
               'diagram_source': 'actual edited renderer poses and camera, saved inside edit_frame',
               'window_m': {'right': XR, 'forward': YR}, 'rgb_pixels_modified': False}
    combined = imageio.get_writer(str(a.output/'interaction_explained.mp4'), fps=10, macro_block_size=2,
                                  ffmpeg_params=['-threads', '4', '-crf', '18'])
    try:
        for name, directory, title in (
            ('bus_stop', a.stop_render, '巴士停车'),
            ('bus_follow_logged', a.go_render, '巴士沿记录轨迹行进')):
            if a.variant not in ('both',name):
                continue
            run = json.loads((directory/'run.json').read_text())
            bundle, manifest = load_bundle(a.experiment/name)
            if run['status'] != 'complete' or run['bundle_sha256'] != manifest['bundle_sha256']:
                raise ValueError('Render and trajectory bundle do not match')
            frames = [im for f in run['frames'] for im in f['images'] if im['camera_id'] == 0]
            if not frames or any('scene_metadata' not in f for f in frames):
                raise ValueError('Render with --save-actor-metadata first')
            current = manifest['current_index']
            anchor = bundle['reference_poses_world'][current, 0]
            yaw = np.arctan2(anchor[1, 0], anchor[0, 0])
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            def project(xy):
                local = (np.asarray(xy)[..., :2] - anchor[:2, 3]) @ R
                right, forward = -local[..., 1], local[..., 0]
                return np.stack((MAP[0] + (right-XR[0])*10,
                                 MAP[3] - (forward-YR[0])*10), axis=-1)
            conditioned_slot = report['conditioned_slot']
            def role(actor):
                if actor['agent_index'] == 0: return 'ego'
                if actor['nexus_slot'] == conditioned_slot: return 'bus'
                if actor['nexus_slot'] == 7: return 'g7'
                return 'generated' if actor['agent_index'] is not None else 'context'
            def label(actor):
                r = role(actor)
                if r == 'ego': return '自车'
                if r == 'bus': return '巴士'
                if r == 'g7': return '对向车7（原G7）'
                if actor['nexus_slot'] == 4: return '左侧面包车'
                if actor['nexus_slot'] == 6: return '右侧车6'
                if actor['nexus_slot'] in (5, 9): return f'模型车{actor["nexus_slot"]}'
                return None
            trail = {}
            name_video = a.output/f'{name}_explained.mp4'
            counts = []
            with imageio.get_writer(str(name_video), fps=10, macro_block_size=2,
                                    ffmpeg_params=['-threads', '4', '-crf', '18']) as writer:
                for index, frame in enumerate(frames):
                    meta = frame['scene_metadata']
                    elapsed = (frame['timestamp_us']-int(bundle['timestamps_us'][current]))/1e6
                    canvas = Image.new('RGB', (W, H), WHITE)
                    d = ImageDraw.Draw(canvas)
                    def text(xy, value, size=24, fill=INK):
                        d.text(xy, value, font=fonts[size], fill=fill)
                    text((58, 25), f'{title}  |  约束后的多车轨迹与3DGS画面', 38)
                    text((58, 84), f't = {elapsed:.1f} 秒     橙色巴士是给定干预；蓝色车来自模型预测并经过离线运动/碰撞约束。', 24, MUTED)
                    text((70, 144), '俯视：上方为起始自车朝向，单位：米', 26)
                    text((785, 144), '前相机：与左图同一时刻、同一组车辆位姿', 26)
                    d.rectangle(MAP, fill='#f0f4f8', outline='#cbd3dd', width=2)
                    # Horizontal field of view, shown as a direction wedge rather than an occlusion mask.
                    cam = np.asarray(meta['camera_world'])
                    K = np.asarray(meta['intrinsics'])
                    rays = np.array([[(u-K[0,2])/K[0,0], 0, 1] for u in (0, meta['width'])]) @ cam[:3,:3].T
                    rays = rays[:, :2] / np.linalg.norm(rays[:, :2], axis=-1, keepdims=True)
                    wedge = project(np.r_[cam[None,:2,3], cam[None,:2,3]+80*rays])
                    overlay = Image.new('RGB', (MAP[2]-MAP[0], MAP[3]-MAP[1]), '#f0f4f8')
                    od = ImageDraw.Draw(overlay)
                    od.polygon([tuple(p-[MAP[0], MAP[1]]) for p in wedge], fill='#e0edf9')
                    canvas.paste(overlay, (MAP[0], MAP[1]));d=ImageDraw.Draw(canvas)
                    for x in range(-20, 41, 10):
                        px = project([anchor[0,3],anchor[1,3]])[0] + x*10
                        d.line((px, MAP[1], px, MAP[3]), fill='#d9e0e9', width=1)
                        text((px-12, MAP[3]+8), str(x), 20, MUTED)
                    for y in range(0, 71, 10):
                        py = MAP[3]-(y-YR[0])*10
                        d.line((MAP[0], py, MAP[2], py), fill='#d9e0e9', width=1)
                        text((27, py-15), str(y), 20, MUTED)
                    text((335, 1012), '← 左侧     右侧 →', 22, MUTED)
                    valid_actors = [actor for actor in meta['actors'] if actor['valid']]
                    visible_in_window = 0
                    labels = []
                    for actor in meta['actors']:
                        key = actor['model_index']
                        if not actor['valid']:
                            trail.pop(key, None);continue
                        center = np.array(actor['center_world'])[:2]
                        xy = project(center)
                        if not (MAP[0]+5 < xy[0] < MAP[2]-5 and MAP[1]+5 < xy[1] < MAP[3]-5):
                            trail.pop(key, None);continue
                        visible_in_window += 1
                        color = COLORS[role(actor)]
                        history = trail.setdefault(key, [])
                        history.append(xy);del history[:-21]
                        if len(history)>1:
                            d.line([tuple(p) for p in history], fill=color, width=2)
                        angle = actor['yaw_world_rad'];c,s=np.cos(angle),np.sin(angle)
                        rotation = np.array([[c,-s],[s,c]])
                        length,width = actor['size_lwh'][:2]
                        corners = project((np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*[length/2,width/2])@rotation.T+center)
                        d.polygon([tuple(p) for p in corners], fill=color, outline=INK, width=1)
                        # White arrow makes vehicle orientation unambiguous even for near-stationary boxes.
                        arrow = project(np.array([[-.25,0],[.38,0]])*length@rotation.T+center)
                        tip = arrow[1];direction=(arrow[1]-arrow[0]);direction/=np.linalg.norm(direction)
                        side=np.array([-direction[1],direction[0]])
                        d.line([tuple(p) for p in arrow], fill='white', width=3)
                        d.polygon([tuple(tip),tuple(tip-8*direction+4*side),tuple(tip-8*direction-4*side)],fill='white')
                        name_label = label(actor)
                        if name_label:
                            labels.append((actor,xy,name_label,color))
                    occupied=[]
                    # Put important labels first and route subsequent callouts around them.
                    labels.sort(key=lambda item: {'g7':0,'bus':1,'ego':2}.get(role(item[0]),3))
                    for actor,xy,value,color in labels:
                        tw=d.textlength(value,font=fonts[18]);th=29
                        candidates=[(xy[0]+16,xy[1]-35),(xy[0]+16,xy[1]+12),
                                    (xy[0]-tw-18,xy[1]-35),(xy[0]-tw-18,xy[1]+12),
                                    (xy[0]+20,xy[1]-70),(xy[0]-tw-18,xy[1]+45)]
                        chosen=None
                        for lx,ly in candidates:
                            box=(lx-4,ly-1,lx+tw+5,ly+th)
                            if box[0]<MAP[0] or box[2]>MAP[2] or box[1]<MAP[1] or box[3]>MAP[3]:continue
                            if any(box[0]<b[2] and box[2]>b[0] and box[1]<b[3] and box[3]>b[1] for b in occupied):continue
                            chosen=(lx,ly,box);break
                        if chosen:
                            lx,ly,box=chosen;occupied.append(box)
                            d.line((tuple(xy),(lx,ly+th/2)),fill=color,width=1)
                            d.rounded_rectangle(box,radius=3,fill=WHITE,outline=color,width=1)
                            text((lx,ly),value,18,color)
                    rgb = Image.open(directory/frame['file']).convert('RGB')
                    rgb = rgb.resize((1120,630),Image.Resampling.LANCZOS)
                    canvas.paste(rgb,(785,195));d=ImageDraw.Draw(canvas)
                    # Only label depth-visible bus/G7 centres; do not alter rendered images on disk.
                    for actor in valid_actors:
                        if role(actor) not in ('bus','g7') or not actor['center_visible']:continue
                        uv=actor['center_uv'];px=785+uv[0]*1120/meta['width'];py=195+uv[1]*630/meta['height']
                        color=COLORS[role(actor)];value='巴士' if role(actor)=='bus' else '模型车7'
                        d.ellipse((px-5,py-5,px+5,py+5),outline=color,width=3)
                        lx=max(800,min(1790,px-28));ly=max(205,py-62)
                        d.line((px,py,lx+25,ly+32),fill=color,width=2)
                        d.rounded_rectangle((lx-5,ly-3,lx+110,ly+36),radius=4,fill=WHITE,outline=color,width=2)
                        text((lx,ly),value,22,color)
                    legend=(('ego','自车'),('bus','给定巴士'),('generated','模型预测+约束'),('context','记录轨迹'),('g7','模型车7'))
                    lx=792
                    for r,value in legend:
                        d.rounded_rectangle((lx,850,lx+18,868),radius=3,fill=COLORS[r])
                        text((lx+26,840),value,22)
                        lx+=int(d.textlength(value,font=fonts[22]))+63
                    text((790,886),'车内白箭头指车头；细线只显示过去2秒；浅蓝区域表示前相机视野方向。',22,MUTED)
                    text((790,924),'俯视图包含画面外及被遮挡的车；这是包围盒示意，不是车道地图。',22,MUTED)
                    g7=next((actor for actor in meta['actors'] if actor['nexus_slot']==7),None)
                    status='模型车7是对向车，车头朝下正常；已约束3.5～4秒的异常急转。' if g7 and g7['valid'] else '模型车7：有效预测在4秒结束；此后没有轨迹，不做外推。'
                    text((790,966),status,22,COLORS['g7'])
                    text((790,1008),f'窗口内 {visible_in_window} 辆 / 本帧有效 {len(valid_actors)} 辆；碰撞检查覆盖全场景23辆。',22,MUTED)
                    pixels=np.asarray(canvas)
                    writer.append_data(pixels);combined.append_data(pixels)
                    counts.append({'frame':index,'valid':len(valid_actors),'in_window':visible_in_window})
                    if index in (0,35,70):canvas.save(a.output/f'{name}_{index:03d}_preview.png')
            with imageio.get_reader(str(name_video),input_params=['-threads','2'],output_params=['-threads','2']) as reader:
                decoded=sum(1 for _ in reader)
            if decoded!=len(frames):raise ValueError('Video decode count differs')
            summary['variants'][name]={'decoded_frames':decoded,'counts':counts,'bundle_sha256':manifest['bundle_sha256']}
    finally:
        combined.close()
    with imageio.get_reader(str(a.output/'interaction_explained.mp4'),input_params=['-threads','2'],output_params=['-threads','2']) as reader:
        summary['combined_decoded_frames']=sum(1 for _ in reader)
    if summary['combined_decoded_frames']!=sum(v['decoded_frames'] for v in summary['variants'].values()):
        raise ValueError('Combined video decode mismatch')
    summary['status']='passed'
    (a.output/'verification.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print('EXPLAINED_VIDEO_OK '+str(a.output),flush=True)


if __name__=='__main__':main()
