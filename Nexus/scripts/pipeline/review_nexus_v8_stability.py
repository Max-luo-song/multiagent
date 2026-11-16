"""Build v7/v8 stability comparisons from complete unfiltered render sequences."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();v=a.experiment;old=v.parent/'optimization_v7';out=v/'review';out.mkdir(exist_ok=False)
    import imageio.v2 as imageio,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from contextlib import ExitStack
    root=Path(__file__).resolve().parents[2];font=ImageFont.truetype(str(root/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'),25)
    audit=json.loads((v/'stability_validation.json').read_text());assert audit['status']=='passed'
    def panel(items,f,caption):
        canvas=Image.new('RGB',(960*len(items),650),'#142136');draw=ImageDraw.Draw(canvas)
        for j,(label,im) in enumerate(items):draw.text((960*j+15,10),label,font=font,fill='white');canvas.paste(Image.fromarray(im),(960*j,50))
        draw.text((15,605),f'{(f-20)/10:.1f} 秒　帧 {f}　'+caption,font=font,fill='#ffd18c');return canvas
    def zoom(before,after,box,f,label):
        canvas=Image.new('RGB',(1760,590),'#142136');draw=ImageDraw.Draw(canvas)
        for j,(name,im) in enumerate([('v7：修正前',before),('v8：修正后',after)]):
            draw.text((j*880+15,10),name,font=font,fill='white');crop=Image.fromarray(im).crop(box).resize((880,480),Image.Resampling.BILINEAR);canvas.paste(crop,(j*880,50))
        draw.text((15,545),f'{(f-20)/10:.1f} 秒　'+label,font=font,fill='#ffd18c');return canvas
    names=['nexus_before_after','heading_ablation','recorded_before_after','black_car_closeup','bus_closeup']
    with ExitStack() as stack:
        writers={name:stack.enter_context(imageio.get_writer(str(out/(name+'.mp4')),format='FFMPEG',fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18'])) for name in names}
        for f in range(20,100):
            original=imageio.imread(old/'raw_nexus_assets_prediction'/f'{f:03d}_0.png');heading=imageio.imread(v/'heading_only'/f'{f:03d}_0.png');stable=imageio.imread(v/'nexus_stable'/f'{f:03d}_0.png');real_old=imageio.imread(old/'assets_motion_validation/new'/f'{f:03d}_0.png');real_new=imageio.imread(v/'observed_validation/new'/f'{f:03d}_0.png');photo=imageio.imread(v/'observed_validation'/f'{f:03d}_photo.png')
            frames={
                'nexus_before_after':panel([('v7 候选：原始预测回放',original),('v8：朝向保护与公交横向约束',stable)],f,'同一相机、同一高斯资产；原始预测保留，右栏是明确修正的派生回放。'),
                'heading_ablation':panel([('v7 原始回放',original),('仅修黑车异常朝向',heading),('v8：再加入车体稳定化',stable)],f,'中栏只改一个黑车朝向采样；右栏另有公交横向修正和观测车体位姿稳定化。'),
                'recorded_before_after':panel([('真实照片',photo),('v7 候选：逐帧位姿',real_old),('v8：有界三维车体稳定化',real_new)],f,'记录相机不变；高斯形状、纹理、透明度与背景保持原值。'),
                'black_car_closeup':zoom(original,heading,(640,215,860,335),f,'右侧黑车放大：只修异常朝向，位置与时序不变；重点看 7～8 秒。'),
                'bus_closeup':zoom(real_old,real_new,(300,90,660,450),f,'记录轨迹中的公交车放大：比较车窗、车顶与车身边缘的左右摆动。')}
            for name,im in frames.items():
                writers[name].append_data(np.array(im))
                if f in [25,55,70,95]:im.save(out/f'{name}_{f:03d}.jpg')
            if f%10==0:print('V8_REVIEW '+str(f),flush=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        before=json.loads((old/'assets_motion_validation/validation.json').read_text())['actual_camera_and_actor_poses'];after=json.loads((v/'observed_validation/validation.json').read_text())['actual_camera_and_actor_poses'];A=np.array([x['actors_world'][6] for x in before]);B=np.array([x['actors_world'][6] for x in after]);ya=np.rad2deg(np.unwrap(np.arctan2(A[:,1,0],A[:,0,0])));yb=np.rad2deg(np.unwrap(np.arctan2(B[:,1,0],B[:,0,0])));time=np.arange(80)/10
        fig,ax=plt.subplots(figsize=(11,4));ax.plot(time,ya-ya[0],label='v7 per-frame body fit',color='#b34a42',alpha=.8);ax.plot(time,yb-ya[0],label='v8 bounded 3D refinement',color='#165db7',linewidth=2);ax.set(xlabel='Time (s)',ylabel='Bus yaw relative to first v7 frame (deg)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'bus_yaw_comparison.png',dpi=150);plt.close(fig)
    except ImportError:pass
    report={'status':'complete','videos':[name+'.mp4' for name in names],'frames_per_video':80,'fps':10,'seconds':8,'no_temporal_pixel_filter_blending_or_sharpening':True,'new_prediction_is_explicit_derived_bundle':True,'audit':audit}
    (out/'review.json').write_text(json.dumps(report,indent=2)+'\n');print('V8_REVIEW_COMPLETE',flush=True)


if __name__=='__main__':main()
