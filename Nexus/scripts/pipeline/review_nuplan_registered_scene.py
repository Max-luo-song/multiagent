"""Create full-length review videos, scores, and an honest candidate status."""
import argparse, json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['validation','old-reference','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--nexus',type=Path);p.add_argument('--old-nexus',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cv2,imageio.v2 as imageio,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    cv2.setNumThreads(4)
    root=Path(__file__).resolve().parents[2]
    font=ImageFont.truetype(str(root/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'),25)
    data=json.loads((a.validation/'validation.json').read_text());rows=data['rows'];assert [r['frame'] for r in rows]==list(range(20,100))
    def panel(images,f,subtitle):
        canvas=Image.new('RGB',(2880,650),'#142136');draw=ImageDraw.Draw(canvas)
        for j,(label,im) in enumerate(images):draw.text((j*960+15,10),label,font=font,fill='white');canvas.paste(Image.fromarray(im),(j*960,50))
        draw.text((15,605),f'帧 {f}　'+subtitle,font=font,fill='#ffd18c')
        return np.array(canvas)
    cases=[('real_motion_comparison','真实照片','旧 v5：记录轨迹','v7 候选：记录轨迹')]
    if a.nexus:
        assert a.old_nexus and json.loads((a.nexus/'run.json').read_text())['status']=='passed'
        cases.append(('nexus_prediction_comparison','真实照片（观察位置不同）','旧 v5：原始 Nexus 条件预测','v7 候选：同一原始 Nexus 条件预测'))
    for name,left,middle,right in cases:
        with imageio.get_writer(str(a.output/(name+'.mp4')),format='FFMPEG',fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18']) as writer:
            for r in rows:
                f=r['frame'];photo=imageio.imread(a.validation/f'{f:03d}_photo.png')
                old=imageio.imread((a.old_reference if name=='real_motion_comparison' else a.old_nexus)/f'{f:03d}_0.png')
                new=imageio.imread((a.validation/'new' if name=='real_motion_comparison' else a.nexus)/f'{f:03d}_0.png')
                subtitle=('相同记录时刻；保留全部 80 帧，无锐化或时间滤波。' if name=='real_motion_comparison' else '预测文件不变；新视角无配对真实照片，不计算预测视角 PSNR。')
                frame=panel([(left,photo),(middle,old),(right,new)],f,subtitle);writer.append_data(frame)
                if f in [20,70,99]:Image.fromarray(frame).save(a.output/f'{name}_{f:03d}.jpg')
    summary={}
    for group,selected in [('stationary',[r for r in rows if r['frame']<=62]),('moving',[r for r in rows if r['frame']>=63]),('moving_rgb_validation',[r for r in rows if r['frame'] in [70,80,90]]),('all',rows)]:
        summary[group]={'frames':len(selected)}
        for version in ['old_native','new']:
            summary[group][version]={key:float(np.mean([r['metrics'][version][key] for r in selected])) for key in ['full_psnr','static_psnr']}
    # This threshold is a review gate, never a claim of novel-view ground truth.
    passed=summary['moving']['new']['full_psnr']>summary['moving']['old_native']['full_psnr'] and summary['stationary']['new']['full_psnr']>=summary['stationary']['old_native']['full_psnr']-1
    report={'status':'complete','candidate_passed_numeric_review_gate':passed,
            'gate':'moving full-image mean PSNR improves and stationary mean drops by at most 1 dB; visual review remains necessary',
            'checkpoint':data['checkpoint'],'frames':80,'fps':10,'seconds':8,'summary':summary,
            'quality_scope':'observed reconstruction fidelity; current RGB excluded from new gradient training at every tenth time; inherited foreground assets saw original full scene; all LiDAR used for pose registration',
            'generated_camera_has_no_paired_ground_truth':True,'no_temporal_filter_or_sharpening':True}
    (a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
