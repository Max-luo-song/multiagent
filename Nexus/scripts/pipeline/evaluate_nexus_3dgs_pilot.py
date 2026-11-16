"""Render measured and saved Nexus motion with one independently trained scene."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'Omniscenegraph'),str(ROOT)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['checkpoint','scene','bundle','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cv2,imageio.v2 as imageio,numpy as np,torch
    from PIL import Image,ImageDraw,ImageFont
    from nexus_3dgs.replay import load_trainer,camera_inputs
    from nexus_3dgs.runtime import TrajectoryController
    torch.set_num_threads(8);cv2.setNumThreads(4)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    checksum=sha(a.checkpoint)
    geom=json.loads((a.checkpoint.parent/'scene_geometry.json').read_text())
    trainer,cfg=load_trainer(a.checkpoint,scene_aabb=geom['scene_aabb']);rigid=trainer.models['RigidNodes']
    ctrl=TrajectoryController(a.bundle,a.scene,rigid,strict=False)
    meta=json.loads((a.scene/'nexus_scene.json').read_text());S=np.linalg.inv(meta['frames'][0]['ego_to_world']);W=np.linalg.inv(S)
    intr=np.loadtxt(a.scene/'intrinsics/0.txt');K=np.array([[intr[0]/2,0,intr[2]/2],[0,intr[1]/2,intr[3]/2],[0,0,1.]])
    extr=np.loadtxt(a.scene/'extrinsics/0.txt');Kt=torch.tensor(K,dtype=torch.float32,device='cuda')
    original={key:getattr(rigid,key).clone() for key in ('instances_trans','instances_quats','instances_fv')}
    fonts=ImageFont.truetype(str(ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'),22)
    rows=[];video_paths=['recorded.mp4','nexus.mp4','comparison.mp4']
    writers=[imageio.get_writer(str(a.output/n),fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','19']) for n in video_paths]
    try:
        for f in range(20,100):
            camera=next(c for c in meta['frames'][f]['images'] if c['camera_id']==0);timestamp=camera['timestamp_us'];Ct=torch.tensor(S@np.asarray(camera['ego_to_world'])@extr,dtype=torch.float32,device='cuda')
            with torch.no_grad():
                Ct=trainer.models['CamPose'](Ct,torch.tensor(f*len(cfg.data.pixel_source.cameras),device='cuda'))
                image,cam=camera_inputs(Ct,Kt,f,540,960,rigid.num_frames,len(cfg.data.pixel_source.cameras))
                result=trainer(image,cam,novel_view=True);result=result[0] if isinstance(result,tuple) else result
                if not torch.isfinite(result['rgb']).all():raise FloatingPointError('Non-finite recorded-view RGB')
                recorded=(result['rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
                with ctrl.edit_frame(f,timestamp) as ego:
                    reference,_=ctrl.sample(timestamp,reference=True)
                    from nexus_3dgs.trajectory_control import planar
                    generated=planar(ego)@np.linalg.inv(planar(reference[0]))@W@Ct.cpu().numpy()
                    image,cam=camera_inputs(torch.tensor(S@generated,dtype=torch.float32,device='cuda'),Kt,f,540,960,rigid.num_frames,len(cfg.data.pixel_source.cameras))
                    result=trainer(image,cam,novel_view=True);result=result[0] if isinstance(result,tuple) else result
                    if not torch.isfinite(result['rgb']).all():raise FloatingPointError('Non-finite predicted-view RGB')
                    predicted=(result['rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
            target=cv2.undistort(np.array(Image.open(a.scene/'images'/f'{f:03d}_0.jpg').convert('RGB').resize((960,540),Image.Resampling.BILINEAR)),K,intr[4:])
            error=((recorded.astype(float)-target)/255)**2
            row={'frame':f,'rgb_gradient_holdout':f in geom['test_timesteps'],'recorded_full_psnr_db':float(-10*np.log10(max(float(error.mean()),1e-12))),'timestamp_us':timestamp}
            rows.append(row)
            panel=Image.new('RGB',(2880,606),'#162135');draw=ImageDraw.Draw(panel)
            for i,(title,rgb) in enumerate([('真实照片',target),('本场景独立 3DGS：记录运动',recorded),('本场景独立 3DGS：保存的 Nexus 目标条件预测',predicted)]):
                draw.text((i*960+12,8),title,font=fonts,fill='white');panel.paste(Image.fromarray(rgb),(i*960,40))
            draw.text((12,580),f'帧 {f} | 新场景初始重建检查；没有达到原场景 v9 的视觉验收结论。预测新视角没有配对照片，不计算 PSNR。',font=fonts,fill='#ffcf90')
            for writer,im in zip(writers,[recorded,predicted,np.array(panel)]):writer.append_data(im)
            if f in (20,50,70,99):
                panel.save(a.output/f'comparison_{f:03d}.jpg');imageio.imwrite(a.output/f'nexus_{f:03d}.png',predicted)
            if f%10==0:print('PILOT_EVALUATION_FRAME',f,flush=True)
    finally:
        for w in writers:w.close()
    for key,value in original.items():assert torch.equal(value,getattr(rigid,key))
    assert sha(a.checkpoint)==checksum
    checks={}
    for name in video_paths:
        with imageio.get_reader(a.output/name) as reader:fps=reader.get_meta_data()['fps'];count=sum(1 for _ in reader)
        assert count==80 and fps==10;checks[name]={'decoded_frames':count,'fps':fps}
    means={}
    for tag,selected in [('all',rows),('rgb_gradient_holdout',[r for r in rows if r['rgb_gradient_holdout']]),('training_rgb',[r for r in rows if not r['rgb_gradient_holdout']])]:
        means[tag]={'frames':len(selected),'mean_recorded_psnr_db':float(np.mean([r['recorded_full_psnr_db'] for r in selected])) if selected else None}
    report={'status':'complete','checkpoint_sha256':checksum,'own_scene_assets':True,'checkpoint_unchanged':True,'temporary_actor_poses_restored':True,
            'scene':str(a.scene),'bundle':str(a.bundle),'bound_prediction_actors':len(ctrl.bindings),'unbound_prediction_tokens':ctrl.missing,
            'rendered_asset_count':rigid.num_instances,'metrics':means,'rows':rows,'video_checks':checks,
            'nexus_task':ctrl.manifest['task'],'trajectory_filtering_applied':False,
            'scope':'Engineering pilot on existing evaluated scenes; independent per-scene reconstruction. RGB gradient holdout shares LiDAR and initialization. No unseen-trip generalization claim.',
            'actor_policy':'bound actors use planar deltas from raw saved prediction, retaining reconstructed height and tilt; other assets retain recorded poses; absent assets explicitly listed',
            'limitations':['No v9-quality acceptance claimed for new scenes.','Pedestrians/cyclists are not separately animated.','Unobserved surfaces and novel viewpoints remain limited.','No predicted-view PSNR or safety guarantee.']}
    (a.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print('PILOT_EVALUATION_COMPLETE',json.dumps(means),flush=True)


if __name__=='__main__':main()
