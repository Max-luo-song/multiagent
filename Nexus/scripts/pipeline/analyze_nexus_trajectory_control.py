"""Build understandable trajectory controls; score only measured camera views."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();v=a.experiment;out=v/'review';out.mkdir(exist_ok=True)
    import cv2,imageio.v2 as imageio,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from skimage.metrics import structural_similarity
    cv2.setNumThreads(4)
    audit=json.loads((v/'audit.json').read_text());assert audit['status']=='passed';rawaudit=json.loads((v/'raw_predictions_audit.json').read_text());assert rawaudit['status']=='passed';conditions=audit['conditions']+rawaudit['conditions'];runs={name:json.loads((v/name/'run.json').read_text()) for name in conditions}
    records={name:{r['frame_idx']:r for r in run['frames']} for name,run in runs.items()};frames=sorted(records['factory_rr']);motion={r['frame']:r for r in audit['motion']}
    for name,run in runs.items():assert run['status']=='complete' and sorted(records[name])==frames
    # Older smoke snapshots may predate explicit size export; full runs require it.
    for name in ['factory_rr','native_rr','native_fitted']:
        assert all('size_lwh' in b for b in records[name][frames[0]]['actors'])
    signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    rows=[];invariance=[]
    def block_actors(blocked,record,padding=0.):
        C=np.array(record['camera_world']);K=np.array(record['intrinsics'])
        for body in record['actors']:
            if not body['valid']:continue
            B=np.array(body['pose_world']);q=((signs*(np.array(body['size_lwh'])+padding)/2)@B[:3,:3].T+B[:3,3]-C[:3,3])@C[:3,:3]
            if q[:,2].min()<=.1:continue
            uv=q[:,:2]/q[:,2,None]@K[:2,:2].T+K[:2,2]
            if np.isfinite(uv).all():cv2.fillConvexPoly(blocked,cv2.convexHull(np.rint(np.clip(uv,-100000,100000)).astype(np.int32)),1)
    for f in frames:
        target=imageio.imread(v/'original'/f'{f:03d}_0.png').astype(np.float32)/255;H,W=target.shape[:2];blocked=np.zeros((H,W),np.uint8)
        for name in ['factory_rr','native_rr','native_fitted']:
            block_actors(blocked,records[name][f])
        blocked=cv2.dilate(blocked,np.ones((17,17),np.uint8));static=blocked==0
        rowmotion=motion[f];active=rowmotion['measured_camera_travel_m']>.1 or rowmotion['measured_camera_rotation_change_deg']>.2
        for name in ['factory_rr','native_rr','native_fitted']:
            rgb=imageio.imread(v/name/f'{f:03d}_0.png').astype(np.float32)/255;error=(rgb-target)**2
            if name=='native_rr':real_rgb=rgb
            rows.append({'frame':f,'condition':name,'phase':'moving' if active else 'stationary','static_pixels':int(static.sum()),'psnr':float(-10*np.log10(max(float(error.mean()),1e-12))),'ssim':float(structural_similarity(target,rgb,channel_axis=-1,data_range=1.)),'background_mask_psnr':float(-10*np.log10(max(float(error[static].mean()),1e-12)))})
        exclusion=np.zeros((H,W),np.uint8)
        for name in ['native_rr','raw_goal_rn']:block_actors(exclusion,records[name][f],padding=.8)
        quiet=cv2.dilate(exclusion,np.ones((25,25),np.uint8))==0
        changed=imageio.imread(v/'raw_goal_rn'/f'{f:03d}_0.png').astype(np.float32)/255;delta=changed[quiet]-real_rgb[quiet]
        invariance.append({'frame':f,'pixels':int(quiet.sum()),'background_difference_psnr':float(-10*np.log10(max(float(np.mean(delta**2)),1e-12))),'mean_abs_8bit':float(np.abs(delta).mean()*255),'fraction_channels_different_by_over_2':float((np.abs(delta)>2.01/255).mean())})
        if f%10==0:print('TRAJECTORY_QUALITY '+str(f),flush=True)
    summary={}
    for name in ['factory_rr','native_rr','native_fitted']:
        summary[name]={}
        for phase in ['all','stationary','moving']:
            rr=[r for r in rows if r['condition']==name and (phase=='all' or r['phase']==phase)]
            summary[name][phase]={'frames':len(rr),**{key:float(np.mean([r[key] for r in rr])) if rr else None for key in ['psnr','ssim','background_mask_psnr']}}
    quality={'status':'complete','scope':'only measured cameras compared with the corresponding measured photos; no PSNR or SSIM assigned to novel generated cameras','phase_policy':'measured ego displacement > 0.1 m OR measured rotation change > 0.2 deg from first exposure','background_mask_policy':'exclude conservative union of projected modeled vehicle boxes in all three measured-camera controls, dilated eight pixels; approximate exclusion, not semantic truth','summary':summary,'rows':rows};(v/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    (v/'fixed_camera_background.json').write_text(json.dumps({'status':'complete','scope':'same real camera, real actors vs unedited goal-predicted actors; compare pixels outside both projected modeled actor volumes with 0.4 m per-side expansion and 12 pixel dilation; this is invariance, not image quality against truth','mean_abs_8bit':float(np.mean([r['mean_abs_8bit'] for r in invariance])),'mean_fraction_channels_different_by_over_2':float(np.mean([r['fraction_channels_different_by_over_2'] for r in invariance])),'rows':invariance},indent=2)+'\n')
    font=lambda n:ImageFont.truetype(str(ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'),n)
    title,small=font(28),font(22)
    layouts=[('real_vs_nexus','native',[('original','原始真实录像'),('native_rr','3DGS：真实轨迹'),('raw_goal_nn','3DGS：Nexus 条件预测原始输出')]),('free_prediction','native',[('original','原始真实录像'),('native_rr','3DGS：真实轨迹'),('raw_free_nn','3DGS：Nexus 自由预测原始输出')]),('current_interaction','native',[('original','原始真实录像'),('native_rr','3DGS：真实轨迹'),('native_nn','3DGS：现有交互轨迹（含离线约束）')]),('factory_real_vs_nexus','factory',[('original','原始真实录像'),('factory_rr','3DGS：真实轨迹'),('factory_nn','3DGS：现有交互轨迹（含离线约束）')]),('fitted_reference','native fitted',[('original','原始真实录像'),('native_fitted','3DGS：记录视角／模型拟合位姿'),('native_rr','3DGS：真实标注／固定资产挂载')])]
    video_counts={};poster_frames=[f for f in [20,55,70,85,99] if f in frames]
    for name,calibration,columns in layouts:
        with imageio.get_writer(str(out/f'{name}.mp4'),fps=10,macro_block_size=2,ffmpeg_params=['-threads','4','-crf','18']) as writer:
            for f in frames:
                canvas=Image.new('RGB',(2880,660),'#142136');d=ImageDraw.Draw(canvas)
                for j,(directory,label) in enumerate(columns):
                    d.text((j*960+18,10),label,font=title,fill='white');canvas.paste(Image.open(v/directory/f'{f:03d}_0.png').convert('RGB'),(j*960,60))
                m=motion[f];predtravel=m['nexus_camera_travel_m']
                if name in ['real_vs_nexus','free_prediction']:
                    key='goal' if name=='real_vs_nexus' else 'free';predtravel=next(x['travel_m'] for x in rawaudit['motion'][key] if x['frame']==f)
                d.text((18,610),f'时刻 {(f-frames[0])/10:.1f} 秒　同一 v5 模型／无训练　真实自车位移 {m["measured_camera_travel_m"]:.2f} 米　右侧自车位移 {predtravel:.2f} 米',font=small,fill='white')
                d.text((1780,610),'轨迹不同，视角和车辆位置可不同',font=small,fill='#ffd18c');writer.append_data(np.array(canvas))
                if f in poster_frames:canvas.save(out/f'{name}_{f:03d}.jpg')
        video_counts[name]=len(frames)
    grid=[('native_rr','真实相机运动＋真实车辆运动'),('raw_goal_nr','Nexus 条件相机运动＋真实车辆运动'),('raw_goal_rn','真实相机运动＋Nexus 条件车辆运动'),('raw_goal_nn','Nexus 条件相机运动＋Nexus 条件车辆运动')]
    with imageio.get_writer(str(out/'motion_factorial.mp4'),fps=10,macro_block_size=2,ffmpeg_params=['-threads','4','-crf','18']) as writer:
        for f in frames:
            canvas=Image.new('RGB',(1920,1240),'#142136');d=ImageDraw.Draw(canvas)
            for j,(directory,label) in enumerate(grid):
                x=(j%2)*960;y=(j//2)*600;d.text((x+16,y+8),label,font=title,fill='white');canvas.paste(Image.open(v/directory/f'{f:03d}_0.png'),(x,y+52))
            d.text((16,1207),f'时刻 {(f-frames[0])/10:.1f} 秒　两侧采用相同学习校正；横向只替换相机运动，纵向只替换其他车辆运动',font=small,fill='white');writer.append_data(np.array(canvas))
            if f in poster_frames:canvas.save(out/f'motion_factorial_{f:03d}.jpg')
    video_counts['motion_factorial']=len(frames)
    # Pose-only observation support: do not present this as a learned perceptual score.
    bus_index=next(b['model_index'] for b in records['native_rr'][frames[0]]['actors'] if b['track_token']=='441830199c6752a5')
    observed=[];camera_support=[]
    for f in frames:
        r=records['native_fitted'][f];C=np.array(r['camera_world']);camera_support.append(C[:3,3]);b=next(b for b in r['actors'] if b['model_index']==bus_index)
        if b['valid']:
            B=np.array(b['pose_world']);observed.append((C[:3,3]-B[:3,3])@B[:3,:3])
    observed=np.array(observed);camera_support=np.array(camera_support);support={}
    for name in ['native_rr','native_nn','raw_free_nn','raw_goal_nn','raw_goal_nr','raw_goal_rn']:
        entries=[]
        for f in frames:
            r=records[name][f];C=np.array(r['camera_world']);b=next(b for b in r['actors'] if b['model_index']==bus_index)
            row={'frame':f,'nearest_observed_front_camera_position_distance_m':float(np.linalg.norm(camera_support-C[:3,3],axis=1).min())}
            if b['valid']:
                B=np.array(b['pose_world']);q=(C[:3,3]-B[:3,3])@B[:3,:3];row.update(bus_center_distance_m=float(np.linalg.norm(q)),nearest_observed_bus_local_camera_position_distance_m=float(np.linalg.norm(observed-q,axis=1).min()))
            entries.append(row)
        support[name]=entries
    support_report={'status':'complete','scope':'geometry-only distances to 80 fitted measured front-camera observations; other training cameras and viewing directions are not represented, so this is not a complete visibility or rendering-quality certificate','observed_front_bus_center_distance_range_m':[float(np.linalg.norm(observed,axis=1).min()),float(np.linalg.norm(observed,axis=1).max())],'conditions':support};(v/'observation_support.json').write_text(json.dumps(support_report,indent=2)+'\n')
    timegroups={phase:[r['frame'] for r in rows if r['condition']=='native_rr' and r['phase']==phase] for phase in ['stationary','moving']}
    for left,right,key in [('native_rr','raw_goal_rn','camera_world'),('raw_goal_nr','raw_goal_nn','camera_world'),('raw_goal_rn','raw_goal_nn','actors')]:
        for f in frames:
            if key=='camera_world':np.testing.assert_allclose(records[left][f][key],records[right][f][key],atol=1e-8,rtol=0)
            else:assert records[left][f][key]==records[right][f][key],(left,right,key,f)
    result={'status':'complete','frames':len(frames),'fps':10,'same_checkpoint_for_all_conditions':True,'raw_factorial_same_cameras_and_generated_actor_states_verified':True,'measured_actor_roundtrip_error_audited_separately':True,'generated_camera_quality_has_no_paired_ground_truth':True,'phase_frames':timegroups,'comparison_videos':video_counts,'main_video_calibration':'same per-frame learned right-multiplied correction in both trajectories','factory_control_available':True,'fitted_training_pose_control_available':True};(out/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2));print('TRAJECTORY_REVIEW_COMPLETE',flush=True)


if __name__=='__main__':main()
