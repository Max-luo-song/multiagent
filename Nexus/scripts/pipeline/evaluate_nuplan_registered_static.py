"""Validate a registered static reconstruction against measured photographs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['checkpoint','scene','old-checkpoint','old-scene','old-reference','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--baseline-label',default='原 v5：原拟合相机与车辆位姿')
    p.add_argument('--candidate-label',default='候选版本：校正数据重新重建')
    p.add_argument('--frames',type=int,nargs='+',default=[20,55,63,65,70,75,80,85,90,95,99,110,120,130]);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cv2,imageio.v2 as imageio,numpy as np,torch
    from PIL import Image,ImageDraw,ImageFont
    from nexus_3dgs.replay import load_trainer,camera_inputs
    from nexus_3dgs.geometry import quaternion_matrix,matrix_quaternion
    cv2.setNumThreads(4);torch.set_num_threads(8);new,cfg=load_trainer(a.checkpoint);old,oldcfg=load_trainer(a.old_checkpoint);rigid=new.models['RigidNodes'];oldrigid=old.models['RigidNodes'];before={k:getattr(oldrigid,k).clone() for k in ['instances_trans','instances_quats','instances_fv']};meta=json.loads((a.scene/'nexus_scene.json').read_text());oldmeta=json.loads((a.old_scene/'nexus_scene.json').read_text());S=np.linalg.inv(meta['frames'][0]['ego_to_world']);W=np.linalg.inv(S);oldS=np.linalg.inv(oldmeta['frames'][0]['ego_to_world']);info=json.loads((a.scene/'instances/instances_info.json').read_text());oldinfo=json.loads((a.old_scene/'instances/instances_info.json').read_text())
    for m,(i,j) in enumerate(zip(rigid.instance_ids.cpu().tolist(),oldrigid.instance_ids.cpu().tolist())):assert info[str(i)]['id']==oldinfo[str(j)]['id']
    intr=np.loadtxt(a.scene/'intrinsics/0.txt');K=np.array([[intr[0]/2,0,intr[2]/2],[0,intr[1]/2,intr[3]/2],[0,0,1.]]);extr=np.loadtxt(a.scene/'extrinsics/0.txt');signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]);font=ImageFont.truetype(str(ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'),24);rows=[];pose_records=[]
    try:
        for f in a.frames:
            cam=next(c for c in meta['frames'][f]['images'] if c['camera_id']==0);Cworld=np.array(cam['ego_to_world'])@extr;C=S@Cworld;Ct=torch.tensor(C,dtype=torch.float32,device='cuda');Kt=torch.tensor(K,dtype=torch.float32,device='cuda')
            with torch.no_grad():
                Ct=new.models['CamPose'](Ct,torch.tensor(f*len(cfg.data.pixel_source.cameras),device='cuda'))
            Cworld=W@Ct.cpu().numpy()
            target=cv2.undistort(np.array(Image.open(a.scene/'images'/f'{f:03d}_0.jpg').convert('RGB').resize((960,540),Image.Resampling.BILINEAR)),K,intr[4:]);rgb={};poses=[]
            new.render_each_class=f in [20,70,99]
            with torch.no_grad():
                image,camera=camera_inputs(Ct,Kt,f,540,960,rigid.num_frames,len(cfg.data.pixel_source.cameras));result=new(image,camera,novel_view=True);result=result[0] if isinstance(result,tuple) else result;rgb['new']=(result['rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
                if f in [20,70,99]:
                    for layer in ['Dynamic_rgb','Dynamic_opacity','Background_rgb','Background_opacity']:
                        if layer in result:
                            value=(result[layer].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
                            if value.shape[-1]==1:value=value[...,0]
                            imageio.imwrite(a.output/f'{f:03d}_{layer}.png',value)
                for m in range(rigid.num_instances):
                    B=np.eye(4);B[:3,:3]=quaternion_matrix(rigid.instances_quats[f,m].detach().cpu().numpy());B[:3,3]=rigid.instances_trans[f,m].detach().cpu().numpy();poses.append(W@B)
                if f<oldrigid.num_frames:
                    for m,B in enumerate(poses):
                        local=oldS@B;oldrigid.instances_trans[f,m].copy_(torch.tensor(local[:3,3],dtype=torch.float32,device='cuda'));oldrigid.instances_quats[f,m].copy_(torch.tensor(matrix_quaternion(local[:3,:3]),dtype=torch.float32,device='cuda'));oldrigid.instances_fv[f,m]=rigid.instances_fv[f,m]
                    image,camera=camera_inputs(torch.tensor(oldS@Cworld,dtype=torch.float32,device='cuda'),Kt,f,540,960,oldrigid.num_frames,len(oldcfg.data.pixel_source.cameras));result=old(image,camera,novel_view=True);result=result[0] if isinstance(result,tuple) else result;rgb['old_same_camera']=(result['rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
            path=a.old_reference/f'{f:03d}_0.png'
            if path.is_file():rgb['old_native']=imageio.imread(path)
            excluded=(cv2.imread(str(a.scene/'vehicle_interior_masks'/f'{f:03d}_0.png'),0)>0).astype(np.uint8)
            for m,B in enumerate(poses):
                if not rigid.instances_fv[f,m]:continue
                size=rigid.instances_size[m].detach().cpu().numpy()+.8;q=((signs*size/2)@B[:3,:3].T+B[:3,3]-Cworld[:3,3])@Cworld[:3,:3]
                if q[:,2].min()<=.1:continue
                uv=q[:,:2]/q[:,2,None]@K[:2,:2].T+K[:2,2];cv2.fillConvexPoly(excluded,cv2.convexHull(np.rint(np.clip(uv,-100000,100000)).astype(np.int32)),1)
            quiet=cv2.dilate(excluded,np.ones((25,25),np.uint8))==0;imageio.imwrite(a.output/f'{f:03d}_static_mask.png',quiet.astype(np.uint8)*255);imageio.imwrite(a.output/f'{f:03d}_photo.png',target);row={'frame':f,'held_out_rgb':f in range(10,len(meta['frames']),10),'static_pixels':int(quiet.sum()),'metrics':{}}
            for name,im in rgb.items():
                (a.output/name).mkdir(exist_ok=True);imageio.imwrite(a.output/name/f'{f:03d}_0.png',im);error=((im.astype(float)-target)/255)**2;row['metrics'][name]={'full_psnr':float(-10*np.log10(max(float(error.mean()),1e-12))),'static_psnr':float(-10*np.log10(max(float(error[quiet].mean()),1e-12)))}
            for tag,baseline,label in [('pipeline','old_native',a.baseline_label),('same_camera','old_same_camera','基准模型：同一相机和车辆位姿')]:
                if baseline not in rgb:continue
                canvas=Image.new('RGB',(2880,650),'#142136');draw=ImageDraw.Draw(canvas)
                for j,(title,im) in enumerate([('真实照片',target),(label,rgb[baseline]),(a.candidate_label,rgb['new'])]):draw.text((j*960+15,10),title,font=font,fill='white');canvas.paste(Image.fromarray(im),(j*960,50))
                draw.text((15,605),f'帧 {f}　'+('留出 RGB 验证帧' if row['held_out_rgb'] else '训练图像重建检查')+'　车辆与背景分开检查；背景指标采用共同车辆排除区域。',font=font,fill='#ffd18c');canvas.save(a.output/f'{tag}_{f:03d}.jpg')
            rows.append(row);pose_records.append({'frame':f,'camera_world':Cworld.tolist(),'actors_world':[B.tolist() for B in poses]});print('REGISTERED_STATIC_VALIDATION '+str(f)+' '+json.dumps(row),flush=True)
    finally:
        with torch.no_grad():
            for k,value in before.items():getattr(oldrigid,k).copy_(value)
        for k,value in before.items():assert torch.equal(getattr(oldrigid,k),value)
    (a.output/'validation.json').write_text(json.dumps({'status':'complete','checkpoint':str(a.checkpoint.resolve()),'scope':'new pipeline vs measured photos; old_native uses its original learned poses; old_same_camera uses identical corrected physical camera and new actual fitted body poses; actor appearance assets are reused','rows':rows,'actual_camera_and_actor_poses':pose_records},indent=2)+'\n');print('REGISTERED_STATIC_VALIDATION_COMPLETE',flush=True)


if __name__=='__main__':main()
