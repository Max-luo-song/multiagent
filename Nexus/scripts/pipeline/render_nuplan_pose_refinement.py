"""Test independent LiDAR pose corrections on the unchanged v5 renderer."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['scene','control','icp','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);print('POSE_REFINEMENT loading unchanged v5 scene',flush=True)
    import cv2,imageio.v2 as imageio,numpy as np,torch
    from PIL import Image,ImageDraw,ImageFont
    from skimage.metrics import structural_similarity
    from nexus_3dgs.replay import load_trainer,camera_inputs
    from nexus_3dgs.geometry import quaternion_matrix
    cv2.setNumThreads(4);torch.set_num_threads(8)
    control=json.loads(a.control.read_text());assert control['status']=='complete';checkpoint=Path(control['checkpoint']);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before_sha=sha(checkpoint);trainer,cfg=load_trainer(checkpoint);rigid=trainer.models['RigidNodes'];versions=[(p,p._version) for model in trainer.models.values() for p in model.parameters()]
    scene=json.loads((a.scene/'nexus_scene.json').read_text());S=np.linalg.inv(scene['frames'][0]['ego_to_world']);worldS=np.linalg.inv(S);extr=np.loadtxt(a.scene/'extrinsics/0.txt');records={r['frame_idx']:r for r in control['frames']};correction={r['frame']:np.array(r['scene_correction_matrix']) for r in json.loads(a.icp.read_text())['rows']};correction[20]=np.eye(4)
    factory=lambda f:S@np.array(next(x for x in scene['frames'][f]['images'] if x['camera_id']==0)['ego_to_world'])@extr
    D20=np.linalg.inv(factory(20))@S@np.array(records[20]['camera_world']);frames=sorted(set(records)&set(correction));frames=[f for f in frames if f in [20,40,55,60,63,65,70,75,80,85,90,99]];signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    font=ImageFont.truetype(str(ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'),25);summary=[];outputs_metadata=[]
    for f in frames:
        K=np.array(records[f]['intrinsics']);Ccases={'native':S@np.array(records[f]['camera_world']),'factory':factory(f),'icp':correction[f]@factory(f)@D20};rendered={};H,W=540,960;blocked=np.zeros((H,W),np.uint8)
        with torch.no_grad():
            for name,C in Ccases.items():
                Ct=torch.tensor(C,dtype=torch.float32,device='cuda');Kt=torch.tensor(K,dtype=torch.float32,device='cuda');image,camera=camera_inputs(Ct,Kt,f,H,W,rigid.num_frames,len(cfg.data.pixel_source.cameras));image['img_idx'].fill_(20*len(cfg.data.pixel_source.cameras));result=trainer(image,camera,novel_view=True);out=result[0] if isinstance(result,tuple) else result;rgb=(out['rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8);rendered[name]=rgb;(a.output/name).mkdir(exist_ok=True);imageio.imwrite(a.output/name/f'{f:03d}_0.png',rgb)
                for m in range(rigid.num_instances):
                    if not rigid.instances_fv[f,m]:continue
                    R=quaternion_matrix(rigid.instances_quats[f,m].detach().cpu().numpy());t=rigid.instances_trans[f,m].detach().cpu().numpy();size=rigid.instances_size[m].detach().cpu().numpy()+.8;q=((signs*size/2)@R.T+t-C[:3,3])@C[:3,:3]
                    if q[:,2].min()<=.1:continue
                    uv=q[:,:2]/q[:,2,None]@K[:2,:2].T+K[:2,2];cv2.fillConvexPoly(blocked,cv2.convexHull(np.rint(np.clip(uv,-100000,100000)).astype(np.int32)),1)
            target=imageio.imread(a.control.parent.parent/'original'/f'{f:03d}_0.png');quiet=cv2.dilate(blocked,np.ones((25,25),np.uint8))==0;imageio.imwrite(a.output/f'{f:03d}_static_mask.png',quiet.astype(np.uint8)*255)
            row={'frame':f,'static_pixels':int(quiet.sum()),'metrics':{}}
            for name,rgb in rendered.items():
                error=((rgb.astype(float)-target)/255)**2;row['metrics'][name]={'full_psnr':float(-10*np.log10(max(float(error.mean()),1e-12))),'static_psnr':float(-10*np.log10(max(float(error[quiet].mean()),1e-12)))}
            summary.append(row);canvas=Image.new('RGB',(W*3,H+105),'#142136');d=ImageDraw.Draw(canvas)
            for j,(name,label) in enumerate([('original','真实照片'),('native','原 v5：学习相机校正'),('icp','候选：独立点云配准校正')]):
                d.text((j*W+15,10),label,font=font,fill='white');canvas.paste(Image.fromarray(target if name=='original' else rendered[name]),(j*W,50))
            d.text((15,H+65),f'帧 {f}　只比较相机校正；高斯和车辆位姿不变。车辆边缘可能错位，先检查建筑、树木和道路。',font=font,fill='#ffd18c');canvas.save(a.output/f'comparison_{f:03d}.jpg')
            outputs_metadata.append({'frame':f,'cameras_scene':{k:C.tolist() for k,C in Ccases.items()}})
        print('POSE_REFINEMENT '+str(f)+' '+json.dumps(row['metrics']),flush=True)
    assert all(p._version==version for p,version in versions);assert sha(checkpoint)==before_sha
    report={'status':'complete','checkpoint_sha256':before_sha,'checkpoint_and_all_parameters_unchanged':True,'scope':'small diagnostic: only camera poses change; actual fitted vehicle poses intentionally fixed, so full RGB is not final registered traffic output; prioritize common vehicle-excluded pixels','correction_policy':'LiDAR-only static registration, followed by fixed first-frame native calibration; no per-frame learned camera correction on candidate','frames':summary,'actual_cameras':outputs_metadata};(a.output/'pose_render_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('POSE_REFINEMENT_COMPLETE',flush=True)


if __name__=='__main__':main()
