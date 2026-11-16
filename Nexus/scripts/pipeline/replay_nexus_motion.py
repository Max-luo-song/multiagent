"""Controlled camera/actor motion diagnostics using recorded actual render states."""
import argparse,copy,gc,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','source','scene','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--frames',type=int,nargs='+',default=[20,40,55,70,85,99])
    p.add_argument('--variants',nargs='+',choices=['fixed','camera','actors','both','rotation','translation'],default=['fixed','camera','actors','both'])
    p.add_argument('--full-video',action='store_true')
    p.add_argument('--logged-reference',action='store_true',help='Keep checkpoint actors and use saved logged cameras')
    p.add_argument('--sky-prior',type=Path,help='Optional native observed distant-sky layer checkpoint')
    p.add_argument('--sky-observations',type=Path)
    p.add_argument('--road-layer',type=Path,help='Observed native road Gaussians with real static occlusion guard')
    p.add_argument('--road-static-guard',type=Path)
    p.add_argument('--measured-background',type=Path)
    p.add_argument('--actor-appearance-checkpoint',type=Path,help='Preserve actor radiance when static fine-tuning learns a new exposure model')
    p.add_argument('--appearance-frame',type=int,help='Use one fixed observed exposure for generated cameras, avoiding learned time-dependent exposure flicker')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import imageio.v2 as imageio
    import numpy as np,torch
    from nexus_3dgs.replay import RecordedStates,camera_inputs,load_trainer
    torch.set_num_threads(8)
    states=RecordedStates(a.source,a.scene);trainer,cfg=load_trainer(a.checkpoint);rigid=trainer.models['RigidNodes'];num_frames=rigid.num_frames
    if a.actor_appearance_checkpoint:
        from nexus_3dgs.replay import preserve_actor_appearance
        preserve_actor_appearance(trainer,a.actor_appearance_checkpoint)
    sky_prior=None
    if a.sky_prior:
        if not a.sky_observations:raise ValueError('--sky-prior requires --sky-observations')
        from nexus_3dgs.replay import ObservedSky
        sky_prior=ObservedSky(a.sky_prior,a.sky_observations)
    road_layer=None
    if a.road_layer:
        from nexus_3dgs.replay import ObservedRoad
        road_layer=ObservedRoad(a.road_layer,trainer,static_guard=a.road_static_guard)
    measured=None
    if a.measured_background:
        from nexus_3dgs.replay import MeasuredBackground
        measured=MeasuredBackground(a.measured_background,trainer)
    originals={k:getattr(rigid,k).clone() for k in ['instances_trans','instances_quats','instances_fv']}
    frames=sorted(states.records) if a.full_video else a.frames
    first=min(states.records);reproduction=[]
    for variant in a.variants:
        directory=a.output/variant;directory.mkdir(exist_ok=True);writer=None
        run={'status':'started','renderer':'Omniscenegraph_3DGS','camera_mode':'logged' if a.logged_reference else 'generated_ego',
             'reference':a.logged_reference,'ground_registration':None if a.logged_reference else states.run['ground_registration'],
             'bundle_sha256':states.run['bundle_sha256'],'checkpoint':str(a.checkpoint.resolve()),
             'state_source':str(a.source.resolve()),'state_source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
             'diagnostic_control':variant,'pose_source':'checkpoint' if a.logged_reference else 'actual saved centres/yaw plus saved fixed 3D mounting',
             'frames':[]}
        if a.actor_appearance_checkpoint:run['actor_radiance_basis_source']=str(a.actor_appearance_checkpoint.resolve())
        if a.appearance_frame is not None:
            if not 0<=a.appearance_frame<num_frames:raise ValueError('Appearance frame outside checkpoint')
            run['fixed_exposure_source_frame']=a.appearance_frame
        if sky_prior:run['observed_sky_layer']={'checkpoint':str(a.sky_prior.resolve()),'observations':str(a.sky_observations.resolve()),'policy':'at least two eroded real sky observations, no confident static contradiction; dynamic Gaussian occlusion retained'}
        if road_layer:run['observed_road_layer']={'directory':str(a.road_layer.resolve()),'static_guard':str(a.road_static_guard.resolve()) if a.road_static_guard else None,'policy':'native metric Gaussians, observed surface ownership, real LiDAR static occlusion and dynamic Gaussian occlusion'}
        if measured:run['measured_background']={'directory':str(a.measured_background.resolve()),'policy':'native measured static Gaussians with depth-tested dynamic occlusion'}
        try:
            for f in frames:
                cf=f if variant in ['camera','both','rotation','translation'] else first
                af=f if variant in ['actors','both'] else first
                record=states.records[f];meta=copy.deepcopy(record['scene_metadata']);cm=states.records[cf]['scene_metadata']
                C=torch.tensor(states.S@np.array(cm['camera_world']),dtype=torch.float32,device='cuda')
                if variant in ['rotation','translation']:
                    initial=torch.tensor(states.S@np.array(states.records[first]['scene_metadata']['camera_world']),dtype=torch.float32,device='cuda')
                    if variant=='rotation':C[:3,3]=initial[:3,3]
                    else:C[:3,:3]=initial[:3,:3]
                K=torch.tensor(cm['intrinsics'],dtype=torch.float32,device='cuda');H,W=cm['height'],cm['width']
                with torch.no_grad():
                    if not a.logged_reference:states.set_actors(rigid,af,f)
                    image,camera=camera_inputs(C,K,f,H,W,num_frames,len(cfg.data.pixel_source.cameras))
                    if a.appearance_frame is not None:image['img_idx'].fill_(a.appearance_frame*len(cfg.data.pixel_source.cameras))
                    result=trainer(image,camera,novel_view=True);outputs=result[0] if isinstance(result,tuple) else result
                    base_rgb=outputs['rgb']
                    if measured:
                        outputs['rgb'],measured_coverage=measured(trainer,image,camera,outputs)
                        if f in a.frames:
                            (directory/'measured_coverage').mkdir(exist_ok=True)
                            imageio.imwrite(directory/'measured_coverage'/f'{f:03d}.png',(measured_coverage.cpu().numpy()*255).round().astype(np.uint8))
                    if sky_prior:
                        outputs['rgb'],sky_coverage=sky_prior(trainer,image,outputs)
                        if f in a.frames:
                            (directory/'sky_coverage').mkdir(exist_ok=True)
                            imageio.imwrite(directory/'sky_coverage'/f'{f:03d}.png',(sky_coverage.cpu().numpy()*255).round().astype(np.uint8))
                    if road_layer:
                        outputs['rgb'],road_owner=road_layer(trainer,image,camera,outputs)
                        if f in a.frames:
                            (directory/'road_coverage').mkdir(exist_ok=True)
                            imageio.imwrite(directory/'road_coverage'/f'{f:03d}.png',(road_owner.cpu().numpy()*255).round().astype(np.uint8))
                    if sky_prior or road_layer or measured:
                        from nexus_3dgs.replay import actor_volume_guard
                        guard=actor_volume_guard(rigid,f,C,K,H,W)
                        outputs['rgb']=torch.where(guard[...,None],base_rgb,outputs['rgb'])
                        assert torch.equal(outputs['rgb'][guard],base_rgb[guard])
                        run['actor_volume_pixels_preserved_exactly']=True
                    rgb=(outputs['rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
                    imageio.imwrite(directory/f'{f:03d}_0.png',rgb)
                    if a.full_video:
                        if writer is None:writer=imageio.get_writer(str(directory/'camera_0.mp4'),fps=10,macro_block_size=2,ffmpeg_params=['-threads','4','-crf','18'])
                        writer.append_data(rgb)
                    if f in a.frames:
                        layers={k:outputs[k].detach().cpu().numpy().astype(np.float32) for k in ['Background_rgb','Background_depth','Background_opacity','Dynamic_rgb','Dynamic_opacity','Dynamic_depth','rgb_sky','rgb_sky_blend','depth','opacity']}
                        (directory/'layers').mkdir(exist_ok=True);np.savez_compressed(directory/'layers'/f'{f:03d}.npz',**layers)
                        for name in ['Background_rgb','Dynamic_rgb','rgb_sky']:
                            imageio.imwrite(directory/'layers'/f'{f:03d}_{name}.png',(layers[name].clip(0,1)*255).round().astype(np.uint8))
                    meta['camera_world']=(np.linalg.inv(states.S)@C.cpu().numpy()).tolist();meta['intrinsics']=cm['intrinsics']
                    if af!=f:meta['actors']=copy.deepcopy(states.records[af]['scene_metadata']['actors'])
                    run['frames'].append({'frame_idx':f,'images':[{'camera_id':0,'timestamp_us':record['timestamp_us'],'file':f'{f:03d}_0.png','scene_metadata':meta}]})
                    source=a.source.parent/record['file']
                    if variant=='both' and source.is_file():
                        old=imageio.imread(source).astype(np.float32);delta=rgb.astype(np.float32)-old;mse=float(np.mean(delta**2));reproduction.append({'frame':f,'psnr':float(-10*np.log10(max(mse/255**2,1e-12))),'mean_abs_8bit':float(abs(delta).mean()),'max_abs_8bit':float(abs(delta).max())})
                    print(f'REPLAY {variant} {f}',flush=True)
                    del outputs,result
            run['status']='complete'
        finally:
            if writer:writer.close()
            (directory/'run.json').write_text(json.dumps(run,indent=2)+'\n')
            with torch.no_grad():
                for k,v in originals.items():getattr(rigid,k).copy_(v)
            for k,v in originals.items():assert torch.equal(getattr(rigid,k),v)
        gc.collect();torch.cuda.empty_cache()
    (a.output/'reproduction.json').write_text(json.dumps(reproduction,indent=2)+'\n')
    print('REPLAY_COMPLETE',flush=True)


if __name__=='__main__':main()
