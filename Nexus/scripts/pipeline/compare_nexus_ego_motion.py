"""Full-camera comparison at identical moving-ego cameras and actor poses."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,nargs='+',required=True);p.add_argument('--new',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--label',default='道路几何修正候选')
    p.add_argument('--old-label',default='原版 v3')
    p.add_argument('--allow-asset-calibration-delta',action='store_true',help='Same prediction and cameras, independently fitted vehicle assets; report calibration differences explicitly')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if len(a.old)!=len(a.new):raise ValueError('Variant counts differ')
    font=lambda size:ImageFont.truetype(str(ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'),size)
    title,small=font(28),font(22);previews=[];count=0
    out=a.output/'ego_motion_comparison.mp4';manifest=[];maximum_actor_delta=0.;actors_identical=True
    with imageio.get_writer(str(out),fps=10,macro_block_size=2,ffmpeg_params=['-threads','4','-crf','18']) as writer:
        for chapter,(old,new) in enumerate(zip(a.old,a.new),1):
            runs=[json.loads((d/'run.json').read_text()) for d in (old,new)]
            if runs[0]['bundle_sha256']!=runs[1]['bundle_sha256']:raise ValueError('Prediction bundles differ')
            def records(r):return {f['frame_idx']:i for f in r['frames'] for i in f['images'] if i['camera_id']==0}
            left,right=map(records,runs)
            if left.keys()!=right.keys():raise ValueError('Frame sets differ')
            first=left[min(left)]['scene_metadata']['camera_world']
            origin=np.array(first)[:3,3]
            for f in sorted(left):
                m0,m1=[items[f]['scene_metadata'] for items in (left,right)]
                for key in ('camera_world','intrinsics'):np.testing.assert_allclose(m0[key],m1[key],atol=1e-8,rtol=0)
                keys=('track_token','valid','center_world','yaw_world_rad','size_lwh')
                physical=lambda m:[{k:actor[k] for k in keys} for actor in m['actors']]
                same=physical(m0)==physical(m1);actors_identical &= same
                if not a.allow_asset_calibration_delta:assert same
                by_token=[{x['track_token']:x for x in m['actors']} for m in (m0,m1)]
                for token in set(by_token[0])&set(by_token[1]):
                    x,y=by_token[0][token],by_token[1][token]
                    if x['valid'] and y['valid']:
                        maximum_actor_delta=max(maximum_actor_delta,float(np.linalg.norm(np.array(x['center_world'])-y['center_world'])))
                canvas=Image.new('RGB',(1920,660),'#141b27');d=ImageDraw.Draw(canvas)
                d.text((20,10),a.old_label,font=title,fill='white');d.text((980,10),a.label,font=title,fill='white')
                for x,directory,items in [(0,old,left),(960,new,right)]:
                    canvas.paste(Image.open(directory/items[f]['file']).convert('RGB'),(x,60))
                travel=np.linalg.norm(np.array(m0['camera_world'])[:2,3]-origin[:2])
                condition='预测与相机相同，车辆资产分别重建' if a.allow_asset_calibration_delta else '两侧相机和车辆位姿相同'
                d.text((20,610),f'交互 {chapter} / {len(a.old)}    时刻 {(f-min(left))/10:.1f} s    相机平移 {travel:.2f} m    {condition}',font=small,fill='white')
                d.text((1270,610),'阶段候选：仍需检查建筑、车辆和接缝',font=small,fill='#ffc66d')
                writer.append_data(np.asarray(canvas));count+=1
                if f in (20,55,70,99):
                    canvas.save(a.output/f'chapter{chapter}_{f:03d}.jpg')
                    if chapter==len(a.old):previews.append(canvas.resize((1280,440)))
            manifest.append({'old':str(old.resolve()),'new':str(new.resolve()),'frames':len(left)})
    if previews:
        sheet=Image.new('RGB',(1280,440*len(previews)))
        for i,im in enumerate(previews):sheet.paste(im,(0,440*i))
        sheet.save(a.output/'contact_sheet.jpg')
    result={'status':'complete','frames':count,'fps':10,'actual_cameras_identical':True,'actual_actor_poses_identical':bool(actors_identical),'prediction_bundles_identical':True,
            'shared_visible_actor_max_center_delta_m':maximum_actor_delta,'chapters':manifest,'video':str(out.resolve()),
            'scope':'same generated trajectories and cameras; no frame blending, sharpening, recorded-camera substitution or trajectory change'}
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
