"""Show every reconstructed traffic actor before/after offline projection."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from nexus_3dgs.scene_traffic import SceneTraffic
from nexus_3dgs.scene_projection import pair_separations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--ground-model',type=Path,required=True);p.add_argument('--stop-render',type=Path,required=True)
    p.add_argument('--go-render',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--focus-conflicts',action='store_true',help='Zoom the diagram to conflicting side/rear tracks; simulation actors are not removed')
    a=p.parse_args();report=json.loads((a.experiment/'run.json').read_text())
    if report['status']!='complete':raise ValueError('Projection incomplete')
    names=('bus_stop','bus_follow_logged');data=[]
    for name,directory in zip(names,(a.stop_render,a.go_render)):
        run=json.loads((directory/'run.json').read_text())
        if run['status']!='complete':raise ValueError('Render incomplete')
        before=SceneTraffic(a.checkpoint,Path(report['source_filter_experiment'])/name,a.ground_model)
        after=SceneTraffic(a.checkpoint,a.experiment/name,a.ground_model)
        if run['bundle_sha256']!=after.controller.manifest['bundle_sha256']:raise ValueError('Render uses another bundle')
        records=[(f['frame_idx'],i) for f in run['frames'] for i in f['images'] if i['camera_id']==0]
        ts=np.array([i['timestamp_us'] for _,i in records]);left,lv=before.sample(ts);right,rv=after.sample(ts)
        np.testing.assert_array_equal(lv,rv)
        separation,pairs=pair_separations(left,lv,before.sizes)
        hit=(separation<=0)&pairs;collided=(hit|hit.swapaxes(1,2)).any(axis=-1)
        separation,pairs=pair_separations(right,rv,after.sizes)
        if ((separation<=0)&pairs).any():raise ValueError('Rendered camera times still have overlapping boxes')
        data.append((before,after,run,records,directory,ts,left,right,rv,collided))
    np.testing.assert_array_equal(data[0][5],data[1][5])
    anchor=data[0][0].controller.bundle['reference_poses_world'][data[0][0].controller.manifest['current_index'],0]
    yaw=np.arctan2(anchor[1,0],anchor[0,0]);R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    def project(xy):
        local=(xy-anchor[:2,3])@R
        return np.stack((-local[...,1],local[...,0]),axis=-1)
    focus_tokens=set()
    for item in report['variants'].values():
        for pair in item['source_ego_filter_audit'].get('other_actor_overlap_details',[]):focus_tokens.update(pair['track_tokens'])
    focus_indices=[i for i,t in enumerate(data[0][0].tokens) if t in focus_tokens]
    if a.focus_conflicts and not focus_indices:raise ValueError('No conflict actors to focus')
    if a.focus_conflicts:
        cloud=np.concatenate([project(row[j][:,focus_indices,:2][row[8][:,focus_indices]]) for row in data for j in (6,7)])
    else:cloud=np.concatenate([project(row[7][...,:2][row[8]]) for row in data])
    lo,hi=cloud.min(axis=0)-7,cloud.max(axis=0)+7
    fig,axes=plt.subplots(2,3,figsize=(22,12),gridspec_kw={'width_ratios':[1,1,1.4]})
    colors=plt.get_cmap('tab20')(np.linspace(0,1,len(data[0][0].tokens)))
    colors[0]=[.9,.1,.1,1]
    artists=[]
    affected=set()
    for name in names:
        affected.update(c['track_token'] for c in report['variants'][name]['free_actor_position_repairs'])
    for row,(before,after,run,records,directory,ts,left,right,valid,collided) in enumerate(data):
        bus=next(i for i,agent in after.agents.items() if agent==report['conditioned_slot'])
        colors[bus]=[.1,.1,.1,1]
        patches=[];labels=[]
        before_count=report['variants'][names[row]]['before_all_actor_overlap_samples_100hz']
        for col,(values,title) in enumerate(((left,f'Before: ego-only filter | overlaps {before_count}'),
                                            (right,'After: all actors checked | overlaps 0'))):
            ax=axes[row,col];ax.set(title=title,xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),
                                   xlabel='Right of anchor ego (m)',ylabel='Forward of anchor ego (m)')
            ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
            group=[];text=[]
            xy=project(values[...,:2])
            for actor,token in enumerate(after.tokens):
                line=xy[:,actor].copy();line[~valid[:,actor]]=np.nan
                ax.plot(line[:,0],line[:,1],color=colors[actor],linewidth=1,alpha=.45)
                poly=Polygon(np.zeros((4,2)),closed=True,facecolor=colors[actor],edgecolor='black',alpha=.75)
                ax.add_patch(poly);group.append(poly)
                agent=after.agents.get(actor)
                role='ego' if actor==0 else ('BUS GIVEN' if actor==bus else
                     (f'G{int(after.controller.bundle["nexus_slots"][agent])}' if agent is not None else f'L{actor-1}'))
                label=ax.text(0,0,role,fontsize=7,clip_on=True)
                text.append(label)
            patches.append(group);labels.append(text)
        image=axes[row,2].imshow(np.asarray(Image.open(directory/records[0][1]['file'])))
        axes[row,2].axis('off');axes[row,2].set_title('Bus GIVEN stopped' if row==0 else 'Bus GIVEN follows logged path')
        artists.append((patches,labels,image))
    stamp=fig.suptitle('')
    fig.text(.5,.025,'Same history, bus condition, ego and recorded context trajectories. Two free Nexus actors adjusted; no actor hidden.\n'
             'G: generated; L: recorded context. Offline projection, not learned closed-loop reactions. Counts: all physical pairs at 100 Hz. Camera cap retained.',
             ha='center',fontsize=12)
    fig.subplots_adjust(left=.04,right=.99,top=.91,bottom=.12,wspace=.2,hspace=.25)
    a.output.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(str(a.output/'all_actor_interaction.mp4'),fps=10,macro_block_size=2,ffmpeg_params=['-threads','4']) as writer:
        for frame in range(len(data[0][3])):
            elapsed=(data[0][5][frame]-data[0][0].controller.bundle['timestamps_us'][data[0][0].controller.manifest['current_index']])/1e6
            title='Conflict-area diagram (all actors retained in simulation)' if a.focus_conflicts else 'Full reconstructed traffic'
            stamp.set_text(f'{title}: before / after joint constraints | t={elapsed:.1f}s | red outlines = overlapping boxes')
            for row,((before,after,run,records,directory,ts,left,right,valid,collided),(patches,labels,image)) in enumerate(zip(data,artists)):
                for col,values in enumerate((left,right)):
                    for actor in range(len(after.tokens)):
                        value=values[frame,actor];L,W=after.sizes[actor]
                        c,s=np.cos(value[2]),np.sin(value[2]);rot=np.array([[c,-s],[s,c]])
                        corners=(np.array([[1,1],[-1,1],[-1,-1],[1,-1]])*[L/2,W/2])@rot.T+value[:2]
                        patches[col][actor].set_xy(project(corners));patches[col][actor].set_visible(bool(valid[frame,actor]))
                        overlap=col==0 and collided[frame,actor]
                        patches[col][actor].set_edgecolor('red' if overlap else 'black')
                        patches[col][actor].set_linewidth(2.5 if overlap else .5)
                        labels[col][actor].set_visible(bool(valid[frame,actor]));labels[col][actor].set_position(project(value[:2])+[.8,.8])
                image.set_data(np.asarray(Image.open(directory/records[frame][1]['file'])))
            fig.canvas.draw();pixels=np.asarray(fig.canvas.buffer_rgba())[...,:3];writer.append_data(pixels)
            if frame==70:Image.fromarray(pixels).save(a.output/'all_actor_preview.png')
    plt.close(fig)
    with imageio.get_reader(str(a.output/'all_actor_interaction.mp4'),input_params=['-threads','2'],output_params=['-threads','2']) as reader:
        count=sum(1 for _ in reader)
    if count!=len(data[0][3]):raise ValueError('Comparison decode mismatch')
    (a.output/'verification.json').write_text(json.dumps({'status':'passed','decoded_frames':count,
        'all_scene_actors_including_ego':len(data[0][0].tokens),'after_overlap_at_render_times':0,
        'diagram_focus_only':a.focus_conflicts,
        'source_experiment':str(a.experiment.resolve()),'scope':'offline projected traffic; sampled geometric check'},indent=2)+'\n')
    print('ALL_ACTOR_VIDEO_OK frames='+str(count),flush=True)


if __name__=='__main__':main()
