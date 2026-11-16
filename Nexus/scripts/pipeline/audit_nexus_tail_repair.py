"""Audit the changed G7 tail against its immutable source, without future GT."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from nexus_3dgs.bundle import load_bundle
from nexus_3dgs.kinematics import angle_difference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--fixed',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    result={'status':'passed','variants':{},'uses_logged_future_as_target':False}
    font=REPO/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'
    f24=ImageFont.truetype(str(font),24);f32=ImageFont.truetype(str(font),32)
    canvas=Image.new('RGB',(1600,950),'#f8fafc');d=ImageDraw.Draw(canvas)
    d.text((40,20),'模型车7（原G7）：末尾低速急转与侧滑修复',font=f32,fill='#202c42')
    d.text((40,70),'只约束自由预测；历史、自车、给定巴士、轨迹有效范围保持不变。未套用记录的未来轨迹。',font=f24,fill='#64748b')
    for row,name in enumerate(('bus_stop','bus_follow_logged')):
        old,meta=load_bundle(a.source/name);new,nm=load_bundle(a.fixed/name)
        for key in old:
            if key!='predicted_poses_world':np.testing.assert_array_equal(old[key],new[key])
        c=meta['current_index'];slots=old['nexus_slots'];j=int(np.flatnonzero(slots==7)[0]);bus=int(np.flatnonzero(slots==1)[0])
        np.testing.assert_array_equal(old['predicted_poses_world'][:c+1],new['predicted_poses_world'][:c+1])
        np.testing.assert_array_equal(old['predicted_poses_world'][:,[0,bus]],new['predicted_poses_world'][:,[0,bus]])
        mask=old['valid'][:,j]&(np.arange(len(old['valid']))>=c)
        t=(old['timestamps_us'][mask]-old['timestamps_us'][c])/1e6
        data={}
        for key,arrays in [('before',old),('after',new)]:
            poses=arrays['predicted_poses_world'][mask,j]
            yaw=np.arctan2(poses[:,1,0],poses[:,0,0]);y=np.rad2deg(np.unwrap(yaw))
            step=np.diff(poses[:,:2,3],axis=0)
            lateral=-step[:,0]*np.sin(yaw[:-1])+step[:,1]*np.cos(yaw[:-1])
            data[key]={'yaw_deg':y.tolist(),'max_heading_step_deg':float(np.rad2deg(np.abs(angle_difference(yaw[1:],yaw[:-1]))).max()),
                       'max_lateral_step_m':float(np.abs(lateral).max()),'xy_world_m':poses[:,:2,3].tolist()}
        if data['after']['max_heading_step_deg']>30:raise ValueError('G7 still has a sharp low-speed tail turn')
        result['variants'][name]={'timestamps_s':t.tolist(),**data,'immutable_fields_verified':True,
                                  'valid_prediction_ends_s':float(t[-1])}
        top=165+row*385;left=130;right=1090;bottom=top+270
        d.text((40,top-42),'巴士停车' if row==0 else '巴士沿记录轨迹行进',font=f24,fill='#202c42')
        d.rectangle((left,top,right,bottom),outline='#bac5d3',width=2)
        low,high=30,220
        def project(x,y):return left+x/4*(right-left),bottom-(y-low)/(high-low)*(bottom-top)
        for deg in (40,80,120,160,200):
            py=project(0,deg)[1];d.line((left,py,right,py),fill='#dce3eb')
            d.text((55,py-15),str(deg)+'°',font=f24,fill='#64748b')
        for sec in range(5):
            px=project(sec,low)[0];d.text((px-8,bottom+8),str(sec)+'s',font=f24,fill='#64748b')
        for key,color in [('before','#db6656'),('after','#3684cc')]:
            xy=[project(x,y) for x,y in zip(t,data[key]['yaw_deg'])]
            d.line(xy,fill=color,width=4)
            for x,y in xy:d.ellipse((x-5,y-5,x+5,y+5),fill=color)
        d.text((1140,top+5),'红：修复前   蓝：修复后',font=f24,fill='#202c42')
        d.text((1140,top+70),'最大相邻朝向变化',font=f24,fill='#64748b')
        d.text((1140,top+110),f'{data["before"]["max_heading_step_deg"]:.1f}° → {data["after"]["max_heading_step_deg"]:.1f}°',font=f32,fill='#202c42')
        d.text((1140,top+175),'最大相邻侧向位移',font=f24,fill='#64748b')
        d.text((1140,top+215),f'{data["before"]["max_lateral_step_m"]:.2f} → {data["after"]["max_lateral_step_m"]:.2f} 米',font=f32,fill='#202c42')
    canvas.save(a.output/'g7_heading_audit.png')
    (a.output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
