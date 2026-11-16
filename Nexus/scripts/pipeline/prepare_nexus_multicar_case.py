"""Prepare a closing-gap cut-in candidate; this does not certify lane topology."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    from PIL import Image,ImageDraw,ImageFont
    from nexus_3dgs.bundle import export_arrays,save_bundle
    selection=json.loads((ROOT/'artifacts/nexus_3dgs/multicar_selection_all_20261026/selection.json').read_text())
    selected=next(r for r in selection['ranked_candidates'] if r['index']==33)
    combo=selected['combinations'][0]
    out=ROOT/'artifacts/nexus_3dgs/multicar_event_033_20261026';out.mkdir(exist_ok=False)
    source=ROOT/'artifacts/nuplan_evaluation/independent256_v1'
    raw=torch.load(source/'raw_features'/f'033_{selected["token"]}.pt',map_location='cpu',weights_only=False)
    gt=raw['tensor'][...,:8].unsqueeze(0)
    arrays,meta=export_arrays({'sampled_tensor':gt,'scene_tensor_gt':gt,'valid_mask':raw['validity'].unsqueeze(0)},
                             {'database':selected['database'],'scenario_token':selected['token']})
    save_bundle(out/'reference',arrays,meta)
    slot=combo['event_slot'];third=combo['third_slot']
    token=str(arrays['track_tokens'][list(arrays['nexus_slots']).index(slot)])
    third_token=str(arrays['track_tokens'][list(arrays['nexus_slots']).index(third)])
    request={'schema_version':1,'source_run':str(source.relative_to(ROOT)),'scenario_token':selected['token'],
             'actor_track_token':token,'behavior':'lateral_shift','start_s':.5,'end_s':2.5,
             'parameters':{'distance_m':3.,'side':'right'},'seeds':[0,1,2],'render_seed':0,
             'description_zh':'左前方 A 在第 0.5～2.5 秒向右横移 3 米，侵入自车行驶区域；右侧近处车辆 C 未来自由生成。自车不指定避让。',
             'third_actor_track_token_for_audit':third_token,
             'scope':'历史几何筛选后的三车切入候选；未核验地图车道、未完成传感器可用性与高斯重建。'}
    (out/'event_request.json').write_text(json.dumps(request,indent=2,ensure_ascii=False)+'\n')
    (out/'selection_evidence.json').write_text(json.dumps({**selected,'selection_reason':'Compared to highest static-distance rank, index33 has closing longitudinal gap and a close opposite-side third actor.',
        'event_token':token,'third_token':third_token,'photorealistic_render_ready':False},indent=2)+'\n')
    im=Image.new('RGB',(1100,1000),'#f7fafc');d=ImageDraw.Draw(im)
    font=ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'
    fonts={s:ImageFont.truetype(str(font),s) for s in (18,22,26,30)}
    def text(xy,value,size=22,fill='#24364b'):d.text(xy,value,font=fonts[size],fill=fill)
    def project(xy):return 550-float(xy[1])*25,750-float(xy[0])*12
    text((35,20),'三车切入候选：033（初始历史几何）',30)
    text((35,65),'上方为自车前方；此图为位置示意，车道拓扑尚未核验。',22)
    for x in range(-10,41,10):
        y=750-x*12;d.line((100,y,1000,y),fill='#d3dee8');text((45,y),f'{x}m',18)
    for actor in selected['actors']:
        i=actor['slot'];x,y=actor['xy_m']
        if not -10<x<40 or abs(y)>16:continue
        q=project([x,y]);color='#d94a4a' if i==0 else '#dd9222' if i==slot else '#287bcd' if i==third else '#c4cdd6'
        d.rectangle((q[0]-24,q[1]-28,q[0]+24,q[1]+28),fill=color,outline='#596a7b')
        if i in (0,slot,third):
            label='E 自车' if i==0 else 'A 指定切入车' if i==slot else 'C 自由交互车'
            speed=float(np.linalg.norm(actor['history_velocity_mps']))
            text((q[0]+30,q[1]-20),f'{label} {speed:.2f}m/s',22,color)
    A=next(r for r in selected['actors'] if r['slot']==slot)
    q=project(A['xy_m']);d.line((q[0]+5,q[1]-55,q[0]+75,q[1]-55),fill='#dd9222',width=5)
    d.polygon([(q[0]+75,q[1]-55),(q[0]+60,q[1]-63),(q[0]+60,q[1]-47)],fill='#dd9222')
    text((35,900),'只指定 A 的局部动作；E 与 C 如何响应交给 Nexus。',26)
    text((35,945),'当前仅有轨迹／DB；图像与 LiDAR 缺失问题需在重建前解决。',22,'#a04d31')
    im.save(out/'initial_geometry.png')
    print(json.dumps({'case':str(out),'event_actor':token,'third_actor':third_token,'request':request},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
