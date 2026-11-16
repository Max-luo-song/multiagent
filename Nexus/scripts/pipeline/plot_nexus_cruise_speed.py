"""Standalone speed plots from saved forecasts; no model/trajectory edits."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from nexus_3dgs.bundle import decode
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    out=ROOT/'artifacts/nexus_kinematic_20261027'
    font=FontProperties(fname=str(ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'))
    fig,axes=plt.subplots(3,1,figsize=(12,11),constrained_layout=True)
    midpoint=np.arange(16)*.5+.25
    for version,stage,color,style in [('上一版','candidate_v1','#d36b48','--'),('本版',args.stage,'#287bcd','-')]:
        for variant in ['normal','event']:
            p=torch.load(out/stage/'kinematic/seed_000'/variant/'prediction.pt',map_location='cpu',weights_only=False)
            s=decode(p['sampled_tensor'][0].numpy())
            for index,slot in [(0,0),(1,0),(2,7)]:
                if (index==1)!=(variant=='event'):continue
                speed=np.linalg.norm(np.diff(s[slot,4:,:2],axis=0)/.5,axis=-1)
                initial=float(np.linalg.norm((s[slot,4,:2]-s[slot,3,:2])/.5))
                axes[index].plot(np.r_[0,midpoint],np.r_[initial,speed],style,color=color,marker='o',markersize=3,label=version)
                if version=='本版':
                    axes[index].axhline(initial,color='#7a8797',linestyle=':',label='当前观测速度软参考')
            if variant=='event' and version=='本版':
                speed=np.linalg.norm(np.diff(s[2,4:,:2],axis=0)/.5,axis=-1)
                axes[1].plot(midpoint,speed,':',color='#b5812a',label='给定前车 A 的位移速度')
    titles=['正常 E：是否仍出现持续慢行','切入 E：额外减速与风险后速度恢复','正常前车（槽7）：检查慢行是否仍在车流中传播']
    for ax,title in zip(axes,titles):
        ax.set_title(title,fontproperties=font);ax.set_xlabel('历史结束后的未来时间 / 秒',fontproperties=font)
        ax.set_ylabel('区间平均速度 / m/s',fontproperties=font);ax.set_xlim(0,8);ax.set_ylim(bottom=0);ax.grid(alpha=.25);ax.legend(prop=font)
    axes[1].axvspan(.5,2.5,color='#e9b44c',alpha=.15)
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination,dpi=150);plt.close(fig)
    print('CRUISE_SPEED_PLOT_COMPLETE',destination,flush=True)


if __name__=='__main__':main()
