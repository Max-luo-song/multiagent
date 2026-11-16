"""Combine two historical reports with the completed independent validation."""

import argparse
import base64
import hashlib
from html import escape
import json
from pathlib import Path
import re
import shutil
import statistics
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    root=repo/'artifacts/nuplan_evaluation'
    old_pilot=args.report_root/'Nexus定量复现进展_2026-10-21'
    old_city=args.report_root/'Nexus四城定量复现结果_2026-10-21'
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory is not empty: '+str(out))
    control=read(root/'training_recipe_control_v1/summary.json')
    safety=read(root/'safety_audit_v1/summary.json')
    road=read(root/'road_kinematics_audit_v1/summary.json')
    current=read(root/'independent256_audit_v1/summary.json')
    cities=read(root/'independent256_v1/common64_displacement_by_recipe_city_seed.json')
    reference=read(root/'multicity128_official_v2/run.json')
    controlled=read(root/'training_recipe_control_v1/run.json')
    holdout=read(root/'independent256_v1/run.json')
    assert reference['status']==controlled['status']==holdout['status']==current['status']=='complete'
    assert len(reference['records'])==384 and len(controlled['records'])==768 and len(holdout['records'])==1536
    assert not ({s['trip'] for s in reference['scenarios']}&{s['trip'] for s in holdout['scenarios']})
    assert len({s['token'] for s in reference['scenarios']+holdout['scenarios']})==384
    assert all(read(root/n/'run.json')['status']=='complete' for n in ('safety_audit_v1','road_kinematics_audit_v1','independent256_audit_v1'))
    assert read(root/'independent256_v1/verification.json')['prediction_files_sha256_verified']==1536
    out.mkdir(parents=True,exist_ok=True)
    for sub in ('assets','sources/assets','data'):
        (out/sub).mkdir(parents=True,exist_ok=True)
    provenance=[]
    def snapshot(source,target):
        destination=out/target
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        assert digest(source)==digest(destination)
        provenance.append(dict(source=str(source.resolve()),copy=target,sha256=digest(destination)))
    for folder,filename,label in (
        (old_pilot,'Nexus定量复现阶段报告.md','pilot_20261021'),
        (old_city,'Nexus四城定量复现结果报告.md','multicity128_20261021')):
        snapshot(folder/filename,'sources/'+filename)
        for p in sorted((folder/'assets').glob('*.png')):
            snapshot(p,'sources/assets/'+p.name)
        for p in sorted((folder/'data').iterdir()):
            if p.is_file():
                snapshot(p,'data/'+label+'/'+p.name)
    snapshot(root/'protocol_audit_20261021.json','data/protocol_audit_20261021.json')
    snapshot(root/'independent256_plan_v1.json','data/independent256_plan_v1.json')
    snapshot(root/'independent256_plan_v1_builder_check.json','data/independent256_plan_v1_builder_check.json')
    snapshots={
        'training_recipe_control_v1':['run.json','summary.json','official_metrics_by_recipe_seed.json','common_agent_comparison.json'],
        'training_recipe_features_v1':['run.json'],
        'safety_audit_v1':['run.json','summary.json','metrics_by_seed_city.json','geometry_regression.json','native_replay_check.json','native_horizon_replay_check.json'],
        'road_kinematics_audit_v1':['run.json','summary.json','metrics_by_seed_city.json','native_replay_check.json','regression_checks.json'],
        'independent256_v1':['run.json','verification.json','frozen_plan.json','config.yaml','official_metrics_by_recipe_seed.json','common64_displacement_by_recipe_city_seed.json','reference_future_points_outside_common64.json'],
        'independent256_audit_v1':['run.json','summary.json','metrics_by_seed_city.json','native_replay_check.json','position_replay_check.json'],
    }
    for directory,names in snapshots.items():
        for name in names:
            snapshot(root/directory/name,'data/'+directory+'/'+name)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    colors=['#376aaf','#159478']
    recipes=['released_test_recipe','causal_training_recipe']
    branches=['saved_test_recipe_common64','causal_history_recipe']
    def bars(ax,labels,series,ylabel,title):
        x=np.arange(len(labels))
        for j,(name,values,errors) in enumerate(series):
            rects=ax.bar(x+(j-.5)*.36,values,.34,color=colors[j],label=name,
                         yerr=errors,capsize=3)
            ax.bar_label(rects,fmt='%.2f',padding=4,fontsize=9)
        ax.set_xticks(x,labels);ax.set_ylabel(ylabel);ax.set_title(title,pad=15)
        ax.set_ylim(0,ax.get_ylim()[1]*1.18);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for ax,task,title in zip(axes,('bp','goal'),('BP: history conditioned','Goal: history + final valid state')):
        series=[]
        for recipe,branch,label in zip(recipes,branches,('Released test recipe','Causal training recipe')):
            a=control['common_displacement']['all'][branch][task]
            b=current['by_recipe'][recipe][task]['displacement_m']
            series.append((label,[a['mean'],b['mean']],[a['seed_sample_sd'],b['seed_sample_sd']]))
        bars(ax,['Previous 128 scenes','New 256 scenes'],series,'Matched future-point error (m)',title)
    axes[0].legend(fontsize=9)
    fig.savefig(out/'assets/05_matched_error_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.5),constrained_layout=True)
    specs=[('Position error','m',lambda v:v['displacement_m']),
           ('Vehicle-frame box overlap','%',lambda v:v['all_agents']['vehicle_frame_collision_fraction_percent']),
           ('Vehicles departing centerline segments','%',lambda v:v['common_segments']['vehicle_deviation_fraction_percent'])]
    for ax,(title,unit,getter) in zip(axes,specs):
        series=[]
        for recipe,label in zip(recipes,('Released test recipe','Causal training recipe')):
            vals=[getter(current['by_recipe'][recipe][task]) for task in ('bp','goal')]
            series.append((label,[v['mean'] for v in vals],[v['seed_sample_sd'] for v in vals]))
        bars(ax,['BP','Goal'],series,unit,title)
    axes[0].legend(fontsize=8)
    fig.savefig(out/'assets/06_independent_tradeoffs.png',dpi=170);plt.close(fig)
    city_order=[('las_vegas','Las Vegas'),('sg-one-north','Singapore'),('us-ma-boston','Boston'),('us-pa-pittsburgh-hazelwood','Pittsburgh')]
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),constrained_layout=True)
    for ax,task,title in zip(axes,('bp','goal'),('BP: new four-city subset','Goal: new four-city subset')):
        series=[]
        for recipe,label in zip(recipes,('Released test recipe','Causal training recipe')):
            vals=[[cities[str(seed)][city][recipe][task]['mean_error_m'] for seed in (0,1,2)] for city,_ in city_order]
            series.append((label,[statistics.mean(v) for v in vals],[statistics.stdev(v) for v in vals]))
        bars(ax,[name for _,name in city_order],series,'Matched future-point error (m)',title)
    axes[0].legend(fontsize=9)
    fig.savefig(out/'assets/07_independent_city_errors.png',dpi=170);plt.close(fig)

    old_labels={'saved_test_recipe_common64':'发布版测试预处理','published_training_recipe':'公开训练预处理（含未来依赖，仅诊断）','causal_history_recipe':'因果历史方案'}
    control_rows=[]
    for branch,label in old_labels.items():
        errors=control['common_displacement']['all'][branch]
        overlaps=safety['by_branch'][branch]
        control_rows.append(f'| {label} | {errors["bp"]["mean"]:.4f} | {errors["goal"]["mean"]:.4f} | '
            f'{overlaps["bp"]["all_agents"]["vehicle_frame_collision_fraction_percent"]["mean"]:.4f} | '
            f'{overlaps["goal"]["all_agents"]["vehicle_frame_collision_fraction_percent"]["mean"]:.4f} |')
    independent_rows=[];road_rows=[];physical_rows=[]
    for recipe,label in zip(recipes,('发布版测试预处理','因果历史方案')):
        for task,task_name in (('bp','普通预测'),('goal','目标条件预测')):
            v=current['by_recipe'][recipe][task]
            independent_rows.append(f'| {label} | {task_name} | {v["displacement_m"]["mean"]:.4f} ± {v["displacement_m"]["seed_sample_sd"]:.4f} | '
                f'{v["all_agents"]["vehicle_frame_collision_fraction_percent"]["mean"]:.4f} | '
                f'{v["all_agents"]["vehicle_any_future_collision_fraction_percent"]["mean"]:.4f} |')
            prior=road['by_branch'][branches[recipes.index(recipe)]][task]
            road_rows.append(f'| {label} | {task_name} | {prior["common_segments"]["vehicle_deviation_fraction"]["mean"]:.4f} | '
                f'{v["common_segments"]["vehicle_deviation_fraction_percent"]["mean"]:.4f} |')
            kin=v['all_agents']['physical_kinematics']
            physical_rows.append('| '+label+' · '+task_name+' | '+' | '.join(f'{kin[m]["mean"]:.4f}' for m in (
                'longitudinal_acceleration_mps2','lateral_acceleration_mps2','longitudinal_jerk_mps3','lateral_jerk_mps3'))+' |')
    gt=current['by_recipe']['recorded_gt']['gt']['all_agents']['physical_kinematics']
    physical_rows.insert(0,'| 记录真值 | '+' | '.join(f'{gt[m]["mean"]:.4f}' for m in (
        'longitudinal_acceleration_mps2','lateral_acceleration_mps2','longitudinal_jerk_mps3','lateral_jerk_mps3'))+' |')
    city_rows=[]
    for city,label in city_order:
        vals=[statistics.mean(cities[str(seed)][city][recipe][task]['mean_error_m'] for seed in (0,1,2))
              for recipe,task in ((recipes[0],'bp'),(recipes[1],'bp'),(recipes[0],'goal'),(recipes[1],'goal'))]
        city_rows.append('| '+label+' | 64 | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    report='''# Nexus 复现工作汇总与独立场景验证

整理日期：2026 年 10 月 22 日。

本报告合并 10 月 21 日的《Nexus 定量复现阶段报告》《Nexus 四城定量复现结果报告》，并补入后续预处理对照、安全与道路运动学审计，以及今日完成的 256 个新场景独立验证。两份原稿和配图保留在 sources 目录，完整指标与清单快照保留在 data 目录。

## 1. 已经复现完成什么

已跑通官方权重加载、真实 nuPlan 场景构建、双任务 GPU 推理、逐场景预测保存、原生指标计算和独立复算。所有实验均使用同一官方 nuplan.ckpt，没有训练或微调。

| 阶段 | 已完成范围 | 主要成果 |
| --- | --- | --- |
| 前期演示 | 行为预测与场景生成演示 | 已有可视化结果；本汇总集中展示定量实验 |
| 8 场景诊断 | 1 次 Las Vegas 行程，速度与约束对照 | 验证指标流程，定位预处理和约束影响 |
| 四城正式基线 | 128 场景、32 条行程、3 种子、双任务 | 384 份预测文件，768 条任务输出 |
| 同集预处理对照 | 上述 128 场景、两套新增预处理、3 种子 | 新增 768 份预测文件，1,536 条任务输出 |
| 128 场景审计 | 基线与两个对照共 1,152 份文件 | 碰撞计数、道路几何、物理运动学核验 |
| 今日独立验证 | 256 个新场景、32 条新行程、4 城市、3 种子 | 1,536 份预测文件，3,072 条任务输出，审计全部通过 |

两批四城场景合计为 **384 个唯一场景、64 条互不重叠的行程**。四城基线、预处理对照和今日验证合计保存 2,688 份预测文件、5,376 条任务输出；同一场景的多种子和多配置不重复计为新场景。上述累计不包含前期 8 场景诊断。

运行环境为 Python 3.9.25、PyTorch 2.3.0+cu121、RTX 4090；地图和数据库来自共享 nuPlan 数据，推理输出和缓存保存在独立工作目录。

## 2. 两份 10 月 21 日报告的主要结果

### 2.1 8 场景诊断：速度预处理影响明显，但安全指标未同步改善

官方 ResetV 将速度通道的相邻时间差写回速度通道，近似匀速车辆的非零速度会变成接近零的数值。保留原始速度的对照使用相同场景和随机数状态，得到了更小的位置误差。

| 设置 | 普通预测整秒误差（米） | 目标条件前七秒整秒误差（米） | 普通预测原生碰撞计数比 ×100 |
| --- | --- | --- | --- |
| 官方 ResetV | 14.3333 | 4.9500 | 2.3604 |
| 保留原始速度 | 7.5204 | 2.8811 | 2.8784 |

![前期 8 场景速度预处理对照，来自原阶段报告](sources/assets/01_velocity_control.png)

固定原始速度后，进一步比较强约束、弱约束和仅裁剪。弱约束的普通预测整秒误差为 7.1693 米，目标条件误差为 1.6771 米；但普通预测道路距离代理从强约束的 4.7917% 升至 15.9028%。这说明降低位置误差与保持道路贴合之间存在取舍，不能只挑误差最低的设置作为复现结论。

![前期 8 场景约束强度对照，来自原阶段报告](sources/assets/02_constraint_control.png)

这里的碰撞计数比按重叠车辆对计数；道路代理基于地图采样点。它们与下文的不同车辆重叠率、中心线线段偏离率口径不同。

### 2.2 四城 128 场景：建立冻结配置的正式基线

正式基线从 1,381 个官方验证数据库中按固定哈希顺序选样，四城各 32 场景、8 条行程，排除前期调试行程。128 个 token 均经真实 nuPlan 场景构建器确认。使用发布版测试预处理、32 步 full_sequence、map / sma / clip 参数 [1,1,1]，每批 4 场景，种子为 0、1、2。

| 原生代码指标 | 三个种子的均值 ± 样本标准差 |
| --- | --- |
| 普通预测整秒误差，1–8 秒（米） | 9.1939 ± 0.0157 |
| 目标条件整秒误差，1–7 秒（米） | 3.1860 ± 0.0144 |
| 普通预测碰撞计数比 ×100 | 0.8533 ± 0.0408 |
| 目标条件碰撞计数比 ×100 | 1.1302 ± 0.0482 |
| 普通预测道路距离代理 ×100 | 6.3618 ± 0.6680 |
| 目标条件道路距离代理 ×100 | 8.3841 ± 0.9502 |

![128 场景基线的整秒时域误差，来自原四城报告](sources/assets/01_horizon_errors.png)

给定终点任务额外获得各车辆最后有效时刻的真实完整状态，因此终点误差接近零来自输入条件。它不能被当成独立终点预测成绩。三个种子的标准差只反映采样随机性，不是全数据分布的置信区间。

![128 场景基线的城市误差，采用有效半秒点审计口径](sources/assets/02_city_errors.png)

首轮 v1 曾因未应用官方首尾场景过滤而失败，未产生预测；正式结果全部来自修正清单后的 v2，没有混入失败运行。

## 3. 后续补充：预处理差异与历史因果边界

核对公开代码后发现，测试与训练配置的差异涉及速度、地图和车辆上限三个方面。

| 方案 | 速度处理 | 地图与车辆输入 | 用途 |
| --- | --- | --- | --- |
| 发布版测试预处理 | 速度通道做时间差分 | 原地图有效性；129 个槽位，包含自车 | 正式基线与独立验证主分支 |
| 公开训练预处理 | 用前向位置差分估计速度 | 仅车道/车道连接线；前 64 个槽位 | 诊断分支 |
| 因果历史方案 | 当前时刻改用最后两个已观测位置差分；无效时保留原始当前速度 | 沿用训练预处理的地图和 64 槽位 | 去除当前历史边界的未来位置依赖 |

公开训练预处理在当前时刻使用了下一帧未来位置。128 场景的未来扰动检查中，公开训练分支的历史速度全部发生变化；因果分支 128/128 均保持历史不变。在 7,443 个有效非自车历史相邻状态对上，平均速度分别为：原始 2.3362 米/秒、测试 ResetV 0.2654 米/秒、公开训练 2.3717 米/秒、因果历史 2.4046 米/秒。发布权重实际训练时采用的完整预处理仍未确认。

为公平比较位置误差，以下统一使用前 64 个槽位，排除自车、历史和无效帧，目标条件任务再排除每辆车已提供的最后有效点，对未来所有半秒点进行加权汇总。原 128 场景的统一结果为：

| 方案 | 普通误差（米） | 目标条件误差（米） | 普通车辆帧重叠率（%） | 目标条件车辆帧重叠率（%） |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(control_rows)+'''

公开训练分支因为存在未来依赖，仅作为诊断证据，不作为最佳有效成绩。因果分支相对发布版仍保留大部分误差改善，但普通预测的重叠率更高。129 与 64 槽位的随机噪声形状不同，同名种子不代表相同噪声；两个 64 槽位诊断分支则共享种子和形状。

![原 128 场景与新 256 场景各自内部的匹配车辆误差对照](assets/05_matched_error_comparison.png)

不同场景集之间不能直接用绝对误差变化证明性能提升，应比较每个场景集内部的冻结方案对照。原生整秒误差 9.1939 米与统一半秒点误差 8.3238 米也不能混作同一个指标。

## 4. 今日完成：256 个全新场景独立验证

在观察结果之前冻结新清单与两个评测方案。每城 64 场景、8 条新行程，每行程 8 个间隔至少 15 秒的锚点；新行程与前期 128 场景及调试行程均无重叠。主分支固定为发布版测试预处理，次分支为因果历史方案，没有依据这次结果挑选最佳预处理。继续使用官方权重、原采样与约束设置、三个种子和批量大小 4。

256/256 场景通过真实场景构建器检查，首 4 个场景完成双方案实际特征预检。全部 1,536 份预测文件已核对 SHA256，256 份输入缓存齐全，模型参数版本和权重文件哈希未变化，训练更新为 0。

### 4.1 总体结果

| 方案 | 任务 | 匹配半秒点位置误差（米） | 车辆帧重叠率（%） | 未来曾重叠的车辆占比（%） |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(independent_rows)+'''

在这些新场景上，因果历史方案的普通预测误差下降 **43.89%**，目标条件预测下降 **41.97%**；这种误差改善在四个城市均出现。但普通预测的车辆帧重叠率从 **1.3463% 升至 2.8479%**，未来曾重叠的车辆占比也从约 **2.99% 升至 6.91%**，不能据此声称整体全面优于发布版。目标条件预测的重叠率则从 1.8091% 降至 1.5378%。

![256 个新场景的位置误差、车辆帧重叠率和道路偏离取舍](assets/06_independent_tradeoffs.png)

位置误差排除自车；本表的重叠率包含自车，按每个未来半秒帧中发生矩形重叠的不同车辆计数。记录真值也存在约 0.9227% 的车辆帧重叠，反映该离散标注几何的局限；这些数值不是独立确认的真实事故率。

有 1 个新场景的有效未来车辆超出前 64 个槽位，多出 16 个有效未来点。共同范围比较在两方案中都排除它，原生指标复算仍保留各方案的全部输入槽位，没有把不同的车辆范围直接混比。

### 4.2 分城市结果

| 城市 | 新场景数 | 发布版普通误差（米） | 因果方案普通误差（米） | 发布版目标条件误差（米） | 因果方案目标条件误差（米） |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(city_rows)+'''

![新四城场景中的两套预处理误差，误差线为种子标准差](assets/07_independent_city_errors.png)

总体误差按有效车辆轨迹点加权，不是四城误差的简单平均。Las Vegas 和 Singapore 在这批场景中误差较大，但城市间速度、道路与车辆密度没有匹配，不能仅凭表格判断差异的因果来源。

## 5. 道路、运动学与指标审计补全

### 5.1 道路几何

原生道路指标基于到离散地图采样点的距离。补充审计统一使用发布版输入中的车道/车道连接线地图，将相邻有效顶点连接为线段，禁止跨无效点连接。只比较真实历史曾在中心线线段 2.75 米内、且具有有效未来的同一批车辆。下表表示这些车辆中未来曾偏离该距离阈值的占比：

| 方案 | 任务 | 原 128 场景（%） | 新 256 场景（%） |
| --- | --- | --- | --- |
'''+ '\n'.join(road_rows)+'''

新场景中，因果方案的普通预测道路偏离占比略升，目标条件预测则下降。这与碰撞审计共同说明收益依赖任务和指标。该定义是最近中心线距离诊断，尚不等同于指定路线合规或可行驶区域多边形检查。

### 5.2 物理运动学

补充审计使用 0.5 秒位置的后向差分计算加速度，再投影到当前车头的纵向与横向；加加速度为上述投影分量的时间差分。只统计差分所需位置都有效的未来帧，可使用已观测历史作为边界，不跨无效帧。新 256 场景中，包含自车的绝对值均值如下：

| 方案与任务 | 纵向加速度（m/s²） | 横向加速度（m/s²） | 纵向加加速度（m/s³） | 横向加加速度（m/s³） |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(physical_rows)+'''

因果方案普通预测的纵向加速度和加加速度下降，横向分量则上升；目标条件任务的四项分量均下降。原生代码部分横向命名指标实际使用航向角变化，不能直接当成横向物理加速度。上述物理诊断单独保留，没有静默替换原生指标。

### 5.3 核验结果

| 检查 | 已完成结果 |
| --- | --- |
| 碰撞几何 | 840 对随机车辆与 Shapely、原生 SAT 交叉核验；覆盖接触、无效帧、多车重叠与长车超过 10 米等情况 |
| 原 128 场景碰撞复算 | 144 个整秒值通过，最大绝对误差约 1.74×10⁻⁹ |
| 道路线段及运动学测试 | 105 个独立线段距离检查，含稀疏点、无效断点、恒速/恒加速度/恒加加速度与旋转检查 |
| 原 128 场景道路/运动学复算 | 126 项通过，最大绝对误差约 3.78×10⁻⁶ |
| 新 256 场景原生指标复算 | 96 项通过，最大绝对误差约 3.86×10⁻⁶ |
| 新 256 场景位置误差复算 | 60 项城市/种子/方案/任务汇总全部一致 |
| 新预测完整性 | 1,536/1,536 份 SHA256 校验通过，权重未变化 |

## 6. 当前可以给出的结论

目前已完成官方权重在真实 nuPlan 四城子集上的双任务推理和定量评测，并有固定场景清单、预测文件、配置、完整指标、核验记录与可携带图文报告。预处理相关的位置误差改善在 32 条全新行程中仍然存在，而非只在原诊断集上出现。

同时需要保留三个结论边界。第一，因果历史方案相对发布版还同时改变地图有效性和车辆上限，不能把全部改善只归因于速度修正。第二，普通预测的重叠和部分道路指标变差，尚不能认为它整体更优。第三，384 个四城唯一场景仍是固定选取的子集，尚未运行全验证集，也没有复现训练过程。

论文全量对齐仍待确认：发布权重真实训练预处理、论文与代码的误差聚合方式、不同车辆与车辆对的碰撞计数、指定中心线与最近地图点的道路定义、物理分量与角度变化的运动学定义，以及约束强度映射。现已核实发布代码按 map → sma → clip 顺序执行；map/sma 按各自系数修正状态，clip 直接裁剪并忽略其 gamma 参数。因此 [1,1,1] 不能直接解释为论文的归一化总引导强度，本轮保持发布设置不变。批量推理时间也不作为论文单场景延迟。

后续若继续扩大复现范围，应先对齐这些协议，再冻结下一轮数据与配置；若研究预处理机制，应分别控制速度、地图和车辆上限，避免用联合改动给出单因素结论。

## 7. 文件与阅读方式

本目录包含汇总 Markdown、图片内嵌的离线 HTML、7 张结果图、两份原报告及 JSON/YAML 证据快照。逐场景预测张量留在服务器，展示包不重复复制；两份历史报告的原目录未改动。

Markdown 使用相对图片路径，下载整个目录或完整包解压后保留 sources、assets、data 的位置即可查看。离线 HTML 的全部图表已内嵌，可单独下载阅读正文与图表；证据附件链接需要完整目录。两份指定的历史报告和今日任务均为定量图表，原演示 GIF 没有在本汇总中重复打包。

飞书提交建议导入汇总 Markdown 为云文档；若图片未自动带入，从 assets 和 sources/assets 插入对应 PNG。单独上传一个 MD 文件不会自动携带本地图片。也可以上传离线 HTML 或完整包作为附件，供下载后查看。

原始实验根目录：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus/artifacts/nuplan_evaluation`。本文主要对应 multicity128_official_v2、training_recipe_control_v1、safety_audit_v1、road_kinematics_audit_v1、independent256_v1、independent256_audit_v1。

原报告保留：[8 场景阶段报告](sources/Nexus定量复现阶段报告.md)、[四城 128 场景报告](sources/Nexus四城定量复现结果报告.md)。新结果详见 [独立验证汇总](data/independent256_audit_v1/summary.json) 和 [文件完整性核验](data/independent256_v1/verification.json)。论文地址保留在 data/protocol_audit_20261021.json。
'''
    title='Nexus复现工作汇总_2026-10-22'
    md=out/(title+'.md');md.write_text(report,encoding='utf-8')
    import markdown
    body=markdown.markdown(report,extensions=['tables','fenced_code'])
    def embed(match):
        path=out/match.group(1)
        assert path.is_file()
        return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')+'"'
    body=re.sub(r'src="([^\"]+\.png)"',embed,body)
    html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+'</title><style>body{font:16px/1.85 system-ui,"Microsoft YaHei",sans-serif;color:#243247;max-width:1140px;margin:30px auto;padding:0 24px}h1{font-size:30px}h2{margin-top:38px;border-bottom:1px solid #d9e0e9;padding-bottom:6px}h3{margin-top:26px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}th,td{border:1px solid #d1d9e3;padding:8px 10px}th{background:#eef3f9}code{overflow-wrap:anywhere}a{color:#2167ad}@media print{body{max-width:none;font-size:11pt}h2,h3{break-after:avoid}img,tr{break-inside:avoid}}</style></head><body>'+body+'</body></html>'
    html_path=out/(title+'_离线预览.html');html_path.write_text(html,encoding='utf-8')
    assert html.count('src="data:image/png;base64,')==7
    note='''# 阅读与提交

主文稿：Nexus复现工作汇总_2026-10-22.md。

本地阅读请下载整个目录或本目录内的完整包 ZIP 并解压，保持 Markdown、assets、sources、data 的相对位置。

离线预览 HTML 的七张结果图均已内嵌，单独下载即可阅读图文；证据附件与原报告链接需配套目录。

飞书导入主 Markdown 为云文档后检查图片；若未自动导入，从 assets 与 sources/assets 插入对应 PNG。单独的 MD 文件不携带本地图片；HTML/ZIP 也可作为附件提交供下载阅读。本汇总使用静态定量图表，未重复收录之前的演示 GIF。

data 保存指标、清单与配置快照，sources 保留两份指定原报告与原图。逐场景 .pt 预测保留在服务器。本目录内 SHA256SUMS.txt 可校验正文、图片和证据文件；交付校验.json 记录检查结果。
'''
    (out/'阅读与提交说明.md').write_text(note,encoding='utf-8')
    (out/'data/source_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    refs=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report)
    assert len(refs)==7 and all((out/path).is_file() for path in refs)
    for source in (out/'sources').glob('*.md'):
        for path in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',source.read_text()):
            assert (source.parent/path).is_file()
    for item in provenance:
        assert digest(out/item['copy'])==item['sha256']
    assert len(list(out.rglob('*.png')))==7
    checks=dict(status='passed',main_report=md.name,offline_html=html_path.name,embedded_images=7,
        markdown_image_references_verified=7,historical_reports_preserved=2,
        copied_source_files_verified=len(provenance),unique_four_city_scenarios=384,
        prediction_files_in_experiments=2688,prediction_tensors_in_package=0,
        new_holdout_prediction_files_verified=1536,native_replay_checks_passed=96,position_replay_checks_passed=60)
    (out/'交付校验.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    files=sorted(p for p in out.rglob('*') if p.is_file())
    (out/'SHA256SUMS.txt').write_text(''.join(digest(p)+'  '+p.relative_to(out).as_posix()+'\n' for p in files))
    files.append(out/'SHA256SUMS.txt')
    archive=out/(title+'_完整包.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path,'2026-10-22/'+path.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)
    print('REPORT='+str(md),flush=True)
    print('HTML='+str(html_path),flush=True)
    print('ZIP='+str(archive),flush=True)
    print('DAILY_REPORT_CHECK_OK copied_sources='+str(len(provenance)),flush=True)


if __name__=='__main__':
    main()
