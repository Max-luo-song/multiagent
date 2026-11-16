"""Publish a reviewed presentation revision while preserving the prior delivery."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
STEM='nexus_3dgs_interaction_v9'
OLD_CAPTION='全部 80 帧、10 FPS、8 秒；俯视图来自实际渲染临时位姿。画面编号是三维中心投影，可能被遮挡。后段预测自车逼近／进入公交车身，会暴露未重建表面；原始视频保留这一结果。'
NEW_CAPTION='全部 80 帧、10 FPS、8 秒；俯视图来自实际渲染临时位姿。车辆编号仅在左侧俯视图显示，右侧高斯画面不叠加标记。后段预测自车逼近／进入公交车身，会暴露未重建表面；原始视频保留这一结果。'
EXPLANATION=(
    '<h3>为什么自车会进入 B1 公交车身？</h3>'
    '<p>这段自车未来由本次官方权重的 Nexus 配对推理生成。B1 的完整未来轨迹是给定条件：一组固定停车，另一组按记录行驶；'
    '自车和其他预测车辆只给历史，未来输入已清零。自车轨迹没有额外车距过滤或避碰修正。</p>'
    '<p>原始 Nexus 输出在接入 3DGS 之前就已检出自车与公交重叠。'
    '这次实际采样配置为 map → sma → clip，没有启用代码中另列的 collision 分支，因而不能将该配置视为硬性无碰撞保证。'
    '公交停车后，自车仍然前行；两组自车八秒路径长度分别约 15.56 m 和 16.35 m，条件响应不足以避免重叠。</p>'
    '<p>3DGS 按保存轨迹更新车辆位姿和自车相机，不运行碰撞后自动刹停的物理控制。'
    '当预测自车进入公交范围，相机也继续进入，出现穿入车身和未重建表面的画面。'
    '这是本场景、本种子和该条件配置下的预测失败案例；具体失效机制尚未确定，不能推广为所有 Nexus 预测都会如此。</p>'
)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args()
    verification=json.loads((a.review/'verification.json').read_text())
    assert verification['status']=='passed' and verification['vehicle_labels_only_in_bev']
    assert verification['unencoded_render_region_exact_source_pixels_all_160_panels']
    paired=json.loads((a.experiment/'raw_pair/run.json').read_text())
    for name,value in paired['variants'].items():
        assert sha(a.experiment/'raw_pair'/value['prediction'])==value['sha256']
        render=json.loads((a.experiment/'renders'/name/'run.json').read_text())
        assert render['source_prediction_sha256']==value['sha256']
        assert not render['later_offline_nexus_trajectory_edits']
    assets=a.report/'assets'/STEM
    manifest_path=a.report/(STEM+'_manifest.json');manifest=json.loads(manifest_path.read_text())
    changed=['paired_interaction.mp4','bus_stop_explained.mp4','bus_follow_logged_explained.mp4',
             'pair_000.jpg','pair_030.jpg','pair_050.jpg','pair_079.jpg']
    a.archive.mkdir(parents=True,exist_ok=False)
    for suffix in ('.html','.md','_manifest.json','_delivery_checks.json'):
        path=a.report/(STEM+suffix)
        if path.exists():shutil.copy2(path,a.archive/path.name)
    (a.archive/'assets').mkdir()
    for name in changed:shutil.copy2(assets/name,a.archive/'assets'/name)
    for name in changed:
        shutil.copy2(a.review/name,assets/name);manifest['assets'][name]=sha(assets/name)
    shutil.copy2(a.review/'verification.json',assets/'presentation_revision.json')
    manifest['assets']['presentation_revision.json']=sha(assets/'presentation_revision.json')
    summary_path=assets/'summary.json';summary=json.loads(summary_path.read_text())
    summary['presentation_revision']={'date':'2026-10-26','render_overlay_labels':False,'labels_only_in_bev':True,
        'source_prediction_sha256':{name:value['sha256'] for name,value in paired['variants'].items()},
        'raw_predictions_and_gaussian_renders_reused':True,'review_directory':str(a.review)}
    summary_path.write_text(json.dumps(summary,indent=2)+'\n');manifest['assets']['summary.json']=sha(summary_path)
    page_path=a.report/(STEM+'.html');page=page_path.read_text()
    assert OLD_CAPTION in page or NEW_CAPTION in page
    page=page.replace(OLD_CAPTION,NEW_CAPTION)
    if EXPLANATION not in page:page=page.replace('<h3>实际渲染车身检查</h3>',EXPLANATION+'\n<h3>实际渲染车身检查</h3>')
    page_path.write_text(page)
    md_path=a.report/(STEM+'.md');md=md_path.read_text()
    md+='\n2026-10-26 展示更新：右侧高斯画面移除所有车辆标记，编号只保留在左侧俯视图。\n'
    md+='\n自车未来来自本次 Nexus 原始配对预测，B1 未来为给定条件。原始输出已存在重叠，3DGS 按轨迹移动相机且不自动避碰，所以会进入公交车身。原始预测、模型和渲染 PNG 保持。\n'
    md_path.write_text(md)
    manifest['presentation_revision']='2026-10-26; labels only in BEV'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('PRESENTATION_UPDATED',page_path,flush=True)


if __name__=='__main__':main()
