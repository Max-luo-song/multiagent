"""Deliver full-sequence v8/v9 bus comparisons and an honest measured-view audit."""
import argparse
import hashlib
import html
import json
import shutil
from contextlib import ExitStack
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--report-root', type=Path, nargs='+', required=True)
    a = p.parse_args()
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    root = Path(__file__).resolve().parents[2]
    v = a.experiment
    baseline = v.parent / 'optimization_v8'
    metrics = json.loads((v / 'analysis/bus_detail_metrics.json').read_text())
    asset = json.loads((v / 'bus_asset_restored/asset_audit.json').read_text())
    new_run = json.loads((v / 'nexus_bus_detail/run.json').read_text())
    old_run = json.loads((baseline / 'nexus_stable/run.json').read_text())
    assert new_run['bundle_sha256'] == old_run['bundle_sha256']
    assert len(new_run['frames']) == len(old_run['frames']) == 80
    for new, old in zip(new_run['frames'], old_run['frames']):
        assert new['frame'] == old['frame']
        np.testing.assert_array_equal(new['camera_world'], old['camera_world'])
    old_poses = json.loads((baseline / 'observed_validation/validation.json').read_text())['actual_camera_and_actor_poses']
    new_poses = json.loads((v / 'observed_validation/validation.json').read_text())['actual_camera_and_actor_poses']
    assert len(old_poses) == len(new_poses) == 80
    for old, new in zip(old_poses, new_poses):
        np.testing.assert_array_equal(old['camera_world'], new['camera_world'])
        np.testing.assert_array_equal(old['actors_world'], new['actors_world'])
    assert metrics['common_static_max_rgb_difference_8bit'] == 0
    assert metrics['black_car_crop_max_rgb_difference_8bit'] == 0
    font = ImageFont.truetype(str(root / 'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'), 24)
    review = v / 'review'
    review.mkdir(exist_ok=False)
    names = ['nexus_before_after', 'bus_closeup', 'recorded_bus_comparison']
    diff_min = np.array([960, 540])
    diff_max = np.array([0, 0])
    nexus_black_max = 0

    def panel(items, f, zoom=False):
        width, height = (540, 540) if zoom else (960, 540)
        canvas = Image.new('RGB', (width*len(items), height+110), '#142136')
        draw = ImageDraw.Draw(canvas)
        for j, (title, im) in enumerate(items):
            draw.text((j*width+12, 8), title, font=font, fill='white')
            im = Image.fromarray(im)
            if zoom:
                im = im.crop((300, 90, 660, 450)).resize((width, height), Image.Resampling.BILINEAR)
            canvas.paste(im, (j*width, 45))
        draw.text((12, height+62), f'{(f-20)/10:.1f} 秒　帧 {f}　同一运动与相机；仅替换公交高斯资产。', font=font, fill='#ffd18c')
        return canvas

    with ExitStack() as stack:
        writers = {name: stack.enter_context(imageio.get_writer(
            str(review / (name+'.mp4')), format='FFMPEG', fps=10,
            macro_block_size=2, ffmpeg_params=['-threads', '2', '-crf', '18'])) for name in names}
        for f in range(20, 100):
            old = imageio.imread(baseline / 'nexus_stable' / f'{f:03d}_0.png')
            new = imageio.imread(v / 'nexus_bus_detail' / f'{f:03d}_0.png')
            delta = abs(new.astype(np.int16)-old.astype(np.int16))
            nexus_black_max = max(nexus_black_max, int(delta[215:335, 690:850].max()))
            yy, xx = np.where(delta.max(-1) > 0)
            if len(xx):
                diff_min = np.minimum(diff_min, [xx.min(), yy.min()])
                diff_max = np.maximum(diff_max, [xx.max(), yy.max()])
            photo = imageio.imread(v / 'observed_validation' / f'{f:03d}_photo.png')
            old_real = imageio.imread(baseline / 'observed_validation/new' / f'{f:03d}_0.png')
            new_real = imageio.imread(v / 'observed_validation/new' / f'{f:03d}_0.png')
            images = {
                'nexus_before_after': panel([('v8：已认可基线', old), ('v9：公交细节候选', new)], f),
                'bus_closeup': panel([('v8：公交放大', old), ('v9：公交放大', new)], f, True),
                'recorded_bus_comparison': panel([('真实照片', photo), ('v8', old_real), ('v9 候选', new_real)], f, True),
            }
            for name, im in images.items():
                writers[name].append_data(np.array(im))
                if f in [20, 55, 70, 95, 99]:
                    im.save(review / f'{name}_{f:03d}.jpg', quality=95)
            if f % 10 == 0:
                print('BUS_DETAIL_REVIEW '+str(f), flush=True)
    assert nexus_black_max == 0
    rows = metrics['rows']
    groups = {'基本静止 43 帧': [r for r in rows if r['frame'] <= 62],
              '运动 37 帧': [r for r in rows if r['frame'] >= 63],
              '全部 80 帧': rows}
    scores = {group: {name: {metric: float(np.mean([r['metrics'][name][metric] for r in subset]))
                             for metric in ['full_psnr', 'bus_box_psnr']}
                      for name in ['v8', 'candidate']} for group, subset in groups.items()}
    audit = {'status': 'passed', 'scope': 'one scene, front camera, complete 80 frames at 10 fps',
             'same_v8_derived_trajectory_bundle': True, 'all_80_cameras_and_recorded_body_poses_exact': True,
             'bus_only_asset_change_verified_after_checkpoint_reload': asset['all_v8_body_poses_camera_background_and_other_assets_bit_exact'],
             'measured_static_background_max_rgb_difference_8bit': 0,
             'measured_and_nexus_black_car_crop_max_rgb_difference_8bit': 0,
             'nexus_changed_pixel_bounds_xy': [diff_min.tolist(), diff_max.tolist()],
             'no_pixel_sharpening_temporal_blending_or_frame_omission': True,
             'nexus_or_3dgs_retrained_this_round': False, 'measured_view_metrics': scores,
             'limitations': 'sharper detail does not guarantee higher pixel PSNR; inherited bus source surface and novel-view limits remain'}
    (v / 'detail_validation.json').write_text(json.dumps(audit, indent=2)+'\n')
    decision = {'status': 'bus_detail_candidate_validated', 'approved_baseline_retained': 'optimization_v8',
                'candidate': 'bus_asset_restored/checkpoint_final.pth', 'promoted_over_v8': False,
                'reason': 'bus detail visibly restored; measured-view fidelity has tradeoffs; full comparisons delivered',
                'no_new_trajectory_edits': True, 'source_assets_and_v8_unchanged': True}
    (v / 'decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    table = '\n'.join(f'| {group} | {s["v8"]["full_psnr"]:.2f} | {s["candidate"]["full_psnr"]:.2f} | {s["v8"]["bus_box_psnr"]:.2f} | {s["candidate"]["bus_box_psnr"]:.2f} |' for group, s in scores.items())
    md = f'''# v9 公交细节恢复候选

v8 保留为你已认可的基线。本轮只把公交车替换为 v7 车辆整体重训之前的高斯资产，保留 v8 的全部运动稳定化、派生轨迹、相机、背景和其他车辆。**没有重新训练 Nexus 或 3DGS，没有视频锐化、补帧、时间融合，也没有把真实视频贴回渲染画面。**公交细节来自此前已有的三维高斯资产；该旧资产的车尾下部曾使用真实观测构建连续曲面，不能据此宣称任意视角均有完整几何。

## 先看哪里

- [完整 Nexus 回放：左 v8，右 v9](assets/nexus_3dgs_v9/nexus_before_after.mp4)
- [公交放大：左 v8，右 v9](assets/nexus_3dgs_v9/bus_closeup.mp4)
- [真实照片／v8／v9 公交放大](assets/nexus_3dgs_v9/recorded_bus_comparison.mp4)
- [仅看 v9 完整渲染](assets/nexus_3dgs_v9/camera_0.mp4)

所有视频都是第 20～99 帧完整 80 帧、10 FPS、8 秒。重点看车尾 “SBS Transit” 字样、车窗底部小字、橙紫车身分界和尾灯。放大视频只对同一区域作等比例显示缩放，不增加原图细节。

![Nexus 公交放大，第 5 秒](assets/nexus_3dgs_v9/bus_closeup_070.jpg)

## 本轮发现与实际改动

公交尾部文字在 v7 的车辆几何与外观整体重训后变软。只恢复公交重训前的资产，在相同 v8 运动与相机下可以恢复可见文字和边缘细节。这说明本例的模糊至少有一部分来自公交资产重训；不能据此判断所有模糊均由同一个原因造成，也不能归结为 Nexus 轨迹必然无法清晰渲染。

本轮恢复的是公交的高斯位置、尺度、旋转、颜色与透明度，共 {asset['bus_gaussians']:,} 个高斯。**全部车体位姿保持 v8 原值**，不会恢复 v7 的晃动或黑车异常翻转。v8 的公交横向轨迹修正也完全保留；这仍是明确标注的派生预测回放。

![记录相机：真实照片、v8、v9，第 5 秒](assets/nexus_3dgs_v9/recorded_bus_comparison_070.jpg)

## 画质有取舍，v8 不覆盖

公交文字更清楚，但与真实照片逐像素的一致性并非每帧提高。公交局部细节的位置、遮挡、反光以及原有几何误差仍会影响分数，因此本版作为细节候选单独交付。下表仅评估有真实照片的记录相机；不对 Nexus 预测视角计算不存在真值的 PSNR。公交区域采用向内收缩的三维框投影，含部分背景，不是精确语义轮廓。分组沿用先前真实运动实验。

| 分组 | v8 整图 PSNR | v9 整图 PSNR | v8 公交区域 PSNR | v9 公交区域 PSNR |
| --- | --- | --- | --- | --- |
{table}

以上是本场景重建保真度检查，旧公交资产见过原场景，不能作为未见场景泛化成绩。

## 已完成的保护与回归检查

- 重新加载输出权重后逐项核对：除公交高斯资产外，所有参数精确保持；全序列车体位姿、可见性和车辆身份保持。
- 两组完整 80 帧使用完全相同的 Nexus 派生轨迹包和相机矩阵；记录相机及实际车体位姿也精确相同。
- 记录视角共同车辆排除区域外，背景像素最大差为 0／255；记录与预测回放中右侧黑车检查区域最大差均为 0／255。
- 原始模型、v8 权重、原始预测和 v8 派生预测保留。本轮没有改依赖。
- 完整视频逐帧解码与报告资源哈希检查见交付检查文件。

## 文件位置

- 本轮目录：`{v.resolve()}`
- 公交候选权重：`{(v/'bus_asset_restored/checkpoint_final.pth').resolve()}`
- 主视频：`{(review/'nexus_before_after.mp4').resolve()}`
- 公交放大：`{(review/'bus_closeup.mp4').resolve()}`
- [参数保护审计](assets/nexus_3dgs_v9/asset_audit.json) · [完整验证](assets/nexus_3dgs_v9/detail_validation.json) · [逐帧质量统计](assets/nexus_3dgs_v9/bus_detail_metrics.json)
- [此前 v8 报告](nexus_3dgs_v8.html)
'''
    (v / 'README.md').write_text(md)
    (root / 'docs/nexus_3dgs_optimization_v9.md').write_text(md)
    full_table = ''.join(f'<tr><td>{html.escape(group)}</td><td>{s["v8"]["full_psnr"]:.2f}</td><td>{s["candidate"]["full_psnr"]:.2f}</td><td>{s["v8"]["bus_box_psnr"]:.2f}</td><td>{s["candidate"]["bus_box_psnr"]:.2f}</td></tr>' for group,s in scores.items())
    for destination in a.report_root:
        destination.mkdir(parents=True, exist_ok=True)
        assets = destination / 'assets/nexus_3dgs_v9'
        assets.mkdir(parents=True, exist_ok=True)
        sources = list(review.glob('*.mp4'))+list(review.glob('*.jpg'))+[
            v/'nexus_bus_detail/camera_0.mp4', v/'detail_validation.json',
            v/'bus_asset_restored/asset_audit.json', v/'analysis/bus_detail_metrics.json']
        for source in sources:
            shutil.copyfile(source, assets/source.name)
        (destination/'nexus_3dgs_v9.md').write_text(md)
        videos = ''.join(f'<h2>{label}</h2><video controls preload="metadata" src="assets/nexus_3dgs_v9/{name}.mp4"></video>' for label,name in [
            ('完整 Nexus：左 v8，右 v9','nexus_before_after'),('公交放大：左 v8，右 v9','bus_closeup'),
            ('真实照片／v8／v9 公交放大','recorded_bus_comparison'),('仅看 v9 完整渲染','camera_0')])
        page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>v9 公交细节恢复候选</title>
<style>body{{max-width:1200px;margin:36px auto;padding:0 20px;background:#101b2d;color:#edf1f7;font:17px/1.8 sans-serif}}video,img{{width:100%;margin:12px 0}}a{{color:#8ac6ff}}td,th{{padding:8px;border:1px solid #586579}}table{{border-collapse:collapse}}.note{{background:#233249;padding:18px}}</style>
<h1>v9 公交细节恢复候选</h1><p class="note">v8 保留为已认可基线。本轮只恢复公交重训前的三维高斯资产，全部 v8 运动、相机、背景和其他车辆保持。没有重新训练，没有视频锐化、补帧或时间融合，也没有把真实视频贴回渲染画面。原公交车尾下部曾使用真实观测构建三维连续曲面，任意新视角仍有局限。</p>
<p>所有视频为完整 80 帧、10 FPS、8 秒。重点看车尾文字、车窗小字、橙紫分界和尾灯。公交放大只是等比例显示缩放。v8 的黑车朝向保护、公交横向修正与车体稳定化全部保留。</p>{videos}
<h2>清晰度与保真度的取舍</h2><p>恢复旧公交资产后，文字和边缘细节明显更清楚；这定位到车辆重训带来的细节损失。但部分帧与真实照片的逐像素误差有所增加，因此作为独立候选交付，不覆盖 v8。以下是记录相机 PSNR，单位 dB；Nexus 预测视角没有对应真值，不计算此分数。公交区域采用收缩后的三维框投影，含部分背景，非精确语义轮廓。旧资产见过原场景，这不是未见场景泛化评测。</p>
<table><tr><th>分组</th><th>v8 整图</th><th>v9 整图</th><th>v8 公交区域</th><th>v9 公交区域</th></tr>{full_table}</table>
<h2>保护检查</h2><p>重载权重逐项核对：仅公交高斯位置、尺度、旋转、颜色和透明度改变；全序列车体位姿、相机、背景和其他车辆参数精确保持。两条完整回放使用同一个 v8 派生轨迹包，80 个预测相机矩阵完全相同。记录相机共同车辆排除区域外背景像素最大差为 0／255；记录与预测回放的右侧黑车检查区域最大差为 0／255。</p>
<p><a href="assets/nexus_3dgs_v9/asset_audit.json">参数审计</a> · <a href="assets/nexus_3dgs_v9/detail_validation.json">完整验证</a> · <a href="assets/nexus_3dgs_v9/bus_detail_metrics.json">逐帧统计</a> · <a href="nexus_3dgs_v9.md">文字说明与路径</a> · <a href="nexus_3dgs_v8.html">v8 报告</a></p></html>'''
        (destination/'nexus_3dgs_v9.html').write_text(page)
        manifest = {'version':'v9','status':'complete','experiment_status':'candidate','frames_per_video':80,'fps':10,
                    'assets': [{'file':str(path.relative_to(destination)), 'bytes':path.stat().st_size,
                                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(assets.iterdir())]}
        (destination/'nexus_3dgs_v9_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('BUS_DETAIL_REPORT '+str(destination), flush=True)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
