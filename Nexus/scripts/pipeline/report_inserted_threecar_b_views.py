"""Package the B-attached native Gaussian videos into a synchronized quad view."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/nexus_3dgs/asset_insert_20261028'
RENDER = OUT / 'b_multiview_final'
REPORT = ROOT / 'report/nexus_threecar_B_multiview_20261028'
sys.path.insert(0, str(ROOT / 'scripts/pipeline'))
from render_inserted_threecar import sha, write


def main():
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    run = json.loads((RENDER / 'run.json').read_text())
    source = json.loads(Path(run['source_render_run']).read_text())
    assert run['status'] == 'complete' and not run['probe_only']
    assert sha(ROOT / 'scripts/pipeline/render_inserted_threecar_b_views.py') == run['renderer_sha256']
    for path, digest in run['source_hashes_unchanged'].items():
        assert sha(path) == digest, path
    media = REPORT / 'media'
    media.mkdir(parents=True, exist_ok=False)
    views = ['front', 'rear', 'left', 'right']
    labels = {'front': 'B车前视：观察A切入', 'rear': 'B车后视：观察C接近',
              'left': 'B车左视：观察C变道超越', 'right': 'B车右视'}
    font_path = ROOT / 'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'
    title_font = ImageFont.truetype(str(font_path), 25)
    label_font = ImageFont.truetype(str(font_path), 22)
    videos = {}
    timeline = {}
    for branch in ('event', 'normal'):
        rows = run['frames'][branch]
        assert len(rows) == 81
        for index, row in enumerate(rows):
            original = source['frames'][branch][index]
            B = np.array(next(a for a in original['actors'] if a['role'] == 'B')['body_world'])
            for view in views:
                mount = np.array(run['camera_mounts_B_body'][view])
                assert np.allclose(np.array(row['views'][view]['camera_world']), B @ mount, atol=1e-9)
        timeline[branch] = [
            {'time_s': row['time_s'], 'A_center_in_views': [view for view in views
                 if row['views'][view]['actor_center_frustum']['A']['center_in_frustum']],
             'C_center_in_views': [view for view in views
                 if row['views'][view]['actor_center_frustum']['C']['center_in_frustum']]}
            for row in rows]
        for view in views:
            src = RENDER / branch / f'{view}.mp4'
            assert sha(src) == run['videos'][branch][view]['sha256']
            dst = media / f'{branch}_B_{view}.mp4'
            shutil.copy2(src, dst)
            videos[dst.name] = dict(run['videos'][branch][view], size=run['size'])
        readers = {view: imageio.get_reader(RENDER / branch / f'{view}.mp4') for view in views}
        path = media / f'{branch}_B_multiview.mp4'
        writer = imageio.get_writer(path, fps=10, macro_block_size=2,
                                    ffmpeg_params=['-threads', '2', '-crf', '18'])
        try:
            for index in range(81):
                canvas = Image.new('RGB', (1920, 1200), '#f3f6fa')
                draw = ImageDraw.Draw(canvas)
                title = '切入组：A切入→B减速→C变道绕行' if branch == 'event' else '未切入组：A在左侧减速，B/C继续跟随'
                draw.text((14, 8), f'{title} | B车四向同步视角 | {index / 10:.1f}s', font=title_font, fill='#24364b')
                for slot, view in enumerate(views):
                    x, y = (slot % 2) * 960, 48 + (slot // 2) * 576
                    draw.text((x + 12, y + 4), labels[view], font=label_font, fill='#24364b')
                    frame = readers[view].get_next_data()
                    assert frame.shape[:2] == (540, 960)
                    canvas.paste(Image.fromarray(frame), (x, y + 36))
                writer.append_data(np.asarray(canvas))
                if index in (0, 20, 40, 60, 80):
                    canvas.save(media / f'{branch}_{index:03d}.jpg')
        finally:
            writer.close()
            for reader in readers.values():
                reader.close()
        with imageio.get_reader(path) as reader:
            count = sum(1 for _ in reader)
            metadata = reader.get_meta_data()
        assert count == 81 and metadata['fps'] == 10
        videos[path.name] = {'frames': count, 'fps': 10, 'size': [1920, 1200], 'sha256': sha(path)}
    # Retain an overview alongside the subjective cameras so the viewer can
    # identify B and understand the full three-car motion.
    overview = ROOT / 'report/nexus_threecar_asset_insert_20261028/media/event_with_bev.mp4'
    shutil.copy2(overview, media / 'event_external_overview.mp4')
    videos['event_external_overview.mp4'] = {'sha256': sha(overview), 'frames': 81, 'fps': 10}
    summary = {
        'status': 'complete', 'camera_host': 'B', 'views': views,
        'video_duration_content_s': [0, 8], 'frames': 81, 'fps': 10,
        'exact_source_actor_poses_preserved': True,
        'camera_mount_check_all_648_samples_passed': True,
        'fixed_intrinsics_no_zoom': True,
        'host_omitted_in_own_cameras': True,
        'source_scope': run['source_scope'], 'videos': videos,
        'background_rear_side_quality_fully_solved': False,
        'visibility_scope': 'frustum center checks only; not an occlusion/recognition guarantee',
    }
    write(REPORT / 'summary.json', summary)
    write(REPORT / 'visibility_timeline.json', timeline)
    shutil.copy2(RENDER / 'run.json', REPORT / 'render_run.json')
    doc = ROOT / 'docs/nexus_threecar_B_multiview_20261028.md'
    if doc.exists():
        shutil.copy2(doc, REPORT / 'implementation.md')
    (REPORT / 'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>B车多视角：三车交互</title><style>body{font-family:system-ui;max-width:1400px;margin:24px auto;padding:0 18px;background:#f3f6fa;color:#24364b}section{padding:22px;background:white;border-radius:12px;margin:18px 0}p{line-height:1.8}video{width:100%;max-height:85vh}a{color:#2364a8}.links{display:flex;flex-wrap:wrap;gap:18px}</style>
<h1>坐到中间的B车上，看三车交互</h1>
<section><p>四宫格依次为：左上前视、右上后视、左下左视、右下右视。四路相机跟随B车同一时刻的位置与转向，右上后视是真实朝后的方向，不做后视镜左右翻转。</p>
<video controls preload="metadata" src="media/event_B_multiview.mp4"></video>
<p>先从前视看A切入，再从后视看C接近、从左视看C变道超越。C约4.0秒完整进入相邻车道，5.2秒完全超过B。单独看前视无法同时看到后方C的反应，因此保留同步四向画面。</p>
<div class="links"><a href="media/event_B_multiview.mp4">打开四视角视频</a><a href="media/event_B_front.mp4">B车前视</a><a href="media/event_B_rear.mp4">B车后视</a><a href="media/event_B_left.mp4">B车左视</a><a href="media/event_B_right.mp4">B车右视</a></div></section>
<section><h2>未切入对照</h2><video controls preload="metadata" src="media/normal_B_multiview.mp4"></video><p>两组相机均刚性安装在各自B车上。B车两组轨迹不同，相机世界路径也随之不同；安装位置、朝向和105°水平视场相同。两组A使用相同纵向减速安排，主要区别为是否切入。</p></section>
<section><h2>需要确认三辆车位置时，参考外部视角</h2><video controls preload="metadata" src="media/event_external_overview.mp4"></video></section>
<section><h2>实现范围</h2><p>复用上次已核验的三车运动和资产，只改变相机。四路相机共用B车顶中心上方15厘米的虚拟安装位，分别朝前、后、左、右，随B车位姿运动；资产没有车内模型，所以自身B车不参与这四路向外渲染。这是B车安装相机视角，未重建座舱，也非该车型的真实标定相机。</p><p>背景来自121路段，三车轨迹来自059的Nexus＋行为引导结果并已适配到该道路；此次没有重新生成轨迹。背景主要由前向及侧前向图像训练，后方、侧方新视角存在缺失和模糊，车辆可见性还受真实遮挡影响。四路为原生3DGS重新光栅化，单独画面不叠加车辆文字。</p><div class="links"><a href="summary.json">视频校验</a><a href="visibility_timeline.json">各时刻视野检查</a><a href="implementation.md">实现说明</a></div></section></html>''')
    print('B_MULTIVIEW_REPORT_COMPLETE', REPORT, flush=True)


if __name__ == '__main__':
    main()
