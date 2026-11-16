"""Check and snapshot the B-hosted multiview delivery without rerendering."""
import json
import shutil
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/nexus_3dgs/asset_insert_20261028'
REPORT = ROOT / 'report/nexus_threecar_B_multiview_20261028'
sys.path.insert(0, str(ROOT / 'scripts/pipeline'))
from render_inserted_threecar import sha, write


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value and not urlparse(value).scheme:
                self.paths.append(urlparse(value).path)


def main():
    import numpy as np
    run = json.loads((OUT / 'b_multiview_final/run.json').read_text())
    summary = json.loads((REPORT / 'summary.json').read_text())
    assert run['status'] == summary['status'] == 'complete' and not run['probe_only']
    assert json.loads((REPORT / 'render_run.json').read_text()) == run
    for path, digest in run['source_hashes_unchanged'].items():
        assert sha(path) == digest, path
    assert sha(ROOT / 'scripts/pipeline/render_inserted_threecar_b_views.py') == run['renderer_sha256']
    source = json.loads(Path(run['source_render_run']).read_text())
    mounts = {key: np.array(value) for key, value in run['camera_mounts_B_body'].items()}
    for matrix in mounts.values():
        assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3))
        assert np.linalg.det(matrix[:3, :3]) > .999999
    assert all(np.array_equal(mounts['front'][:3, 3], matrix[:3, 3]) for matrix in mounts.values())
    mount_checks = 0
    for branch in ('normal', 'event'):
        rows = run['frames'][branch]
        assert len(rows) == 81
        for index, row in enumerate(rows):
            original = source['frames'][branch][index]
            assert row['frame'] == index and row['time_s'] == original['time_s']
            B = np.array(next(a for a in original['actors'] if a['role'] == 'B')['body_world'])
            for view, mount in mounts.items():
                assert np.allclose(np.array(row['views'][view]['camera_world']), B @ mount, atol=1e-9)
                mount_checks += 1
            for role in ('A', 'C'):
                assert any(view['actor_center_frustum'][role]['center_in_frustum'] for view in row['views'].values())
    assert mount_checks == 648
    for name, record in summary['videos'].items():
        assert sha(REPORT / 'media' / name) == record['sha256'], name
        assert record['frames'] == 81 and record['fps'] == 10
    links = Links()
    links.feed((REPORT / 'index.html').read_text())
    for path in links.paths:
        assert (REPORT / path).is_file(), path
    doc = ROOT / 'docs/nexus_threecar_B_multiview_20261028.md'
    assert sha(doc) == sha(REPORT / 'implementation.md')
    snapshot = OUT / 'b_multiview_reproduction'
    snapshot.mkdir(exist_ok=True)
    files = ['docs/nexus_threecar_B_multiview_20261028.md',
             'scripts/pipeline/render_inserted_threecar_b_views.py',
             'scripts/pipeline/report_inserted_threecar_b_views.py',
             'scripts/pipeline/verify_inserted_threecar_b_views.py',
             'scripts/pipeline/nexus_3dgs.sh',
             'scripts/pipeline/render_inserted_threecar.py',
             'nexus_3dgs/inserted_vehicle.py', 'nexus_3dgs/replay.py']
    manifest = {}
    for relative in files:
        dst = snapshot / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dst)
        manifest[relative] = sha(dst)
        assert manifest[relative] == sha(ROOT / relative)
    write(snapshot / 'manifest.json', manifest)
    record = {
        'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'B_camera_attachment_checks': mount_checks,
        'source_trajectories_assets_background_and_prior_render_unchanged': True,
        'each_A_C_center_in_at_least_one_view_every_sample': True,
        'source_video_frame_counts_and_fps_producer_decoded': True,
        'delivery_video_hashes_verified': len(summary['videos']),
        'html_local_paths_verified': len(links.paths),
        'source_snapshot_manifest': str(snapshot / 'manifest.json'),
        'visibility_including_occlusion_proven': False,
        'side_rear_background_quality_fully_solved': False,
    }
    write(REPORT / 'verification.json', record)
    write(OUT / 'b_multiview_verification.json', record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
