"""Verify and archive the completed 20261028 synthetic three-car delivery."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/nexus_3dgs/asset_insert_20261028'
REPORT = ROOT / 'report/nexus_threecar_asset_insert_20261028'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ('src', 'href') and value:
                parsed = urlparse(value)
                if not parsed.scheme and parsed.path:
                    self.paths.append(unquote(parsed.path))


def main():
    summary = read(REPORT / 'summary.json')
    run = read(OUT / 'render_final/run.json')
    protocol = read(OUT / 'insertion_protocol.json')
    assert summary['status'] == run['status'] == 'complete'
    assert not run['probe_only']
    assert read(REPORT / 'render_run.json') == run
    assert sha(run['checkpoint']) == run['checkpoint_sha256']
    assert sha(ROOT / 'scripts/pipeline/render_inserted_threecar.py') == run['renderer_sha256']
    assert sha(ROOT / 'nexus_3dgs/inserted_vehicle.py') == run['asset_loader_sha256']
    source_hashes = dict(protocol['source_prediction_sha256'])
    source_hashes.update({a['source']: a['source_sha256'] for a in run['asset_info'].values()})
    for path, expected in source_hashes.items():
        assert sha(path) == expected, path
    assert all(run[key] for key in (
        'background_and_original_models_unchanged',
        'asset_sources_unchanged', 'Nexus_predictions_unchanged'))
    pose = read(OUT / 'reconstruction/nuplan/background8000/pose_protection.json')
    assert pose['camera_and_recorded_vehicle_poses_bitwise_unchanged']
    for branch in ('normal', 'event'):
        audit = summary['geometry_audit'][branch]
        assert audit['accepted']
        assert audit['actual_3D_body_overlap_samples'] == 0
        assert audit['actual_3D_body_offroad_samples'] == 0
        assert audit['projected_actor_centers_inside_image'] == 243
        assert len(run['frames'][branch]) == 81
        assert sha(OUT / f'render_final/{branch}/camera_0.mp4') == run['videos'][branch]['sha256']
    for normal, event in zip(run['frames']['normal'], run['frames']['event']):
        assert normal['camera_scene'] == event['camera_scene']
        assert normal['intrinsics'] == event['intrinsics']
    for name, video in summary['videos'].items():
        assert video['frames'] == 81 and video['fps'] == 10
        assert sha(REPORT / 'media' / name) == video['sha256'], name
    assert summary['geometry_audit']['event']['C_complete_target_lane_1s_hold_start_s'] == 4.0
    assert summary['geometry_audit']['event']['C_rear_fully_passes_B_front_s'] == 5.2
    assert not summary['geometry_audit']['normal']['C_final_in_target_lane']

    doc = ROOT / 'docs/nexus_threecar_asset_insert_20261028.md'
    shutil.copy2(doc, REPORT / 'implementation.md')
    parser = Links()
    parser.feed((REPORT / 'index.html').read_text())
    for name in parser.paths:
        assert (REPORT / name).is_file(), name

    snapshot = OUT / 'reproduction'
    snapshot.mkdir(exist_ok=True)
    sources = [
        'docs/nexus_threecar_asset_insert_20261028.md',
        'nexus_3dgs/inserted_vehicle.py', 'nexus_3dgs/replay.py',
        'nexus_3dgs/geometry.py', 'nexus_3dgs/bundle.py',
        'nexus_3dgs/scene_projection.py',
        'Omniscenegraph/models/gaussians/support.py',
    ]
    sources += ['scripts/pipeline/' + name + '.py' for name in (
        'preview_inserted_gaussian_assets', 'prepare_inserted_threecar',
        'prepare_inserted_threecar_frenet', 'render_inserted_threecar',
        'probe_inserted_road_depth', 'report_inserted_threecar',
        'verify_inserted_threecar_delivery', 'prepare_nuplan_3dgs',
        'train_nexus_3dgs_pilot', 'audit_nexus_omega_v2')]
    manifest = {}
    for relative in sources:
        source = ROOT / relative
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative] = sha(source)
        assert sha(target) == manifest[relative]
    shutil.copy2(OUT / 'background_config.yaml', snapshot / 'background_config.yaml')
    manifest['background_config.yaml'] = sha(snapshot / 'background_config.yaml')
    write(snapshot / 'manifest.json', manifest)
    verification = {
        'status': 'passed',
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint_sha256': run['checkpoint_sha256'],
        'original_source_hashes_verified': source_hashes,
        'renderer_and_loader_match_render_time_hashes': True,
        'recorded_background_poses_unchanged': True,
        'camera_and_intrinsics_identical_between_branches': True,
        'actual_rendered_three_car_sampled_geometry_passed': True,
        'video_hashes_match_producer_decoded_81_frame_10fps_records': True,
        'html_local_links_exist': parser.paths,
        'documentation_sha256': sha(doc),
        'source_snapshot_manifest': str(snapshot / 'manifest.json'),
        'source_snapshot_files': len(manifest),
        'strict_causal_chain_proven': False,
        'full_scene_safety_proven': False,
        'background_novel_view_quality_fully_solved': False,
    }
    write(OUT / 'final_verification.json', verification)
    write(REPORT / 'verification.json', verification)
    print(json.dumps({'status': 'passed', 'snapshot_files': len(manifest),
                      'videos': len(summary['videos']), 'html_links': len(parser.paths)}))


if __name__ == '__main__':
    main()
