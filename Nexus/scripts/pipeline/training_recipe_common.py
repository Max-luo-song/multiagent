"""Frozen inputs for the published-training-preprocessing diagnostic."""

import json
from pathlib import Path

from run_repro_demo import file_sha256


BRANCHES = ('published_training_recipe', 'causal_history_recipe')
SOURCE_FILES = (
    'nuplan_extent/planning/training/data_augmentation/only_centerline_reset_v.py',
    'nuplan_extent/planning/training/data_augmentation/reset_v.py',
    'nuplan_extent/planning/script/experiments/training/training_nuplan_nexus.yaml',
    'nuplan_extent/planning/script/experiments/training/testing_nuplan_nexus.yaml',
    'nuplan_extent/planning/training/preprocessing/features/scene_tensor.py',
)


def check_prepared(directory, reference):
    prepared = json.loads((directory/'run.json').read_text())
    baseline = json.loads((reference/'run.json').read_text())
    assert prepared['status'] == baseline['status'] == 'complete'
    assert prepared['plan_sha256'] == baseline['plan_sha256']
    assert prepared['reference_manifest_sha256'] == file_sha256(reference/'run.json')
    assert prepared['branches'] == list(BRANCHES)
    assert prepared['preparation_runner_sha256'] == file_sha256(Path(__file__).with_name('prepare_nuplan_training_recipe.py'))
    assert prepared['prepared_scenarios'] == len(prepared['records']) == 128
    assert prepared['scenarios'] == baseline['scenarios']
    for path, digest in prepared['source_sha256'].items():
        assert file_sha256(Path(path)) == digest, 'Preprocessing source changed: '+path
    for s in baseline['scenarios']:
        record = prepared['records'][str(s['index'])]
        assert record['token'] == s['token']
        for branch in BRANCHES:
            path = directory/record['features'][branch]['file']
            assert path.is_file() and file_sha256(path) == record['features'][branch]['sha256']
    return prepared, baseline
