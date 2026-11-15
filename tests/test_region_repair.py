"""Stage-4 region repair: policy, outputs, checks and failure safety (no real model)."""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from mural_agent import workspace
from mural_agent.config import Settings
from mural_agent.engine import Engine
from mural_agent.imaging import image_array, mask_array, sha256
from mural_agent.region_repair import region_policy, run_region_candidates
from mural_agent.storage import Store, new_id


def png(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return buffer.getvalue()


def make_case(tmp_path, kind='paint_loss', confirmed=True, with_mask=True, recommendation='telea'):
    settings = Settings(
        data_dir=tmp_path / 'data', lama_model=tmp_path / 'missing.pt',
        sam_model=tmp_path / 'missing-sam', diffusion_model=tmp_path / 'missing-sd',
        controlnet_model=tmp_path / 'missing-cn', vision_model=tmp_path / 'missing-qwen')
    store = Store(settings.data_dir)
    engine = Engine(settings, store)
    submitted = []
    engine.submit = lambda job_id: submitted.append(job_id)
    height, width = 90, 120
    image = np.zeros((height, width, 3), np.uint8)
    image[..., 0] = np.arange(width, dtype=np.uint8)[None, :] + 60
    image[..., 1] = 120
    image[..., 2] = 80
    image[18:44, 28:54] = [250, 231, 198]
    record = workspace.add_image(store, png(image), 'region-case.png')
    mask = np.zeros((height, width), np.uint8)
    mask[20:42, 30:52] = 255
    proposal_id = new_id()
    if with_mask:
        Image.fromarray(mask).save(store.folder(proposal_id) / 'mask.png')
    store.put('proposal', {
        'id': proposal_id, 'image_id': record['id'], 'created_at': '2026-11-16T00:00:00+00:00',
        'fraction': float(mask.mean()), 'confirmed': confirmed, 'mask_source': 'manual',
        'region_class': 'fillable_lacuna', 'repair_allowed': True,
        'metadata': {'type': kind, 'bbox_pixels': [30, 20, 52, 42],
                     'bbox': [250, 222, 433, 466],
                     'recommendation': {'backend': recommendation, 'method': 'test'},
                     'local_detection': True},
    })
    return store, engine, record, proposal_id, submitted, image, mask


def test_region_policy_prefers_lama_only_for_geometric_damage(tmp_path, monkeypatch):
    store, engine, image, proposal_id, _, _, _ = make_case(tmp_path, kind='paint_loss')
    proposal = store.get('proposal', proposal_id)
    # LaMa weights are absent in this fixture: fall back but still record the intent.
    assert region_policy(engine, proposal) == ('telea', 'lama', region_policy(engine, proposal)[2])
    monkeypatch.setattr(engine.lama, 'available', lambda: True)
    assert region_policy(engine, proposal)[:2] == ('lama', 'lama')

    for kind in ('crack', 'flaking', 'manual'):
        proposal['metadata']['type'] = kind
        assert region_policy(engine, proposal)[:2] == ('lama', 'lama')

    # Observations keep their own path and are never routed through LaMa.
    proposal['metadata'].update(type='fading', recommendation={'backend': 'lama'})
    assert region_policy(engine, proposal) == ('color', 'analysis_only', region_policy(engine, proposal)[2])
    proposal['metadata'].update(type='stain', recommendation={'backend': 'color'})
    assert region_policy(engine, proposal)[0] == 'color'


def test_region_repair_saves_full_region_and_difference_with_checks(tmp_path, monkeypatch):
    store, engine, image, proposal_id, submitted, original, mask = make_case(tmp_path)
    result = run_region_candidates(store, engine, image, [proposal_id])
    assert result['skipped'] == []
    assert len(result['jobs']) == 1
    job = result['jobs'][0]
    assert submitted == [job['id']]
    assert job['region_repair'] is True
    assert job['defect_type'] == 'paint_loss'
    assert job['region_backend'] == 'telea'          # LaMa missing in this fixture
    assert job['region_preferred'] == 'lama'
    assert job['defect_bbox'] == [30, 20, 52, 42]

    def fake_execute(step, base, active_mask, job_record, folder, timeout):
        prediction = base.copy()
        prediction[active_mask > 0] = [12, 200, 40]
        return prediction, {'device': 'cpu', 'weight_sha256': 'fake-weight', 'torch': 'test'}

    monkeypatch.setattr(engine.tools, 'execute', fake_execute)
    engine.run(job['id'])

    job = store.get('job', job['id'])
    assert job['status'] == 'review', job.get('error')
    assert job['checks'] == {'original_unchanged': True, 'output_size': [120, 90],
                             'size_match': True, 'outside_unchanged': True, 'candidates_checked': 1}
    candidate = job['candidates'][0]
    assert candidate['tool'] == 'telea'
    folder = store.folder(candidate['id'])
    result_image = np.asarray(Image.open(folder / 'result.png'))
    assert result_image.shape[:2] == (90, 120)
    assert np.array_equal(result_image[mask == 0], original[mask == 0])
    assert candidate['checks']['size_match'] and candidate['checks']['outside_unchanged']
    assert candidate['checks']['output_size'] == [120, 90]
    assert candidate['versions']['weight_sha256'] == 'fake-weight'

    region = np.asarray(Image.open(folder / 'region.png'))
    assert candidate['region']['box'] == [22, 12, 60, 50]
    assert region.shape[:2] == (38, 38)
    assert (folder / 'region_difference.png').is_file()
    assert candidate['region']['url'].endswith('/region.png')

    # The working original is never overwritten by a repair.
    assert sha256(store.folder(image['id']) / 'image.png') == image['working_sha256']


def test_region_repair_skips_unconfirmed_and_missing_masks(tmp_path):
    store, engine, image, proposal_id, submitted, _, _ = make_case(tmp_path, confirmed=False)
    result = run_region_candidates(store, engine, image, [proposal_id])
    assert result['jobs'] == []
    assert '尚未人工确认' in result['skipped'][0]['reason']
    assert submitted == []

    store.update('proposal', proposal_id, confirmed=True)
    missing_mask = new_id()
    store.put('proposal', {'id': missing_mask, 'image_id': image['id'], 'confirmed': True,
                           'created_at': '2026-11-16T00:00:01+00:00',
                           'metadata': {'type': 'crack', 'bbox_pixels': [10, 10, 20, 20]}})
    result = run_region_candidates(store, engine, image, [missing_mask])
    assert result['jobs'] == []
    assert '还没有修复范围' in result['skipped'][0]['reason']

    with pytest.raises(ValueError):
        run_region_candidates(store, engine, image, [])


def test_observation_type_keeps_analysis_path_without_lama(tmp_path, monkeypatch):
    store, engine, image, proposal_id, _, _, _ = make_case(
        tmp_path, kind='fading', recommendation='lama')
    monkeypatch.setattr(engine.lama, 'available', lambda: True)
    job = run_region_candidates(store, engine, image, [proposal_id])['jobs'][0]
    assert job['region_backend'] == 'color'
    assert job['region_preferred'] == 'analysis_only'
    assert '不做自动补洞' in job['region_reason']


def test_failed_repair_records_failure_without_touching_original(tmp_path, monkeypatch):
    store, engine, image, proposal_id, _, _, _ = make_case(tmp_path)
    job = run_region_candidates(store, engine, image, [proposal_id])['jobs'][0]

    def broken_execute(*args, **kwargs):
        raise RuntimeError('模拟模型崩溃')

    monkeypatch.setattr(engine.tools, 'execute', broken_execute)
    engine.run(job['id'])
    job = store.get('job', job['id'])
    assert job['status'] == 'failed'
    assert job['candidates'] == []
    assert job['checks']['original_unchanged'] is True
    assert sha256(store.folder(image['id']) / 'image.png') == image['working_sha256']
    assert not any(store.folder(job['id']).glob('candidate-*'))
