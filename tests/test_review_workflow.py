"""Stage-6 review workflow: separated states, audit trail, stale results and export gates."""
import json

import numpy as np
import pytest
from PIL import Image

from mural_agent import review_workflow, workspace
from mural_agent.config import Settings
from mural_agent.desktop_web import Bridge
from mural_agent.engine import Engine
from mural_agent.imaging import image_array
from mural_agent.storage import Store, new_id


def make_region_case(tmp_path, risk='high', speculative=False):
    height, width = 90, 120
    original = np.zeros((height, width, 3), np.uint8)
    original[..., 0] = 60
    original[..., 1] = 90
    original[..., 2] = 70
    store = Store(tmp_path / 'data')
    image = workspace.add_image(store, _png(original), 'review-case.png')
    proposal_id, job_id, candidate_id = new_id(), new_id(), new_id()
    mask = np.zeros((height, width), np.uint8)
    mask[20:44, 20:44] = 255
    Image.fromarray(mask).save(store.folder(proposal_id) / 'mask.png')
    store.put('proposal', {
        'id': proposal_id, 'image_id': image['id'], 'created_at': '2026-11-16T00:00:00+00:00',
        'fraction': float(mask.mean()), 'confirmed': True, 'mask_source': 'manual',
        'region_class': risk, 'revision': 1,
        'metadata': {'type': 'paint_loss', 'bbox_pixels': [20, 20, 44, 44],
                     'recommendation': {'backend': 'lama', 'method': 'LaMa 纹理补全'},
                     'local_detection': True},
    })
    result = original.copy()
    result[20:44, 20:44] = [200, 120, 90]
    Image.fromarray(result).save(store.folder(candidate_id) / 'result.png')
    Image.fromarray(np.zeros_like(original)).save(store.folder(candidate_id) / 'difference.png')
    candidate = {'id': candidate_id, 'tool': 'lama', 'parameters': {'tool': 'lama'},
                 'speculative': speculative, 'proposal_revision': 1,
                 'metrics': {'outside_unchanged': True}}
    store.put('job', {
        'id': job_id, 'image_id': image['id'], 'mask_id': new_id(), 'status': 'review',
        'defect_proposal_id': proposal_id, 'defect_risk_class': risk,
        'selected_id': None, 'selected_as': None, 'candidates': [candidate],
        'mode': 'conservative', 'workflow': 'professional_step',
        'created_at': '2026-11-16T00:00:01+00:00',
    })
    store.put('candidate', {'id': candidate_id, 'job_id': job_id, 'image_id': image['id'],
                            'tool': 'lama', 'parameters': {'tool': 'lama'},
                            'proposal_id': proposal_id, 'proposal_revision': 1,
                            'speculative': speculative,
                            'result_sha256': 'x', 'raw_sha256': 'x',
                            'url': f'/api/assets/{candidate_id}/result.png',
                            'difference_url': f'/api/assets/{candidate_id}/difference.png',
                            'metrics': {'outside_unchanged': True}})
    settings = Settings(data_dir=store.root, lama_model=tmp_path / 'missing.pt',
                        deepseek_api_key='')
    engine = Engine(settings, store)
    bridge = Bridge(settings, store, engine)
    return bridge, engine, store, image, proposal_id, job_id, candidate_id


def _png(array):
    from io import BytesIO
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return buffer.getvalue()


def review(bridge, job_id, candidate_id, action, note='', **extra):
    return json.loads(bridge.reviewCandidate(json.dumps({
        'job_id': job_id, 'candidate_id': candidate_id, 'action': action, 'note': note, **extra})))


def test_identification_and_candidate_review_are_separate_states(tmp_path):
    bridge, engine, store, image, proposal_id, job_id, candidate_id = make_region_case(tmp_path)
    # Pending identification blocks accepting a candidate.
    store.update('proposal', proposal_id, confirmed=False, identification={'status': 'pending'})
    blocked = review(bridge, job_id, candidate_id, 'accept')
    assert 'error' in blocked and '确认病害识别' in blocked['error']
    # Confirming identification does not silently accept the candidate.
    defect = json.loads(bridge.editLocalDefect(image['id'], json.dumps({
        'action': 'confirm', 'proposal_id': proposal_id})))['defect']
    assert defect['identification_status'] == 'confirmed'
    assert review_workflow.candidate_state(store.get('job', job_id)['candidates'][0]) == 'pending'
    accepted = review(bridge, job_id, candidate_id, 'accept', note='边缘可接受')
    assert accepted['candidate']['review']['status'] == 'accepted'
    assert accepted['job']['selected_as'] == 'accepted'
    assert accepted['job']['review']['decision'] == 'adopt'
    events = json.loads(bridge.listReviewEvents(image['id']))['events']
    actions = [event['action'] for event in events]
    assert 'identification_confirm' in actions and 'candidate_accept' in actions
    engine.executor.shutdown(wait=False)
    bridge.aux_executor.shutdown(wait=False)


def test_reject_expert_review_and_preview_states(tmp_path):
    bridge, engine, store, image, proposal_id, job_id, candidate_id = make_region_case(tmp_path)
    preview = review(bridge, job_id, candidate_id, 'experimental_preview', note='只看一眼')
    assert preview['candidate']['review']['status'] == 'experimental_preview'
    assert preview['job']['preview_candidate_id'] == candidate_id
    with pytest.raises(ValueError, match='试验性预览'):
        review_workflow.export_guard(store, store.get('job', job_id),
                                     store.get('job', job_id)['candidates'][0])

    rejected = review(bridge, job_id, candidate_id, 'reject', note='纹理不合理')
    assert rejected['candidate']['review']['status'] == 'rejected'
    with pytest.raises(ValueError, match='驳回'):
        review_workflow.export_guard(store, store.get('job', job_id),
                                     store.get('job', job_id)['candidates'][0])

    expert = review(bridge, job_id, candidate_id, 'expert_review', note='需要专家')
    assert expert['candidate']['review']['status'] == 'expert_review'
    assert review_workflow.identification_state(store.get('proposal', proposal_id)) == 'expert_review'
    with pytest.raises(ValueError, match='专家复核'):
        review_workflow.export_guard(store, store.get('job', job_id),
                                     store.get('job', job_id)['candidates'][0])
    # Explicit acceptance resolves the pending expert flag and is recorded.
    resolved = review(bridge, job_id, candidate_id, 'accept', note='专家口头同意')
    assert resolved['candidate']['review']['status'] == 'accepted'
    assert review_workflow.identification_state(store.get('proposal', proposal_id)) == 'confirmed'
    engine.executor.shutdown(wait=False)
    bridge.aux_executor.shutdown(wait=False)


def test_preview_requires_high_risk_or_speculative(tmp_path):
    bridge, engine, store, image, proposal_id, job_id, candidate_id = make_region_case(tmp_path, risk='low')
    blocked = review(bridge, job_id, candidate_id, 'experimental_preview')
    assert 'error' in blocked and '高风险' in blocked['error']
    engine.executor.shutdown(wait=False)
    bridge.aux_executor.shutdown(wait=False)


def test_manual_mask_change_marks_candidates_and_ai_review_stale(tmp_path):
    bridge, engine, store, image, proposal_id, job_id, candidate_id = make_region_case(tmp_path)
    accepted = review(bridge, job_id, candidate_id, 'accept', note='先接受')
    assert accepted['candidate']['review']['status'] == 'accepted'

    # Simulate a stage-5 advisory review attached to this job.
    store.update('job', job_id, ai_review={'assessment_id': new_id(), 'advisory': True})
    store.put('assessment', {'id': new_id(), 'image_id': image['id'], 'created_at': '2026-11-16T00:00:02+00:00',
                             'region_ids': [proposal_id], 'advisory': True, 'regions': []})

    mask = np.zeros((90, 120), np.uint8)
    mask[20:46, 20:46] = 255
    from io import BytesIO
    buffer = BytesIO()
    Image.fromarray(mask).save(buffer, format='PNG')
    import base64
    edited = json.loads(bridge.editLocalDefect(image['id'], json.dumps({
        'action': 'update', 'proposal_id': proposal_id,
        'mask_data': base64.b64encode(buffer.getvalue()).decode(), 'note': '修正边界'})))
    assert edited['defect']['revision'] == 2

    job = store.get('job', job_id)
    assert job['candidates'][0]['stale'] is True
    assert '掩膜' in job['candidates'][0]['stale_reason']
    assert job['ai_review']['stale'] is True
    assessments = store.list('assessment')
    assert assessments and assessments[0]['stale'] is True
    with pytest.raises(ValueError, match='过期'):
        review_workflow.export_guard(store, job, job['candidates'][0])
    stale_accept = review(bridge, job_id, candidate_id, 'accept')
    assert 'error' in stale_accept and '过期' in stale_accept['error']
    engine.executor.shutdown(wait=False)
    bridge.aux_executor.shutdown(wait=False)


def test_recommendation_change_also_invalidates_old_reviews(tmp_path):
    bridge, engine, store, image, proposal_id, job_id, candidate_id = make_region_case(tmp_path)
    edited = json.loads(bridge.editLocalDefect(image['id'], json.dumps({
        'action': 'recommendation', 'proposal_id': proposal_id,
        'method': '局部色调匹配', 'note': '改建议'})))['defect']
    assert edited['recommendation']['method'] == '局部色调匹配'
    assert edited['revision'] == 2
    assert store.get('job', job_id)['candidates'][0]['stale'] is True
    engine.executor.shutdown(wait=False)
    bridge.aux_executor.shutdown(wait=False)


def test_audit_events_persist_and_restore_after_reopen(tmp_path):
    bridge, engine, store, image, proposal_id, job_id, candidate_id = make_region_case(tmp_path)
    review(bridge, job_id, candidate_id, 'accept', note='第一轮采纳')
    api = review(bridge, job_id, candidate_id, 'reject', note='复核后又驳回')

    reopened = Store(store.root)
    events = review_workflow.list_events(reopened, image['id'])
    assert [event['action'] for event in events] == ['candidate_accept', 'candidate_reject']
    for event in events:
        assert event['proposal_id'] == proposal_id
        assert event['created_at'] and 'changes' in event and 'decision' in event
    job = reopened.get('job', job_id)
    assert job['selected_as'] == 'rejected'
    assert job['candidates'][0]['review']['status'] == 'rejected'
    assert job['review']['decision'] == 'reject'
    defects = json.loads(bridge.listLocalDefects(image['id']))['defects']
    assert defects[0]['identification_status'] == 'confirmed'
    assert defects[0]['revision'] == 1
    engine.executor.shutdown(wait=False)
    bridge.aux_executor.shutdown(wait=False)


def test_export_guard_requires_confirmed_identification_and_clean_candidate(tmp_path):
    bridge, engine, store, image, proposal_id, job_id, candidate_id = make_region_case(tmp_path)
    job = store.get('job', job_id)
    candidate = job['candidates'][0]
    store.update('proposal', proposal_id, confirmed=False, identification={'status': 'pending'})
    with pytest.raises(ValueError, match='识别尚未确认'):
        review_workflow.export_guard(store, store.get('job', job_id),
                                     store.get('job', job_id)['candidates'][0])
    store.update('proposal', proposal_id, confirmed=True, identification={'status': 'confirmed'})
    review(bridge, job_id, candidate_id, 'accept', note='ok')
    allowed = review_workflow.export_guard(store, store.get('job', job_id),
                                           store.get('job', job_id)['candidates'][0])
    assert allowed['selected_as'] == 'accepted'

    # Speculative output selected as a hypothesis still needs the archive flag.
    hypothesis_job = {**store.get('job', job_id), 'selected_as': 'hypothesis',
                      'review': {'archive_as_hypothesis': False, 'decision': 'adopt'}}
    with pytest.raises(ValueError, match='假设'):
        review_workflow.export_guard(store, hypothesis_job,
                                     store.get('job', job_id)['candidates'][0])
    engine.executor.shutdown(wait=False)
    bridge.aux_executor.shutdown(wait=False)


def test_engine_export_blocks_stale_region_candidates(tmp_path):
    bridge, engine, store, image, proposal_id, job_id, candidate_id = make_region_case(tmp_path)
    review(bridge, job_id, candidate_id, 'accept', note='先接受')
    mask_record = workspace.add_mask(store, image,
                                     _png(np.pad(np.full((20, 20), 255, np.uint8),
                                                 ((20, 50), (20, 80)))),
                                     [proposal_id], mask_source='manual', user_confirmed=True)
    store.update('job', job_id, mask_id=mask_record['id'], mask_file_url=None)
    review_workflow.bump_revision(store, proposal_id, '掩膜或区域已人工修改', action='region_update')
    with pytest.raises(ValueError, match='过期'):
        engine.export(job_id)
    engine.executor.shutdown(wait=False)
    bridge.aux_executor.shutdown(wait=False)


def test_standard_job_selection_still_works_and_is_audited(tmp_path):
    """Legacy manual job (no defect region) keeps the old one-click flow."""
    store = Store(tmp_path / 'data')
    height, width = 40, 60
    image = workspace.add_image(store, _png(np.full((height, width, 3), 120, np.uint8)), 'manual.png')
    partial = np.zeros((height, width), np.uint8)
    partial[10:30, 10:30] = 255
    mask = workspace.add_mask(store, image, _png(partial),
                              mask_source='manual', user_confirmed=True, region_class='fillable_lacuna')
    candidate_id = new_id()
    Image.fromarray(np.full((height, width, 3), 140, np.uint8)).save(store.folder(candidate_id) / 'result.png')
    Image.fromarray(np.zeros((height, width, 3), np.uint8)).save(store.folder(candidate_id) / 'difference.png')
    job_id = new_id()
    store.put('job', {'id': job_id, 'image_id': image['id'], 'mask_id': mask['id'], 'status': 'review',
                      'workflow': 'standard', 'candidates': [{'id': candidate_id, 'tool': 'telea',
                                                              'metrics': {'outside_unchanged': True}}],
                      'selected_id': None})
    settings = Settings(data_dir=store.root, lama_model=tmp_path / 'missing.pt')
    engine = Engine(settings, store)
    bridge = Bridge(settings, store, engine)
    selected = json.loads(bridge.selectCandidate(job_id, candidate_id, '旧流程选定'))
    assert selected['selected_as'] == 'accepted'
    assert selected['candidates'][0]['review']['status'] == 'accepted'
    events = json.loads(bridge.listReviewEvents(image['id']))['events']
    assert events[0]['action'] == 'candidate_select'
    assert events[0]['proposal_id'] == ''
    engine.executor.shutdown(wait=False)
    bridge.aux_executor.shutdown(wait=False)
