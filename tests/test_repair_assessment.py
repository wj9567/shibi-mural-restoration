"""Stage-5 DeepSeek repair review: mocked HTTP, region binding, local rules, no side effects."""
from io import BytesIO
import json
import time

import httpx
import numpy as np
import pytest
from PIL import Image

from mural_agent import repair_assessment, workspace
from mural_agent.config import Settings
from mural_agent.deepseek_vision import DeepSeekVisionError, DeepSeekVisionProvider
from mural_agent.desktop_web import Bridge
from mural_agent.engine import Engine
from mural_agent.storage import Store, new_id
from mural_agent.vision_schema import RepairAssessment, RepairRegionReview

MOCK_KEY = 'fixture-repair-review-key-not-real'


@pytest.fixture
def mock_http(monkeypatch):
    client = httpx.Client

    def install(handler):
        monkeypatch.setattr('mural_agent.deepseek_vision.httpx.Client',
                            lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))

    return install


def png(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return buffer.getvalue()


def completion(payload, finish_reason='stop'):
    return httpx.Response(200, json={'choices': [{'finish_reason': finish_reason,
                                                 'message': {'content': payload}}]})


def review(region_id, **changes):
    return {
        'region_id': region_id,
        'observation': '修复区域纹理连续，边缘仍有轻微色差。',
        'repair_suggestion': '建议人工对比原图后，局部降低补全强度。',
        'repair_assessment': '局部补全基本连续，但依据仅来自图像外观。',
        'basis': '对比修复前后边缘与周边颜料衔接。',
        'risk_level': 'medium',
        'uncertainty': '缺少该壁画的颜料与年代资料。',
        'human_review_required': True,
        'review_reason': '依据不足，需要修复人员核对。',
        **changes,
    }


def make_case(tmp_path, two_regions=True, confirmed=True):
    height, width = 120, 160
    original = np.zeros((height, width, 3), np.uint8)
    original[..., 0] = np.arange(width, dtype=np.uint8)[None, :] + 40
    original[..., 1] = 90
    original[..., 2] = 60
    store = Store(tmp_path / 'data')
    image = workspace.add_image(store, png(original), 'repair-case.png')
    entries = []

    def add(kind, box, color, ordinal=0):
        proposal_id, job_id, candidate_id = new_id(), new_id(), new_id()
        x0, y0, x1, y1 = box
        mask = np.zeros((height, width), np.uint8)
        mask[y0:y1, x0:x1] = 255
        Image.fromarray(mask).save(store.folder(proposal_id) / 'mask.png')
        store.put('proposal', {
            'id': proposal_id, 'image_id': image['id'],
            'created_at': f'2026-11-16T00:00:0{ordinal}+00:00',
            'fraction': float(mask.mean()), 'confirmed': confirmed, 'mask_source': 'manual',
            'metadata': {'type': kind, 'bbox_pixels': list(box), 'evidence': f'{kind} 可见依据',
                         'recommendation': {'backend': 'lama', 'method': 'LaMa 纹理补全'},
                         'local_detection': True},
        })
        result = original.copy()
        result[y0:y1, x0:x1] = color
        Image.fromarray(result).save(store.folder(candidate_id) / 'result.png')
        Image.fromarray(np.zeros_like(original)).save(store.folder(candidate_id) / 'difference.png')
        store.put('job', {'id': job_id, 'image_id': image['id'], 'status': 'review',
                          'defect_proposal_id': proposal_id, 'defect_bbox': list(box),
                          'mask_id': new_id(), 'mask_file_url': None,
                          'mode': 'conservative', 'review': {'note': '未审核'},
                          'candidates': [{'id': candidate_id, 'tool': 'lama',
                                          'parameters': {'tool': 'lama'}, 'speculative': False}]})
        entries.append((proposal_id, job_id, candidate_id))

    add('paint_loss', (20, 20, 44, 44), [200, 120, 90], ordinal=0)
    if two_regions:
        add('crack', (90, 60, 120, 72), [180, 150, 120], ordinal=1)
    return store, image, entries, original


def test_request_sends_original_repaired_regions_and_context(mock_http):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        assert MOCK_KEY.encode() not in request.content
        return completion(json.dumps({'regions': [review('r1')]}))

    mock_http(handler)
    original = Image.new('RGB', (600, 400), (30, 40, 50))
    repaired = Image.new('RGB', (600, 400), (35, 45, 55))
    crop_before = Image.new('RGB', (60, 60), (20, 20, 20))
    crop_after = Image.new('RGB', (60, 60), (60, 60, 60))
    result = DeepSeekVisionProvider(MOCK_KEY, timeout=20).assess_repair(
        original, repaired,
        [{'region_id': 'r1', 'type': 'crack', 'bbox_0_1000': [10, 10, 20, 20],
          'method': 'LaMa 纹理补全', 'mask_fraction': .02}],
        context='参考纹样为卷草纹',
        crops=[{'region_id': 'r1', 'before': crop_before, 'after': crop_after}])
    assert result.advisory is True and result.model_source == 'deepseek-flash'
    assert result.regions[0].region_id == 'r1'
    body, = bodies
    assert 'response_format' not in body
    assert body['safety_check'] == {'level': 'high'}
    content = body['messages'][0]['content']
    texts = [item['text'] for item in content if item['type'] == 'text']
    images = [item for item in content if item['type'] == 'image_url']
    assert len(images) == 3          # original, repaired, one before/after crop
    assert all(image['image_url']['url'].startswith('data:image/jpeg;base64,') for image in images)
    assert any('r1' in text and '参考纹样为卷草纹' in text for text in texts)
    assert any('每个 region_id' in text for text in texts)


def test_unknown_or_missing_region_ids_are_rejected(mock_http):
    def unknown(request):
        return completion(json.dumps({'regions': [review('not-a-local-region')]}))

    mock_http(unknown)
    provider = DeepSeekVisionProvider(MOCK_KEY)
    with pytest.raises(DeepSeekVisionError, match='不一致'):
        provider.assess_repair(Image.new('RGB', (64, 64)), Image.new('RGB', (64, 64)),
                               [{'region_id': 'local-1', 'type': 'crack'}])

    def missing(request):
        return completion('{"regions": []}')

    mock_http(missing)
    with pytest.raises(DeepSeekVisionError, match='缺失'):
        provider.assess_repair(Image.new('RGB', (64, 64)), Image.new('RGB', (64, 64)),
                               [{'region_id': 'local-1', 'type': 'crack'}])


@pytest.mark.parametrize('payload', [
    'not json at all',
    '{"regions": [{"region_id": "local-1"}]}',                    # missing required fields
    '{"regions": [%s]}' % str(review('local-1')),                 # python repr, not JSON
])
def test_invalid_payloads_are_rejected(mock_http, payload):
    mock_http(lambda request: completion(payload))
    with pytest.raises(DeepSeekVisionError):
        DeepSeekVisionProvider(MOCK_KEY).assess_repair(
            Image.new('RGB', (64, 64)), Image.new('RGB', (64, 64)),
            [{'region_id': 'local-1', 'type': 'crack'}])


def test_credential_echo_is_rejected(mock_http):
    body = {'regions': [review('local-1', basis=MOCK_KEY)]}
    mock_http(lambda request: completion(json.dumps(body)))
    with pytest.raises(DeepSeekVisionError, match='敏感信息'):
        DeepSeekVisionProvider(MOCK_KEY).assess_repair(
            Image.new('RGB', (64, 64)), Image.new('RGB', (64, 64)),
            [{'region_id': 'local-1', 'type': 'crack'}])


def test_local_rules_force_review_for_faces_area_and_sensitive_context(tmp_path, monkeypatch):
    store, image, entries, original = make_case(tmp_path)
    cases = repair_assessment.collect_cases(store, image)
    assert len(cases) == 2
    assessment = type('A', (), {
        'regions': [
            RepairRegionReview.model_validate(review(entries[0][0], risk_level='high', basis='短')),
            RepairRegionReview.model_validate(review(entries[1][0], risk_level='low',
                                                      human_review_required=False, review_reason='',
                                                      basis='对比了修复前后边缘和周边颜料，判断衔接自然。')),
        ],
        'limitations': '模型只看到缩略图。',
        'model_source': 'deepseek-flash',
        'source': 'deepseek',
    })()
    monkeypatch.setattr(repair_assessment, 'detect_faces', lambda picture: [[20, 20, 44, 44]])
    rows = repair_assessment.merge(assessment, cases, original, '该壁画有宗教题材与题记文字')
    by_id = {row['region_id']: row for row in rows}
    first, second = by_id[entries[0][0]], by_id[entries[1][0]]
    assert first['human_review_required'] is True
    for reason in ('面部', '高风险', '依据不足', '宗教'):
        assert any(reason in flag for flag in first['local_flags']), first
    # The second region also touches a declared sensitive reference, so local
    # rules override the model's "no review" answer.
    assert second['model_review_required'] is False
    assert second['human_review_required'] is True
    assert '用户提供的背景' in second['review_reason']


def test_clean_region_is_not_forced_into_review(tmp_path, monkeypatch):
    store, image, entries, original = make_case(tmp_path, two_regions=False)
    cases = repair_assessment.collect_cases(store, image)
    monkeypatch.setattr(repair_assessment, 'detect_faces', lambda picture: [])
    assessment = type('A', (), {
        'regions': [RepairRegionReview.model_validate(review(
            entries[0][0], risk_level='low', human_review_required=False, review_reason='',
            basis='对比修复前后边缘与周边颜料，衔接自然。'))],
        'limitations': '', 'model_source': 'deepseek-flash', 'source': 'deepseek',
    })()
    first, = repair_assessment.merge(assessment, cases, original, '')
    assert first['human_review_required'] is False
    assert first['review_reason'] == ''


def test_compose_and_record_never_touch_masks_jobs_or_candidates(tmp_path):
    store, image, entries, original = make_case(tmp_path)
    cases = repair_assessment.collect_cases(store, image)
    composite_original, composite = repair_assessment.compose_repaired(store, image, cases)
    assert np.array_equal(composite_original, original)
    union = np.zeros(original.shape[:2], bool)
    for case in cases:
        x0, y0, x1, y1 = case['bbox_pixels']
        color = np.array([200, 120, 90] if x1 == 44 else [180, 150, 120])
        assert np.all(composite[y0:y1, x0:x1] == color)
        union |= case['mask'] > 0
    assert np.array_equal(composite[~union], original[~union])

    before = {kind: len(store.list(kind)) for kind in ('mask', 'candidate', 'proposal', 'job')}
    mask_hashes = {pid: (store.folder(pid) / 'mask.png').read_bytes() for pid, _, _ in entries}
    jobs = {job_id: store.get('job', job_id) for _, job_id, _ in entries}
    assessment = type('A', (), {
        'regions': [RepairRegionReview.model_validate(review(pid)) for pid, _, _ in entries],
        'limitations': '仅供参考', 'model_source': 'deepseek-flash', 'source': 'deepseek',
    })()
    rows = repair_assessment.merge(assessment, cases, original, '')
    record = repair_assessment.assessment_record(store, image, cases[0]['job'], cases, assessment, rows, '')
    assert record['advisory'] is True
    assert {kind: len(store.list(kind)) for kind in ('mask', 'candidate', 'proposal', 'job')} == before
    for pid, _, _ in entries:
        assert (store.folder(pid) / 'mask.png').read_bytes() == mask_hashes[pid]
    for _, job_id, _ in entries:
        job = store.get('job', job_id)
        assert job['review'] == jobs[job_id]['review']
        assert job['candidates'][0]['id'] == jobs[job_id]['candidates'][0]['id']
    assert store.get('job', entries[0][1])['ai_review']['advisory'] is True


def test_collect_cases_only_includes_confirmed_typed_defects(tmp_path):
    store, image, entries, _ = make_case(tmp_path)
    pending = new_id()
    Image.fromarray(np.full((120, 160), 255, np.uint8)).save(store.folder(pending) / 'mask.png')
    store.put('proposal', {'id': pending, 'image_id': image['id'], 'confirmed': False,
                           'created_at': '2026-11-16T00:00:02+00:00',
                           'metadata': {'type': 'crack', 'bbox_pixels': [1, 1, 10, 10]}})
    ids = {case['region_id'] for case in repair_assessment.collect_cases(store, image)}
    assert pending not in ids
    assert ids == {proposal_id for proposal_id, _, _ in entries}


def wait_aux(bridge, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = json.loads(bridge.snapshot())
        if snapshot['aux'].get('status') in {'done', 'error'}:
            return snapshot
        time.sleep(.02)
    raise AssertionError('aux did not finish')


def make_bridge(tmp_path):
    store, image, entries, original = make_case(tmp_path)
    settings = Settings(data_dir=store.root, lama_model=tmp_path / 'missing.pt',
                        deepseek_api_key='mock-bridge-key')
    engine = Engine(settings, store)
    return Bridge(settings, store, engine), engine, store, image, entries, original


def fake_review(self, original, repaired, regions, context='', crops=None):
    ids = [region['region_id'] for region in regions]
    return RepairAssessment(source='deepseek', model_source='deepseek-flash', limitations='仅缩略图。',
                            regions=[RepairRegionReview(
                                region_id=region_id, observation='修复区域纹理连续。',
                                repair_suggestion='建议人工核对边缘。',
                                repair_assessment='局部补全基本合理。', basis='对比修复前后区域。',
                                risk_level='low', uncertainty='缺少历史资料。',
                                human_review_required=False, review_reason='') for region_id in ids])


def test_bridge_assessment_persists_advisory_record_only(tmp_path, monkeypatch):
    bridge, engine, store, image, entries, _ = make_bridge(tmp_path)
    monkeypatch.setattr(DeepSeekVisionProvider, 'assess_repair', fake_review)
    mask_before = (store.folder(entries[0][0]) / 'mask.png').read_bytes()
    job_before = store.get('job', entries[0][1])
    started = json.loads(bridge.assessRepair(json.dumps({
        'image_id': image['id'], 'job_id': entries[0][1], 'context': '参考纹样'})))
    assert started['status'] == 'running'
    snapshot = wait_aux(bridge)
    assert snapshot['aux']['status'] == 'done', snapshot['aux']
    record = snapshot['aux']['result']['assessment']
    assert record['advisory'] is True and record['model_source'] == 'deepseek-flash'
    assert {row['region_id'] for row in record['regions']} == {pid for pid, _, _ in entries}
    assert all(row['candidate']['tool'] == 'lama' for row in record['regions'])
    assert 'mock-bridge-key' not in json.dumps(snapshot)
    job_after = store.get('job', entries[0][1])
    assert job_after['ai_review']['advisory'] is True
    assert job_after['review'] == job_before['review']
    assert job_after.get('selected_id') is None and job_after['candidates'] == job_before['candidates']
    assert (store.folder(entries[0][0]) / 'mask.png').read_bytes() == mask_before
    engine.executor.shutdown(wait=False)
    bridge.aux_executor.shutdown(wait=False)


def test_bridge_assessment_failure_is_reported_and_local_results_stay(tmp_path, monkeypatch):
    bridge, engine, store, image, entries, _ = make_bridge(tmp_path)

    def broken(*args, **kwargs):
        raise DeepSeekVisionError('DeepSeek 网络请求失败，请检查连接或改用手动圈选。')

    monkeypatch.setattr(DeepSeekVisionProvider, 'assess_repair', broken)
    started = json.loads(bridge.assessRepair(json.dumps({
        'image_id': image['id'], 'job_id': entries[0][1]})))
    assert started['status'] == 'running'
    snapshot = wait_aux(bridge)
    assert snapshot['aux']['status'] == 'error'
    assert '网络请求失败' in snapshot['aux']['error']
    assert store.get('job', entries[0][1])['candidates']
    assert store.list('assessment') == []
    engine.executor.shutdown(wait=False)
    bridge.aux_executor.shutdown(wait=False)


def test_bridge_assessment_requires_key_and_existing_candidates(tmp_path):
    bridge, engine, store, image, entries, _ = make_bridge(tmp_path)
    bridge.settings.deepseek_api_key = ''
    assert 'error' in json.loads(bridge.assessRepair(json.dumps({'image_id': image['id']})))
    bridge.settings.deepseek_api_key = 'mock-bridge-key'
    # A confirmed region without any local candidate cannot be assessed.
    for _, job_id, _ in entries:
        store.update('job', job_id, candidates=[])
    result = json.loads(bridge.assessRepair(json.dumps({'image_id': image['id']})))
    assert 'error' in result and '修复候选' in result['error']
    engine.executor.shutdown(wait=False)
    bridge.aux_executor.shutdown(wait=False)
