"""Stage-7 PDF outputs: page counts, Chinese text, provenance, truncation, demo status."""
from io import BytesIO
import json

import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader

from mural_agent import competition_output, workspace
from mural_agent.config import ROOT, Settings
from mural_agent.desktop_web import Bridge
from mural_agent.engine import Engine
from mural_agent.storage import Store, new_id, timestamp


def make_case(tmp_path, evidence=None):
    store = Store(tmp_path / 'data')
    payload = (ROOT / '图片' / '01-cma-1963.252.jpg').read_bytes()
    image = workspace.add_image(store, payload, '01-cma-1963.252.jpg')
    width, height = image['width'], image['height']
    proposal_ids = []
    for index, kind in enumerate(('paint_loss', 'crack', 'flaking', 'fading', 'stain')):
        proposal_id, job_id, candidate_id = new_id(), new_id(), new_id()
        mask = np.zeros((height, width), np.uint8)
        x0 = 10 + index * 25
        mask[80:300, x0:x0 + 20] = 255
        Image.fromarray(mask).save(store.folder(proposal_id) / 'mask.png')
        text = evidence if evidence is not None and index == 0 else f'{kind} 在左上边缘可见连续缺失与色差。'
        store.put('proposal', {
            'id': proposal_id, 'image_id': image['id'], 'created_at': f'2026-11-16T00:00:0{index}+00:00',
            'fraction': float((mask > 0).mean()), 'confirmed': True, 'mask_source': 'manual',
            'region_class': 'fillable_lacuna', 'revision': 1,
            'metadata': {'type': kind, 'bbox_pixels': [x0, 80, x0 + 20, 300], 'evidence': text,
                         'mask_status': 'sam' if index < 3 else 'observation',
                         'model_source': 'Qwen3-VL local',
                         'recommendation': {'backend': 'lama', 'method': 'LaMa 纹理补全'}},
        })
        original = np.asarray(Image.open(store.folder(image['id']) / 'image.png').convert('RGB'))
        result = original.copy()
        result[mask > 0] = [200, 160, 130]
        Image.fromarray(result).save(store.folder(candidate_id) / 'result.png')
        Image.fromarray(np.zeros_like(original)).save(store.folder(candidate_id) / 'difference.png')
        Image.fromarray(result[80:300, x0:x0 + 20]).save(store.folder(candidate_id) / 'region.png')
        if index == 0:
            store.put('job', {
                'id': job_id, 'image_id': image['id'], 'mask_id': new_id(), 'status': 'review',
                'defect_proposal_id': proposal_id, 'defect_bbox': [x0, 80, x0 + 20, 300],
                'workflow': 'professional_step', 'selected_id': candidate_id, 'selected_as': 'accepted',
                'review': {'decision': 'adopt', 'note': '边缘补全可接受', 'archive_as_hypothesis': False},
                'candidates': [{'id': candidate_id, 'tool': 'lama',
                                'parameters': {'tool': 'lama', 'radius': 3}, 'elapsed_seconds': 8.46,
                                'versions': {'device': 'cpu', 'weight_sha256': 'f' * 64},
                                'metrics': {'masked_fraction': float((mask > 0).mean())},
                                'checks': {'size_match': True, 'outside_unchanged': True},
                                'review': {'status': 'accepted', 'decision': 'adopt'}}]})
            store.put('assessment', {
                'id': new_id(), 'image_id': image['id'], 'created_at': timestamp(),
                'model_source': 'deepseek-flash', 'advisory': True, 'region_ids': [proposal_id],
                'limitations': '只能目视整图。',
                'regions': [{'index': 1, 'region_id': proposal_id, 'risk_level': 'medium',
                             'human_review_required': True,
                             'review_reason': '补全涉及边饰结构推测，需人工核对。',
                             'repair_suggestion': '建议降低补全强度并保留边缘纹理。'}]})
            store.put('review_event', {
                'id': new_id(), 'image_id': image['id'], 'proposal_id': proposal_id,
                'action': 'candidate_accept', 'created_at': timestamp(), 'changes': {},
                'decision': 'adopt', 'note': '边缘补全可接受'})
            proposal_ids.append((proposal_id, job_id, candidate_id))
        else:
            store.put('job', {'id': job_id, 'image_id': image['id'], 'mask_id': new_id(),
                              'status': 'review', 'defect_proposal_id': proposal_id,
                              'candidates': [{'id': candidate_id, 'tool': 'telea',
                                              'parameters': {'tool': 'telea', 'radius': 3},
                                              'metrics': {'masked_fraction': float((mask > 0).mean())}}]})
    return store, image, proposal_ids[0]


def pdf_text(path):
    reader = PdfReader(str(path))
    return reader, ''.join(page.extract_text() or '' for page in reader.pages)


def test_observation_card_is_one_readable_page_with_provenance(tmp_path):
    store, image, (proposal_id, job_id, candidate_id) = make_case(tmp_path)
    output = tmp_path / 'card.pdf'
    result = competition_output.build_observation_card(store, image, output)
    assert result['pages'] == 1 and result['regions'] == 3
    reader, text = pdf_text(output)
    assert len(reader.pages) == 1
    for token in ('AI 壁画观察卡', '重点标注', '颜料层缺失', '可见依据', '保护与数字展示建议',
                  'AI 使用情况', 'Qwen3-VL', 'SAM 2.1', 'LaMa', 'DeepSeek', '不确定性与人工复核',
                  'CC0', 'clevelandart.org', '人工复核'):
        assert token in text, token


def test_card_truncates_long_text_without_overflow(tmp_path):
    store, image, _ = make_case(tmp_path, evidence='边缘缺失与色差。' * 400)
    output = tmp_path / 'card-long.pdf'
    result = competition_output.build_observation_card(store, image, output)
    reader, text = pdf_text(output)
    assert len(reader.pages) == 1
    assert result['truncated'] is True
    assert '…' in text


def test_repair_report_is_two_pages_and_records_review(tmp_path):
    store, image, (proposal_id, job_id, candidate_id) = make_case(tmp_path)
    output = tmp_path / 'report.pdf'
    result = competition_output.build_repair_report(store, image, store.get('job', job_id), output)
    assert result['pages'] == 2
    reader, text = pdf_text(output)
    assert len(reader.pages) == 2
    for token in ('修复成果说明', '修复前原图', '本地修复结果', '差异图', '修复方式',
                  'LaMa 纹理补全', '输出检查', '掩膜外像素不变', '人工审核记录',
                  'candidate_accept', 'deepseek-flash', 'CC0'):
        assert token in text, token


def test_repair_report_requires_candidate(tmp_path):
    store, image, _ = make_case(tmp_path)
    empty_job = {'id': new_id(), 'image_id': image['id'], 'candidates': []}
    with pytest.raises(ValueError):
        competition_output.build_repair_report(store, image, empty_job, tmp_path / 'x.pdf')


def test_bridge_exports_with_explicit_path_and_reports_demo_progress(tmp_path):
    store, image, (proposal_id, job_id, candidate_id) = make_case(tmp_path)
    settings = Settings(data_dir=store.root, lama_model=tmp_path / 'missing.pt')
    engine = Engine(settings, store)
    bridge = Bridge(settings, store, engine)
    card = json.loads(bridge.exportObservationCard(json.dumps({
        'image_id': image['id'], 'path': str(tmp_path / 'bridge-card.pdf')})))
    assert card['saved'] is True and card['pages'] == 1
    report = json.loads(bridge.exportRepairReport(json.dumps({
        'job_id': job_id, 'path': str(tmp_path / 'bridge-report.pdf')})))
    assert report['saved'] is True and report['pages'] == 2
    assert len(PdfReader(str(tmp_path / 'bridge-card.pdf')).pages) == 1
    assert len(PdfReader(str(tmp_path / 'bridge-report.pdf')).pages) == 2

    status = json.loads(bridge.demoStatus(image['id']))
    steps = {step['key']: step for step in status['steps']}
    assert steps['import']['done'] is True
    assert steps['identify']['done'] is True
    assert steps['repair']['done'] is True
    assert steps['assess']['done'] is True
    assert steps['export']['done'] is False
    assert status['total'] == 7
    engine.executor.shutdown(wait=False)
    bridge.aux_executor.shutdown(wait=False)
