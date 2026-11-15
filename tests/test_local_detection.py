"""Exercise local detection and mask validation without installing model weights."""
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mural_agent import vision_analysis, workspace
from mural_agent.damage_assessment import parse_model_output
from mural_agent.storage import Store


@pytest.fixture
def mural(tmp_path):
    store = Store(tmp_path / 'data')
    source = BytesIO()
    Image.new('RGB', (1501, 941), '#b7957a').save(source, 'PNG')
    record = workspace.add_image(store, source.getvalue(), 'mural.png')
    with Image.open(store.folder(record['id']) / 'image.png') as preview:
        preview.thumbnail((1024, 1024))
        preview.save(store.folder(record['id']) / 'damage-assessment-input.jpg')
    return store, record


def raw_item(kind, bbox=(101, 181, 241, 341)):
    return {'type': kind, 'bbox': list(bbox), 'size': 'medium',
            'confidence': 0.92, 'evidence': '图像上可观察到色块与边界变化。'}


def test_local_detection_reuses_full_work_shape_and_distinguishes_damage_types(mural, monkeypatch):
    store, record = mural
    expected_types = ['crack', 'flaking', 'paint_loss', 'stain', 'fading']
    boxes = [[101 + 150 * index, 181, 241 + 150 * index, 341] for index in range(5)]
    monkeypatch.setattr(vision_analysis, 'assess_image', lambda engine, image_record: {
        'items': [raw_item(kind, bbox) for kind, bbox in zip(expected_types, boxes)],
    })
    prompted = []

    def predict(work_image, *, box):
        assert work_image.shape == (941, 1501, 3)
        prompted.append(list(box))
        x0, y0, x1, y1 = box
        mask = np.zeros(work_image.shape[:2], np.uint8)
        mask[y0 + 10:y1 - 15, x0 + 20:x1 - 25] = 255
        return mask, {'predicted_iou': 0.95}

    engine = SimpleNamespace(sam=SimpleNamespace(available=lambda: True, predict=predict))
    assessment, candidates = vision_analysis.detect_local(store, engine, record)

    assert assessment.source == 'local'
    assert assessment.model_source == 'Qwen3-VL local'
    assert (assessment.image.original_width, assessment.image.original_height) == (1501, 941)
    assert (assessment.image.analysis_width, assessment.image.analysis_height) == (1024, 642)
    assert assessment.image.scale_x == pytest.approx(1024 / 1501)
    expected_bbox = [151, 170, 362, 321]
    assert assessment.items[0].bbox == expected_bbox
    assert prompted[0] == expected_bbox
    assert len(prompted) == 3
    assert [entry['candidate'].type for entry in candidates] == expected_types
    assert [entry['mask_status'] for entry in candidates] == ['sam', 'sam', 'sam', 'observation', 'observation']
    assert candidates[0]['candidate'].evidence == raw_item('crack')['evidence']
    pixels = candidates[0]['mask']
    assert pixels.shape == (941, 1501) and pixels.dtype == np.uint8
    assert set(np.unique(pixels)) == {0, 255}
    assert pixels[180, 171] == 255
    assert pixels[179, 171] == 0
    assert pixels[180, 170] == 0
    assert all(entry['mask'] is None and entry['mask_notice'] for entry in candidates[3:])
    assert not store.list('proposal'), 'generation and persistence are separate operations'


@pytest.mark.parametrize('failure', ['missing', 'prediction_error', 'wrong_shape', 'empty', 'leak', 'low_score'])
def test_unreliable_sam_leaves_only_a_candidate_box(mural, monkeypatch, failure):
    store, record = mural
    monkeypatch.setattr(vision_analysis, 'assess_image', lambda engine, image_record: {
        'items': [raw_item('paint_loss')],
    })

    def predict(work_image, *, box):
        if failure == 'missing':
            pytest.fail('SAM predict must not run when weights are missing')
        if failure == 'prediction_error':
            raise RuntimeError('SAM could not load its checkpoint')
        x0, y0, _, _ = box
        mask = np.zeros(work_image.shape[:2], np.uint8)
        if failure != 'empty':
            mask[y0 + 10:y0 + 50, x0 + 20:x0 + 80] = 255
        if failure == 'leak':
            mask[600:700, 800:1000] = 255
        if failure == 'wrong_shape':
            mask = mask.T
        return mask, {'predicted_iou': 0.1 if failure == 'low_score' else 0.95}

    engine = SimpleNamespace(sam=SimpleNamespace(available=lambda: failure != 'missing', predict=predict))
    assessment, candidates = vision_analysis.detect_local(store, engine, record)

    assert len(assessment.items) == len(candidates) == 1
    assert candidates[0]['candidate'].bbox == [151, 170, 362, 321]
    assert candidates[0]['mask_status'] == 'needs_review'
    assert candidates[0]['mask'] is None and candidates[0]['mask_notice']


def test_no_qwen_candidates_is_a_valid_empty_result(mural, monkeypatch):
    store, record = mural
    monkeypatch.setattr(vision_analysis, 'assess_image', lambda engine, image_record: {'items': []})

    def predict(*args, **kwargs):
        pytest.fail('No boxes must not yield an invocation of SAM')

    engine = SimpleNamespace(sam=SimpleNamespace(available=lambda: True, predict=predict))
    assessment, candidates = vision_analysis.detect_local(store, engine, record)

    assert assessment.items == candidates == []


def test_qwen_parser_keeps_fading_and_flaking_separate_from_paint_loss():
    parsed = parse_model_output('''{"items": [
        {"type": "fading", "bbox": [40, 60, 120, 150]},
        {"type": "flaking", "bbox": [240, 260, 320, 350]},
        {"type": "paint_loss", "bbox": [440, 460, 520, 550]}
    ]}''')
    assert [item['type'] for item in parsed['items']] == ['fading', 'flaking', 'paint_loss']


def test_percent_boxes_are_scaled_only_when_requested():
    """The pinned 2B Qwen answers in 0..100 percent; callers opt into that unit."""
    text = '{"items": [{"type": "crack", "bbox": [10, 20, 100, 101]}]}'
    assert parse_model_output(text)['items'][0]['bbox'] == [10, 20, 100, 101]
    scaled = parse_model_output(text, bbox_units='auto')['items'][0]['bbox']
    assert scaled == [100, 200, 1000, 1000]


def test_coarse_box_skips_sam_and_asks_for_manual_selection(mural, monkeypatch):
    store, record = mural
    monkeypatch.setattr(vision_analysis, 'assess_image', lambda engine, image_record: {
        'items': [raw_item('crack', (0, 0, 1000, 1000)), raw_item('fading', (10, 10, 60, 60))],
    })

    def predict(*args, **kwargs):
        pytest.fail('SAM must not run for a box covering most of the picture')

    engine = SimpleNamespace(sam=SimpleNamespace(available=lambda: True, predict=predict))
    assessment, candidates = vision_analysis.detect_local(store, engine, record)

    assert len(assessment.items) == len(candidates) == 2
    assert [entry['mask_status'] for entry in candidates] == ['needs_review', 'observation']
    assert candidates[0]['mask'] is None
    assert candidates[0]['mask_notice'] == vision_analysis.COARSE_NOTICE


def test_duplicated_boxes_skip_sam(mural, monkeypatch):
    store, record = mural
    monkeypatch.setattr(vision_analysis, 'assess_image', lambda engine, image_record: {
        'items': [raw_item('crack'), raw_item('paint_loss')],
    })

    def predict(*args, **kwargs):
        pytest.fail('SAM must not run when several candidates share one box')

    engine = SimpleNamespace(sam=SimpleNamespace(available=lambda: True, predict=predict))
    _, candidates = vision_analysis.detect_local(store, engine, record)

    assert [entry['mask_status'] for entry in candidates] == ['needs_review', 'needs_review']
    assert all(entry['mask_notice'] == vision_analysis.DUPLICATE_NOTICE for entry in candidates)


def test_mask_extending_far_beyond_box_is_rejected(mural, monkeypatch):
    """A thin line running far outside the box means SAM grabbed an edge, not the lesion."""
    store, record = mural
    monkeypatch.setattr(vision_analysis, 'assess_image', lambda engine, image_record: {
        'items': [raw_item('crack', (101, 181, 241, 341))],
    })

    def predict(work_image, *, box):
        x0, y0, x1, y1 = box
        mask = np.zeros(work_image.shape[:2], np.uint8)
        mask[y0 + 20:y1 - 15, x0:x1 - 25] = 255
        mask[y0 + 40:y0 + 42, x0:x1 + 200] = 255
        return mask, {'predicted_iou': 0.95}

    engine = SimpleNamespace(sam=SimpleNamespace(available=lambda: True, predict=predict))
    _, candidates = vision_analysis.detect_local(store, engine, record)
    assert candidates[0]['mask_status'] == 'needs_review'
    assert candidates[0]['mask'] is None
    assert '远超候选框' in candidates[0]['mask_notice']


def test_small_mask_overshoot_is_tolerated_but_object_leak_is_not(mural, monkeypatch):
    store, record = mural
    monkeypatch.setattr(vision_analysis, 'assess_image', lambda engine, image_record: {
        'items': [raw_item('paint_loss', (101, 181, 241, 341))],
    })

    def predict(work_image, *, box):
        x0, y0, x1, y1 = box
        mask = np.zeros(work_image.shape[:2], np.uint8)
        # 8 px of organic overshoot on the left is acceptable.
        mask[y0 + 10:y1 - 15, x0 - 8:x1 - 25] = 255
        return mask, {'predicted_iou': 0.9}

    engine = SimpleNamespace(sam=SimpleNamespace(available=lambda: True, predict=predict))
    _, candidates = vision_analysis.detect_local(store, engine, record)
    assert candidates[0]['mask_status'] == 'sam'
    assert candidates[0]['mask'] is not None
