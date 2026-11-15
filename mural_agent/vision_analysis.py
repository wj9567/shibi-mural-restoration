"""Adapt structured vision output to the workbench's working-image canvas.

Two paths share one defect vocabulary:

* ``cloud_defects``     — DeepSeek boxes, always requiring a hand-drawn mask.
* ``detect_local``      — local Qwen3-VL boxes, then SAM 2.1 masks for the
  geometric damage types that a hole-filling repair can use.

A candidate box is never treated as a precise lesion: only a SAM mask that
passed the size, confidence and containment checks is stored as ``sam``.
Everything else keeps the box and asks the user to correct the mask.
"""
from math import ceil, floor

import numpy as np
from PIL import Image

from .damage_assessment import assess_image, recommendation_for
from .storage import new_id
from .vision_schema import CulturalContext, DefectCandidate, VisionAssessment, VisionImage

MODEL_SOURCE = 'Qwen3-VL local'
# Damage with a visible geometry that an inpainting mask can describe. Fading
# and staining are observations, not holes; they must not become LaMa fills by
# default.
SAM_TYPES = frozenset({'crack', 'flaking', 'paint_loss'})
TYPE_ALIASES = {'plaster_loss': 'paint_loss'}
CANDIDATE_TYPES = frozenset({'crack', 'paint_loss', 'fading', 'stain', 'flaking', 'other'})
SAM_MIN_IOU = .5
SAM_BOX_PADDING = .05
SAM_LEAK_RATIO = .25
# SAM may return the object that contains the box; a mask whose extent runs far
# past the candidate box is an object/border grab, not the candidate's damage.
SAM_EXTENT_PADDING = .15
# A box covering most of the picture cannot localize a lesion; SAM on it would
# either return the whole image or an unrelated object.
SAM_COARSE_FRACTION = .5
MAX_CANDIDATES = 8
OBSERVATION_NOTICE = '该类型为表面观察（褪色/污染），不默认补洞；请人工决定是否做局部色调匹配。'
COARSE_NOTICE = '候选框覆盖超过一半画面，SAM 无法给出局部掩膜；请手动框选实际缺损范围。'
DUPLICATE_NOTICE = '模型对多个候选给出相同定位框，定位不可信；请手动框选实际缺损范围。'


def _analysis_size(store, image_record):
    """The size of the copy Qwen actually looked at, for provenance only."""
    preview_path = store.folder(image_record['id']) / 'damage-assessment-input.jpg'
    if preview_path.is_file():
        with Image.open(preview_path) as preview:
            return preview.size
    return (image_record['width'], image_record['height'])


def normalized_box(item, width, height):
    """Map a model box (0..1000 of the image) to half-open working-image pixels.

    Corners are rounded outward so a thin crack is not clipped by rounding, and
    the result is clipped to the image so a bad model box cannot escape it.
    """
    try:
        x0, y0, x1, y1 = (max(0, min(1000, int(value))) for value in item['bbox'])
    except (KeyError, TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    box = [x0 * width // 1000, y0 * height // 1000,
           min(width, (x1 * width + 999) // 1000), min(height, (y1 * height + 999) // 1000)]
    # A model box smaller than one output pixel still describes a real spot.
    box[2] = min(width, max(box[2], box[0] + 1))
    box[3] = min(height, max(box[3], box[1] + 1))
    return box


def pixels_to_normalized(bbox, width, height):
    """Inverse of ``normalized_box`` for the 0..1000 boxes the canvas uses."""
    x0, y0, x1, y1 = bbox
    return [x0 * 1000 // width, y0 * 1000 // height,
            min(1000, (x1 * 1000 + width - 1) // width),
            min(1000, (y1 * 1000 + height - 1) // height)]


def _mask_leaks(binary, box, shape):
    """True when most mask pixels fall well outside their own candidate box.

    SAM often returns the whole object that contains the prompt box, so a small
    organic overshoot is tolerated; an object-sized or distant region is not.
    """
    height, width = shape
    x0, y0, x1, y1 = box
    pad = max(4, round(max(x1 - x0, y1 - y0) * SAM_BOX_PADDING))
    allowed = np.zeros_like(binary)
    allowed[max(0, y0-pad):min(height, y1+pad), max(0, x0-pad):min(width, x1+pad)] = True
    total = int(binary.sum())
    outside = int((binary & ~allowed).sum())
    return total > 0 and outside / total > SAM_LEAK_RATIO


def _mask_extent_leaks(binary, box, shape):
    """True when the mask reaches far past the box in any direction."""
    height, width = shape
    x0, y0, x1, y1 = box
    pad = max(4, round(max(x1 - x0, y1 - y0) * SAM_EXTENT_PADDING))
    ys, xs = np.where(binary)
    return bool(xs.min() < x0 - pad or xs.max() > x1 + pad
                or ys.min() < y0 - pad or ys.max() > y1 + pad)


def _sam_mask(engine, work_image, box):
    """Return ``(mask, status, notice)``; only a verified mask is ``sam``."""
    sam = getattr(engine, 'sam', None)
    if sam is None or not sam.available():
        return None, 'needs_review', 'SAM 2.1 权重未安装，仅保留候选框；请手动修正掩膜。'
    try:
        mask, metadata = sam.predict(work_image, box=list(box))
    except Exception as exc:  # noqa: BLE001 — a failed split is a normal fallback
        return None, 'needs_review', f'SAM 分割未完成（{str(exc)[:120]}），仅保留候选框；请手动修正。'
    if mask is None:
        return None, 'needs_review', 'SAM 没有返回掩膜，仅保留候选框；请手动修正。'
    array = np.asarray(mask)
    if array.shape != work_image.shape[:2]:
        return None, 'needs_review', 'SAM 返回的掩膜尺寸与原图不一致，已丢弃；请手动修正。'
    binary = array > 127
    if not binary.any():
        return None, 'needs_review', 'SAM 未分割出有效像素，仅保留候选框；请手动修正。'
    score = float((metadata or {}).get('predicted_iou') or 0)
    if score < SAM_MIN_IOU:
        return None, 'needs_review', f'SAM 置信分偏低（{score:.2f}），仅保留候选框；请手动修正。'
    if _mask_leaks(binary, box, work_image.shape[:2]):
        return None, 'needs_review', 'SAM 掩膜明显超出候选框，已丢弃；请手动修正。'
    if _mask_extent_leaks(binary, box, work_image.shape[:2]):
        return None, 'needs_review', 'SAM 掩膜范围远超候选框（可能选中了整条边线），已丢弃；请手动修正。'
    return binary.astype(np.uint8) * 255, 'sam', ''


def _cultural_context(result):
    context = result.get('cultural_context')
    if not isinstance(context, dict):
        return None
    if context.get('confidence') not in {'low', 'medium', 'high'}:
        context = {**context, 'confidence': ''}
    try:
        return CulturalContext.model_validate(context)
    except Exception:  # noqa: BLE001 — cultural notes are optional, never fatal
        return None


def detect_local(store, engine, image_record):
    """Run Qwen candidate detection and optional SAM masks in working pixels.

    Returns ``(assessment, entries)``. ``entries`` keeps the raw mask and status
    so the bridge can persist proposals without re-running either model. No
    record is written here: generation and persistence stay separate steps.
    """
    result = assess_image(engine, image_record)
    width, height = image_record['width'], image_record['height']
    image = VisionImage.from_sizes((width, height), _analysis_size(store, image_record))
    work_image = None
    raw_items = [raw for raw in result.get('items', [])[:MAX_CANDIDATES] if isinstance(raw, dict)]
    box_counts = {}
    for raw in raw_items:
        box = normalized_box(raw, width, height)
        if box is not None:
            key = tuple(box)
            box_counts[key] = box_counts.get(key, 0) + 1
    items, entries = [], []
    for raw in raw_items:
        box = normalized_box(raw, width, height)
        if box is None:
            continue
        raw_kind = str(raw.get('type', 'other')).strip().lower()
        kind = TYPE_ALIASES.get(raw_kind, raw_kind)
        kind = kind if kind in CANDIDATE_TYPES else 'other'
        evidence = (raw.get('evidence') or '').strip() or '模型未提供具体视觉依据。'
        recommendation = recommendation_for({**raw, 'type': kind})
        candidate = DefectCandidate(
            candidate_id=new_id(),
            bbox=box,
            type=kind,
            observation=evidence,
            evidence=evidence,
            repair_suggestion=recommendation['method'],
            uncertainty='本地视觉候选，需人工复核；掩膜只是分割候选。',
            risk_level='unknown',
            model_source=MODEL_SOURCE,
        )
        items.append(candidate)
        area_fraction = (box[2] - box[0]) * (box[3] - box[1]) / max(1, width * height)
        if candidate.type in SAM_TYPES:
            if box_counts.get(tuple(box), 0) > 1:
                mask, status, notice = None, 'needs_review', DUPLICATE_NOTICE
            elif area_fraction > SAM_COARSE_FRACTION:
                mask, status, notice = None, 'needs_review', COARSE_NOTICE
            else:
                if work_image is None:
                    from .imaging import image_array
                    work_image = image_array(store.folder(image_record['id']) / 'image.png')
                mask, status, notice = _sam_mask(engine, work_image, box)
        else:
            mask, status, notice = None, 'observation', OBSERVATION_NOTICE
        entries.append({'candidate': candidate, 'mask': mask, 'mask_status': status,
                        'mask_notice': notice, 'size': str(raw.get('size') or 'uncertain')[:20],
                        'confidence': raw.get('confidence'),
                        'recommendation': recommendation})
    limitations = (result.get('limitations') or '').strip() or '本地视觉候选需人工核对；掩膜不是修复定稿。'
    return VisionAssessment(source='local', model_source=MODEL_SOURCE, image=image, items=items,
                            limitations=limitations, cultural_context=_cultural_context(result)), entries


def cloud_defects(assessment: VisionAssessment) -> list[dict]:
    width, height = assessment.image.original_width, assessment.image.original_height
    defects = []
    for candidate in assessment.items:
        x0, y0, x1, y1 = candidate.bbox
        defect = candidate.model_dump(mode='json')
        defect.update(
            bbox=[floor(x0 * 1000 / width), floor(y0 * 1000 / height),
                  ceil(x1 * 1000 / width), ceil(y1 * 1000 / height)],
            bbox_pixels=list(candidate.bbox),
            recommendation=recommendation_for(defect),
            requires_manual_selection=True,
            confirmed=False,
        )
        defects.append(defect)
    return defects


def local_assessment(store, image_record, result) -> VisionAssessment:
    """Keep existing Qwen masks and routing while returning the common candidate fields."""
    width, height = image_record['width'], image_record['height']
    preview_path = store.folder(image_record['id']) / 'damage-assessment-input.jpg'
    analysis_size = (width, height)
    if preview_path.is_file():
        with Image.open(preview_path) as preview:
            analysis_size = preview.size
    image = VisionImage.from_sizes((width, height), analysis_size)
    model_source = 'Qwen3-VL local'
    aliases = {'plaster_loss': 'paint_loss'}
    risk_levels = {'fillable_lacuna': 'low', 'uncertain': 'medium', 'unrestorable': 'high'}
    items = []
    for defect in result.get('defects', []):
        evidence = (defect.get('evidence') or '').strip() or '模型未提供具体视觉依据。'
        x0, y0, x1, y1 = defect['bbox']
        candidate = DefectCandidate(
            candidate_id=defect.get('proposal_id') or new_id(),
            bbox=[floor(x0 * width / 1000), floor(y0 * height / 1000),
                  ceil(x1 * width / 1000), ceil(y1 * height / 1000)],
            type=aliases.get(defect['type'], defect['type']),
            observation=evidence,
            evidence=evidence,
            repair_suggestion=(defect.get('recommendation') or {}).get('method') or '请人工核对后选择修补方法。',
            uncertainty=defect.get('risk_notice') or '本地视觉候选，需人工复核。',
            risk_level=risk_levels.get(defect.get('region_class'), 'unknown') if defect.get('proposal_id') else 'unknown',
            model_source=model_source,
        )
        items.append(candidate)
        fields = candidate.model_dump(mode='json', exclude={'bbox', 'type'})
        defect.update(fields, bbox_pixels=list(candidate.bbox))
    context = result.get('cultural_context')
    if context and context.get('confidence') not in {'low', 'medium', 'high'}:
        context = {**context, 'confidence': ''}
    return VisionAssessment(source='local', model_source=model_source, image=image, items=items,
                            limitations=result.get('limitations') or '本地视觉候选需人工核对。',
                            cultural_context=CulturalContext.model_validate(context) if context else None)
