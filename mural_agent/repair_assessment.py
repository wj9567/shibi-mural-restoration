"""Stage 5: assemble DeepSeek repair-review inputs and merge deterministic rules.

The language model may only return advisory text keyed by existing local region
ids. Everything that gates the UI (which region needs human review and why) is
also computed here from local, inspectable rules: face overlap, repaired area,
speculative tool, model risk/basis quality and user-declared sensitive content.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .imaging import image_array, mask_array
from .storage import new_id, timestamp

MAX_REGIONS = 8
MAX_CROPS = 3
LARGE_AREA_FRACTION = .12
SPECULATIVE_TOOLS = {'diffusion', 'imagine'}
TYPE_LABELS = {'crack': '裂隙', 'paint_loss': '颜料层缺失', 'plaster_loss': '地仗/灰泥缺失',
               'flaking': '起甲/剥落', 'fading': '褪色', 'stain': '污染',
               'manual': '手动区域', 'other': '其他疑似缺损'}
# Content the user explicitly mentions in their background note. Local code
# cannot read "this motif is religious" from pixels, so a declared sensitive
# reference forces human review instead of trusting the model's risk label.
SENSITIVE_CONTEXT_WORDS = ('人物', '面部', '脸部', '眼睛', '纹样', '图案', '装饰', '文字', '题记',
                           '铭文', '经文', '佛', '菩萨', '宗教', '神像', '供养', '年代')


def type_label(kind):
    return TYPE_LABELS.get(str(kind or 'other'), str(kind or '其他疑似缺损'))


def bbox_to_1000(bbox, width, height):
    x0, y0, x1, y1 = [int(value) for value in bbox]
    return [x0 * 1000 // width, y0 * 1000 // height,
            min(1000, (x1 * 1000 + width - 1) // width),
            min(1000, (y1 * 1000 + height - 1) // height)]


def detect_faces(image):
    """Deterministic face boxes with OpenCV's bundled Haar cascade (may be empty)."""
    try:
        import cv2
        cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'))
        if cascade.empty():
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        scale = min(1., 720 / max(gray.shape[:2]))
        small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
        found = cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        return [[int(x / scale), int(y / scale), int((x + w) / scale), int((y + h) / scale)]
                for x, y, w, h in found]
    except Exception:  # noqa: BLE001 — face detection is a safety aid, never fatal
        return []


def face_overlap(faces, mask):
    for x0, y0, x1, y1 in faces:
        height, width = mask.shape[:2]
        x0, x1 = max(0, min(width, x0)), max(0, min(width, x1))
        y0, y1 = max(0, min(height, y0)), max(0, min(height, y1))
        if x0 < x1 and y0 < y1 and (mask[y0:y1, x0:x1] > 0).any():
            return True
    return False


def candidate_for(store, proposal_id, jobs=None):
    """Newest review candidate produced from this defect record, if any."""
    for job in (jobs if jobs is not None else store.list('job')):
        if job.get('defect_proposal_id') != proposal_id or job.get('status') != 'review':
            continue
        for candidate in job.get('candidates') or []:
            if (store.folder(candidate['id']) / 'result.png').is_file():
                return job, candidate
    return None, None


def crop_pair(store, image, job, candidate, region_mask):
    """Before/after crop around the repaired region, or None when unusable."""
    bbox = job.get('defect_bbox') or []
    try:
        x0, y0, x1, y1 = [int(value) for value in bbox]
    except (TypeError, ValueError):
        ys, xs = np.where(region_mask > 0)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    pad = max(12, round(max(x1 - x0, y1 - y0) * .15))
    width, height = image['width'], image['height']
    box = [max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)]
    if box[2] - box[0] < 48 or box[3] - box[1] < 48:
        return None
    original = np.asarray(Image.open(store.folder(image['id']) / 'image.png').convert('RGB'))
    result_path = store.folder(candidate['id']) / 'result.png'
    if not result_path.is_file():
        return None
    before = Image.fromarray(original[box[1]:box[3], box[0]:box[2]])
    after = Image.fromarray(np.asarray(Image.open(result_path).convert('RGB'))[box[1]:box[3], box[0]:box[2]])
    return before, after, box


def build_case(store, image, proposal, job=None, candidate=None):
    metadata = proposal.get('metadata') or {}
    mask = mask_array(store.folder(proposal['id']) / 'mask.png')
    if job is None and candidate is None:
        job, candidate = candidate_for(store, proposal['id'])
    if job is not None and not candidate:
        candidate = (job.get('candidates') or [None])[0]
    tool = (candidate or {}).get('tool') or ''
    parameters = (candidate or {}).get('parameters') or {}
    bbox = metadata.get('bbox_pixels') or []
    return {
        'region_id': proposal['id'],
        'proposal': proposal,
        'mask': mask,
        'job': job,
        'candidate': candidate,
        'tool': tool,
        'method': (metadata.get('recommendation') or {}).get('method') or tool or '未记录',
        'type': str(metadata.get('type') or 'other'),
        'evidence': metadata.get('evidence') or '',
        'mask_fraction': float((mask > 0).mean()),
        'bbox_pixels': [int(value) for value in bbox] if len(bbox) == 4 else [],
        'bbox_0_1000': bbox_to_1000(bbox, image['width'], image['height']) if len(bbox) == 4 else [],
        'speculative': bool((candidate or {}).get('speculative') or (job or {}).get('mode') == 'speculative'
                            or tool in SPECULATIVE_TOOLS),
        'parameters': parameters,
    }


def compose_repaired(store, image, cases):
    """Full repaired image built only from each region's own mask pixels."""
    original = image_array(store.folder(image['id']) / 'image.png')
    composite = original.copy()
    for case in cases:
        candidate = case['candidate']
        if not candidate:
            continue
        path = store.folder(candidate['id']) / 'result.png'
        if not path.is_file():
            continue
        repaired = np.asarray(Image.open(path).convert('RGB'))
        if repaired.shape != composite.shape:
            continue
        active = case['mask'] > 0
        composite[active] = repaired[active]
    return original, composite


def local_flags(case, faces, context, review):
    flags = []
    if face_overlap(faces, case['mask']):
        flags.append('修复区域与人物面部重叠，需专家核对')
    if case['mask_fraction'] > LARGE_AREA_FRACTION:
        flags.append(f'修复面积较大（{case["mask_fraction"]:.1%}），需专家核对')
    if case['speculative']:
        flags.append('推测性/生成式补全，只能按假设性建议复核')
    if review.risk_level == 'high':
        flags.append('模型判定为高风险')
    if review.risk_level == 'unknown':
        flags.append('模型无法判断风险')
    if len(review.basis.strip()) < 8:
        flags.append('模型判断依据不足')
    if context and any(word in context for word in SENSITIVE_CONTEXT_WORDS):
        flags.append('用户提供的背景涉及人物/纹样/文字/宗教内容，需人工核对')
    if not case['candidate']:
        flags.append('该区域还没有本地修复候选')
    return flags


def merge(assessment, cases, image_array_, context):
    """Attach local evidence and deterministic review flags to model answers."""
    review_by_id = {review.region_id: review for review in assessment.regions}
    faces = detect_faces(image_array_)
    rows = []
    for index, case in enumerate(cases, 1):
        review = review_by_id[case['region_id']]
        flags = local_flags(case, faces, context, review)
        reasons = []
        if review.human_review_required and review.review_reason.strip():
            reasons.append(review.review_reason.strip())
        reasons.extend(flag for flag in flags if flag not in reasons)
        rows.append({
            'index': index,
            'region_id': case['region_id'],
            'type': case['type'],
            'type_label': type_label(case['type']),
            'bbox_0_1000': case['bbox_0_1000'],
            'local_evidence': case['evidence'],
            'local_mask_fraction': round(case['mask_fraction'], 6),
            'candidate': {
                'tool': case['tool'],
                'method': case['method'],
                'parameters': case['parameters'],
            } if case['candidate'] else None,
            'observation': review.observation,
            'repair_suggestion': review.repair_suggestion,
            'repair_assessment': review.repair_assessment,
            'basis': review.basis,
            'risk_level': review.risk_level,
            'uncertainty': review.uncertainty,
            'model_review_required': review.human_review_required,
            'human_review_required': bool(review.human_review_required or flags),
            'review_reason': '；'.join(reasons),
            'local_flags': flags,
        })
    return rows


def assessment_record(store, image, job, cases, assessment, rows, context):
    identifier = new_id()
    candidate_id = None
    if job and job.get('candidates'):
        candidate_id = job['candidates'][0]['id']
    record = {
        'id': identifier, 'image_id': image['id'], 'created_at': timestamp(),
        'job_id': (job or {}).get('id'), 'candidate_id': candidate_id,
        'model_source': assessment.model_source, 'source': assessment.source,
        'advisory': True, 'context_used': bool(context.strip()),
        'limitations': assessment.limitations,
        'region_ids': [case['region_id'] for case in cases],
        'revisions': {case['region_id']: int((store.get('proposal', case['region_id']) or {}).get('revision') or 1)
                      for case in cases},
        'regions': rows,
    }
    store.put('assessment', record)
    if job:
        # Advisory only: never changes masks, pixels, candidate selection or review.
        store.update('job', job['id'], ai_review={'assessment_id': identifier,
                                                   'created_at': record['created_at'],
                                                   'model_source': assessment.model_source,
                                                   'advisory': True})
    return record


def collect_cases(store, image, focus_job_id=None):
    """Confirmed local defects of this image, newest repair candidate first."""
    jobs = store.list('job')
    proposals = [proposal for proposal in store.list('proposal')
                 if proposal.get('image_id') == image['id']
                 and proposal.get('confirmed')
                 and (proposal.get('metadata') or {}).get('type')]
    proposals.sort(key=lambda proposal: (proposal.get('created_at') or '', proposal['id']))
    if focus_job_id:
        focus = next((job for job in jobs if job['id'] == focus_job_id), None)
        focus_id = (focus or {}).get('defect_proposal_id')
        proposals.sort(key=lambda proposal: proposal['id'] != focus_id)
    cases = [build_case(store, image, proposal) for proposal in proposals[:MAX_REGIONS]]
    return [case for case in cases if (case['mask'] > 0).any()]
