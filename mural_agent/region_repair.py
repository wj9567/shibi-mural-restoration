"""Stage 4: turn confirmed defect masks into local repair candidates.

The bridge keeps the policy here so the engine stays a generic executor:

* geometric damage (crack, paint loss, plaster loss, flaking, or a user-drawn
  region) prefers the local LaMa model and falls back to Telea only when the
  checkpoint is missing;
* fading/stain are observations. They keep a colour-matching or manual path and
  are never pushed through hole filling;
* every candidate is a single-region job, so the result, the region crop and the
  difference image stay attributable to one mask and one model call.
"""
from . import workspace
from .imaging import mask_array

INPAINT_TYPES = {'crack', 'paint_loss', 'plaster_loss', 'flaking', 'manual'}
TYPE_LABELS = {'crack': '裂隙', 'paint_loss': '颜料层缺失', 'plaster_loss': '地仗/灰泥缺失',
               'flaking': '起甲/剥落', 'fading': '褪色', 'stain': '污染',
               'manual': '手动区域', 'other': '其他疑似缺损'}
LAMA_REASON = '几何缺损适合图像补全，优先使用本地 LaMa 纹理补全。'
FALLBACK_REASON = '本地 LaMa 未安装，回退 OpenCV 保守补全；结果同样需要人工复核。'
NON_INPAINT_REASON = '该类型是表面观察，不做自动补洞；保留分析与人工处理入口。'


def region_policy(engine, proposal):
    """Return ``(backend, preferred, reason)`` for one defect record."""
    metadata = proposal.get('metadata') or {}
    kind = str(metadata.get('type') or 'other')
    availability = engine.tools.availability()
    recommendation = str((metadata.get('recommendation') or {}).get('backend') or 'telea')
    if kind in INPAINT_TYPES:
        if availability.get('lama'):
            return 'lama', 'lama', LAMA_REASON
        return 'telea', 'lama', FALLBACK_REASON
    if recommendation == 'lama':
        # Never route an observation type through hole filling just because a
        # legacy recommendation said LaMa.
        recommendation = 'color' if kind in {'fading', 'stain'} else 'telea'
    return recommendation, 'analysis_only', NON_INPAINT_REASON


def validated_region(store, image, proposal):
    """Return ``(mask_array, metadata)`` or raise a user-facing ValueError."""
    if not proposal:
        raise ValueError('缺损记录不存在。')
    if proposal.get('image_id') != image['id']:
        raise ValueError('缺损项不属于当前图像。')
    if not proposal.get('confirmed'):
        raise ValueError('缺损项尚未人工确认，请先在缺损清单中确认。')
    mask_path = store.folder(proposal['id']) / 'mask.png'
    if not mask_path.is_file():
        raise ValueError('该缺损项还没有掩膜，请先手动圈选或修正掩膜。')
    mask = mask_array(mask_path)
    if mask.shape != (image['height'], image['width']):
        raise ValueError('掩膜尺寸与原图不一致，请重新保存掩膜。')
    if not (mask > 0).any():
        raise ValueError('掩膜为空，请先圈选实际缺损像素。')
    return mask, proposal.get('metadata') or {}


def tight_box(mask, metadata):
    """Tight candidate box in working pixels; the engine adds crop context."""
    height, width = mask.shape[:2]
    bbox = metadata.get('bbox_pixels') or []
    try:
        x0, y0, x1, y1 = [int(value) for value in bbox]
    except (TypeError, ValueError):
        x0 = y0 = x1 = y1 = 0
    if x1 <= x0 or y1 <= y0:
        # Fall back to the mask extent when a record has no usable box.
        import numpy as np
        ys, xs = np.where(mask > 0)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    box = [max(0, x0), max(0, y0), min(width, x1), min(height, y1)]
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    return box


def run_region_candidates(store, engine, image, proposal_ids, instructions=''):
    """Create one conservative repair job per confirmed defect and return them."""
    requested = list(dict.fromkeys(proposal_ids or []))
    if not requested:
        raise ValueError('没有指定要生成修复候选的缺损项。')
    jobs, skipped = [], []
    for proposal_id in requested:
        proposal = store.get('proposal', proposal_id)
        try:
            mask, metadata = validated_region(store, image, proposal)
        except ValueError as exc:
            skipped.append({'proposal_id': proposal_id, 'reason': str(exc)})
            continue
        kind = str(metadata.get('type') or 'other')
        backend, preferred, reason = region_policy(engine, proposal)
        box = tight_box(mask, metadata)
        mask_record = workspace.add_mask(
            store, image, (store.folder(proposal_id) / 'mask.png').read_bytes(),
            [proposal_id], mask_source=proposal.get('mask_source') or 'ai_candidate',
            user_confirmed=True, region_class='fillable_lacuna', risk_regions=[])
        label = TYPE_LABELS.get(kind, kind)
        rationale = (f'缺损清单中的{label}区域：{reason} '
                     f'掩膜占比 {mask_record["fraction"]:.2%}；结果须人工复核。')
        job = workspace.create_job(
            store, engine, image, mask_record, backend, 'rules', 1,
            instructions or f'仅修复该缺损区域（{label}），保持掩膜外像素不变。',
            mode='conservative', budget={'candidates': 1, 'revisions': 0, 'timeout_seconds': 300},
            workflow='professional_step', rationale=rationale,
            evidence_level='conservative', mask_source=proposal.get('mask_source') or 'ai_candidate',
            mask_gate_confirmed=True, extra={
                'region_repair': True,
                'defect_proposal_id': proposal_id,
                'defect_type': kind,
                'defect_bbox': box,
                'defect_bbox_pixels': list(metadata.get('bbox_pixels') or []),
                'region_backend': backend,
                'region_preferred': preferred,
                'region_reason': reason,
                'repair_model_plan': {
                    'preferred': preferred, 'backend': backend, 'reason': reason,
                    'mask_sha256': mask_record['sha256'],
                    'mask_fraction': mask_record['fraction'],
                },
            })
        jobs.append(job)
    return {'jobs': jobs, 'skipped': skipped}
