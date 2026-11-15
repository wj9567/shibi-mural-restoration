"""Adapt the structured analysis to the workbench's existing normalized canvas."""
from math import ceil, floor

from PIL import Image

from .damage_assessment import recommendation_for
from .storage import new_id
from .vision_schema import CulturalContext, DefectCandidate, VisionAssessment, VisionImage


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
    aliases = {'flaking': 'paint_loss', 'plaster_loss': 'paint_loss'}
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
