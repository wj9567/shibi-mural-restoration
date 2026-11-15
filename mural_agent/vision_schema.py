"""Shared, provider-independent result of inspecting a mural working image."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StringConstraints, model_validator


DefectType = Literal['crack', 'paint_loss', 'fading', 'stain', 'flaking', 'other']
RiskLevel = Literal['low', 'medium', 'high', 'unknown']
Box = Annotated[list[int], Field(min_length=4, max_length=4)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
ContextText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)]
ContextList = Annotated[list[ContextText], Field(max_length=8)]
MAX_CANDIDATES = 8


class VisionModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, hide_input_in_errors=True)


class VisionImage(VisionModel):
    """Dimensions and actual x/y scale of the analyzed copy of the working image."""

    original_width: PositiveInt
    original_height: PositiveInt
    analysis_width: PositiveInt
    analysis_height: PositiveInt
    scale_x: Annotated[float, Field(gt=0, le=1)]
    scale_y: Annotated[float, Field(gt=0, le=1)]

    @classmethod
    def from_sizes(cls, original_size: tuple[int, int], analysis_size: tuple[int, int]) -> 'VisionImage':
        width, height = original_size
        analysis_width, analysis_height = analysis_size
        return cls(original_width=width, original_height=height,
                   analysis_width=analysis_width, analysis_height=analysis_height,
                   scale_x=analysis_width / width, scale_y=analysis_height / height)

    def restore_bbox(self, bbox: Box) -> list[int]:
        """Clip a half-open analysis box, then round outward in original pixels."""
        x0, y0, x1, y1 = bbox
        width, height = self.analysis_width, self.analysis_height
        x0, x1 = (max(0, min(width, x)) for x in (x0, x1))
        y0, y1 = (max(0, min(height, y)) for y in (y0, y1))
        if x0 >= x1 or y0 >= y1:
            raise ValueError('候选框坐标反向、面积为零或完全超出分析图像。')
        return [x0 * self.original_width // width, y0 * self.original_height // height,
                (x1 * self.original_width + width - 1) // width,
                (y1 * self.original_height + height - 1) // height]


class CulturalObservations(VisionModel):
    visible_subject: ContextText = ''
    possible_period_or_tradition: ContextText = ''
    visual_clues: ContextList = Field(default_factory=list)
    verification_needed: ContextList = Field(default_factory=list)
    preservation_constraints: ContextList = Field(default_factory=list)


class CulturalContext(CulturalObservations):
    confidence: Literal['low', 'medium', 'high', ''] = ''
    verified: Literal[False] = False


class DefectObservation(VisionModel):
    candidate_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
    bbox: Box
    type: DefectType
    observation: ShortText
    evidence: ShortText
    repair_suggestion: ShortText
    uncertainty: ShortText
    risk_level: RiskLevel


class DefectCandidate(DefectObservation):
    """A suggested area in integer, half-open working-image coordinates."""

    model_source: ShortText

    @model_validator(mode='after')
    def validate_bbox(self) -> 'DefectCandidate':
        x0, y0, x1, y1 = self.bbox
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError('候选框坐标反向、面积为零或超出原图。')
        return self


class VisionAssessment(VisionModel):
    source: Literal['deepseek', 'local']
    model_source: ShortText
    image: VisionImage
    items: Annotated[list[DefectCandidate], Field(max_length=MAX_CANDIDATES)]
    cultural_context: CulturalContext | None = None
    limitations: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)] = ''

    @model_validator(mode='after')
    def validate_items(self) -> 'VisionAssessment':
        if len({item.candidate_id for item in self.items}) != len(self.items):
            raise ValueError('候选 ID 重复。')
        for item in self.items:
            if item.bbox[2] > self.image.original_width or item.bbox[3] > self.image.original_height:
                raise ValueError('候选框坐标超出原图。')
            if item.model_source != self.model_source:
                raise ValueError('候选和分析的模型来源不一致。')
        return self


ReviewText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=600)]


class RepairRegionReview(VisionModel):
    """One model review of an existing local repair region (advisory only)."""

    region_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
    observation: ReviewText
    repair_suggestion: ReviewText
    repair_assessment: ReviewText
    basis: ReviewText
    risk_level: RiskLevel
    uncertainty: ReviewText
    human_review_required: bool
    review_reason: ReviewText


class RepairAssessment(VisionModel):
    """Provider-independent repair review; never an instruction to edit pixels."""

    source: Literal['deepseek']
    model_source: ShortText
    regions: Annotated[list[RepairRegionReview], Field(max_length=MAX_CANDIDATES)]
    limitations: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)] = ''
    advisory: Literal[True] = True

    @model_validator(mode='after')
    def validate_regions(self) -> 'RepairAssessment':
        if len({item.region_id for item in self.regions}) != len(self.regions):
            raise ValueError('评估区域 ID 重复。')
        return self
