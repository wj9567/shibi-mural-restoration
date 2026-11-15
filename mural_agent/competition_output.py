"""Stage 7: one-page AI mural observation card and optional repair report PDFs.

The PDFs are generated from stored records only. They show what actually ran:
local model sources, chosen repair tools, review states, stale flags and image
provenance. Nothing invents confidence scores, accuracy or expert conclusions.
"""
from pathlib import Path
from datetime import datetime
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from .config import ROOT
from . import review_workflow

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 36
FOOTER_Y = 42
TYPE_LABELS = {'crack': '裂隙', 'paint_loss': '颜料层缺失', 'plaster_loss': '地仗/灰泥缺失',
               'flaking': '起甲/剥落', 'fading': '褪色', 'stain': '污染',
               'manual': '手动区域', 'other': '其他疑似缺损'}
TOOL_LABELS = {'lama': 'LaMa 纹理补全', 'telea': 'Telea 纹理传播', 'ns': 'Navier–Stokes 细线修补',
               'color': '局部色调匹配', 'diffusion': '扩散生成（推测）', 'imagine': '云端全图重绘（推测）',
               'denoise': '局部降噪', 'deblur': '局部锐化', 'reference': '核验参考转移'}
FONT_CANDIDATES = [
    ('MuralCJK', 'C:/Windows/Fonts/msyh.ttc', 0),
    ('MuralCJK', 'C:/Windows/Fonts/msyh.ttc', 0),
    ('MuralCJK', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 0),
    ('MuralCJK', '/System/Library/Fonts/PingFang.ttc', 0),
    ('MuralCJK', 'C:/Windows/Fonts/simhei.ttf', None),
]
BOLD_CANDIDATES = [
    ('MuralCJKBold', 'C:/Windows/Fonts/msyhbd.ttc', 0),
    ('MuralCJKBold', 'C:/Windows/Fonts/simhei.ttf', None),
]


def _register(candidates):
    registered = set(pdfmetrics.getRegisteredFontNames())
    for name, path, index in candidates:
        if name in registered:
            return name, path
        if not Path(path).is_file():
            continue
        try:
            if index is None:
                pdfmetrics.registerFont(TTFont(name, path))
            else:
                pdfmetrics.registerFont(TTFont(name, path, subfontIndex=index))
            return name, path
        except Exception:  # noqa: BLE001 — try the next known-good Chinese font
            continue
    return None, None


_FONTS = None


def fonts():
    """Register a Chinese font once; raise a clear error when none is available."""
    global _FONTS
    if _FONTS is None:
        override = os.environ.get('MURAL_PDF_FONT')
        candidates = list(FONT_CANDIDATES)
        if override:
            candidates.insert(0, ('MuralCJK', override, 0))
        regular, path = _register(candidates)
        if not regular:
            raise RuntimeError('未找到可用的中文字体，无法生成 PDF。请安装微软雅黑/黑体，或设置 MURAL_PDF_FONT 指向字体文件。')
        bold, _ = _register(BOLD_CANDIDATES)
        _FONTS = (regular, bold or regular, path)
    return _FONTS


def _pil_font(size, bold=False):
    regular, bold_name, path = fonts()
    index = 0
    try:
        return ImageFont.truetype(path, size, index=index)
    except Exception:  # noqa: BLE001 — fall back to a TTF sibling if the TTC index fails
        for fallback in ('C:/Windows/Fonts/simhei.ttf', path):
            try:
                return ImageFont.truetype(fallback, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _clean(text, limit=2000):
    return ' '.join(str(text or '').split())[:limit]


def wrap_text(text, font, size, width, max_lines):
    """Character wrapping for CJK; returns (lines, truncated)."""
    text = _clean(text)
    if not text:
        return ['（未提供）'], False
    lines, current = [], ''
    for char in text:
        if pdfmetrics.stringWidth(current + char, font, size) <= width:
            current += char
            continue
        lines.append(current)
        current = char
        if len(lines) >= max_lines:
            break
    if len(lines) < max_lines and current:
        lines.append(current)
    truncated = len(''.join(lines)) < len(text)
    if truncated:
        last = lines[-1] if lines else ''
        while last and pdfmetrics.stringWidth(last + '…', font, size) > width:
            last = last[:-1]
        lines[-1] = last + '…'
    return lines[:max_lines], truncated


def _now():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


class _Page:
    def __init__(self, pdf, title, subtitle):
        self.pdf = pdf
        self.regular, self.bold, _ = fonts()
        self.truncated = False
        width = PAGE_WIDTH - 2*MARGIN
        self.pdf.setTitle(title)
        self.pdf.setAuthor('拾壁 mural-restoration-agent')
        self.pdf.setFillColorRGB(.12, .13, .15)
        self.pdf.setFont(self.bold, 17)
        self.pdf.drawString(MARGIN, PAGE_HEIGHT - 52, title)
        self.pdf.setFont(self.regular, 8.5)
        self.pdf.setFillColorRGB(.42, .45, .5)
        self.pdf.drawString(MARGIN, PAGE_HEIGHT - 68, subtitle)
        self.pdf.setFont(self.regular, 7)
        self.pdf.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 52,
                                 '拾壁 · AI 辅助观察，不是鉴定结论')
        self.pdf.setStrokeColorRGB(.78, .8, .84)
        self.pdf.setLineWidth(.6)
        self.pdf.line(MARGIN, PAGE_HEIGHT - 76, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 76)
        self.y = PAGE_HEIGHT - 92
        self.width = width

    def section(self, title, top_gap=6):
        self.y -= top_gap
        if self.y < FOOTER_Y + 30:
            self.truncated = True
            return False
        self.pdf.setFillColorRGB(.15, .18, .22)
        self.pdf.setFont(self.bold, 10.5)
        self.pdf.drawString(MARGIN, self.y, title)
        self.y -= 4
        self.pdf.setStrokeColorRGB(.85, .87, .9)
        self.pdf.line(MARGIN, self.y, PAGE_WIDTH - MARGIN, self.y)
        self.y -= 12
        return True

    def paragraph(self, text, size=8.6, leading=12, max_lines=3, indent=0, color=(.22, .24, .28)):
        if self.y < FOOTER_Y + 24:
            self.truncated = True
            return
        lines, truncated = wrap_text(text, self.regular, size, self.width - indent, max_lines)
        self.truncated = self.truncated or truncated
        self.pdf.setFont(self.regular, size)
        self.pdf.setFillColorRGB(*color)
        for line in lines:
            if self.y < FOOTER_Y + 20:
                self.truncated = True
                break
            self.pdf.drawString(MARGIN + indent, self.y, line)
            self.y -= leading

    def footer(self, note, page_label=''):
        self.pdf.setStrokeColorRGB(.85, .87, .9)
        self.pdf.line(MARGIN, FOOTER_Y + 12, PAGE_WIDTH - MARGIN, FOOTER_Y + 12)
        self.pdf.setFont(self.regular, 6.8)
        self.pdf.setFillColorRGB(.45, .48, .52)
        self.pdf.drawString(MARGIN, FOOTER_Y, _clean(note, 160))
        suffix = f'{page_label} · ' if page_label else ''
        self.pdf.drawRightString(PAGE_WIDTH - MARGIN, FOOTER_Y, f'{suffix}生成于 {_now()}')


def _draw_image(pdf, picture, x, y_top, max_width, max_height, caption=None):
    width, height = picture.size
    scale = min(max_width / width, max_height / height, 1.0)
    draw_width, draw_height = width * scale, height * scale
    left = x + (max_width - draw_width) / 2
    top = y_top - draw_height
    pdf.drawImage(ImageReader(picture), left, top, draw_width, draw_height, mask='auto')
    if caption:
        pdf.setFont(fonts()[0], 7)
        pdf.setFillColorRGB(.42, .45, .5)
        pdf.drawString(left, top - 9, caption)
    return top - (16 if caption else 2)


# ---------------------------------------------------------------- data assembly

def provenance_for(image):
    """Match the imported working copy against the licensed sample manifest."""
    manifest = ROOT / '图片' / '清单.jsonl'
    if not manifest.is_file():
        return None
    digest = image.get('original_sha256') or ''
    name = Path(str(image.get('name') or '')).name
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get('sha256') == digest or (name and entry.get('image') == name):
            return entry
    return None


def _latest_assessment(store, image):
    records = [record for record in store.list('assessment') if record.get('image_id') == image['id']]
    records.sort(key=lambda record: record.get('created_at') or '')
    return records[-1] if records else None


def _region_records(store, image, limit=3):
    proposals = [proposal for proposal in store.list('proposal')
                 if proposal.get('image_id') == image['id']
                 and (proposal.get('metadata') or {}).get('type')]
    jobs = store.list('job')

    def has_candidate(proposal):
        return any(job.get('defect_proposal_id') == proposal['id'] and job.get('candidates') for job in jobs)

    proposals.sort(key=lambda proposal: (not proposal.get('confirmed'), not has_candidate(proposal),
                                         proposal.get('created_at') or ''))
    assessment = _latest_assessment(store, image)
    assessment_rows = {}
    if assessment and not assessment.get('stale'):
        assessment_rows = {row.get('region_id'): row for row in assessment.get('regions') or []}
    records = []
    for proposal in proposals[:limit]:
        metadata = proposal.get('metadata') or {}
        job = next((job for job in jobs if job.get('defect_proposal_id') == proposal['id']
                    and job.get('candidates') and job.get('status') == 'review'), None)
        candidate = (job or {}).get('candidates', [None])[0] if job else None
        records.append({
            'proposal_id': proposal['id'],
            'type': str(metadata.get('type') or 'other'),
            'label': TYPE_LABELS.get(str(metadata.get('type') or 'other'), '其他疑似缺损'),
            'evidence': metadata.get('evidence') or metadata.get('observation') or '',
            'identification': review_workflow.identification_state(proposal),
            'mask_status': metadata.get('mask_status') or '',
            'mask_notice': metadata.get('mask_notice') or '',
            'bbox_pixels': metadata.get('bbox_pixels') or [],
            'candidate': candidate,
            'job': job,
            'assessment': assessment_rows.get(proposal['id']),
            'revision': review_workflow.revision(proposal),
        })
    return records, assessment


def _annotated(image_array, regions):
    picture = Image.fromarray(image_array).convert('RGB')
    draw = ImageDraw.Draw(picture)
    size = max(14, round(picture.width / 42))
    font = _pil_font(size, bold=True)
    palette = [(214, 62, 44), (30, 110, 170), (150, 96, 20)]
    for index, region in enumerate(regions, 1):
        bbox = region['bbox_pixels']
        if len(bbox) != 4:
            continue
        color = palette[(index - 1) % len(palette)]
        x0, y0, x1, y1 = [int(value) for value in bbox]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, size // 5))
        text = f'{index} {region["label"]}'
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tag_width, tag_height = right - left + 8, bottom - top + 6
        tag_y = max(0, y0 - tag_height)
        draw.rectangle([x0, tag_y, x0 + tag_width, tag_y + tag_height], fill=color)
        draw.text((x0 + 4, tag_y + 2), text, font=font, fill=(255, 255, 255))
    return picture


def _ai_usage(store, image, regions, assessment):
    detection = sorted({(proposal.get('metadata') or {}).get('model_source')
                        for proposal in store.list('proposal')
                        if proposal.get('image_id') == image['id']
                        and (proposal.get('metadata') or {}).get('model_source')})
    sam_used = any(region['mask_status'] == 'sam' for region in regions)
    tools = sorted({(region['candidate'] or {}).get('tool') for region in regions if region['candidate']})
    lines = []
    lines.append('识别：' + ('、'.join(detection) if detection else '未运行本地识别') + '（候选，需人工核对）')
    if sam_used:
        lines.append('分割：SAM 2.1 Hiera Tiny（本地）')
    if tools:
        lines.append('修复：' + '、'.join(TOOL_LABELS.get(tool, str(tool)) for tool in tools))
    if assessment:
        state = '（已过期，需重新评估）' if assessment.get('stale') else ''
        lines.append(f'评估：DeepSeek {assessment.get("model_source") or "deepseek-flash"}（建议，非定稿）{state}')
    else:
        lines.append('评估：本次未调用 DeepSeek 修复评估')
    lines.append('未编造识别置信度、实验精度或审核结论；所有结果均来自本机记录。')
    return lines


def _advice(regions, assessment):
    tips = []
    kinds = {region['type'] for region in regions}
    if kinds & {'crack', 'paint_loss', 'plaster_loss', 'flaking'}:
        tips.append('裂隙与剥落区域可做局部保守数字补全，仅作展示，必须在修复范围内并保留原画线条。')
    if kinds & {'fading', 'stain'}:
        tips.append('褪色/污染优先做色调整理与记录，不建议直接补洞。')
    if any(region['assessment'] and region['assessment'].get('human_review_required') for region in regions):
        tips.append('AI 评估要求人工复核，展示前应由修复人员确认补全范围与依据。')
    if assessment:
        for row in assessment.get('regions') or []:
            suggestion = _clean(row.get('repair_suggestion'), 160)
            if suggestion:
                tips.append('DeepSeek 建议：' + suggestion)
                break
    if not tips:
        tips.append('本卡仅记录可见现象；任何补全或展示方案都需由专业人员确认。')
    return tips[:2]


def _uncertainties(regions, assessment):
    notes = []
    pending = [region['label'] for region in regions if region['identification'] == 'pending']
    if pending:
        notes.append('尚未确认识别：' + '、'.join(dict.fromkeys(pending)) + '。')
    review = [region['label'] for region in regions if region['mask_status'] in {'needs_review', 'observation'}]
    if review:
        notes.append('修复范围需人工修正或属于观察类：' + '、'.join(dict.fromkeys(review)) + '。')
    if assessment is None:
        notes.append('本次未调用 DeepSeek 修复评估；风险与复核意见仅来自本地规则。')
    elif assessment.get('stale'):
        notes.append('AI 评估基于旧修复范围，已标记过期，需重新生成候选后再次评估。')
    else:
        flagged = [row for row in assessment.get('regions') or [] if row.get('human_review_required')]
        if flagged:
            notes.append('AI 评估要求人工复核：' + _clean(flagged[0].get('review_reason'), 180))
    if not notes:
        notes.append('未发现额外机器风险提示；仍需人工核对原图与修复范围。')
    return notes[:3]


def _source_lines(image):
    entry = provenance_for(image)
    if entry:
        return [f'演示素材：{Path(str(image.get("name") or "")).name}',
                f'来源：{entry.get("source")}',
                f'许可：{entry.get("license")}（{entry.get("group")}）']
    return ['来源：用户本地导入的图像，来源与使用许可由用户自行负责。',
            '参考模型：Qwen3-VL-2B-Instruct、SAM 2.1 Hiera Tiny、LaMa（IOPaint 发布权重）、DeepSeek API。',
            '本卡只用于数字修复实验与人工复核。']


# ---------------------------------------------------------------- PDF builders

def build_observation_card(store, image, output_path):
    """One-page A4 observation card. Returns metadata about the exported file."""
    picture = Image.open(store.folder(image['id']) / 'image.png').convert('RGB')
    image_array = np.asarray(picture)
    regions, assessment = _region_records(store, image)
    annotated = _annotated(image_array, regions)
    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    page = _Page(pdf, 'AI 壁画观察卡', f'{Path(str(image.get("name") or "")).name} · {image["width"]}×{image["height"]} px')
    page.y = _draw_image(pdf, annotated, MARGIN, page.y,
                         PAGE_WIDTH - 2*MARGIN, 250,
                         '原图与重点标注（编号与下方清单对应）') - 14

    if page.section('一、重大可疑病害与可见依据'):
        if not regions:
            page.paragraph('未记录带类型的病害区域；可在缺损清单中确认或手动画选。', max_lines=2)
        for index, region in enumerate(regions, 1):
            state = {'confirmed': '识别已确认', 'expert_review': '识别已确认 · 需专家复核',
                     'pending': '识别待确认'}.get(region['identification'], region['identification'])
            mask = {'sam': 'SAM 修复范围候选', 'manual': '人工修复范围', 'needs_review': '修复范围待修正',
                    'observation': '观察类，不生成修复范围'}.get(region['mask_status'], region['mask_status'] or '无修复范围')
            header = f'{index}. {region["label"]} · {state} · {mask}'
            if region['candidate']:
                header += f' · 候选：{TOOL_LABELS.get(region["candidate"].get("tool"), region["candidate"].get("tool"))}'
            page.paragraph(header, size=8.8, leading=11.5, max_lines=1)
            page.paragraph('依据：' + (region['evidence'] or '未记录可观察依据'), size=8.2, leading=10.8,
                           max_lines=2, indent=10)

    if page.section('二、保护与数字展示建议'):
        for tip in _advice(regions, assessment):
            page.paragraph('· ' + tip, size=8.4, leading=11, max_lines=2, indent=2)

    if page.section('三、AI 使用情况'):
        for line in _ai_usage(store, image, regions, assessment):
            page.paragraph(line, size=8.2, leading=10.8, max_lines=2)

    if page.section('四、不确定性与人工复核提示'):
        for note in _uncertainties(regions, assessment):
            page.paragraph('· ' + note, size=8.2, leading=10.8, max_lines=2, indent=2,
                           color=(.48, .25, .1))

    if page.section('五、图片来源与参考资料'):
        for line in _source_lines(image):
            page.paragraph(line, size=7.8, leading=10.2, max_lines=2)

    page.footer('仅供数字修复实验与人工复核；AI 观察不是文物病害鉴定，也不代表历史原貌。', '第 1/1 页')
    pdf.showPage()
    pdf.save()
    return {'path': str(output_path), 'pages': 1, 'regions': len(regions),
            'truncated': page.truncated, 'has_assessment': bool(assessment)}


def _candidate_pictures(store, job, candidate):
    folder = store.folder(candidate['id'])
    before = Image.open(store.folder(job['image_id']) / 'image.png').convert('RGB')
    after = Image.open(folder / 'result.png').convert('RGB')
    difference = Image.open(folder / 'difference.png').convert('RGB') if (folder / 'difference.png').is_file() else None
    region = Image.open(folder / 'region.png').convert('RGB') if (folder / 'region.png').is_file() else None
    return before, after, difference, region


def build_repair_report(store, image, job, output_path):
    """Two-page repair report: result comparison + method/checks, risks + review log."""
    if not job.get('candidates'):
        raise ValueError('该任务还没有修复候选，无法生成修复说明。')
    candidate_id = job.get('selected_id') or job['candidates'][0]['id']
    candidate = next((item for item in job['candidates'] if item['id'] == candidate_id), job['candidates'][0])
    proposal = store.get('proposal', job.get('defect_proposal_id')) if job.get('defect_proposal_id') else None
    before, after, difference, region = _candidate_pictures(store, job, candidate)
    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    page = _Page(pdf, '修复成果说明', f'{Path(str(image.get("name") or "")).name} · '
                                     f'{"区域修复" if proposal else "手动修复"} · '
                                     f'{job.get("workflow") or "standard"}')
    width = (PAGE_WIDTH - 2*MARGIN - 16) / 3
    top = page.y
    tops = [_draw_image(pdf, before, MARGIN, top, width, 190, '修复前原图'),
            _draw_image(pdf, after, MARGIN + width + 8, top, width, 190, '本地修复结果'),
            _draw_image(pdf, difference if difference is not None else after,
                        MARGIN + 2*(width + 8), top, width, 190, '差异图（放大）')]
    page.y = min(tops) - 10
    if region is not None:
        page.y = _draw_image(pdf, region, MARGIN, page.y, 200, 80, '区域前后裁剪（如有）') - 4

    if page.section('一、修复方式'):
        method = TOOL_LABELS.get(candidate.get('tool'), candidate.get('tool')) or '未知工具'
        parameters = candidate.get('parameters') or {}
        versions = candidate.get('versions') or {}
        page.paragraph(f'候选 {candidate_id[:8]} · 方法：{method} · 设备：{versions.get("device", "未记录")}', max_lines=1)
        if parameters.get('radius') is not None:
            page.paragraph(f'参数：半径 {parameters.get("radius")} · 耗时 {candidate.get("elapsed_seconds", "-")} 秒', max_lines=1)
        else:
            page.paragraph(f'参数：{_clean(json.dumps(parameters, ensure_ascii=False), 300)}', max_lines=2)
        weight = versions.get('weight_sha256')
        if weight:
            page.paragraph(f'权重 SHA-256：{weight}', max_lines=1)
        fraction = (candidate.get('metrics') or {}).get('masked_fraction')
        if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
            fraction = (proposal or {}).get('fraction') or 0
        page.paragraph(f'修复范围占比：{float(fraction):.2%} · '
                       f'区域等级：{(proposal or {}).get("region_class") or "未记录"}', max_lines=1)

    if page.section('二、输出检查'):
        checks = candidate.get('checks') or {}
        page.paragraph(f'输出尺寸一致：{"是" if checks.get("size_match") else "否"} · '
                       f'修复范围之外的像素不变：{"是" if checks.get("outside_unchanged") else "否"}', max_lines=1)
        page.paragraph('工程检查只证明没有改动修复范围之外的像素，不证明补全内容符合历史原貌。', max_lines=2,
                       color=(.48, .25, .1))

    if page.section('三、修复摘要'):
        if proposal:
            metadata = proposal.get('metadata') or {}
            page.paragraph(f'区域类型：{TYPE_LABELS.get(str(metadata.get("type")), "其他疑似缺损")} · '
                           f'版本 {review_workflow.revision(proposal)} · '
                           f'识别状态 {review_workflow.identification_state(proposal)}', max_lines=1)
            page.paragraph('可见依据：' + (metadata.get('evidence') or '未记录可观察依据'),
                           max_lines=2, size=8.2, leading=10.6)
        else:
            page.paragraph('本次为手动框选修复，没有病害类型记录。', max_lines=1)
        page.paragraph('本报告只描述一次本地修复候选；工程检查不证明补全内容符合历史原貌。',
                       max_lines=2, color=(.48, .25, .1))
    page.footer('修复结果为数字修复候选，不是历史复原；请结合复核记录使用。', '第 1/2 页')
    pdf.showPage()

    second = _Page(pdf, '修复成果说明（复核记录）', f'任务 {job.get("id", "")[:8]} · '
                                                   f'{Path(str(image.get("name") or "")).name}')
    if second.section('四、风险与人工审核状态'):
        if proposal:
            state = review_workflow.identification_state(proposal)
            second.paragraph(f'病害识别：{state} · 区域版本：{review_workflow.revision(proposal)}', max_lines=1)
            second.paragraph(f'候选审核：{review_workflow.candidate_state(candidate)}', max_lines=1)
        else:
            second.paragraph('该任务不是按病害区域生成，沿用原有手动复核流程。', max_lines=2)
        if candidate.get('stale'):
            second.paragraph('注意：该候选已过期，不能作为正式导出结果。', max_lines=1, color=(.55, .2, .1))
        if candidate.get('speculative'):
            second.paragraph('该候选为推测性补全，只能按假设性建议展示。', max_lines=1, color=(.55, .2, .1))
        assessment = _latest_assessment(store, image)
        if assessment:
            state = '（已过期）' if assessment.get('stale') else ''
            second.paragraph(f'AI 修复评估：DeepSeek {assessment.get("model_source")}{state}', max_lines=1)
            for row in (assessment.get('regions') or [])[:2]:
                second.paragraph(
                    f'区域 {row.get("index")} · 风险 {row.get("risk_level")} · '
                    f'{"需要人工复核" if row.get("human_review_required") else "未标记必须复核"}：'
                    f'{row.get("review_reason") or row.get("repair_assessment") or ""}',
                    max_lines=2, size=8, leading=10.4)
        else:
            second.paragraph('本次未调用 DeepSeek 修复评估。', max_lines=1)

    if second.section('五、人工审核记录'):
        proposal_id = (proposal or {}).get('id', '')
        events = review_workflow.list_events(store, image['id'], proposal_id) if proposal_id else []
        events = events[-6:]
        if not events:
            second.paragraph('没有审核记录；请先在结果页确认候选或记录复核意见。', max_lines=2)
        for event in events:
            second.paragraph(f'{event["created_at"][5:16].replace("T", " ")} · {event["action"]} · '
                             f'{event.get("decision") or "—"} · {event.get("note") or ""}', max_lines=2,
                             size=7.8, leading=10.2)

    if second.section('六、来源与模型记录'):
        for line in _source_lines(image)[:3]:
            second.paragraph(line, size=7.6, leading=9.8, max_lines=2)
        second.paragraph('本地模型：Qwen3-VL-2B-Instruct、SAM 2.1 Hiera Tiny、LaMa（big-lama）；'
                         '云端评估：DeepSeek deepseek-flash（若启用）。', size=7.6, leading=9.8, max_lines=2)
    second.footer('本说明由拾壁根据本机记录生成，不代表历史原貌。', '第 2/2 页')
    pdf.showPage()
    pdf.save()
    return {'path': str(output_path), 'pages': 2, 'candidate_id': candidate['id'],
            'truncated': page.truncated or second.truncated, 'has_assessment': bool(_latest_assessment(store, image))}
