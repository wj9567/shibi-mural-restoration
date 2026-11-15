"""Qt WebEngine desktop shell for the local HTML workbench."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

from PySide6.QtCore import QObject, QLockFile, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QMainWindow
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView

from . import backup, datasets, references, workspace
from .config import ROOT, Settings, save_env_fields, valid_deepseek_key
from .deepseek_vision import DeepSeekVisionProvider
from .engine import Engine
from .imaging import decode_image, image_array, mask_array
from .qt_setup import configure_application
from .segmentation import damage_proposal
from .storage import Store, new_id, timestamp
from .vision_analysis import cloud_defects, detect_local, local_assessment, pixels_to_normalized


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str)


# Sizes quoted in the install hints match the pinned checkpoints recorded by
# scripts/download_models.py (Qwen3-VL-2B weights and SAM 2.1 Hiera Tiny).
QWEN_WEIGHTS = 'model.safetensors'
SAM_WEIGHTS = 'model.safetensors'


def local_model_setup(settings):
    """Readiness of the local detector plus actionable install hints.

    Readiness is decided by the checkpoint files the two workers actually load,
    not by a provenance marker alone, so a half-finished download cannot be
    mistaken for a usable model.
    """
    qwen_ready = (settings.vision_model / QWEN_WEIGHTS).is_file()
    sam_ready = (settings.sam_model / SAM_WEIGHTS).is_file()
    qwen_hint = '' if qwen_ready else (
        '本地 Qwen3-VL 未安装：请把 Qwen3-VL-2B-Instruct 的 model.safetensors（约 4.27 GB）等文件放到 '
        f'{settings.vision_model}，或运行 .\\scripts\\download_models.py vision。')
    sam_hint = '' if sam_ready else (
        'SAM 2.1 未安装：请把 sam2.1-hiera-tiny 的 model.safetensors（约 156 MB）放到 '
        f'{settings.sam_model}，或运行 .\\scripts\\download_models.py sam2；缺少 SAM 时只保留候选框。')
    return {'qwen_ready': qwen_ready, 'sam_ready': sam_ready,
            'qwen_hint': qwen_hint, 'sam_hint': sam_hint}


def select_full_generation_backend(available):
    """Choose the full-image backend before checking its dependencies.

    Cloud image generation does not need a local diffusion install. Local
    diffusion is required only when that backend is the one that will run.
    """
    if available.get('imagine'):
        return 'imagine'
    if available.get('diffusion'):
        return 'diffusion'
    raise ValueError('全图 AI 生成需要已配置云端生图，或已安装本地扩散模型（models/sd-inpainting）。')


class Bridge(QObject):
    notify = Signal(str)
    auxCompleted = Signal()

    def __init__(self, settings, store, engine):
        super().__init__()
        self.settings, self.store, self.engine = settings, store, engine
        self.aux_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mural-aux')
        self.aux = {'status': 'idle'}
        self.aux_future = None
        self.aux_version = 0
        self.current_image_id = None
        self.cloud_analysis = None

    def _result(self, function):
        try:
            return _json(function())
        except Exception as exc:
            return _json({'error': str(exc)[:500]})

    def _file_url(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.store.root.resolve()):
            raise ValueError('资源路径不在工作空间内。')
        return QUrl.fromLocalFile(str(path)).toString()

    def _image(self, record):
        if not record:
            return None
        result = dict(record)
        result['file_url'] = self._file_url(self.store.folder(record['id']) / 'image.png')
        return result

    def _job(self, record):
        if not record:
            return None
        result = dict(record)
        result['mask_file_url'] = self._file_url(self.store.folder(record['mask_id']) / 'mask.png')
        result['candidates'] = [self._candidate(item) for item in record.get('candidates', [])]
        return result

    def _candidate(self, record):
        result = dict(record)
        folder = self.store.folder(record['id'])
        result['file_url'] = self._file_url(folder / 'result.png')
        result['difference_url'] = self._file_url(folder / 'difference.png')
        if (folder / 'region.png').is_file():
            result['region_url'] = self._file_url(folder / 'region.png')
        if (folder / 'region_difference.png').is_file():
            result['region_difference_url'] = self._file_url(folder / 'region_difference.png')
        return result

    def _save_proposal(self, image_id, mask, metadata):
        identifier = new_id()
        folder = self.store.folder(identifier)
        Image.fromarray(mask).save(folder / 'mask.png')
        proposal = self.store.put('proposal', {'id': identifier, 'image_id': image_id, 'created_at': timestamp(),
            'fraction': float((mask > 0).mean()), 'metadata': metadata, 'confirmed': False})
        return {'proposal': proposal}

    def _mask_overlay(self, array):
        """RGBA canvas overlay: damaged pixels tinted, everything else transparent."""
        import numpy as np
        rgba = np.zeros((array.shape[0], array.shape[1], 4), dtype=np.uint8)
        rgba[array > 0] = (214, 62, 44, 150)
        from io import BytesIO
        buffer = BytesIO()
        Image.fromarray(rgba, 'RGBA').save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    def _local_defect(self, proposal):
        """One local-detection record in the shape the canvas and list expect."""
        from .review_workflow import identification_state, revision
        metadata = proposal.get('metadata') or {}
        defect = {
            'index': 0, 'type': metadata.get('type') or 'other',
            'bbox': list(metadata.get('bbox') or []),
            'bbox_pixels': list(metadata.get('bbox_pixels') or []),
            'size': metadata.get('size') or 'uncertain',
            'confidence': metadata.get('confidence'),
            'evidence': metadata.get('evidence') or '',
            'observation': metadata.get('observation') or metadata.get('evidence') or '',
            'repair_suggestion': metadata.get('repair_suggestion') or '',
            'uncertainty': metadata.get('uncertainty') or '',
            'model_source': metadata.get('model_source') or 'Qwen3-VL local',
            'risk_level': metadata.get('risk_level') or 'unknown',
            'recommendation': metadata.get('recommendation') or {},
            'mask_status': metadata.get('mask_status') or 'observation',
            'mask_notice': metadata.get('mask_notice') or '',
            'confirmed': bool(proposal.get('confirmed')),
            'identification_status': identification_state(proposal),
            'revision': revision(proposal),
            'proposal_id': proposal['id'],
        }
        mask_path = self.store.folder(proposal['id']) / 'mask.png'
        if mask_path.is_file():
            defect['mask_file_url'] = self._file_url(mask_path)
            defect['mask_overlay_data'] = self._mask_overlay(mask_array(mask_path))
        return defect

    def _persist_local_defects(self, image, entries):
        """Store one reviewable proposal per detected candidate (never a job)."""
        width, height = image['width'], image['height']
        defects, proposals = [], []
        for entry in entries:
            candidate = entry['candidate']
            identifier = new_id()
            folder = self.store.folder(identifier)
            if entry['mask'] is not None:
                Image.fromarray(entry['mask']).save(folder / 'mask.png')
            metadata = {
                'type': candidate.type,
                'bbox': pixels_to_normalized(candidate.bbox, width, height),
                'bbox_pixels': list(candidate.bbox),
                'size': entry.get('size') or 'uncertain',
                'confidence': entry.get('confidence'),
                'evidence': candidate.evidence,
                'observation': candidate.observation,
                'repair_suggestion': candidate.repair_suggestion,
                'uncertainty': candidate.uncertainty,
                'model_source': candidate.model_source,
                'risk_level': candidate.risk_level,
                'recommendation': entry.get('recommendation') or {},
                'mask_status': entry['mask_status'],
                'mask_notice': entry['mask_notice'],
                'local_detection': True,
                # Local candidate regions are suggestions; risk classification is
                # intentionally left to the existing review gates.
                'region_class': 'fillable_lacuna', 'risk_regions': [],
                'risk_notice': entry['mask_notice'],
            }
            record = self.store.put('proposal', {
                'id': identifier, 'image_id': image['id'], 'created_at': timestamp(),
                'metadata': metadata, 'adopted': False, 'confirmed': False,
                'mask_source': 'sam' if entry['mask_status'] == 'sam' else 'manual',
                'region_class': 'fillable_lacuna', 'repair_allowed': True,
                'fraction': round(float((entry['mask'] > 0).mean()), 6) if entry['mask'] is not None else 0.0,
            })
            proposals.append(record)
            defects.append(self._local_defect(record))
        for index, defect in enumerate(defects, 1):
            defect['index'] = index
        return defects, proposals

    def _new_local_defect(self, image, payload):
        """Create a user-drawn local defect (normalized 0..1000 box) as a proposal."""
        import numpy as np
        try:
            bbox = [int(value) for value in payload.get('bbox') or []]
        except (TypeError, ValueError) as exc:
            raise ValueError('缺损框无效。') from exc
        if len(bbox) != 4:
            raise ValueError('缺损框无效。')
        width, height = image['width'], image['height']
        x0, y0, x1, y1 = [max(0, min(1000, value)) for value in bbox]
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            raise ValueError('缺损框太小，请重新框选。')
        px0, px1 = x0 * width // 1000, min(width, max(x0 * width // 1000 + 1, (x1 * width + 999) // 1000))
        py0, py1 = y0 * height // 1000, min(height, max(y0 * height // 1000 + 1, (y1 * height + 999) // 1000))
        array = np.zeros((height, width), dtype=np.uint8)
        array[py0:py1, px0:px1] = 255
        identifier = new_id()
        Image.fromarray(array).save(self.store.folder(identifier) / 'mask.png')
        kind = str(payload.get('type') or 'other')[:40]
        from .damage_assessment import recommendation_for
        metadata = {
            'type': kind, 'bbox': [x0, y0, x1, y1], 'bbox_pixels': [px0, py0, px1, py1],
            'size': str(payload.get('size') or 'manual')[:20],
            'evidence': '人工新增或修正的缺损区域。',
            'mask_status': 'manual', 'mask_notice': '人工确认的修复范围，需复核。',
            'recommendation': recommendation_for({'type': kind, 'size': payload.get('size') or 'manual'}),
            'model_source': 'manual', 'risk_level': 'unknown',
            'local_detection': True, 'region_class': 'fillable_lacuna', 'risk_regions': [],
            'risk_notice': '人工编辑，未做自动风险分类。',
        }
        record = self.store.put('proposal', {
            'id': identifier, 'image_id': image['id'], 'created_at': timestamp(),
            'metadata': metadata, 'adopted': False, 'confirmed': True,
            'mask_source': 'manual', 'region_class': 'fillable_lacuna', 'repair_allowed': True,
            'fraction': round(float(array.mean()), 6)})
        return self._local_defect(record)

    def _snapshot(self):
        images = [self._image(item) for item in self.store.list('image')]
        if self.current_image_id is None and images:
            latest_job = next(iter(self.store.list('job')), None)
            self.current_image_id = latest_job['image_id'] if latest_job else images[0]['id']
        jobs = [self._job(item) for item in self.store.list('job')]
        availability = self.engine.tools.availability()
        return {'images': images, 'jobs': jobs, 'models': {**availability, 'vision': (self.settings.vision_model / 'provenance.json').exists(),
            'deepseek_vision': bool(self.settings.deepseek_api_key),
            'local_detection_setup': local_model_setup(self.settings),
            'api_configured': self.settings.api_ready}, 'aux': self.aux, 'aux_version': self.aux_version,
            'api': {'kind': self.settings.api_kind, 'base_url': self.settings.api_base_url,
                    'model': self.settings.api_model, 'json_mode': self.settings.api_json_mode},
            'imagine': {'provider': self.settings.imagine_provider,
                        'base_url': self.settings.imagine_base_url,
                        'model': self.settings.imagine_model,
                        'configured': self.settings.imagine_ready},
            'registration': self._registration(self.store.get('registration', self.aux.get('registration_id', '')))
                if self.aux.get('registration_id') else None}

    def _registration(self, record):
        if not record:
            return None
        result = dict(record)
        folder = self.store.folder(record['id'])
        result['overlay_url'] = self._file_url(folder / 'overlay.png')
        return result

    def _aux(self, function, label, *, image_id=None):
        if self.aux_future and not self.aux_future.done():
            raise ValueError('辅助工具仍在运行。')
        if any(job['status'] in {'queued', 'planning', 'running', 'evaluating'}
               for job in self.store.list('job')):
            raise ValueError('修复任务运行中，请完成或取消后再分析。')
        self.aux = {'status': 'running', 'label': label}
        if image_id is not None:
            self.aux['image_id'] = image_id
        self.aux_version += 1
        self.aux_future = self.aux_executor.submit(function)
        return {'status': 'running'}

    def poll_aux(self):
        if self.aux_future and self.aux_future.done():
            future, self.aux_future = self.aux_future, None
            image_id = self.aux.get('image_id')
            try:
                self.aux = {'status': 'done', 'result': future.result()}
                value = self.aux['result']
                assessment = value.get('auto', {}).get('assessment')
                if image_id is not None and assessment and assessment['source'] == 'deepseek':
                    self.cloud_analysis = (image_id, assessment)
                if value.get('proposal'):
                    url = self._file_url(self.store.folder(value['proposal']['id']) / 'mask.png')
                    value['proposal']['file_url'] = url
                    value['proposal']['mask_file_url'] = url
                if value.get('registration'):
                    self.aux['registration_id'] = value['registration']['id']
                    value['registration'] = self._registration(value['registration'])
                if value.get('references'):
                    for item in value['references']:
                        image = self.store.get('image', item['image_id'])
                        item['file_url'] = self._file_url(self.store.folder(image['id']) / 'image.png')
            except Exception as exc:
                # Keep the label so the UI can tell which auxiliary action failed.
                self.aux = {'status': 'error', 'error': str(exc)[:500], 'label': self.aux.get('label')}
            if image_id is not None:
                self.aux['image_id'] = image_id
            self.aux_version += 1
            self.auxCompleted.emit()

    @Slot(result=str)
    def snapshot(self):
        self.poll_aux()
        return self._result(self._snapshot)

    @Slot(result=str)
    def openImage(self):
        def action():
            path, _ = QFileDialog.getOpenFileName(None, '打开壁画图像', '', '图像 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)')
            if not path: return None
            record = workspace.add_image(self.store, Path(path).read_bytes(), Path(path).name)
            self.current_image_id = record['id']
            return self._image(record)
        return self._result(action)

    @Slot(result=str)
    def loadDemo(self):
        from .workspace import sample_image
        def action():
            record = workspace.add_image(self.store, sample_image(), '合成纹样练习图（非文物照片）.png')
            self.current_image_id = record['id']
            return self._image(record)
        return self._result(action)

    @Slot(str, str, str, result=str)
    def saveMask(self, image_id, data, proposal_ids):
        def action():
            payload = base64.b64decode(data, validate=True)
            image = self.store.get('image', image_id)
            if not image:
                raise ValueError('图像记录不存在。')
            self.current_image_id = image_id
            # Desktop save after brush/edit is always an expert-confirmed mask.
            ids = json.loads(proposal_ids) if proposal_ids else []
            source = 'ai_candidate' if ids else 'manual'
            mask = workspace.add_mask(self.store, image, payload, ids, mask_source=source, user_confirmed=True)
            return {**mask, 'file_url': self._file_url(self.store.folder(mask['id']) / 'mask.png')}
        return self._result(action)

    @Slot(str, result=str)
    def generateFullImage(self, image_id):
        """Generate a full-image AI reconstruction guided by the original image."""
        def action():
            from io import BytesIO
            import numpy as np
            image = self.store.get('image', image_id)
            if not image:
                raise ValueError('图像记录不存在。')
            available = self.engine.tools.availability()
            backend = select_full_generation_backend(available)
            with Image.open(self.store.folder(image['id']) / 'image.png') as source:
                source = source.convert('RGB')
                width, height = source.size
            border = max(2, min(width, height) // 100)
            mask = np.full((height, width), 255, dtype=np.uint8)
            mask[:border, :] = mask[-border:, :] = 0
            mask[:, :border] = mask[:, -border:] = 0
            buffer = BytesIO()
            Image.fromarray(mask).save(buffer, format='PNG')
            mask_record = workspace.add_mask(
                self.store, image, buffer.getvalue(), mask_source='manual',
                user_confirmed=True, region_class='fillable_lacuna')
            if backend == 'imagine':
                from .imagine import FULL_REDRAW_PROMPT
                instructions = FULL_REDRAW_PROMPT
                structure = False
            else:
                from .professional import FULL_GENERATION_PROMPT
                instructions = FULL_GENERATION_PROMPT
                structure = bool(available.get('controlnet'))
            job = workspace.create_job(
                self.store, self.engine, image, mask_record, backend, 'rules', 1,
                instructions,
                mode='speculative', workflow='professional_step',
                rationale='用户要求按原图进行全图 AI 生成；结果不是历史原貌，仅作方案示意。',
                evidence_level='speculative', mask_gate_confirmed=True,
                risk_acknowledged=True, allow_generative_attempt=True,
                structure_control=structure,
                full_generation=True)
            return self._job(job)
        return self._result(action)

    @Slot(str, result=str)
    def generateLargeRepair(self, image_id):
        """Run experimental diffusion over the union of all detected defects."""
        def action():
            from io import BytesIO
            import numpy as np
            image = self.store.get('image', image_id)
            if not image:
                raise ValueError('图像记录不存在。')
            available = self.engine.tools.availability()
            if not available.get('diffusion'):
                raise ValueError('AI 大面积修补需要已安装扩散模型（models/sd-inpainting）。')
            masks = []
            for proposal in self.store.list('proposal'):
                if proposal.get('image_id') != image_id:
                    continue
                path = self.store.folder(proposal['id']) / 'mask.png'
                if path.is_file():
                    masks.append(mask_array(path) > 0)
            if not masks:
                raise ValueError('当前图像还没有缺损候选；请先做 AI 分析或手动添加缺损。')
            height, width = masks[0].shape
            union = np.zeros((height, width), dtype=bool)
            for item in masks:
                if item.shape == union.shape:
                    union |= item
            if not union.any() or union.all():
                raise ValueError('缺损候选修复范围不可用，请重新分析或手动画选区。')
            buffer = BytesIO()
            Image.fromarray(union.astype(np.uint8) * 255).save(buffer, format='PNG')
            mask_record = workspace.add_mask(
                self.store, image, buffer.getvalue(), proposal_ids=[],
                mask_source='manual', user_confirmed=True, region_class='fillable_lacuna')
            from .professional import LARGE_GENERATION_PROMPT
            job = workspace.create_job(
                self.store, self.engine, image, mask_record, 'diffusion', 'rules', 1,
                LARGE_GENERATION_PROMPT,
                mode='speculative', workflow='professional_step',
                rationale='用户要求 AI 修补大面积缺损；并集区域为推测性生成，不是历史复原。',
                evidence_level='speculative', mask_gate_confirmed=True,
                risk_acknowledged=True, allow_generative_attempt=True)
            return self._job(job)
        return self._result(action)

    @Slot(str, str, result=str)
    def addManualDefect(self, image_id, payload_json):
        """Add a user-drawn rectangular defect (normalized bbox 0..1000)."""
        def action():
            image = self.store.get('image', image_id)
            if not image:
                raise ValueError('图像记录不存在。')
            payload = json.loads(payload_json or '{}')
            candidate_id = payload.get('candidate_id')
            candidate = None
            if candidate_id is not None:
                analysis_image_id, assessment = self.cloud_analysis or (None, None)
                if analysis_image_id == image_id:
                    candidate = next((item for item in assessment['items'] if item['candidate_id'] == candidate_id), None)
                if candidate is None:
                    raise ValueError('该候选不属于当前图像的 DeepSeek 分析，请重新分析或手动添加缺损。')
            try:
                bbox = [int(value) for value in payload.get('bbox', [])]
            except (TypeError, ValueError) as exc:
                raise ValueError('手动缺损框无效。') from exc
            if len(bbox) != 4:
                raise ValueError('手动缺损框无效。')
            x0, y0, x1, y1 = [max(0, min(1000, value)) for value in bbox]
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                raise ValueError('手动缺损框太小，请重新框选。')
            import numpy as np
            with Image.open(self.store.folder(image['id']) / 'image.png') as source:
                width, height = source.size
            px0, px1 = round(x0 * width / 1000), round(x1 * width / 1000)
            py0, py1 = round(y0 * height / 1000), round(y1 * height / 1000)
            px0, px1 = max(0, px0), min(width, max(px0 + 1, px1))
            py0, py1 = max(0, py0), min(height, max(py0 + 1, py1))
            array = np.zeros((height, width), dtype=np.uint8)
            array[py0:py1, px0:px1] = 255
            if not array.any():
                raise ValueError('手动缺损框无效。')
            identifier = new_id()
            folder = self.store.folder(identifier)
            Image.fromarray(array).save(folder / 'mask.png')
            Image.fromarray(array).save(folder / 'proposal.png')
            metadata = {
                'type': str(payload.get('type') or 'manual')[:40],
                'bbox': [x0, y0, x1, y1],
                'size': str(payload.get('size') or 'manual')[:20],
                'evidence': '用户手动框选的缺损区域。',
                'recommendation': {'method': 'Telea 局部纹理传播', 'backend': 'telea'},
                'region_class': 'fillable_lacuna',
                'risk_regions': [],
                'risk_notice': '手动添加，未做自动风险分类。',
                'manual': True,
            }
            if candidate:
                from .damage_assessment import recommendation_for
                metadata.update(type=candidate['type'], evidence=candidate['evidence'],
                                recommendation=recommendation_for(candidate),
                                vision_candidate=candidate, vision_image=assessment['image'])
            proposal = self.store.put('proposal', {
                'id': identifier, 'image_id': image['id'], 'created_at': timestamp(),
                'fraction': round(float(array.mean()), 6), 'metadata': metadata,
                'adopted': False, 'confirmed': True, 'mask_source': 'manual',
                'region_class': 'fillable_lacuna', 'repair_allowed': True,
            })
            defect = {
                'index': 0, 'type': metadata['type'], 'bbox': metadata['bbox'],
                'size': metadata['size'],
                'evidence': metadata['evidence'], 'recommendation': metadata['recommendation'],
                'proposal_id': identifier, 'region_class': 'fillable_lacuna',
                'repair_allowed': True, 'risk_notice': metadata['risk_notice'], 'confirmed': True,
                'mask_status': 'manual', 'mask_notice': '人工框选的修复范围，需复核。',
                'model_source': 'manual',
                'mask_file_url': self._file_url(folder / 'mask.png'),
                'mask_overlay_data': self._mask_overlay(array),
            }
            return {'proposal': proposal, 'defect': defect}
        return self._result(action)

    @Slot(str, result=str)
    def createJob(self, request_json):
        def action():
            if self.aux_future and not self.aux_future.done():
                raise ValueError('AI 分析仍在运行，请完成后再启动修复。')
            request = json.loads(request_json)
            image = self.store.get('image', request['image_id'])
            mask = self.store.get('mask', request['mask_id'])
            if not image or not mask:
                raise ValueError('图像或修复范围不存在。')
            job = workspace.create_job(
                self.store, self.engine, image, mask, request['backend'], request['planner'],
                int(request['candidate_count']), request.get('instructions', ''),
                request.get('mode', 'conservative'), request.get('registration_id'),
                request.get('budget'), request.get('structure_control', False),
                request.get('quality_thresholds'),
                workflow=request.get('workflow', 'standard'),
                rationale=request.get('rationale', ''),
                evidence_level=request.get('evidence_level'),
                expert_notes=request.get('expert_notes', ''),
                mask_source=request.get('mask_source') or mask.get('mask_source'),
                allow_enhanced=bool(request.get('allow_enhanced')),
                risk_acknowledged=bool(request.get('risk_acknowledged')),
                mask_gate_confirmed=bool(request.get('mask_gate_confirmed')),
                has_unrestorable=bool(request.get('has_unrestorable')),
                has_uncertain=bool(request.get('has_uncertain')),
                force_override=bool(request.get('force_override')),
                force_override_reason=request.get('force_override_reason') or '',
                allow_generative_attempt=bool(request.get('allow_generative_attempt')))
            return self._job(job)
        return self._result(action)

    @Slot(str, result=str)
    def cancelJob(self, job_id):
        return self._result(lambda: self.store.update('job', job_id, cancel_requested=True))

    @Slot(str, result=str)
    def resumeJob(self, job_id):
        return self._result(lambda: (self.engine.resume(job_id), self._job(self.store.get('job', job_id)))[1])

    @Slot(str, str, str, result=str)
    def selectCandidate(self, job_id, candidate_id, note):
        def action():
            from .professional import validate_candidate_selection
            job = self.store.get('job', job_id)
            if not job or candidate_id not in [item['id'] for item in job['candidates']]:
                raise ValueError('候选不属于当前任务。')
            if job['status'] in {'queued', 'planning', 'running', 'evaluating'}:
                raise ValueError('任务完成前不能选定候选。')
            archive_as_hypothesis = False
            review_note = note
            decision = None
            if isinstance(note, str) and note.strip().startswith('{'):
                try:
                    payload = json.loads(note)
                    if isinstance(payload, dict):
                        archive_as_hypothesis = bool(payload.get('archive_as_hypothesis'))
                        review_note = payload.get('note', '')
                        decision = payload.get('decision')
                except json.JSONDecodeError:
                    pass
            candidate = next(item for item in job['candidates'] if item['id'] == candidate_id)
            selection = validate_candidate_selection(
                job, candidate, archive_as_hypothesis=archive_as_hypothesis,
                decision=decision)
            # An empty box must not erase a note restored from the archive/archive history.
            review_note = str(review_note).strip()[:2000] or (job.get('review') or {}).get('note', '')
            self.store.update(
                'job', job_id, selected_id=candidate_id, selected_as=selection['selected_as'],
                review={'note': review_note, 'time': timestamp(),
                        'archive_as_hypothesis': selection['archive_as_hypothesis'],
                        'decision': selection.get('decision')})
            from .review_workflow import record_event, set_candidate_review
            if selection['selected_as'] == 'rejected':
                status = 'rejected'
            elif selection['selected_as'] == 'hypothesis':
                status = 'accepted_hypothesis'
            else:
                status = 'accepted'
            set_candidate_review(self.store, job_id, candidate_id, status, review_note,
                                 decision=selection.get('decision') or 'adopt')
            record_event(self.store, image_id=job['image_id'],
                         proposal_id=job.get('defect_proposal_id') or '',
                         action='candidate_select',
                         changes={'candidate_id': candidate_id, 'candidate_status': status,
                                  'selected_as': selection['selected_as']},
                         decision=selection.get('decision') or 'adopt', note=review_note)
            return self._job(self.store.get('job', job_id))
        return self._result(action)

    @Slot(str, result=str)
    def autoConfirmDefect(self, request_json):
        """One click: confirm the region and make sure a usable repair range exists."""
        def action():
            request = json.loads(request_json)
            image = self.store.get('image', request.get('image_id'))
            if not image:
                raise ValueError('图像记录不存在。')
            ids = request.get('proposal_ids') or ([request.get('proposal_id')] if request.get('proposal_id') else [])
            ids = [identifier for identifier in dict.fromkeys(ids) if identifier]
            if not ids:
                raise ValueError('没有指定要确认的区域。')
            import numpy as np
            from .imaging import image_array
            from .review_workflow import bump_revision, set_identification

            def ensure_range(proposal):
                metadata = dict(proposal.get('metadata') or {})
                mask_path = self.store.folder(proposal['id']) / 'mask.png'
                if mask_path.is_file():
                    existing = mask_array(mask_path)
                    if existing.shape == (image['height'], image['width']) and (existing > 0).any():
                        return 'kept', metadata
                bbox = metadata.get('bbox_pixels') or []
                try:
                    x0, y0, x1, y1 = [int(value) for value in bbox]
                except (TypeError, ValueError):
                    raise ValueError('该区域没有可用的定位框，请先手动框选。') from None
                width, height = image['width'], image['height']
                x0, x1 = max(0, min(width - 1, x0)), max(1, min(width, x1))
                y0, y1 = max(0, min(height - 1, y0)), max(1, min(height, y1))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    raise ValueError('该区域定位框太小，请先手动框选。')
                generated = 'rectangle'
                array = None
                if self.engine.sam.available():
                    try:
                        working = image_array(self.store.folder(image['id']) / 'image.png')
                        candidate, info = self.engine.sam.predict(working, box=[x0, y0, x1, y1])
                        candidate = np.asarray(candidate)
                        score = float((info or {}).get('predicted_iou') or 0)
                        if candidate.shape == (height, width) and (candidate > 127).any() and score >= .5:
                            array = (candidate > 127).astype(np.uint8) * 255
                            generated = 'sam'
                    except Exception:  # noqa: BLE001 — fall back to the confirmed box
                        array = None
                if array is None:
                    array = np.zeros((height, width), np.uint8)
                    array[y0:y1, x0:x1] = 255
                Image.fromarray(array).save(mask_path)
                metadata.update(
                    mask_status='sam' if generated == 'sam' else 'manual',
                    mask_notice='SAM 自动生成，需人工核对。' if generated == 'sam'
                    else '按确认框自动生成矩形范围，可再用画笔修正。')
                return generated, metadata

            defects, generated, skipped = [], [], []
            for proposal_id in ids:
                proposal = self.store.get('proposal', proposal_id)
                if not proposal or proposal.get('image_id') != image['id']:
                    skipped.append({'proposal_id': proposal_id, 'reason': '缺损记录不存在或不属于当前图像'})
                    continue
                try:
                    kind, metadata = ensure_range(proposal)
                    mask = mask_array(self.store.folder(proposal_id) / 'mask.png')
                    self.store.update('proposal', proposal_id, metadata=metadata, confirmed=True,
                                      fraction=round(float((mask > 0).mean()), 6))
                    proposal = set_identification(self.store, self.store.get('proposal', proposal_id),
                                                  'confirmed', '一键确认：识别与修复范围已生成')
                    if kind == 'kept':
                        from .review_workflow import record_event
                        record_event(self.store, image_id=image['id'], proposal_id=proposal_id,
                                     action='auto_confirm', changes={'mask': 'kept'},
                                     decision='confirmed', note='一键确认（沿用已有修复范围）')
                    else:
                        bump_revision(self.store, proposal_id, '自动确认并生成修复范围',
                                      changes={'mask': kind}, action='auto_confirm',
                                      note='一键确认并生成修复范围')
                    proposal = self.store.get('proposal', proposal_id)
                    defects.append(self._local_defect(proposal))
                    generated.append(kind)
                except ValueError as exc:
                    skipped.append({'proposal_id': proposal_id, 'reason': str(exc)[:200]})
            return {'defects': defects, 'generated': generated, 'skipped': skipped}
        return self._result(action)

    @Slot(str, result=str)
    def acceptDetectedRegion(self, request_json):
        """Turn one AI box (even without a stored region) into a ready repair region."""
        def action():
            request = json.loads(request_json)
            image = self.store.get('image', request.get('image_id'))
            if not image:
                raise ValueError('图像记录不存在。')
            proposal_id = request.get('proposal_id')
            if proposal_id:
                proposal = self.store.get('proposal', proposal_id)
                if not proposal or proposal.get('image_id') != image['id']:
                    raise ValueError('缺损记录不存在或不属于当前图像。')
            else:
                try:
                    bbox = [int(value) for value in request.get('bbox') or []]
                except (TypeError, ValueError) as exc:
                    raise ValueError('AI 区域框无效。') from exc
                if len(bbox) != 4:
                    raise ValueError('AI 区域框无效。')
                x0, y0, x1, y1 = [max(0, min(1000, value)) for value in bbox]
                x0, x1 = min(x0, x1), max(x0, x1)
                y0, y1 = min(y0, y1), max(y0, y1)
                if x1 - x0 < 3 or y1 - y0 < 3:
                    raise ValueError('AI 区域框太小，请手动框选。')
                width, height = image['width'], image['height']
                metadata = {
                    'type': str(request.get('type') or 'other')[:40],
                    'bbox': [x0, y0, x1, y1],
                    'bbox_pixels': [x0 * width // 1000, y0 * height // 1000,
                                    min(width, (x1 * width + 999) // 1000),
                                    min(height, (y1 * height + 999) // 1000)],
                    'evidence': str(request.get('evidence') or 'AI 标出的疑似病害区域')[:500],
                    'recommendation': request.get('recommendation') or {'method': 'LaMa 纹理补全', 'backend': 'lama'},
                    'model_source': str(request.get('model_source') or 'AI 检测框')[:80],
                    'local_detection': True, 'mask_status': '',
                }
                proposal_id = new_id()
                self.store.put('proposal', {
                    'id': proposal_id, 'image_id': image['id'], 'created_at': timestamp(),
                    'fraction': 0.0, 'confirmed': False, 'mask_source': 'manual',
                    'region_class': 'fillable_lacuna', 'revision': 1,
                    'metadata': metadata,
                })
            try:
                result = self.autoConfirmDefect(json.dumps({
                    'image_id': image['id'], 'proposal_ids': [proposal_id]}))
                payload = json.loads(result)
                if payload.get('error'):
                    raise ValueError(payload['error'])
                if not payload.get('defects'):
                    raise ValueError((payload.get('skipped') or [{}])[0].get('reason', '未能确认该区域。'))
                return {'defect': payload['defects'][0], 'generated': payload['generated'][0],
                        'proposal_id': proposal_id}
            except ValueError:
                if not request.get('proposal_id'):
                    self.store.delete('proposal', proposal_id)
                raise
        return self._result(action)

    @Slot(str, result=str)
    def composeFinalRepair(self, request_json):
        """Merge every finished region repair into one full-image result."""
        def action():
            request = json.loads(request_json)
            image = self.store.get('image', request.get('image_id'))
            if not image:
                raise ValueError('图像记录不存在。')
            import numpy as np
            from .imaging import image_array, make_difference, sha256
            from . import review_workflow
            folder = self.store.folder(image['id'])
            original = image_array(folder / 'image.png')
            height, width = original.shape[:2]
            composite = original.copy()
            union = np.zeros((height, width), bool)
            rows, skipped = [], []
            jobs = self.store.list('job')
            proposals = [proposal for proposal in self.store.list('proposal')
                         if proposal.get('image_id') == image['id']
                         and (proposal.get('metadata') or {}).get('type')]
            proposals.sort(key=lambda proposal: proposal.get('created_at') or '')
            for proposal in proposals:
                mask_path = self.store.folder(proposal['id']) / 'mask.png'
                if not mask_path.is_file():
                    continue
                mask = mask_array(mask_path)
                if mask.shape != (height, width) or not (mask > 0).any():
                    skipped.append({'proposal_id': proposal['id'], 'reason': '修复范围不可用'})
                    continue
                candidates = [job for job in jobs if job.get('defect_proposal_id') == proposal['id']
                              and job.get('status') == 'review' and job.get('candidates')]
                if not candidates:
                    skipped.append({'proposal_id': proposal['id'], 'reason': '还没有完成的修复候选'})
                    continue
                job = candidates[0]
                candidate = next((item for item in job['candidates'] if item['id'] == job.get('selected_id')),
                                 job['candidates'][0])
                stale, stale_reason = review_workflow.candidate_stale(self.store, job, candidate)
                if stale:
                    skipped.append({'proposal_id': proposal['id'], 'reason': stale_reason})
                    continue
                result_path = self.store.folder(candidate['id']) / 'result.png'
                if not result_path.is_file():
                    skipped.append({'proposal_id': proposal['id'], 'reason': '修复结果文件缺失'})
                    continue
                repaired = np.asarray(Image.open(result_path).convert('RGB'))
                if repaired.shape != original.shape:
                    skipped.append({'proposal_id': proposal['id'], 'reason': '修复结果尺寸不一致'})
                    continue
                active = mask > 0
                composite[active] = repaired[active]
                union |= active
                metadata = proposal.get('metadata') or {}
                rows.append({
                    'proposal_id': proposal['id'],
                    'type': metadata.get('type'),
                    'candidate_id': candidate['id'],
                    'job_id': job['id'],
                    'tool': candidate.get('tool'),
                    'review_status': review_workflow.candidate_state(candidate),
                    'accepted': review_workflow.candidate_state(candidate) in {'accepted', 'accepted_hypothesis'},
                })
            if not rows:
                raise ValueError('还没有可以合并的修复结果；请先生成并完成至少一个区域的修复候选。')
            difference = make_difference(original, composite)
            Image.fromarray(composite).save(folder / 'final-repair.png')
            Image.fromarray(difference).save(folder / 'final-difference.png')
            outside_unchanged = bool(np.array_equal(composite[~union], original[~union]))
            checks = {
                'size_match': composite.shape[:2] == (height, width),
                'outside_unchanged': outside_unchanged,
                'original_unchanged': sha256(folder / 'image.png') == image['working_sha256'],
            }
            summary = {
                'image_id': image['id'], 'created_at': timestamp(), 'regions': rows,
                'skipped': skipped, 'checks': checks,
                'accepted_regions': sum(1 for row in rows if row['accepted']),
            }
            (folder / 'final-repair.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                      encoding='utf-8')
            return {
                'file_url': self._file_url(folder / 'final-repair.png'),
                'difference_url': self._file_url(folder / 'final-difference.png'),
                'regions': rows, 'skipped': skipped, 'checks': checks,
                'accepted_regions': summary['accepted_regions'],
            }
        return self._result(action)

    @Slot(str, result=str)
    def saveFinalRepair(self, request_json):
        """Save the merged final image (optional explicit path for automation)."""
        def action():
            request = json.loads(request_json)
            image = self.store.get('image', request.get('image_id'))
            if not image:
                raise ValueError('图像记录不存在。')
            source = self.store.folder(image['id']) / 'final-repair.png'
            if not source.is_file():
                raise ValueError('还没有生成修复后总图，请先生成。')
            path = request.get('path')
            if not path:
                path, _ = QFileDialog.getSaveFileName(None, '保存修复后总图', 'final-repair.png',
                                                      'PNG 图像 (*.png)')
            if not path:
                return {'saved': False}
            shutil.copy2(source, path)
            self.notify.emit('修复后总图已保存。')
            return {'saved': True, 'path': str(path)}
        return self._result(action)

    @Slot(str, result=str)
    def reviewCandidate(self, request_json):
        """Stage 6: accept / reject / expert-review / preview one repair candidate."""
        def action():
            request = json.loads(request_json)
            job = self.store.get('job', request.get('job_id'))
            if not job:
                raise ValueError('修复任务不存在。')
            from .review_workflow import review_candidate
            job, candidate, event = review_candidate(
                self.store, job, request.get('candidate_id'), request.get('action') or '',
                note=str(request.get('note') or ''),
                archive_as_hypothesis=bool(request.get('archive_as_hypothesis')))
            self.current_image_id = job['image_id']
            return {'job': self._job(job), 'candidate': self._candidate(candidate), 'event': event}
        return self._result(action)

    @Slot(str, result=str)
    def listReviewEvents(self, image_id):
        """Stage 6 audit trail for one image, newest first."""
        def action():
            if not self.store.get('image', image_id):
                raise ValueError('图像记录不存在。')
            from .review_workflow import list_events
            return {'events': list(reversed(list_events(self.store, image_id)))[:100]}
        return self._result(action)

    @Slot(str, result=str)
    def saveResult(self, candidate_id):
        def action():
            job = next((j for j in self.store.list('job') if any(c.get('id') == candidate_id for c in j.get('candidates') or [])), None)
            if not job:
                raise ValueError('修复候选不存在。')
            if job.get('selected_id') != candidate_id:
                raise ValueError('请先在候选对照中选择并确认该候选，再保存修复图像。')
            candidate = next(c for c in job['candidates'] if c['id'] == candidate_id)
            from .review_workflow import export_guard
            selection = export_guard(self.store, job, candidate)
            if selection['selected_as'] in {'rejected', 'hypothesis'}:
                raise ValueError('驳回或假设性结果不能保存为修复图像；请改用「导出修复档案」并标注为假设。')
            path, _ = QFileDialog.getSaveFileName(None, '保存修复结果', 'restored.png', 'PNG 图像 (*.png)')
            if path:
                shutil.copy2(self.store.folder(candidate_id) / 'result.png', path)
                self.notify.emit('结果已保存。')
            return {'saved': bool(path)}
        return self._result(action)

    @Slot(str, result=str)
    def exportArchive(self, job_id):
        def action():
            job = self.store.get('job', job_id)
            if not job:
                raise ValueError('修复任务不存在。')
            path, _ = QFileDialog.getSaveFileName(None, '导出修复档案', f'mural-{job_id[:8]}.zip', 'ZIP 档案 (*.zip)')
            if path:
                source = self.engine.export(job_id)
                if Path(path).resolve() != source.resolve():
                    shutil.copy2(source, path)
                self.notify.emit('修复档案已导出。')
            return {'saved': bool(path)}
        return self._result(action)

    @Slot(result=str)
    def importMask(self):
        def action():
            path, _ = QFileDialog.getOpenFileName(None, '导入选区', '', '图像 (*.png *.jpg *.jpeg)')
            return {'mask_data': base64.b64encode(Path(path).read_bytes()).decode()} if path else None
        return self._result(action)

    @Slot(str, result=str)
    def setCurrentImage(self, image_id):
        def action():
            if not self.store.get('image', image_id):
                raise ValueError('图像记录不存在。')
            self.current_image_id = image_id
            return {'image_id': image_id}
        return self._result(action)

    @Slot(str, result=str)
    def proposeDamage(self, image_id):
        def action():
            self.current_image_id = image_id
            image = image_array(self.store.folder(image_id) / 'image.png')
            array, metadata = damage_proposal(image)
            return self._save_proposal(image_id, array, metadata)
        return self._result(lambda: self._aux(action, '分析细线候选…'))

    @Slot(str, str, str, str, result=str)
    def runSam(self, image_id, points_json, labels_json, box_json):
        def action():
            self.current_image_id = image_id
            points, labels, box = json.loads(points_json), json.loads(labels_json), json.loads(box_json)
            array = image_array(self.store.folder(image_id) / 'image.png')
            mask, metadata = self.engine.sam.predict(array, points, labels, box)
            return self._save_proposal(image_id, mask, metadata)
        return self._result(lambda: self._aux(action, 'SAM 分割中…'))

    @Slot(str, result=str)
    def assessDamage(self, image_id):
        def action():
            image = self.store.get('image', image_id)
            if not image:
                raise ValueError('图像记录不存在。')
            from .damage_assessment import assess_image, recommendation_for
            result = assess_image(self.engine, image)
            items = []
            for item in result.get('items', []):
                enriched = dict(item)
                enriched['recommendation'] = recommendation_for(item)
                items.append(enriched)
            return {'items': items, 'limitations': result.get('limitations', '')}
        return self._result(lambda: self._aux(action, '本地视觉模型识别缺损…'))

    @Slot(str, str, result=str)
    def damagePreview(self, image_id, item_json):
        def action():
            if not self.store.get('image', image_id):
                raise ValueError('图像记录不存在。')
            bbox = [int(value) for value in json.loads(item_json).get('bbox', [])]
            if len(bbox) != 4:
                raise ValueError('候选框无效。')
            with Image.open(self.store.folder(image_id) / 'image.png') as source:
                preview = source.convert('RGB').copy()
            width, height = preview.size
            x0, y0, x1, y1 = bbox
            box = (round(x0 * width / 1000), round(y0 * height / 1000),
                   round(x1 * width / 1000), round(y1 * height / 1000))
            from PIL import ImageDraw
            ImageDraw.Draw(preview).rectangle(box, outline=(230, 70, 52), width=max(2, min(width, height) // 350))
            target = self.store.folder(image_id) / 'damage-preview.png'
            preview.save(target)
            return {'file_url': self._file_url(target)}
        return self._result(action)

    @Slot(str, str, result=str)
    def refineDamage(self, image_id, item_json):
        def action():
            if not self.store.get('image', image_id):
                raise ValueError('图像记录不存在。')
            self.current_image_id = image_id
            item = json.loads(item_json)
            bbox = [int(value) for value in item.get('bbox', [])]
            if len(bbox) != 4:
                raise ValueError('候选框无效。')
            array = image_array(self.store.folder(image_id) / 'image.png')
            height, width = array.shape[:2]
            x0, y0, x1, y1 = bbox
            pad_x, pad_y = max(3, round(width * .005)), max(3, round(height * .005))
            box = [max(0, round(x0 * width / 1000) - pad_x), max(0, round(y0 * height / 1000) - pad_y),
                   min(width, round(x1 * width / 1000) + pad_x), min(height, round(y1 * height / 1000) + pad_y)]
            if item.get('type') == 'crack':
                from .segmentation import crack_mask_proposal
                mask, metadata = crack_mask_proposal(array, box)
            else:
                mask, metadata = self.engine.sam.predict(array, box=box)
            from .damage_assessment import recommendation_for
            metadata.update({key: item[key] for key in ('index', 'type', 'size', 'confidence', 'evidence', 'bbox') if key in item})
            metadata['recommendation'] = recommendation_for(item)
            return self._save_proposal(image_id, mask, metadata)
        return self._result(lambda: self._aux(action, '细化缺损修复范围…'))

    @Slot(str, result=str)
    def adoptProposal(self, proposal_id):
        def action():
            proposal = self.store.get('proposal', proposal_id)
            if not proposal or proposal['image_id'] != self.current_image_id:
                raise ValueError('辅助选区不属于当前图像。')
            return proposal
        return self._result(action)

    @Slot(str, result=str)
    def autoRestore(self, image_id):
        def action():
            from .professional import WORKFLOW_PROFESSIONAL, normalize_workflow

            # The workbench always posts a JSON payload. A bare image ID keeps the
            # older local autonomous workflow for callers that still use it.
            legacy_auto = not image_id.lstrip().startswith('{')
            payload = {'image_id': image_id, 'workflow': 'standard', 'provider': 'local'} if legacy_auto else json.loads(image_id)
            identifier = payload.get('image_id') or payload.get('id')
            image = self.store.get('image', identifier)
            if not image:
                raise ValueError('图像记录不存在。')
            provider = payload.get('provider', 'deepseek')
            api_key = ''
            if provider == 'deepseek':
                api_key = self.settings.deepseek_api_key
                if not api_key:
                    raise ValueError('请在 API 设置保存 DeepSeek 密钥，或切换本地模型/手动圈选。')
                workflow = WORKFLOW_PROFESSIONAL
            elif provider == 'local':
                workflow = normalize_workflow(payload.get('workflow', WORKFLOW_PROFESSIONAL))
                if legacy_auto and not (self.settings.vision_model / 'provenance.json').exists():
                    raise ValueError('本地 Qwen3-VL 权重未安装，可切换 DeepSeek 云端或手动圈选。')
            else:
                raise ValueError('不支持的视觉分析来源。')

            self.current_image_id = identifier
            if provider == 'local' and not legacy_auto:
                return self._aux(lambda: analyze_local(image, workflow),
                                 '本地 Qwen3-VL 识别缺损…', image_id=identifier)
            return self._aux(lambda: analyze(image, provider, workflow, api_key),
                             'AI 正在分析缺损与文化线索…', image_id=identifier)

        def analyze_local(image, workflow):
            from .auto_restore import context_instructions

            setup = local_model_setup(self.settings)
            if not setup['qwen_ready']:
                # Never fall back to the cloud silently: report the missing model
                # and keep the manual workflow available.
                return {'auto': {'image_id': image['id'], 'assessment': None, 'mask': None, 'job': None,
                                 'items': [], 'steps': [], 'defects': [], 'backend': '',
                                 'summary': f'{setup["qwen_hint"]}仍可手动圈选缺损区域。',
                                 'cultural_context': {}, 'planning_notes': '',
                                 'workflow': workflow, 'awaiting_confirmation': False, 'proposals': []}}
            assessment, entries = detect_local(self.store, self.engine, image)
            defects, proposals = self._persist_local_defects(image, entries)
            masked = sum(1 for entry in entries if entry['mask_status'] == 'sam')
            if defects:
                summary = (f'本地 Qwen3-VL 识别到 {len(defects)} 处疑似病害，其中 {masked} 处已由 SAM 2.1 '
                           '生成修复范围候选；其余保留候选框，请逐项核对后确认。')
            else:
                summary = '本地 Qwen3-VL 未发现明确缺损；可手动圈选缺损区域。'
            context = assessment.cultural_context.model_dump(mode='json') if assessment.cultural_context else {}
            return {'auto': {'image_id': image['id'], 'assessment': assessment.model_dump(mode='json'),
                             'mask': None, 'job': None, 'items': [], 'steps': [],
                             'defects': defects, 'backend': '', 'summary': summary,
                             'cultural_context': context,
                             'planning_notes': context_instructions(context),
                             'workflow': workflow, 'awaiting_confirmation': bool(defects),
                             'proposals': proposals}}

        def analyze(image, provider, workflow, api_key):
            from .auto_restore import auto_restore, context_instructions

            if provider == 'deepseek':
                with Image.open(self.store.folder(image['id']) / 'image.png') as source:
                    assessment = DeepSeekVisionProvider(api_key).analyze(source)
                defects = cloud_defects(assessment)
                context = assessment.cultural_context.model_dump(mode='json') if assessment.cultural_context else {}
                summary = (f'DeepSeek 给出 {len(defects)} 处疑似缺损。请核对原图并手动圈选后再修补。'
                           if defects else 'DeepSeek 未发现明确缺损。请检查原图，可手动圈选。')
                result = {'items': [], 'defects': defects, 'steps': [], 'backend': '', 'summary': summary,
                          'cultural_context': context, 'planning_notes': context_instructions(context),
                          'awaiting_confirmation': bool(defects)}
            else:
                result = auto_restore(self.store, self.engine, image, workflow=workflow)
                assessment = local_assessment(self.store, image, result)
            mask = dict(result['mask']) if result.get('mask') else None
            if mask:
                mask['file_url'] = self._file_url(self.store.folder(mask['id']) / 'mask.png')
            job = result.get('job')
            return {'auto': {'image_id': image['id'], 'assessment': assessment.model_dump(mode='json'),
                             'mask': mask, 'job': self._job(job) if job else None,
                             'items': result['items'], 'steps': result['steps'],
                             'defects': result.get('defects') or [],
                             'backend': result['backend'], 'summary': result['summary'],
                             'cultural_context': result.get('cultural_context') or {},
                             'planning_notes': result.get('planning_notes') or '',
                             'workflow': result.get('workflow', workflow),
                             'awaiting_confirmation': bool(result.get('awaiting_confirmation')),
                             'proposals': result.get('proposals') or []}}
        return self._result(action)

    @Slot(str, result=str)
    def listLocalDefects(self, image_id):
        """All persisted local defects of one image, in creation order."""
        def action():
            if not self.store.get('image', image_id):
                raise ValueError('图像记录不存在。')
            records = [proposal for proposal in self.store.list('proposal')
                       if proposal.get('image_id') == image_id
                       and (proposal.get('metadata') or {}).get('local_detection')]
            records.sort(key=lambda proposal: proposal.get('created_at') or '')
            defects = []
            for index, record in enumerate(records, 1):
                defect = self._local_defect(record)
                defect['index'] = index
                defects.append(defect)
            return {'defects': defects}
        return self._result(action)

    @Slot(str, str, result=str)
    def editLocalDefect(self, image_id, payload_json):
        """Confirm, re-mask, re-box or delete one local defect."""
        def action():
            image = self.store.get('image', image_id)
            if not image:
                raise ValueError('图像记录不存在。')
            payload = json.loads(payload_json or '{}')
            action_name = payload.get('action')
            if action_name == 'update' and not payload.get('proposal_id'):
                return {'defect': self._new_local_defect(image, payload)}
            proposal_id = payload.get('proposal_id')
            if not proposal_id:
                raise ValueError('缺少缺损项编号。')
            proposal = self.store.get('proposal', proposal_id)
            if not proposal:
                raise ValueError('缺损项不存在。')
            if proposal['image_id'] != image['id']:
                raise ValueError('缺损项不属于当前图像。')
            from .review_workflow import (bump_revision, mark_stale, record_event,
                                          revision, set_identification)
            note = str(payload.get('note') or '')[:2000]
            if action_name == 'remove':
                mark_stale(self.store, proposal_id, '病害区域已删除')
                record_event(self.store, image_id=image['id'], proposal_id=proposal_id,
                             action='region_remove', changes={'revision': revision(proposal)},
                             decision='removed', note=note)
                self.store.delete('proposal', proposal_id)
                return {'removed': True, 'proposal_id': proposal_id}
            if action_name == 'confirm':
                proposal = set_identification(self.store, proposal, 'confirmed', note)
                record_event(self.store, image_id=image['id'], proposal_id=proposal_id,
                             action='identification_confirm', decision='confirmed', note=note)
                return {'defect': self._local_defect(proposal)}
            if action_name == 'expert_review':
                proposal = set_identification(self.store, proposal, 'expert_review', note)
                record_event(self.store, image_id=image['id'], proposal_id=proposal_id,
                             action='identification_expert_review', decision='expert_review', note=note)
                return {'defect': self._local_defect(proposal)}
            if action_name == 'recommendation':
                method = str(payload.get('method') or '').strip()[:200]
                if not method:
                    raise ValueError('请填写或选择修复建议。')
                metadata = dict(proposal.get('metadata') or {})
                recommendation = dict(metadata.get('recommendation') or {})
                recommendation['method'] = method
                metadata['recommendation'] = recommendation
                self.store.update('proposal', proposal_id, metadata=metadata)
                bump_revision(self.store, proposal_id, '修复建议已人工修改',
                              changes={'method': method}, action='recommendation_update', note=note)
                return {'defect': self._local_defect(self.store.get('proposal', proposal_id))}
            if action_name != 'update':
                raise ValueError('不支持的缺损编辑操作。')
            changed = {'mask': bool(payload.get('mask_data')), 'bbox': bool(payload.get('bbox')),
                       'type': bool(payload.get('type'))}
            metadata = dict(proposal.get('metadata') or {})
            fraction = proposal.get('fraction')
            mask_source = proposal.get('mask_source')
            if action_name == 'update':
                if payload.get('mask_data'):
                    import numpy as np
                    mask = decode_image(base64.b64decode(payload['mask_data'], validate=True), mask=True)
                    if mask.size != (image['width'], image['height']):
                        raise ValueError('修复范围尺寸必须与原图一致。')
                    array = (np.asarray(mask) > 127).astype(np.uint8) * 255
                    if not array.any():
                        raise ValueError('修复范围不能为空，请先圈选要修复的像素。')
                    Image.fromarray(array).save(self.store.folder(proposal_id) / 'mask.png')
                    fraction = round(float(array.mean()), 6)
                    mask_source = 'manual'
                    metadata.update(mask_status='manual', mask_notice='人工修正的修复范围，需复核。')
                if payload.get('bbox'):
                    try:
                        bbox = [int(value) for value in payload['bbox']]
                    except (TypeError, ValueError) as exc:
                        raise ValueError('缺损框无效。') from exc
                    if len(bbox) != 4:
                        raise ValueError('缺损框无效。')
                    x0, y0, x1, y1 = [max(0, min(1000, value)) for value in bbox]
                    x0, x1 = min(x0, x1), max(x0, x1)
                    y0, y1 = min(y0, y1), max(y0, y1)
                    if x1 - x0 < 3 or y1 - y0 < 3:
                        raise ValueError('缺损框太小，请重新框选。')
                    width, height = image['width'], image['height']
                    metadata['bbox'] = [x0, y0, x1, y1]
                    metadata['bbox_pixels'] = [
                        x0 * width // 1000, y0 * height // 1000,
                        min(width, max(x0 * width // 1000 + 1, (x1 * width + 999) // 1000)),
                        min(height, max(y0 * height // 1000 + 1, (y1 * height + 999) // 1000))]
                if payload.get('type'):
                    metadata['type'] = str(payload['type'])[:40]
            proposal = self.store.update('proposal', proposal_id, metadata=metadata, confirmed=True,
                                         fraction=fraction, mask_source=mask_source)
            if any(changed.values()):
                # Any manual region change invalidates candidates made from the old mask.
                bump_revision(self.store, proposal_id, '修复范围或区域已人工修改',
                              changes={key: value for key, value in changed.items() if value},
                              action='region_update', note=note)
                proposal = self.store.get('proposal', proposal_id)
            return {'defect': self._local_defect(proposal)}
        return self._result(action)

    @Slot(str, result=str)
    def generateRegionRepairs(self, request_json):
        """Stage 4: create one LaMa/OpenCV candidate per confirmed defect mask."""
        def action():
            request = json.loads(request_json)
            if self.aux_future and not self.aux_future.done():
                raise ValueError('AI 分析仍在运行，请完成后再生成修复候选。')
            image = self.store.get('image', request.get('image_id'))
            if not image:
                raise ValueError('图像记录不存在。')
            active = [job for job in self.store.list('job')
                      if job.get('image_id') == image['id']
                      and job.get('status') in {'queued', 'planning', 'running', 'evaluating'}]
            if active:
                raise ValueError('该图像已有修复任务在运行，请完成后再生成新的候选。')
            from .region_repair import run_region_candidates
            result = run_region_candidates(
                self.store, self.engine, image, request.get('proposal_ids') or [],
                instructions=str(request.get('instructions') or '')[:2000])
            self.current_image_id = image['id']
            result['jobs'] = [self._job(job) for job in result['jobs']]
            return result
        return self._result(action)

    @Slot(str, result=str)
    def exportObservationCard(self, request_json):
        """Stage 7: one-page AI observation card PDF (optional explicit path for automation)."""
        def action():
            request = json.loads(request_json)
            image = self.store.get('image', request.get('image_id'))
            if not image:
                raise ValueError('图像记录不存在。')
            path = request.get('path')
            if not path:
                path, _ = QFileDialog.getSaveFileName(None, '导出 AI 壁画观察卡',
                                                      'mural-observation-card.pdf', 'PDF 文件 (*.pdf)')
            if not path:
                return {'saved': False}
            from .competition_output import build_observation_card
            result = build_observation_card(self.store, image, Path(path))
            self.notify.emit('AI 壁画观察卡已导出。')
            return {'saved': True, **result}
        return self._result(action)

    @Slot(str, result=str)
    def exportRepairReport(self, request_json):
        """Stage 7: optional two-page repair report PDF."""
        def action():
            request = json.loads(request_json)
            job = self.store.get('job', request.get('job_id'))
            if not job:
                raise ValueError('修复任务不存在。')
            image = self.store.get('image', job['image_id'])
            path = request.get('path')
            if not path:
                path, _ = QFileDialog.getSaveFileName(None, '导出修复成果说明',
                                                     f'mural-repair-{job["id"][:8]}.pdf', 'PDF 文件 (*.pdf)')
            if not path:
                return {'saved': False}
            from .competition_output import build_repair_report
            result = build_repair_report(self.store, image, job, Path(path))
            self.notify.emit('修复成果说明已导出。')
            return {'saved': True, **result}
        return self._result(action)

    @Slot(str, result=str)
    def demoStatus(self, image_id):
        """Honest demo guide: report which real steps are already complete."""
        def action():
            image = self.store.get('image', image_id)
            if not image:
                raise ValueError('图像记录不存在。')
            proposals = [proposal for proposal in self.store.list('proposal')
                         if proposal.get('image_id') == image['id']]
            jobs = [job for job in self.store.list('job') if job.get('image_id') == image['id']]
            assessments = [record for record in self.store.list('assessment')
                           if record.get('image_id') == image['id']]
            candidate_reviewed = any(
                (candidate.get('review') or {}).get('status') for job in jobs
                for candidate in job.get('candidates') or [])
            steps = [
                {'key': 'import', 'label': '导入有明确许可的壁画图像', 'done': True,
                 'hint': '样例素材与许可见 图片/来源与许可.md。'},
                {'key': 'identify', 'label': '本地 Qwen3-VL 识别疑似病害', 'done': bool(proposals),
                 'hint': '点击“AI 分析缺损与文化线索”。'},
                {'key': 'mask', 'label': 'SAM 生成或人工修正修复范围', 'done': any(
                    (proposal.get('metadata') or {}).get('mask_status') for proposal in proposals),
                 'hint': '在缺损清单中确认区域、修正修复范围。'},
                {'key': 'repair', 'label': '本地 LaMa/OpenCV 生成修复候选', 'done': any(
                    job.get('candidates') for job in jobs),
                 'hint': '点“生成修复候选（本地补全）”。'},
                {'key': 'assess', 'label': 'DeepSeek 修复评估（可选）', 'done': bool(assessments),
                 'hint': '结果页点“AI 修复评估”，需本机密钥。'},
                {'key': 'review', 'label': '人工审核并接受/驳回候选', 'done': candidate_reviewed,
                 'hint': '选择候选后点“确认复核决定”或标记专家复核。'},
                {'key': 'export', 'label': '导出观察卡与修复说明 PDF', 'done': False,
                 'hint': '点击“导出 AI 观察卡 PDF”或“导出修复说明 PDF”。'},
            ]
            return {'steps': steps, 'done': sum(step['done'] for step in steps), 'total': len(steps)}
        return self._result(action)

    @Slot(str, result=str)
    def assessRepair(self, request_json):
        """Stage 5: advisory DeepSeek review of existing local repair results."""
        def action():
            request = json.loads(request_json)
            image = self.store.get('image', request.get('image_id'))
            if not image:
                raise ValueError('图像记录不存在。')
            if not self.settings.deepseek_api_key:
                raise ValueError('请先在 API 设置保存 DeepSeek 密钥，再使用 AI 修复评估。')
            job = self.store.get('job', request['job_id']) if request.get('job_id') else None
            if request.get('job_id') and not job:
                raise ValueError('修复任务不存在。')
            if job and job['image_id'] != image['id']:
                raise ValueError('修复任务不属于当前图像。')
            from . import repair_assessment
            cases = repair_assessment.collect_cases(self.store, image, (job or {}).get('id'))
            if not cases:
                raise ValueError('请先确认缺损区域并生成修复范围，再请求 AI 修复评估。')
            if not any(case['candidate'] for case in cases):
                raise ValueError('还没有本地修复候选可供评估，请先生成修复候选。')
            context = str(request.get('context') or '')[:800]

            def run_assessment():
                from PIL import Image
                from .deepseek_vision import DeepSeekVisionProvider
                from . import repair_assessment as analysis
                original, composite = analysis.compose_repaired(self.store, image, cases)
                regions, crops = [], []
                for case in cases:
                    regions.append({
                        'region_id': case['region_id'], 'type': case['type'],
                        'type_label': analysis.type_label(case['type']),
                        'bbox_0_1000': case['bbox_0_1000'], 'method': case['method'],
                        'mask_fraction': round(case['mask_fraction'], 4),
                        'has_candidate': bool(case['candidate']),
                        'local_evidence': (case['evidence'] or '')[:200],
                    })
                    if case['candidate'] and len(crops) < analysis.MAX_CROPS:
                        pair = analysis.crop_pair(self.store, image, case['job'], case['candidate'], case['mask'])
                        if pair:
                            crops.append({'region_id': case['region_id'], 'before': pair[0], 'after': pair[1]})
                provider = DeepSeekVisionProvider(self.settings.deepseek_api_key, timeout=90)
                assessment = provider.assess_repair(
                    Image.fromarray(original), Image.fromarray(composite), regions,
                    context=context, crops=crops)
                rows = analysis.merge(assessment, cases, original, context)
                record = analysis.assessment_record(
                    self.store, image, job or (cases[0]['job'] if cases[0]['job'] else None),
                    cases, assessment, rows, context)
                return {'assessment': record}

            started = self._aux(run_assessment, 'DeepSeek 正在评估修复结果…', image_id=image['id'])
            self.current_image_id = image['id']
            return started
        return self._result(action)

    @Slot(str, result=str)
    def runDefectRepairs(self, request_json):
        def action():
            request = json.loads(request_json)
            if self.aux_future and not self.aux_future.done():
                raise ValueError('AI 分析仍在运行，请完成后再启动修复。')
            image_id = request.get('image_id')
            if not self.store.get('image', image_id):
                raise ValueError('图像记录不存在。')
            requested = list(dict.fromkeys(request.get('proposal_ids') or []))
            confirmed_ids, skipped = [], []
            for proposal_id in requested:
                proposal = self.store.get('proposal', proposal_id)
                if not proposal or proposal.get('image_id') != image_id:
                    skipped.append({'proposal_id': proposal_id, 'reason': '缺损记录不存在或不属于当前图像'})
                elif not proposal.get('confirmed'):
                    skipped.append({'proposal_id': proposal_id, 'reason': '该缺损项尚未人工确认，请先在清单中确认'})
                elif not (self.store.folder(proposal_id) / 'mask.png').is_file():
                    skipped.append({'proposal_id': proposal_id, 'reason': '该缺损项还没有修复范围，请先手动圈选或修正修复范围'})
                else:
                    confirmed_ids.append(proposal_id)
            if not confirmed_ids:
                return {'jobs': [], 'skipped': skipped}
            from .defect_jobs import run_confirmed_defects
            result = run_confirmed_defects(
                self.store, self.engine, image_id, confirmed_ids,
                request.get('methods'), generative_attempt=bool(request.get('generative_attempt')))
            result['jobs'] = [self._job(job) for job in result['jobs']]
            result['skipped'] = skipped + list(result.get('skipped') or [])
            return result
        return self._result(action)

    @Slot(str, str, result=str)
    def mergeProposal(self, proposal_id, current_mask):
        def action():
            proposal = self.store.get('proposal', proposal_id)
            if not proposal or proposal['image_id'] != self.current_image_id:
                raise ValueError('辅助选区不属于当前图像。')
            incoming = mask_array(self.store.folder(proposal_id) / 'mask.png') > 0
            from .imaging import decode_image
            import numpy as np
            current = np.asarray(decode_image(base64.b64decode(current_mask), mask=True)) > 127
            if current.shape != incoming.shape:
                raise ValueError('当前修复范围与辅助选区尺寸不一致。')
            merged = incoming | current
            identifier = new_id()
            Image.fromarray(merged.astype('uint8') * 255).save(self.store.folder(identifier) / 'mask.png')
            result = self.store.put('proposal', {'id': identifier, 'image_id': proposal['image_id'],
                'created_at': timestamp(), 'fraction': float(merged.mean()),
                'metadata': {**proposal['metadata'], 'merged_from': proposal_id, 'confirmed': False}})
            url = self._file_url(self.store.folder(identifier) / 'mask.png')
            result['file_url'] = url
            result['mask_file_url'] = url
            return result
        return self._result(action)

    @Slot(str, result=str)
    def addReference(self, metadata_json):
        def action():
            metadata = json.loads(metadata_json)
            path, _ = QFileDialog.getOpenFileName(None, '导入参考图', '', '图像 (*.png *.jpg *.jpeg *.tif *.tiff)')
            return references.add_reference(self.store, Path(path).read_bytes(), Path(path).name, metadata) if path else None
        return self._result(action)

    @Slot(str, str, result=str)
    def searchReferences(self, query, group):
        def action():
            image_record = self.store.get('image', self.current_image_id) if self.current_image_id else None
            if not image_record:
                image_record = next(iter(self.store.list('image')), None)
            if not image_record:
                raise ValueError('请先打开一幅壁画，再检索参考资料。')
            image = image_array(self.store.folder(image_record['id']) / 'image.png')
            records = references.retrieve(self.store, image, query, group)
            for item in records:
                image_record = self.store.get('image', item['image_id'])
                item['file_url'] = self._file_url(self.store.folder(image_record['id']) / 'image.png')
            return records
        return self._result(lambda: self._aux(action, '检索参考资料…'))

    @Slot(str, str, str, result=str)
    def registerReference(self, image_id, reference_id, mask_data):
        def action():
            from PIL import Image
            image = self.store.get('image', image_id); reference = self.store.get('reference', reference_id)
            self.current_image_id = image_id
            if not image or not reference:
                raise ValueError('图像或参考资料不存在。')
            from .imaging import decode_image
            import numpy as np
            mask = np.asarray(decode_image(base64.b64decode(mask_data), mask=True))
            record = references.register(self.store, image, reference, mask)
            return {'registration': record}
        return self._result(lambda: self._aux(action, '参考图配准中…'))

    @Slot(str, str, str, result=str)
    def approveReference(self, registration_id, mask_data, note):
        def action():
            payload = base64.b64decode(mask_data)
            return references.approve_registration(self.store, registration_id, payload, note)
        return self._result(action)

    @Slot(result=str)
    def importDataset(self):
        def action():
            path, _ = QFileDialog.getOpenFileName(None, '导入数据清单', '', 'JSONL (*.jsonl *.txt)')
            if not path: return {'notice': '已取消。'}
            records = datasets.import_manifest(self.store, path)
            return {'notice': f'已导入 {len(records)} 条样本。'}
        return self._result(action)

    @Slot(result=str)
    def exportSplits(self):
        def action():
            path, _ = QFileDialog.getSaveFileName(None, '导出分组划分', 'splits.json', 'JSON (*.json)')
            if not path: return {'notice': '已取消。'}
            Path(path).write_text(_json(datasets.grouped_split(self.store.list('dataset'))), encoding='utf-8')
            return {'notice': '分组划分已导出。'}
        return self._result(action)

    @Slot(str, result=str)
    def evaluatePair(self, candidate_id):
        def action():
            from . import datasets
            path, _ = QFileDialog.getOpenFileName(None, '选择已对齐的配对目标', '', '图像 (*.png *.jpg *.jpeg *.tif *.tiff)')
            if not path: return {'cancelled': True}
            candidate = self.store.get('candidate', candidate_id)
            if not candidate: raise ValueError('候选记录不存在。')
            report = datasets.evaluate_pair(self.store, candidate, path, perceptual=True)
            QMessageBox.information(None, '配对评测', _json(report))
            return report
        return self._result(action)

    @Slot(str, result=str)
    def batchJobs(self, settings_json):
        def action():
            directory = QFileDialog.getExistingDirectory(None, '选择批量目录：每幅图需配套同名 .mask.png')
            if not directory: return {'cancelled': True}
            settings = json.loads(settings_json)
            if settings.get('mode', 'conservative') != 'conservative':
                raise ValueError('批量任务仅支持保守修补模式。')
            paths = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'} and '.mask.' not in p.name)
            pairs = [(path, path.with_name(path.stem + '.mask.png')) for path in paths]
            if not pairs or len(pairs) > 50 or any(not mask_path.is_file() for _, mask_path in pairs):
                raise ValueError('目录需提供 1 至 50 组图像及同名 .mask.png 文件。')
            from .imaging import decode_image
            import numpy as np
            for path, mask_path in pairs:
                image = decode_image(path.read_bytes())
                mask = decode_image(mask_path.read_bytes(), mask=True)
                pixels = np.asarray(mask) > 127
                if image.size != mask.size or not pixels.any() or pixels.all():
                    raise ValueError('无效的批量图像或修复范围：' + path.name)
            queued = []
            for path, mask_path in pairs:
                image = workspace.add_image(self.store, path.read_bytes(), path.name)
                record = workspace.add_mask(self.store, image, mask_path.read_bytes())
                job = workspace.create_job(self.store, self.engine, image, record, settings['backend'], settings['planner'],
                    settings['count'], settings.get('instructions', ''), budget={'candidates': settings['count'], 'revisions': settings['rounds'], 'timeout_seconds': settings['timeout']})
                queued.append(job['id'])
            return {'notice': f'已创建 {len(queued)} 个批量修复任务。'}
        return self._result(action)

    @Slot(result=str)
    def showModels(self):
        def action():
            QMessageBox.information(None, '模型与环境', _json(self.engine.tools.availability()))
            return {'shown': True}
        return self._result(action)

    @Slot(result=str)
    def backupWorkspace(self):
        def action():
            path, _ = QFileDialog.getSaveFileName(None, '备份工作空间', 'shibi-backup.zip', 'ZIP 档案 (*.zip)')
            return {'path': backup.backup_workspace(self.store, path)} if path else {'cancelled': True}
        return self._result(action)

    @Slot(str, result=str)
    def saveDeepSeekKey(self, key):
        def action():
            normalized = key.strip()
            if not normalized:
                raise ValueError('请填写 DeepSeek API 密钥后再保存。')
            if not valid_deepseek_key(normalized):
                raise ValueError('DeepSeek API 密钥格式无效，请检查后重试。')
            try:
                save_env_fields({'DEEPSEEK_API_KEY': normalized})
            except (OSError, UnicodeError):
                raise ValueError('无法写入本机配置，请检查项目目录的写入权限及 .env 文件编码。') from None
            self.settings.deepseek_api_key = normalized
            return {'configured': True}
        return self._result(action)

    @Slot(str, result=str)
    def saveApiSettings(self, settings_json):
        def action():
            values = json.loads(settings_json)
            kind = values.get('kind')
            base = values.get('base_url', '').strip().rstrip('/')
            model = values.get('model', '').strip()
            key = values.get('key', '')
            json_mode = bool(values.get('json_mode', True))
            if kind not in {'responses', 'chat_completions'}:
                raise ValueError('不支持的 API 协议。')
            if not base.startswith(('https://', 'http://127.0.0.1', 'http://localhost')):
                raise ValueError('请填写 HTTPS 服务根地址，或本机模型服务地址。')
            fields = {'MURAL_API_KIND': kind, 'MURAL_API_BASE_URL': base,
                      'MURAL_API_MODEL': model, 'MURAL_API_JSON_MODE': str(json_mode).lower()}
            if key: fields['MURAL_API_KEY'] = key
            imagine = values.get('imagine')
            if isinstance(imagine, dict):
                from .imagine import imagine_env_updates
                fields.update(imagine_env_updates(imagine))
            save_env_fields(fields)
            self.settings.api_kind, self.settings.api_base_url = kind, base
            self.settings.api_model, self.settings.api_json_mode = model, json_mode
            if key: self.settings.api_key = key
            if isinstance(imagine, dict):
                self.settings.imagine_provider = fields['MURAL_IMAGINE_PROVIDER']
                self.settings.imagine_base_url = fields['MURAL_IMAGINE_BASE_URL']
                self.settings.imagine_model = fields['MURAL_IMAGINE_MODEL']
                if fields.get('MURAL_IMAGINE_API_KEY'):
                    self.settings.imagine_api_key = fields['MURAL_IMAGINE_API_KEY']
            return {'saved': True}
        return self._result(action)


class WebMainWindow(QMainWindow):
    def __init__(self, settings):
        super().__init__()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(settings.data_dir)
        self.engine = Engine(settings, self.store)
        self.setWindowTitle('拾壁 · 壁画修复工作室')
        self.resize(1440, 920)
        self.setMinimumSize(980, 640)
        icon = ROOT / 'assets' / 'app.ico'
        if icon.exists(): self.setWindowIcon(QIcon(str(icon)))
        self.view = QWebEngineView(self)
        self.channel = QWebChannel(self.view.page())
        self.bridge = Bridge(settings, self.store, self.engine)
        self.channel.registerObject('bridge', self.bridge)
        self.view.page().setWebChannel(self.channel)
        self.view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self.view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        self.setCentralWidget(self.view)
        self.view.load(QUrl.fromLocalFile(str(ROOT / 'ui' / 'index.html')))
        self.closing = False
        self.close_timer = QTimer(self)
        self.close_timer.setInterval(250)
        self.close_timer.timeout.connect(self._finish_close)

    def _finish_close(self):
        active = any(j['status'] in {'queued', 'planning', 'running', 'evaluating'} for j in self.store.list('job'))
        auxiliary = self.bridge.aux_future and not self.bridge.aux_future.done()
        if not active and not auxiliary:
            self.close_timer.stop()
            self.close()

    def closeEvent(self, event):
        if self.closing:
            # Second close attempt while cancellation is still pending: leave now.
            # Every finished attempt is checkpointed in SQLite and the next start
            # marks running jobs as interrupted, so work can be resumed from history.
            try:
                self.engine.executor.shutdown(wait=False, cancel_futures=True)
                self.bridge.aux_executor.shutdown(wait=False, cancel_futures=True)
            finally:
                os._exit(0)
        active = [j for j in self.store.list('job') if j['status'] in {'queued', 'planning', 'running', 'evaluating'}]
        auxiliary = self.bridge.aux_future and not self.bridge.aux_future.done()
        if active or auxiliary:
            self.closing = True
            self.bridge.notify.emit('正在取消运行中的任务，完成后自动退出；再点一次关闭可强制退出。')
            for job in active: self.store.update('job', job['id'], cancel_requested=True)
            if auxiliary and self.bridge.aux_future:
                self.bridge.aux_future.cancel()
            self.close_timer.start()
            event.ignore()
            return
        self.engine.executor.shutdown(wait=False, cancel_futures=True)
        event.accept()


def main():
    os.chdir(ROOT)
    app = QApplication(sys.argv)
    app.setApplicationName('拾壁'); app.setOrganizationName('MuralAtelier')
    configure_application(app)
    settings = Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(settings.data_dir / 'desktop.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        QMessageBox.information(None, '拾壁', '拾壁已经在运行，请切换到已有的应用窗口。')
        return 0
    window = WebMainWindow(settings); window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
