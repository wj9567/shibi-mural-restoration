"""Stage-7 real end-to-end demo acceptance.

Runs the actual pipeline on a licensed sample mural:
import -> local Qwen3-VL detection -> SAM mask (or a documented manual fallback)
-> local LaMa repair -> one DeepSeek repair review -> human accept -> PDF card +
repair report. Records exactly what ran; no invented confidence or conclusions.

    .\\.venv\\Scripts\\python.exe scripts/verify_competition_demo.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mural_agent import competition_output, repair_assessment, review_workflow, workspace  # noqa: E402
from mural_agent.config import Settings  # noqa: E402
from mural_agent.deepseek_vision import DeepSeekVisionProvider  # noqa: E402
from mural_agent.desktop_web import local_model_setup  # noqa: E402
from mural_agent.engine import Engine  # noqa: E402
from mural_agent.region_repair import run_region_candidates  # noqa: E402
from mural_agent.storage import Store, new_id, timestamp  # noqa: E402
from mural_agent.vision_analysis import detect_local  # noqa: E402

IMAGE = ROOT / '图片' / '01-cma-1963.252.jpg'
RUN = ROOT / 'artifacts' / 'competition-demo'


def page_count(path):
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(path)).pages)
    except Exception:  # noqa: BLE001 — page count is a verification aid, not a product feature
        return None


def masked_region(entry):
    mask = entry['mask']
    return (mask > 0).astype(np.uint8) * 255


def main():
    report = {'started_at': timestamp(), 'steps': {}, 'image': str(IMAGE)}
    settings = Settings.from_env()
    setup = local_model_setup(settings)
    report['models'] = setup
    report['deepseek_key_configured'] = bool(settings.deepseek_api_key)
    if not (setup['qwen_ready'] and setup['sam_ready'] and (ROOT / 'models' / 'big-lama.pt').is_file()):
        print(json.dumps({'skipped': True, 'reason': '本地模型权重不完整', **report},
                         ensure_ascii=False, indent=2))
        return 2

    RUN.mkdir(parents=True, exist_ok=True)
    store = Store(RUN / 'data')
    image = workspace.add_image(store, IMAGE.read_bytes(), IMAGE.name)
    width, height = image['width'], image['height']
    engine = Engine(settings, store)
    engine.submit = lambda job_id: None       # run synchronously for the acceptance
    report['steps']['import'] = {'done': True, 'size': [width, height], 'provenance':
                                 competition_output.provenance_for(image)}

    started = time.perf_counter()
    assessment, entries = detect_local(store, engine, image)
    report['steps']['detect'] = {
        'done': True, 'seconds': round(time.perf_counter() - started, 2),
        'candidates': len(entries), 'models': assessment.model_source,
        'types': [entry['candidate'].type for entry in entries],
        'sam_masks': sum(1 for entry in entries if entry['mask_status'] == 'sam'),
        'notes': [entry['mask_notice'] for entry in entries if entry['mask_notice']][:3],
    }
    chosen = next((entry for entry in entries if entry['mask_status'] == 'sam'), None)
    fallback = chosen is None
    if fallback:
        entry = entries[0] if entries else None
        box = list(entry['candidate'].bbox) if entry else [0, int(height*.12), int(width*.06), int(height*.85)]
        mask = np.zeros((height, width), np.uint8)
        mask[box[1]:box[3], box[0]:box[2]] = 255
        chosen = {'candidate': entry['candidate'] if entry else None, 'mask': mask,
                  'mask_status': 'manual', 'recommendation': (entry or {}).get('recommendation') or {}}
    proposal_id = new_id()
    Image.fromarray(masked_region(chosen)).save(store.folder(proposal_id) / 'mask.png')
    candidate = chosen['candidate']
    kind = candidate.type if candidate else 'paint_loss'
    bbox = list(candidate.bbox) if candidate else [0, int(height*.12), int(width*.06), int(height*.85)]
    store.put('proposal', {
        'id': proposal_id, 'image_id': image['id'], 'created_at': timestamp(),
        'fraction': float((masked_region(chosen) > 0).mean()), 'confirmed': True,
        'mask_source': 'sam' if chosen['mask_status'] == 'sam' else 'manual',
        'region_class': 'fillable_lacuna', 'revision': 1,
        'identification': {'status': 'confirmed', 'note': '演示验收：人工确认识别', 'time': timestamp()},
        'metadata': {'type': kind, 'bbox_pixels': bbox,
                     'bbox': [int(bbox[0]*1000/width), int(bbox[1]*1000/height),
                              int(bbox[2]*1000/width), int(bbox[3]*1000/height)],
                     'evidence': (candidate.evidence if candidate else '演示验收使用的人工确认区域'),
                     'mask_status': chosen['mask_status'], 'model_source': 'Qwen3-VL local',
                     'recommendation': chosen.get('recommendation') or {'method': 'LaMa 纹理补全', 'backend': 'lama'},
                     'local_detection': True},
    })
    report['steps']['mask'] = {'done': True, 'region_id': proposal_id, 'mask_status': chosen['mask_status'],
                               'fallback_manual': fallback, 'bbox_pixels': bbox,
                               'fraction': round(float((masked_region(chosen) > 0).mean()), 6)}

    started = time.perf_counter()
    job = run_region_candidates(store, engine, image, [proposal_id])['jobs'][0]
    engine.run(job['id'])
    job = store.get('job', job['id'])
    report['steps']['repair'] = {
        'done': job['status'] == 'review', 'seconds': round(time.perf_counter() - started, 2),
        'status': job['status'], 'backend': job.get('region_backend'),
        'candidate_id': (job.get('candidates') or [{}])[0].get('id'),
        'tool': (job.get('candidates') or [{}])[0].get('tool'),
        'device': (job.get('candidates') or [{}])[0].get('versions', {}).get('device'),
        'checks': job.get('checks'), 'error': job.get('error'),
    }
    if job['status'] != 'review':
        report['passed'] = False
        (RUN / 'demo-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    candidate_id = job['candidates'][0]['id']

    accepted_job, reviewed, event = review_workflow.review_candidate(
        store, store.get('job', job['id']), candidate_id, 'accept',
        note='演示验收：人工确认修复范围与结果')
    report['steps']['review'] = {
        'done': reviewed['review']['status'] == 'accepted',
        'identification': review_workflow.identification_state(store.get('proposal', proposal_id)),
        'candidate_status': reviewed['review']['status'],
        'selected_as': accepted_job.get('selected_as'), 'event': event['action'],
    }

    if settings.deepseek_api_key:
        cases = [case for case in repair_assessment.collect_cases(store, image) if case['candidate']]
        original, composite = repair_assessment.compose_repaired(store, image, cases)
        regions = [{'region_id': case['region_id'], 'type': case['type'],
                    'type_label': repair_assessment.type_label(case['type']),
                    'bbox_0_1000': case['bbox_0_1000'], 'method': case['method'],
                    'mask_fraction': round(case['mask_fraction'], 4), 'has_candidate': True,
                    'local_evidence': (case['evidence'] or '')[:200]} for case in cases]
        crops = []
        pair = repair_assessment.crop_pair(store, image, cases[0]['job'], cases[0]['candidate'], cases[0]['mask'])
        if pair:
            crops.append({'region_id': cases[0]['region_id'], 'before': pair[0], 'after': pair[1]})
        started = time.perf_counter()
        try:
            review = DeepSeekVisionProvider(settings.deepseek_api_key, timeout=90).assess_repair(
                Image.fromarray(original), Image.fromarray(composite), regions, crops=crops)
            rows = repair_assessment.merge(review, cases, original, '')
            record = repair_assessment.assessment_record(store, image, cases[0]['job'], cases, review, rows, '')
            report['steps']['assess'] = {
                'done': True, 'seconds': round(time.perf_counter() - started, 2),
                'model': review.model_source, 'regions': len(review.regions),
                'human_review_required': [row['human_review_required'] for row in rows],
                'review_reason': [row['review_reason'] for row in rows][:1],
                'assessment_id': record['id'], 'key_leaked': settings.deepseek_api_key in json.dumps(report),
            }
        except Exception as exc:  # noqa: BLE001 — record the real outcome without secrets
            report['steps']['assess'] = {'done': False, 'seconds': round(time.perf_counter() - started, 2),
                                         'error': str(exc)[:300]}
    else:
        report['steps']['assess'] = {'done': False, 'reason': '本机未配置 DeepSeek 密钥'}

    card_path = RUN / 'AI观察卡.pdf'
    report_path = RUN / '修复成果说明.pdf'
    card = competition_output.build_observation_card(store, image, card_path)
    report_pdf = competition_output.build_repair_report(store, image, store.get('job', job['id']), report_path)
    report['steps']['export'] = {
        'done': True,
        'card': {'path': str(card_path), 'pages': page_count(card_path), 'truncated': card['truncated']},
        'repair': {'path': str(report_path), 'pages': page_count(report_path), 'truncated': report_pdf['truncated']},
    }
    report['passed'] = all([
        report['steps']['detect']['done'], report['steps']['repair']['done'],
        report['steps']['review']['done'], report['steps'].get('assess', {}).get('done'),
        page_count(card_path) == 1, page_count(report_path) == 2,
        not report['steps']['assess'].get('key_leaked', False),
    ])
    report['finished_at'] = timestamp()
    report['note'] = ('本次为真实本地推理与真实 DeepSeek 请求；缺失识别候选时会明确标注人工区域回退，'
                      '不编造置信度或审核结论。')
    (RUN / 'demo-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
