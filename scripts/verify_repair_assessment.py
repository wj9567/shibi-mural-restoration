"""Stage-5 real acceptance: one DeepSeek repair-review request on existing local results.

Uses the key already configured in the local ``.env`` (no key is printed or
written). Reuses the local repair produced by scripts/verify_region_repair.py so
the run costs a single API request.

    .\\.venv\\Scripts\\python.exe scripts/verify_repair_assessment.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mural_agent import repair_assessment  # noqa: E402
from mural_agent.config import Settings, valid_deepseek_key  # noqa: E402
from mural_agent.deepseek_vision import DeepSeekVisionProvider  # noqa: E402
from mural_agent.storage import Store  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'artifacts' / 'verify-region' / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts' / 'repair-assessment-verification.json')
    arguments = parser.parse_args()

    settings = Settings.from_env()
    if not settings.deepseek_api_key:
        print(json.dumps({'skipped': True, 'reason': '本机 .env 未配置 DEEPSEEK_API_KEY'},
                         ensure_ascii=False, indent=2))
        return 2
    if not valid_deepseek_key(settings.deepseek_api_key):
        print(json.dumps({'skipped': True, 'reason': 'DEEPSEEK_API_KEY 格式无效'},
                         ensure_ascii=False, indent=2))
        return 2
    if not arguments.data.exists():
        print(json.dumps({'skipped': True, 'reason': f'缺少本地修复数据 {arguments.data}'},
                         ensure_ascii=False, indent=2))
        return 2

    store = Store(arguments.data)
    image = next((record for record in store.list('image')
                  if any(job.get('defect_proposal_id') for job in store.list('job')
                         if job.get('image_id') == record['id'])), None)
    if not image:
        print(json.dumps({'skipped': True, 'reason': '本地修复数据中没有可评估的图像'},
                         ensure_ascii=False, indent=2))
        return 2
    cases = repair_assessment.collect_cases(store, image)
    cases = [case for case in cases if case['candidate']]
    if not cases:
        print(json.dumps({'skipped': True, 'reason': '没有已完成的本地修复候选'},
                         ensure_ascii=False, indent=2))
        return 2

    original, composite = repair_assessment.compose_repaired(store, image, cases)
    regions, crops = [], []
    for case in cases:
        regions.append({
            'region_id': case['region_id'], 'type': case['type'],
            'type_label': repair_assessment.type_label(case['type']),
            'bbox_0_1000': case['bbox_0_1000'], 'method': case['method'],
            'mask_fraction': round(case['mask_fraction'], 4),
            'has_candidate': bool(case['candidate']),
            'local_evidence': (case['evidence'] or '')[:200],
        })
        pair = repair_assessment.crop_pair(store, image, case['job'], case['candidate'], case['mask'])
        if pair and len(crops) < repair_assessment.MAX_CROPS:
            crops.append({'region_id': case['region_id'], 'before': pair[0], 'after': pair[1]})

    started = time.perf_counter()
    report = {'key_configured': True, 'requests': 1, 'image_size': [image['width'], image['height']],
              'regions': len(regions), 'crops': len(crops)}
    try:
        assessment = DeepSeekVisionProvider(settings.deepseek_api_key, timeout=90).assess_repair(
            Image.fromarray(original), Image.fromarray(composite), regions, crops=crops)
        elapsed = time.perf_counter() - started
        rows = repair_assessment.merge(assessment, cases, original, '')
        report.update({
            'status': 'success', 'elapsed_seconds': round(elapsed, 2),
            'model_source': assessment.model_source,
            'limitations': assessment.limitations,
            'region_ids_matched': sorted(region.region_id for region in assessment.regions),
            'rows': [{key: row[key] for key in ('index', 'region_id', 'type_label', 'risk_level',
                                                'human_review_required', 'review_reason', 'local_flags')}
                     for row in rows],
            'advisory': True,
        })
        report['key_leaked'] = settings.deepseek_api_key in json.dumps(report, ensure_ascii=False)
        report['passed'] = bool(report['regions'] == len(rows)) and not report['key_leaked']
    except Exception as exc:  # noqa: BLE001 — record the real outcome without secrets
        report.update({'status': 'failed', 'elapsed_seconds': round(time.perf_counter() - started, 2),
                       'error': str(exc)[:300], 'passed': False})
        report['key_leaked'] = settings.deepseek_api_key in json.dumps(report, ensure_ascii=False)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
