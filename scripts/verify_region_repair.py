"""Stage-4 real local inference check: confirmed mask -> LaMa -> checked candidate.

Runs only when models/big-lama.pt exists. The mask is a user-style confirmed
region (not a model guess), so the check isolates the repair engine: real LaMa
weights, real GPU/CPU inference, and the same checks the application enforces.

    .\\.venv\\Scripts\\python.exe scripts/verify_region_repair.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mural_agent import workspace  # noqa: E402
from mural_agent.config import Settings  # noqa: E402
from mural_agent.engine import Engine  # noqa: E402
from mural_agent.imaging import image_array, sha256  # noqa: E402
from mural_agent.region_repair import run_region_candidates  # noqa: E402
from mural_agent.storage import Store, new_id  # noqa: E402


def side_by_side(original, result, difference):
    height, width = original.shape[:2]
    canvas = np.zeros((height, width * 3, 3), np.uint8)
    canvas[:, :width] = original
    canvas[:, width:width * 2] = result
    canvas[:, width * 2:] = difference
    preview = Image.fromarray(canvas)
    preview.thumbnail((1800, 900))
    return preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=ROOT / '图片' / '01-cma-1963.252.jpg')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts' / 'region-repair-verification.json')
    arguments = parser.parse_args()

    settings = Settings.from_env()
    engine = Engine(settings, Store(ROOT / 'artifacts' / 'verify-region' / 'data'))
    if not engine.lama.available():
        print(json.dumps({'skipped': True, 'reason': 'models/big-lama.pt 未安装'},
                         ensure_ascii=False, indent=2))
        return 2

    image_payload = arguments.image.read_bytes()
    record = workspace.add_image(engine.store, image_payload, arguments.image.name)
    width, height = record['width'], record['height']
    # The torn/damaged left edge of this mural is the region repaired here.
    mask = np.zeros((height, width), np.uint8)
    mask[int(height * .12):int(height * .82), 0:int(width * .05)] = 255
    proposal_id = new_id()
    Image.fromarray(mask).save(engine.store.folder(proposal_id) / 'mask.png')
    engine.store.put('proposal', {
        'id': proposal_id, 'image_id': record['id'], 'created_at': '2026-11-16T00:00:00+00:00',
        'fraction': float(mask.mean()), 'confirmed': True, 'mask_source': 'manual',
        'region_class': 'fillable_lacuna', 'repair_allowed': True,
        'metadata': {'type': 'paint_loss', 'bbox_pixels': [0, int(height * .12),
                                                           int(width * .05), int(height * .82)],
                     'recommendation': {'backend': 'lama', 'method': 'LaMa 纹理补全'},
                     'local_detection': False},
    })
    # Keep this check synchronous: no background executor races with the assertions.
    engine.submit = lambda job_id: None

    started = time.perf_counter()
    job = run_region_candidates(engine.store, engine, record, [proposal_id])['jobs'][0]
    engine.run(job['id'])
    elapsed = time.perf_counter() - started

    job = engine.store.get('job', job['id'])
    if job['status'] != 'review':
        print(json.dumps({'passed': False, 'status': job['status'], 'error': job.get('error'),
                          'events': job.get('events')}, ensure_ascii=False, indent=2))
        return 1
    candidate = job['candidates'][0]
    folder = engine.store.folder(candidate['id'])
    original = image_array(engine.store.folder(record['id']) / 'image.png')
    result = np.asarray(Image.open(folder / 'result.png').convert('RGB'))
    difference = np.asarray(Image.open(folder / 'difference.png').convert('RGB'))
    region = np.asarray(Image.open(folder / 'region.png').convert('RGB'))
    outside = np.abs(result[mask == 0].astype(np.int16) - original[mask == 0].astype(np.int16))
    report = {
        'image': str(arguments.image),
        'working_size': [width, height],
        'backend': job['region_backend'],
        'preferred': job['region_preferred'],
        'tool': candidate['tool'],
        'parameters': candidate['parameters'],
        'versions': candidate['versions'],
        'elapsed_seconds': round(elapsed, 2),
        'mask_pixels': int((mask > 0).sum()),
        'result_size': [int(result.shape[1]), int(result.shape[0])],
        'size_match': result.shape[:2] == (height, width),
        'region_box': candidate['region']['box'],
        'region_size': [int(region.shape[1]), int(region.shape[0])],
        'outside_max_difference': int(outside.max()) if outside.size else 0,
        'outside_unchanged': bool(outside.size and outside.max() == 0 or not outside.size),
        'original_untouched': sha256(engine.store.folder(record['id']) / 'image.png') == record['working_sha256'],
        'job_checks': job['checks'],
        'candidate_checks': candidate['checks'],
        'weight_sha256': candidate['versions'].get('weight_sha256'),
        'weight_matches_file': candidate['versions'].get('weight_sha256') == sha256(settings.lama_model),
        'preview': 'artifacts/region-repair-preview.png',
    }
    report['passed'] = all([report['size_match'], report['outside_unchanged'],
                            report['original_untouched'], report['weight_matches_file'],
                            job['checks']['original_unchanged'], job['checks']['outside_unchanged']])
    side_by_side(original, result, difference).save(ROOT / 'artifacts' / 'region-repair-preview.png')
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
