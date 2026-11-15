"""Verify the real local Qwen3-VL + SAM 2.1 detection on this machine.

The check never scores conservation accuracy. It records what actually ran:

* how long Qwen inference and SAM splitting took;
* which health types were returned;
* whether every accepted mask is in working-image coordinates;
* whether masks stay inside their candidate box.

Run with the project virtualenv, because the Qwen worker uses ``sys.executable``:

    .\\.venv\\Scripts\\python.exe scripts/verify_local_detection.py
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
from mural_agent.desktop_web import local_model_setup  # noqa: E402
from mural_agent.engine import Engine  # noqa: E402
from mural_agent.storage import Store  # noqa: E402
from mural_agent.vision_analysis import (SAM_BOX_PADDING, SAM_EXTENT_PADDING,  # noqa: E402
                                         SAM_LEAK_RATIO, detect_local)


def overlay_preview(image, entries):
    """Tint accepted masks on the working image so offsets are visible."""
    canvas = image.copy()
    for entry in entries:
        mask = entry['mask']
        if mask is None:
            continue
        region = mask > 0
        tint = np.array([214, 62, 44], dtype=np.float32)
        canvas[region] = (.45*canvas[region] + .55*tint).astype(np.uint8)
    return canvas


def mask_checks(entry, width, height):
    """Re-check an accepted mask with the same thresholds the pipeline uses."""
    mask = entry['mask']
    box = list(entry['candidate'].bbox)
    problems = []
    if mask is None:
        return problems
    if mask.shape != (height, width):
        problems.append(f'mask shape {mask.shape} != {(height, width)}')
    if mask.dtype != np.uint8:
        problems.append(f'mask dtype {mask.dtype}')
    if set(np.unique(mask)) - {0, 255}:
        problems.append('mask is not binary')
    binary = mask > 0
    ys, xs = np.where(binary)
    if not len(xs):
        problems.append('mask is empty')
        return problems
    x0, y0, x1, y1 = box
    pad = max(4, round(max(x1 - x0, y1 - y0) * SAM_BOX_PADDING))
    allowed = np.zeros_like(binary)
    allowed[max(0, y0-pad):min(height, y1+pad), max(0, x0-pad):min(width, x1+pad)] = True
    outside = float((binary & ~allowed).sum()) / max(1, int(binary.sum()))
    extent_pad = max(4, round(max(x1 - x0, y1 - y0) * SAM_EXTENT_PADDING))
    if outside > SAM_LEAK_RATIO:
        problems.append(f'mask outside box ratio {outside:.2f} > {SAM_LEAK_RATIO}')
    if (xs.min() < x0-extent_pad or xs.max() > x1+extent_pad
            or ys.min() < y0-extent_pad or ys.max() > y1+extent_pad):
        problems.append(f'mask extent [{xs.min()},{ys.min()},{xs.max()},{ys.max()}] far past box {box}')
    entry['mask_outside_ratio'] = round(outside, 4)
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=ROOT / '图片' / '01-cma-1963.252.jpg')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts' / 'local-detection-verification.json')
    parser.add_argument('--max-preview', type=int, default=1600,
                        help='Downscale only the saved preview; inference still uses the working image.')
    arguments = parser.parse_args()

    settings = Settings.from_env()
    setup = local_model_setup(settings)
    if not (setup['qwen_ready'] and setup['sam_ready']):
        print(json.dumps({'skipped': True, 'setup': setup}, ensure_ascii=False, indent=2))
        return 2

    data_dir = ROOT / 'artifacts' / 'verify-local' / 'data'
    store = Store(data_dir)
    source = arguments.image.read_bytes()
    record = workspace.add_image(store, source, arguments.image.name)
    engine = Engine(settings, store)
    print(f'image={record["width"]}x{record["height"]} device={settings.device}', flush=True)

    started = time.perf_counter()
    assessment, entries = detect_local(store, engine, record)
    elapsed = time.perf_counter() - started

    width, height = record['width'], record['height']
    defects = []
    for entry in entries:
        candidate = entry['candidate']
        problems = mask_checks(entry, width, height)
        defects.append({
            'candidate_id': candidate.candidate_id,
            'type': candidate.type,
            'bbox_pixels': list(candidate.bbox),
            'bbox_0_1000': [round(candidate.bbox[0]*1000/width), round(candidate.bbox[1]*1000/height),
                            round(candidate.bbox[2]*1000/width), round(candidate.bbox[3]*1000/height)],
            'evidence': candidate.evidence,
            'mask_status': entry['mask_status'],
            'mask_notice': entry['mask_notice'],
            'mask_pixels': int((entry['mask'] > 0).sum()) if entry['mask'] is not None else 0,
            'mask_outside_ratio': entry.get('mask_outside_ratio'),
            'mask_problems': problems,
        })

    preview_dir = ROOT / 'artifacts'
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview = overlay_preview(np.asarray(Image.open(store.folder(record['id']) / 'image.png').convert('RGB')), entries)
    preview_image = Image.fromarray(preview)
    preview_image.thumbnail((arguments.max_preview, arguments.max_preview))
    preview_path = preview_dir / 'local-detection-preview.png'
    preview_image.save(preview_path)

    report = {
        'image': str(arguments.image),
        'working_size': [width, height],
        'model_source': assessment.model_source,
        'qwen_seconds_including_startup': round(elapsed, 2),
        'candidates': len(entries),
        'masks_generated': sum(1 for entry in entries if entry['mask_status'] == 'sam'),
        'limitations': assessment.limitations,
        'defects': defects,
        'preview': str(preview_path.relative_to(ROOT)),
        'passed': all(not defect['mask_problems'] for defect in defects),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
