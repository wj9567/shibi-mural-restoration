"""Startup smoke check: build the shipped window offscreen and load the workbench page.

    .\\.venv\\Scripts\\python.exe scripts/verify_startup.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --no-sandbox')
os.environ.setdefault('QTWEBENGINE_DISABLE_SANDBOX', '1')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from mural_agent.config import Settings  # noqa: E402
from mural_agent.desktop_web import WebMainWindow  # noqa: E402


def main():
    app = QApplication.instance() or QApplication([])
    shell = WebMainWindow(Settings.from_env())
    loaded = {}
    shell.view.loadFinished.connect(lambda ok: loaded.update(done=True, ok=ok))
    shell.show()
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline and not loaded.get('done'):
        app.processEvents()
        time.sleep(0.02)
    snapshot = json.loads(shell.bridge.snapshot())
    report = {
        'page_loaded': bool(loaded.get('ok')),
        'url': shell.view.url().toString(),
        'images': len(snapshot.get('images', [])),
        'models': {key: bool(value) for key, value in (snapshot.get('models') or {}).items()
                   if key in {'lama', 'sam2', 'vision', 'deepseek_vision'}},
        'title': shell.windowTitle(),
    }
    if not report['page_loaded']:
        report['error'] = 'ui/index.html 未在 40 秒内加载完成'
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    shell.engine.executor.shutdown(wait=False, cancel_futures=True)
    shell.bridge.aux_executor.shutdown(wait=False, cancel_futures=True)
    shell.close()
    app.processEvents()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
