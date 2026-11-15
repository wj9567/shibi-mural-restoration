import json
import os
import time
import zipfile

import pytest

from mural_agent import config


def test_deepseek_key_stays_in_local_settings(tmp_path, monkeypatch):
    key = 'mock-vision-credential'
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'unrelated-process-key')
    assert not config.Settings.from_env().deepseek_api_key
    monkeypatch.delenv('DEEPSEEK_API_KEY')
    (tmp_path / '.env').write_text(f'DEEPSEEK_API_KEY={key}\n', encoding='utf-8')

    settings = config.Settings.from_env()

    assert settings.deepseek_api_key == key
    assert key not in repr(settings)
    assert 'DEEPSEEK_API_KEY' not in os.environ


def test_save_deepseek_key_updates_only_its_local_setting(tmp_path, monkeypatch, qt_app):
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from mural_agent.desktop_web import Bridge
    from mural_agent.engine import Engine
    from mural_agent.storage import Store

    original_key = 'mock-previous-deepseek-key'
    new_key = 'mock-replacement-deepseek-key'
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    monkeypatch.setenv('MURAL_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('MURAL_API_KEY', 'mock-planner-process-key')
    monkeypatch.setenv('MURAL_IMAGINE_API_KEY', 'mock-imagine-process-key')
    env_path = tmp_path / '.env'
    unchanged = ['# Existing API configuration', 'MURAL_API_KEY=mock-planner-file-key',
                 'MURAL_IMAGINE_API_KEY=mock-imagine-file-key']
    original = '\n'.join([*unchanged, f'DEEPSEEK_API_KEY={original_key}']) + '\n'
    env_path.write_text(original, encoding='utf-8')
    settings = config.Settings.from_env()
    store = Store(settings.data_dir)
    engine = Engine(settings, store)
    bridge = Bridge(settings, store, engine)

    try:
        for invalid in ('', 'invalid-key\nMURAL_API_KEY=mock-injected-key', 'a' * 513):
            response = bridge.saveDeepSeekKey(invalid)
            assert 'error' in json.loads(response)
            if invalid:
                assert invalid.splitlines()[0] not in response
            assert env_path.read_text(encoding='utf-8') == original
            assert settings.deepseek_api_key == original_key

        def failed_replace(*args):
            raise OSError(f'write failure containing {new_key}')

        with monkeypatch.context() as patch:
            patch.setattr(config.Path, 'replace', failed_replace)
            response = bridge.saveDeepSeekKey(new_key)
        assert 'error' in json.loads(response) and new_key not in response
        assert env_path.read_text(encoding='utf-8') == original
        assert settings.deepseek_api_key == original_key
        assert list(tmp_path.glob('.env.*.tmp')) == []

        response = bridge.saveDeepSeekKey(new_key)
        assert json.loads(response) == {'configured': True}
        assert new_key not in response
        assert settings.deepseek_api_key == new_key
        assert settings.api_key == 'mock-planner-process-key'
        assert settings.imagine_api_key == 'mock-imagine-process-key'
        saved = env_path.read_text(encoding='utf-8-sig').splitlines()
        assert [line for line in saved if not line.startswith('DEEPSEEK_API_KEY=')] == unchanged
        assert saved.count(f'DEEPSEEK_API_KEY={new_key}') == 1
        assert config.Settings.from_env().deepseek_api_key == new_key
        snapshot = bridge.snapshot()
        assert json.loads(snapshot)['models']['deepseek_vision'] is True
        assert new_key not in snapshot and original_key not in snapshot
    finally:
        bridge.aux_executor.shutdown(wait=True)
        engine.executor.shutdown(wait=True)


def test_cloud_candidate_requires_manual_area_and_archive_keeps_only_provenance(tmp_path, monkeypatch, qt_app):
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from mural_agent.desktop_web import Bridge, DeepSeekVisionProvider
    from mural_agent.engine import Engine
    from mural_agent.storage import Store
    from mural_agent.vision_schema import DefectCandidate, VisionAssessment, VisionImage

    key = 'mock-archive-credential'
    settings = config.Settings(tmp_path / 'data', tmp_path / 'missing.pt',
                               vision_model=tmp_path / 'missing-vision', deepseek_api_key=key)
    store = Store(settings.data_dir)
    engine = Engine(settings, store)
    bridge = Bridge(settings, store, engine)
    image = json.loads(bridge.loadDemo())
    candidate = DefectCandidate(candidate_id='crack-1', bbox=[130, 90, 165, 200], type='crack',
                                observation='表面有细线', evidence='边缘不连续', repair_suggestion='人工核对后局部修补',
                                uncertainty='也可能为原画线条', risk_level='medium', model_source='deepseek-flash')

    def analyze(self, source):
        return VisionAssessment(source='deepseek', model_source='deepseek-flash',
                                image=VisionImage.from_sizes(source.size, source.size), items=[candidate])

    def wait_for(record, complete):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            data = record()
            if complete(data):
                return data
            qt_app.processEvents()
            time.sleep(.01)
        pytest.fail('Timed out waiting for the workbench task')

    monkeypatch.setattr(DeepSeekVisionProvider, 'analyze', analyze)
    try:
        request = json.dumps({'image_id': image['id'], 'provider': 'deepseek'})
        assert json.loads(bridge.autoRestore(request))['status'] == 'running'
        snapshot = wait_for(lambda: json.loads(bridge.snapshot()), lambda data: data['aux']['status'] != 'running')
        assert snapshot['aux']['status'] == 'done', snapshot['aux']
        assert snapshot['models']['deepseek_vision'] and not snapshot['models']['vision']
        assert key not in json.dumps(snapshot)
        auto = snapshot['aux']['result']['auto']
        assert auto['assessment']['items'][0]['bbox'] == [130, 90, 165, 200]
        assert auto['defects'][0]['requires_manual_selection'] is True
        assert auto['mask'] is None and auto['job'] is None and not store.list('proposal')

        other = json.loads(bridge.loadDemo())
        manual = {'bbox': [130, 140, 180, 300], 'candidate_id': 'crack-1'}
        assert 'error' in json.loads(bridge.addManualDefect(other['id'], json.dumps(manual)))
        response = json.loads(bridge.addManualDefect(image['id'], json.dumps(manual)))
        proposal = response['proposal']
        assert proposal['confirmed'] is True and proposal['mask_source'] == 'manual'
        assert proposal['metadata']['vision_candidate']['candidate_id'] == 'crack-1'
        assert 'confidence' not in proposal['metadata']

        run = json.loads(bridge.runDefectRepairs(json.dumps({
            'image_id': image['id'], 'proposal_ids': [proposal['id']],
            'methods': {proposal['id']: 'telea'},
        })))
        job_id = run['jobs'][0]['id']
        job = wait_for(lambda: store.get('job', job_id), lambda item: item['status'] in {'review', 'failed'})
        assert job['status'] == 'review', job.get('error')
        selection = json.loads(bridge.selectCandidate(job_id, job['candidates'][0]['id'], '{"decision":"adopt"}'))
        assert selection['review']['decision'] == 'adopt'
        with zipfile.ZipFile(engine.export(job_id)) as archive:
            entry = json.loads(archive.read(f'proposals/{proposal["id"]}.json'))
            assert entry['metadata']['vision_candidate']['bbox'] == [130, 90, 165, 200]
            assert entry['metadata']['vision_image']['original_width'] == image['width']
            for name in archive.namelist():
                assert key.encode() not in archive.read(name)
    finally:
        bridge.aux_executor.shutdown(wait=True)
        engine.executor.shutdown(wait=True)
