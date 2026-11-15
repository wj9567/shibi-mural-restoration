"""DeepSeek vision coverage using only synthetic images and mocked HTTP."""
import base64
import json
import logging
import traceback
from io import BytesIO

import httpx
import pytest
from PIL import Image

from mural_agent.deepseek_vision import DeepSeekVisionError, DeepSeekVisionProvider


MOCK_KEY = 'fixture-credential-not-a-real-key'


@pytest.fixture
def mock_http(monkeypatch):
    client = httpx.Client

    def install(handler):
        monkeypatch.setattr('mural_agent.deepseek_vision.httpx.Client',
                            lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))

    return install


def candidate(**changes):
    return {
        'candidate_id': 'crack-1',
        'bbox': [2, 3, 12, 15],
        'type': 'crack',
        'observation': '可见不连续的细暗线',
        'evidence': '暗线穿过两个相邻颜料区域',
        'repair_suggestion': '人工确认裂隙后可比较局部保守修补方案',
        'uncertainty': '可能是拍摄阴影',
        'risk_level': 'medium',
        **changes,
    }


def completion(content, finish_reason='stop', refusal=None):
    return httpx.Response(200, json={
        'model': 'deepseek-flash',
        'choices': [{
            'finish_reason': finish_reason,
            'message': {'role': 'assistant', 'content': content, 'refusal': refusal},
        }],
    })


def test_chat_request_and_inverse_bbox_mapping(mock_http, caplog):
    original = Image.new('RGB', (2051, 513), (20, 45, 95))
    requests = []
    caplog.set_level(logging.DEBUG)

    def handler(request):
        requests.append(request)
        assert request.method == 'POST'
        assert str(request.url) == 'https://api.deepseek.com/chat/completions'
        assert request.headers['Authorization'] == f'Bearer {MOCK_KEY}'
        assert MOCK_KEY.encode() not in request.content
        body = json.loads(request.content)
        assert body['model'] == 'deepseek-flash'
        assert body['response_format'] == {'type': 'json_object'}
        assert body['safety_check'] == {'level': 'high'}
        message, = body['messages']
        assert message['role'] == 'user'
        text, image = message['content']
        assert text['type'] == 'text'
        assert text['text'].strip()
        assert image['type'] == 'image_url'
        assert image['image_url']['detail'] == 'high'
        data_url = image['image_url']['url']
        assert data_url.startswith('data:image/jpeg;base64,')
        with Image.open(BytesIO(base64.b64decode(data_url.split(',', 1)[1], validate=True))) as sent:
            assert sent.format == 'JPEG'
            assert sent.size == (1024, 256)
        transport_timeout = request.extensions['timeout']
        assert transport_timeout['read'] == 12.5
        items = [candidate(bbox=[-7, 9, 513, 999])]
        return completion('```json\n' + json.dumps({'items': items}) + '\n```')

    mock_http(handler)
    payload = DeepSeekVisionProvider(MOCK_KEY, timeout=12.5).analyze(original).model_dump(mode='json')

    assert len(requests) == 1
    assert original.size == (2051, 513)
    assert payload['source'] == 'deepseek'
    assert payload['model_source'] == 'deepseek-flash'
    assert payload['image'] == {
        'original_width': 2051, 'original_height': 513,
        'analysis_width': 1024, 'analysis_height': 256,
        'scale_x': pytest.approx(1024 / 2051), 'scale_y': pytest.approx(256 / 513),
    }
    result, = payload['items']
    assert result['bbox'] == [0, 18, 1028, 513]
    assert all(type(coordinate) is int for coordinate in result['bbox'])
    assert result['candidate_id'] == 'crack-1'
    assert result['risk_level'] == 'medium'
    assert result['model_source'] == 'deepseek-flash'
    assert result.get('confidence') is None
    assert MOCK_KEY not in repr(payload)
    assert MOCK_KEY not in caplog.text


def test_deduplication_is_limited_to_overlapping_same_type(mock_http):
    items = [candidate(bbox=[2, 3, 25, 28]),
             candidate(candidate_id='crack-2', bbox=[3, 3, 25, 28]),
             candidate(candidate_id='stain-1', bbox=[2, 3, 25, 28], type='stain'),
             candidate(candidate_id='crack-3', bbox=[26, 28, 31, 31])]
    mock_http(lambda request: completion(json.dumps({'items': items})))

    assessment = DeepSeekVisionProvider(MOCK_KEY).analyze(Image.new('RGB', (32, 32)))

    assert [item.candidate_id for item in assessment.items] == ['crack-1', 'stain-1', 'crack-3']
    assert assessment.image.scale_x == assessment.image.scale_y == 1


def test_conflicting_candidate_ids_are_rejected(mock_http):
    items = [candidate(), candidate(bbox=[16, 18, 24, 28])]
    mock_http(lambda request: completion(json.dumps({'items': items})))

    with pytest.raises(DeepSeekVisionError, match='重复候选 ID'):
        DeepSeekVisionProvider(MOCK_KEY).analyze(Image.new('RGB', (32, 32)))


def test_explicit_empty_candidates_is_a_valid_result(mock_http):
    mock_http(lambda request: completion('{"items": []}'))
    assessment = DeepSeekVisionProvider(MOCK_KEY).analyze(Image.new('RGB', (32, 32)))
    assert assessment.items == []


def test_cultural_observations_remain_unverified(mock_http):
    context = {
        'visible_subject': '几何色带',
        'possible_period_or_tradition': '待核实',
        'visual_clues': ['可见重复边线'],
        'verification_needed': ['核对来源说明'],
        'preservation_constraints': ['保留现有色带边界'],
    }
    mock_http(lambda request: completion(json.dumps({'items': [], 'cultural_context': context})))

    assessment = DeepSeekVisionProvider(MOCK_KEY).analyze(Image.new('RGB', (32, 32)))

    assert assessment.model_dump(mode='json')['cultural_context'] == dict(context, verified=False, confidence='')


@pytest.mark.parametrize(('item', 'message'), [
    (candidate(type='invented'), 'JSON'),
    (candidate(bbox=[True, 3, 12, 15]), 'JSON'),
    (candidate(bbox=['2', 3, 12, 15]), 'JSON'),
    (candidate(bbox=[2.5, 3, 12, 15]), 'JSON'),
    (candidate(bbox=[2, 3, 12]), 'JSON'),
    (candidate(bbox=[12, 3, 2, 15]), '坐标'),
    (candidate(bbox=[33, 3, 45, 15]), '坐标'),
    (candidate(risk_level='maybe'), 'JSON'),
    (candidate(evidence='   '), 'JSON'),
    (candidate(confidence=0.99), 'JSON'),
    (candidate(model_source='other-model'), 'JSON'),
], ids=['type-enum', 'boolean-coordinate', 'string-coordinate', 'fractional-coordinate',
        'short-box', 'reversed-box', 'outside-box', 'risk-enum', 'empty-evidence',
        'invented-confidence', 'untrusted-model-source'])
def test_invalid_candidate_is_not_returned_as_analysis(mock_http, item, message):
    invalid = dict(item, candidate_id='invalid')
    mock_http(lambda request: completion(json.dumps({'items': [candidate(), invalid]})))

    with pytest.raises(DeepSeekVisionError, match=message):
        DeepSeekVisionProvider(MOCK_KEY).analyze(Image.new('RGB', (32, 32)))


@pytest.mark.parametrize(('reply', 'message'), [
    (lambda: httpx.Response(401, json={'error': {'message': MOCK_KEY}}), '401'),
    (lambda: httpx.Response(429, json={'error': {'message': MOCK_KEY}}), '429'),
    (lambda: completion(None), '为空'),
    (lambda: completion('   '), '为空'),
    (lambda: httpx.Response(200, content=b'<html>Proxy error</html>'), 'JSON'),
    (lambda: completion('这不是 JSON'), 'JSON'),
    (lambda: completion('{}'), 'JSON'),
    (lambda: completion('{"items": [], "items": []}'), 'JSON'),
    (lambda: httpx.Response(400, json={'error': {'message': 'Content Exists Risk',
                                                'type': 'invalid_request_error'}}), '审核'),
    (lambda: completion('{"items": []}', 'content_filter'), '审核'),
    (lambda: completion('{"items": []}', refusal='image rejected'), '审核'),
    (lambda: completion('{"items": []}', 'length'), '截断'),
], ids=['401', '429', 'null', 'blank', 'non-json-http-body', 'bad-json', 'missing-items', 'duplicate-fields',
        'input-audit', 'content-filter', 'refusal', 'truncated'])
def test_response_failure_is_actionable_and_sanitized(mock_http, caplog, reply, message):
    caplog.set_level(logging.DEBUG)
    requests = []

    def handler(request):
        requests.append(request)
        return reply()

    mock_http(handler)

    with pytest.raises(DeepSeekVisionError, match=message) as captured:
        DeepSeekVisionProvider(MOCK_KEY).analyze(Image.new('RGB', (32, 32)))

    assert len(requests) == 1
    assert MOCK_KEY not in ''.join(traceback.format_exception(captured.value))
    assert MOCK_KEY not in caplog.text


@pytest.mark.parametrize(('error_type', 'message'), [(httpx.ConnectError, '网络'), (httpx.ReadTimeout, '超时')])
def test_transport_error_does_not_expose_request_or_provider_detail(mock_http, caplog, error_type, message):
    caplog.set_level(logging.DEBUG)

    def handler(request):
        raise error_type(f'simulated SDK failure with {MOCK_KEY}', request=request)

    mock_http(handler)
    with pytest.raises(DeepSeekVisionError, match=message) as captured:
        DeepSeekVisionProvider(MOCK_KEY).analyze(Image.new('RGB', (32, 32)))

    assert MOCK_KEY not in ''.join(traceback.format_exception(captured.value))
    assert MOCK_KEY not in caplog.text


@pytest.mark.parametrize('escape_first_character', [False, True], ids=['plain', 'json-escaped'])
def test_credential_echo_is_not_returned_or_logged(mock_http, caplog, escape_first_character):
    caplog.set_level(logging.DEBUG)
    content = json.dumps({'items': [candidate(evidence=MOCK_KEY)]})
    if escape_first_character:
        escaped_key = f'\\u{ord(MOCK_KEY[0]):04x}' + MOCK_KEY[1:]
        content = content.replace(MOCK_KEY, escaped_key)
    mock_http(lambda request: completion(content))

    with pytest.raises(DeepSeekVisionError) as captured:
        DeepSeekVisionProvider(MOCK_KEY).analyze(Image.new('RGB', (32, 32)))

    assert MOCK_KEY not in ''.join(traceback.format_exception(captured.value))
    assert MOCK_KEY not in caplog.text


def test_missing_credential_does_not_send_a_request(mock_http):
    def unexpected_request(request):
        pytest.fail('缺少可用配置时不应调用服务')

    mock_http(unexpected_request)

    with pytest.raises(DeepSeekVisionError, match='DEEPSEEK_API_KEY'):
        DeepSeekVisionProvider('').analyze(Image.new('RGB', (32, 32)))


def test_redirect_is_not_followed_with_the_authorization_header(mock_http):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(302, headers={'Location': 'https://redirect.example.test/collect'})

    mock_http(handler)
    with pytest.raises(DeepSeekVisionError):
        DeepSeekVisionProvider(MOCK_KEY).analyze(Image.new('RGB', (32, 32)))

    assert urls == ['https://api.deepseek.com/chat/completions']
