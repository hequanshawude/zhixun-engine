import io
import json
from types import SimpleNamespace
import urllib.error

import pytest

from selftrain.actions import ACTION_NAMES
from selftrain.config import HarnessConfig, SystemConfig
from selftrain.data import SyntheticConfig, SyntheticStream
from selftrain.harness import LLMClient, LLMError, LLMUsage, PolicyRouter, parse_action
from selftrain.loop import SelfEvolvingSystem
from selftrain.playbook import Playbook
from selftrain.types import DriftKind, Fingerprint, TaskType


@pytest.mark.parametrize('text', [
    '{"action":"finetune","params":{},"reason":"ok"}',
    '```json\n{"action":"finetune","params":{},"reason":"ok"}\n```',
    '决策如下：{"action":"finetune","params":{},"reason":"{括号}和\\\"引号"}',
])
def test_complete_nested_decision_is_parsed(text):
    assert parse_action(text)['action'] == 'finetune'


@pytest.mark.parametrize('text', [None, [], '', '{"action":"noop","params":{}',
    '[{"action":"noop"}]', '{"action":"noop"}{"action":"retrain"}',
    '{"params":{"action":"noop"}}', '{"action":1}', '没有决策'])
def test_malformed_or_ambiguous_decision_is_rejected(text):
    assert parse_action(text) is None


def response(content, usage=None):
    body = {'choices': [{'message': {'content': content}}]}
    if usage is not None:
        body['usage'] = usage
    return io.BytesIO(json.dumps(body).encode())


def test_known_usage_survives_invalid_response_envelope(monkeypatch):
    calls = []
    def request(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b'{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7}}')
    monkeypatch.setattr('urllib.request.urlopen', request)
    client = LLMClient(HarnessConfig(retries=2))
    with pytest.raises(LLMError) as error:
        client.chat([])
    assert error.value.usage.total == 18
    assert error.value.usage.unknown_calls == 0
    assert client.prompt_tokens == 11
    assert len(calls) == 1  # 已返回响应的结构错误不重新发请求。


def test_network_retry_retains_unknown_usage(monkeypatch):
    calls = []
    def request(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError('timeout')
        return response('{"action":"noop"}', {'prompt_tokens': 4, 'completion_tokens': 3})
    monkeypatch.setattr('urllib.request.urlopen', request)
    monkeypatch.setattr('selftrain.harness.time.sleep', lambda _: None)
    _, usage = LLMClient(HarnessConfig(retries=1)).chat([])
    assert usage.total == 7 and usage.unknown_calls == 1


def test_auth_failure_does_not_retry_or_expose_server_error(monkeypatch):
    calls = []
    def request(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError('https://local.test', 401, 'secret-from-server', {}, None)
    monkeypatch.setattr('urllib.request.urlopen', request)
    with pytest.raises(LLMError) as error:
        LLMClient(HarnessConfig(retries=2)).chat([])
    assert len(calls) == 1
    assert 'secret-from-server' not in str(error.value)
    assert error.value.usage.unknown_calls == 1


def test_model_server_unavailable_is_classified_without_exposing_body(monkeypatch):
    def request(*args, **kwargs):
        raise urllib.error.HTTPError('https://pytrio.com', 503, 'unavailable', {},
                                     io.BytesIO(b'no available server\n'))
    monkeypatch.setattr('urllib.request.urlopen', request)
    with pytest.raises(LLMError) as error:
        LLMClient(HarnessConfig(retries=0)).chat([])
    assert error.value.http_status == 503
    assert error.value.provider_code == 'no_available_server'
    assert 'no available server' not in str(error.value)


def test_missing_usage_is_not_claimed_as_zero_cost(monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: response('{"action":"noop"}'))
    _, usage = LLMClient(HarnessConfig()).chat([])
    assert usage.total == 0
    assert usage.unknown_calls == 1


def test_retry_does_not_restart_full_time_budget(monkeypatch):
    clock = [0.0]
    timeouts = []
    monkeypatch.setattr('selftrain.harness.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('selftrain.harness.time.sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    def request(*args, **kwargs):
        timeouts.append(kwargs['timeout'])
        clock[0] += kwargs['timeout']
        raise TimeoutError('timeout')
    monkeypatch.setattr('urllib.request.urlopen', request)
    with pytest.raises(LLMError) as error:
        LLMClient(HarnessConfig(timeout_s=1, retries=2)).chat([])
    assert timeouts == [1]
    assert error.value.usage.unknown_calls == 1


def test_invalid_endpoint_downgrades_without_request(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('must not make a request')
    monkeypatch.setattr('urllib.request.urlopen', unexpected)
    with pytest.raises(LLMError, match='地址无效'):
        LLMClient(HarnessConfig(base_url='not-a-url')).chat([])


def router_with_reply(text):
    router = PolicyRouter(HarnessConfig(enabled=True), Playbook(ACTION_NAMES), model='llm')
    calls = []
    def chat(messages):
        calls.append(messages)
        return text, LLMUsage(11, 7)
    router.big = router.small = SimpleNamespace(chat=chat)
    ctx = SimpleNamespace(spec=SimpleNamespace(task=TaskType.BINARY))
    fp = Fingerprint(DriftKind.CONCEPT, 0.5, 0.1)
    return router, calls, ctx, fp


def test_valid_decision_cache_and_action_mask():
    router, calls, ctx, fp = router_with_reply('{"action":"retrain","params":{},"reason":"ok"}')
    state = {'budget_left': {'tokens': 1000}}
    first, trace = router.decide(fp, ctx, state, allowed=['noop', 'finetune', 'retrain'], best_available=False)
    assert first.action == 'retrain' and trace.tokens == 18
    second, trace = router.decide(fp, ctx, state, allowed=['noop', 'finetune', 'retrain'], best_available=False)
    assert second.action == 'retrain' and trace.source == 'llm-cache' and trace.tokens == 0
    fallback, trace = router.decide(fp, ctx, state, allowed=['noop', 'finetune'], best_available=False)
    assert fallback.action == 'finetune' and trace.source == 'rule'
    assert trace.tokens == 18 and trace.fallback_reason
    assert len(calls) == 2


@pytest.mark.parametrize('text', ['not json', '{"action":"finetune","params":[]}',
    '{"action":"finetune","params":{"seed":1}}', '{"action":"delete_all"}'])
def test_invalid_reply_uses_rule_and_preserves_cost(text):
    router, calls, ctx, fp = router_with_reply(text)
    state = {'budget_left': {'tokens': 1000}}
    for _ in range(4):
        plan, trace = router.decide(fp, ctx, state, allowed=['noop', 'finetune'], best_available=False)
        assert plan.action == 'finetune' and trace.source == 'rule'
        assert trace.fallback_reason
    assert len(calls) == 3  # 连续失败后熔断，不再请求服务。
    assert trace.tokens == 0


@pytest.mark.parametrize('tokens,allowed', [(0, ['noop', 'finetune']), (1000, ['noop'])])
def test_no_remote_call_without_budget_or_update_permission(tokens, allowed):
    router, calls, ctx, fp = router_with_reply('{"action":"noop"}')
    router.decide(fp, ctx, {'budget_left': {'tokens': tokens}}, allowed=allowed, best_available=False)
    assert calls == []


def test_loop_records_paid_invalid_reply_and_unknown_usage(monkeypatch):
    cfg = SystemConfig()
    cfg.model.default_kind = 'dummy'
    cfg.budget.bootstrap_timeout_s = None
    cfg.budget.action_timeout_s = None
    cfg.harness.enabled = True
    stream = SyntheticStream(SyntheticConfig(n_batches=2, batch_size=40, n_features=4))
    system = SelfEvolvingSystem(stream.spec, cfg, policy_mode='llm')
    def fail(messages):
        raise LLMError('response invalid', LLMUsage(11, 7, 1))
    system.policy.big = system.policy.small = SimpleNamespace(chat=fail)
    monkeypatch.setattr(system, '_allowed_actions', lambda *args: ['noop', 'finetune'])
    result = system.run(iter(stream))
    assert result.summary['budget']['tokens'] == 18
    assert result.summary['unknown_llm_usage_calls'] == 1
    assert result.decisions[0]['plan']['fallback_reason'] == 'response invalid'
