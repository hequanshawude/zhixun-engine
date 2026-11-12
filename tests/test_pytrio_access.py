import pytest

from selftrain.config import SystemConfig, apply_env_overrides


def test_pytrio_environment_key(monkeypatch):
    monkeypatch.delenv('HARNESS_API_KEY', raising=False)
    monkeypatch.delenv('HARNESS_BASE_URL', raising=False)
    monkeypatch.setenv('PYTRIO_API_KEY', 'test-key')
    cfg = apply_env_overrides(SystemConfig.load('configs/pytrio.yaml'))
    assert cfg.harness.api_key == 'test-key'
    assert cfg.harness.model == cfg.harness.small_model == 'Qwen/Qwen3.8-27B'


def test_pytrio_key_never_forwarded_to_other_endpoint(monkeypatch):
    monkeypatch.setenv('PYTRIO_API_KEY', 'test-key')
    monkeypatch.setenv('HARNESS_BASE_URL', 'https://example.com/v1')
    with pytest.raises(ValueError, match='官方'):
        apply_env_overrides(SystemConfig.load('configs/pytrio.yaml'))


def test_default_config_does_not_load_pytrio(monkeypatch):
    monkeypatch.delenv('HARNESS_API_KEY', raising=False)
    monkeypatch.setenv('PYTRIO_API_KEY', 'test-key')
    assert apply_env_overrides(SystemConfig()).harness.api_key == 'EMPTY'


def test_pytrio_budget_reserves_whole_request_before_call():
    from types import SimpleNamespace
    from selftrain.actions import ACTION_NAMES
    from selftrain.config import HarnessConfig
    from selftrain.harness import PolicyRouter
    from selftrain.playbook import Playbook
    from selftrain.types import DriftKind, Fingerprint, TaskType

    cfg = HarnessConfig(enabled=True, credential_source='pytrio', max_tokens=1024)
    router = PolicyRouter(cfg, Playbook(ACTION_NAMES), model='llm')
    router.big = router.small = SimpleNamespace(chat=lambda _: pytest.fail('unexpected paid call'))
    plan, trace = router.decide(
        Fingerprint(DriftKind.CONCEPT, 0.5, 0.1),
        SimpleNamespace(spec=SimpleNamespace(task=TaskType.BINARY)),
        {'budget_left': {'tokens': 1000}},
        allowed=['noop', 'finetune'], best_available=False,
    )
    assert plan.source == 'rule'
    assert trace.tokens == 0
    assert 'Token' in trace.fallback_reason
