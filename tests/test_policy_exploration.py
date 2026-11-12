from types import SimpleNamespace

import pytest

from selftrain.actions import ACTION_NAMES
from selftrain.config import HarnessConfig, SystemConfig
from selftrain.data import SyntheticConfig, SyntheticStream
from selftrain.harness import PolicyRouter
from selftrain.loop import SelfEvolvingSystem
from selftrain.playbook import Playbook, context_key
from selftrain.types import DriftKind, Fingerprint, TaskType


def test_router_explores_and_learns_from_feedback():
    book = Playbook(ACTION_NAMES)
    router = PolicyRouter(HarnessConfig(), book, model='bandit')
    fp = Fingerprint(DriftKind.CONCEPT, 0.5, 0.1)
    ctx = SimpleNamespace(spec=SimpleNamespace(task=TaskType.BINARY))
    key = context_key(fp, 'binary')
    actions = []
    for _ in range(100):
        plan, _ = router.decide(fp, ctx, {}, allowed=['noop', 'finetune', 'retrain'],
                                best_available=False)
        actions.append(plan.action)
        book.update(key, plan.action, 1.0 if plan.action == 'retrain' else -1.0)
    assert actions[0] == 'finetune'  # 冷启动仍由规则提供先验。
    assert set(actions[:3]) == {'noop', 'finetune', 'retrain'}
    assert actions[-20:].count('retrain') >= 18


def test_exploration_respects_current_mask_and_keeps_rule_rollback():
    book = Playbook(ACTION_NAMES)
    router = PolicyRouter(HarnessConfig(), book, model='bandit')
    ctx = SimpleNamespace(spec=SimpleNamespace(task=TaskType.BINARY))
    fp = Fingerprint(DriftKind.CONCEPT, 0.5, 0.1)
    book.update(context_key(fp, 'binary'), 'retrain', 100.0)
    for _ in range(6):
        plan, _ = router.decide(fp, ctx, {}, allowed=['noop', 'finetune'], best_available=False)
        assert plan.action in {'noop', 'finetune'}
        book.update(context_key(fp, 'binary'), plan.action, 0)
    for kind, drop, expected in [(DriftKind.NONE, 0, 'noop'), (DriftKind.MIXED, 0.3, 'rollback')]:
        fp = Fingerprint(kind, 0.5, drop)
        plan, _ = router.decide(fp, ctx, {}, allowed=ACTION_NAMES, best_available=True)
        assert plan.action == expected


def test_empty_mask_does_not_expand_to_all_actions():
    book = Playbook(['retrain', 'noop'])
    assert book.select('context', allowed=[]) == 'noop'


@pytest.mark.parametrize('unlabeled', [False, True])
def test_bandit_keeps_predicting_without_update_permission(unlabeled):
    stream = SyntheticStream(SyntheticConfig(n_batches=4, batch_size=40, n_features=4, seed=19))
    batches = list(stream)
    cfg = SystemConfig()
    if not unlabeled:
        cfg.budget.max_compute_seconds = 0
    else:
        for batch in batches[1:]:
            batch.y = None
    emitted = []
    system = SelfEvolvingSystem(stream.spec, cfg, policy_mode='bandit',
                                emit=lambda k, p: emitted.append(k))
    result = system.run(iter(batches))
    assert emitted == [1, 2, 3]
    assert all(d['plan']['action'] == 'noop' for d in result.decisions)
    assert not any(d['promoted'] for d in result.decisions)
