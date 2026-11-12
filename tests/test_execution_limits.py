import json
import time
from pathlib import Path

import numpy as np
import pytest

from selftrain.cli import _run
from selftrain.config import SystemConfig
from selftrain.data import SyntheticConfig, SyntheticStream
from selftrain.loop import SelfEvolvingSystem
from selftrain.types import ActionPlan
from selftrain.harness import PolicyTrace
from selftrain.worker import ExecutionTimeout, isolated_call


def test_worker_returns_result_and_reports_exception():
    assert isolated_call(pow, 2, 5, timeout_s=15) == 32
    with pytest.raises(RuntimeError, match='ZeroDivisionError'):
        isolated_call(divmod, 1, 0, timeout_s=15)


def test_worker_terminates_stuck_operation():
    started = time.perf_counter()
    with pytest.raises(ExecutionTimeout, match='工作进程已终止'):
        isolated_call(time.sleep, 30, timeout_s=0.5)
    assert time.perf_counter() - started < 10


def test_initialization_is_counted_once_without_consuming_iterations():
    stream = SyntheticStream(SyntheticConfig(n_batches=3, batch_size=40, n_features=4))
    batches = list(stream)
    system = SelfEvolvingSystem(stream.spec)
    system.bootstrap(batches[0].X, batches[0].y)
    initial = system.budget.initialization_seconds
    assert initial > 0
    assert system.budget.compute_seconds == initial
    assert system.budget.iterations == 0
    system.run(iter(batches[1:]))
    assert system.budget.initialization_seconds == initial
    assert system.budget.compute_seconds == pytest.approx(
        initial + sum(r.compute_seconds for r in system.budget.records))
    assert system.budget.iterations == 2


def test_timed_out_candidates_never_promote_and_predictions_continue(monkeypatch):
    stream = SyntheticStream(SyntheticConfig(n_batches=4, batch_size=40, n_features=4))
    cfg = SystemConfig()
    cfg.budget.action_timeout_s = 0.001
    emitted = []
    system = SelfEvolvingSystem(stream.spec, cfg, emit=lambda k, p: emitted.append((k, p)))
    monkeypatch.setattr(system.policy, 'decide', lambda *args, **kw:
                        (ActionPlan('finetune'), PolicyTrace('rule')))
    monkeypatch.setattr(system, '_allowed_actions', lambda *args: ['noop', 'finetune'])
    result = system.run(iter(stream))
    assert [k for k, _ in emitted] == [1, 2, 3]
    assert all(np.isfinite(p).all() for _, p in emitted)
    assert result.summary['n_timeouts'] == 3
    assert result.summary['n_promotions'] == 0
    assert all(d['metrics_candidate'] == {} for d in result.decisions)


def test_initialization_timeout_preserves_failure_record(tmp_path):
    cfg = SystemConfig()
    cfg.budget.bootstrap_timeout_s = 0.001
    cfg.log.backend = 'jsonl'
    cfg.log.out_dir = str(tmp_path)
    stream = SyntheticStream(SyntheticConfig(n_batches=2, batch_size=40, n_features=4))
    with pytest.raises(ExecutionTimeout):
        _run(cfg, stream.spec, iter(stream), 'rule', 'init-timeout', quiet=True)
    failure = json.loads(next(Path(tmp_path).glob('*/failure.json')).read_text(encoding='utf-8'))
    assert failure['status'] == 'failed'
    assert failure['initialization']['status'] == 'failed'
    assert failure['initialization']['compute_s'] > 0
