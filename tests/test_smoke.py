"""快速自检：跑 `pytest -q` 应该全绿。

重点守住三个不变量：
    1. 当前批绝不进训练集（防泄漏）
    2. 静态基线永不更新
    3. 闸门在退化时一定拒绝
"""

from __future__ import annotations

import numpy as np
import pytest

from selftrain.actions import ActionExecutor
from selftrain.config import GateConfig, SystemConfig
from selftrain.data import SyntheticConfig, SyntheticStream
from selftrain.drift import UnlabeledDriftDetector, ks_2samp, psi
from selftrain.evaluation import PromotionGate
from selftrain.experiment import NullLogger
from selftrain.loop import SelfEvolvingSystem
from selftrain.playbook import Playbook
from selftrain.types import DatasetSpec, TaskType


# --------------------------------------------------------------------------


def _spec(task=TaskType.BINARY, d=8) -> DatasetSpec:
    return DatasetSpec(name="t", task=task, feature_names=[f"f{i}" for i in range(d)])


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    same = rng.normal(0, 1, 2000)
    shifted = rng.normal(1.5, 1, 2000)
    assert psi(ref, same) < 0.05
    assert psi(ref, shifted) > 0.5


def test_ks_pvalue_direction():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 500)
    b = rng.normal(0, 1, 500)
    c = rng.normal(2, 1, 500)
    _, p_same = ks_2samp(a, b)
    _, p_diff = ks_2samp(a, c)
    assert p_same > 0.01
    assert p_diff < 1e-6


def test_detector_severity_monotone():
    rng = np.random.default_rng(2)
    X0 = rng.normal(0, 1, (400, 6))
    det = UnlabeledDriftDetector(SystemConfig().drift)
    det.fit(X0, [f"f{i}" for i in range(6)])
    s_same = det.score(rng.normal(0, 1, (400, 6))).severity
    s_shift = det.score(rng.normal(2.0, 1, (400, 6))).severity
    assert s_shift > s_same
    assert 0.0 <= s_same <= 1.0 and 0.0 <= s_shift <= 1.0


def test_gate_rejects_degradation():
    gate = PromotionGate(GateConfig(min_gain=0.005, max_drop=0.01))
    r = gate.check(TaskType.BINARY, {"f1_macro": 0.8}, {"f1_macro": 0.7})
    assert not r.accept
    r2 = gate.check(TaskType.BINARY, {"f1_macro": 0.8}, {"f1_macro": 0.85})
    assert r2.accept


def test_gate_respects_static_floor():
    gate = PromotionGate(GateConfig(min_gain=0.001, static_guard=0.0))
    r = gate.check(
        TaskType.BINARY, {"f1_macro": 0.5}, {"f1_macro": 0.52}, {"f1_macro": 0.6}
    )
    assert not r.accept        # 比锚点好，但低于静态基线


def test_playbook_explores_then_exploits():
    """UCB1：先轮流试遍所有臂，再收敛到平均收益最高的那个。"""
    pb = Playbook(["noop", "calibrate", "retrain"])
    key = "concept|hi"

    seen = []
    for _ in range(6):
        a = pb.select(key)
        seen.append(a)
        pb.update(key, a, 0.0)
    assert len(set(seen)) >= 2, f"应当探索多个臂，实际只用了 {set(seen)}"

    # 再让 calibrate 稳定拿高收益，看它是否被选中
    pb2 = Playbook(["noop", "calibrate", "retrain"])
    for _ in range(12):
        a = pb2.select(key)
        pb2.update(key, a, 0.05 if a == "calibrate" else 0.0)
    assert pb2.best_action(key) == "calibrate"


# --------------------------------------------------------------------------
# 端到端 + 泄漏不变量
# --------------------------------------------------------------------------


class _SpyExecutor(ActionExecutor):
    """记录每次执行时训练集大小，用来验证"当前批没进训练"。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls = []

    def execute(self, name, ctx):
        self.calls.append({"action": name, "n_train": len(ctx.train_X), "batch": ctx.batch_index})
        return super().execute(name, ctx)


@pytest.mark.parametrize('policy', ['rule', 'bandit'])
def test_no_current_batch_in_training(policy):
    cfg = SystemConfig()
    cfg.budget.action_timeout_s = None  # 观察父进程中的训练参数；隔离执行另测。
    cfg.seed = 0
    stream = SyntheticStream(
        SyntheticConfig(task=TaskType.BINARY, n_batches=8, batch_size=100,
                        n_features=8, scenario="covariate", drift_start=2, seed=0)
    )
    sys_ = SelfEvolvingSystem(_spec(), cfg, logger=NullLogger(), policy_mode=policy)
    spy = _SpyExecutor(cfg.model)
    sys_.executor = spy
    sys_.run(stream)

    assert spy.calls, "应当至少执行过一次动作"
    # bootstrap 用掉第 0 批；处理第 k 批时训练集 = k 批
    for c in spy.calls:
        expected = c["batch"] * 100
        assert c["n_train"] == expected, f"第{c['batch']}批训练集应含 {expected} 行，实际 {c['n_train']}"


@pytest.mark.parametrize('policy', ['rule', 'bandit'])
def test_static_model_never_updates(policy):
    cfg = SystemConfig()
    stream = SyntheticStream(
        SyntheticConfig(n_batches=8, batch_size=100, n_features=8, scenario="mixed", seed=1)
    )
    sys_ = SelfEvolvingSystem(_spec(), cfg, logger=NullLogger(), policy_mode=policy)
    sys_.run(stream)
    assert sys_.static is not None
    assert sys_.static.version == "static"
    assert sys_.static.trained_on == 100      # 永远只有 bootstrap 那 100 行


def test_end_to_end_beats_static_on_drift():
    cfg = SystemConfig()
    stream = SyntheticStream(
        SyntheticConfig(n_batches=14, batch_size=250, n_features=12,
                        scenario="mixed", drift_start=2, seed=7)
    )
    sys_ = SelfEvolvingSystem(_spec(TaskType.BINARY, 12), cfg,
                              logger=NullLogger(), policy_mode="rule")
    res = sys_.run(stream)
    s = res.summary
    assert s["n_batches"] >= 10
    assert s["mean_system"] >= s["mean_static"] - 0.01
    assert s["budget"]["total_cost"] > 0


def test_regression_path():
    cfg = SystemConfig()
    stream = SyntheticStream(
        SyntheticConfig(task=TaskType.REGRESSION, n_batches=10, batch_size=200,
                        n_features=10, scenario="concept", seed=3)
    )
    sys_ = SelfEvolvingSystem(_spec(TaskType.REGRESSION, 10), cfg,
                              logger=NullLogger(), policy_mode="rule")
    res = sys_.run(stream)
    assert res.summary["mean_static"] > 0


def test_prediction_sink(tmp_path):
    from selftrain.sink import PredictionSink

    sink = PredictionSink(str(tmp_path))
    sink(0, np.array([1, 0, 1]))
    assert (tmp_path / "pred_0000.csv").exists()
    assert (tmp_path / "index.json").exists()
