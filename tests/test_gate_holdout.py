import numpy as np

from selftrain import loop
from selftrain.actions import ExecutionResult
from selftrain.config import SystemConfig
from selftrain.harness import PolicyTrace
from selftrain.loop import SelfEvolvingSystem
from selftrain.types import ActionPlan, Batch, DatasetSpec, TaskType


class _OneRegressor:
    version = "candidate-one"

    def predict(self, X):
        return np.ones(len(X))


def test_independent_gate_rejects_candidate_and_stays_out_of_training(monkeypatch):
    cfg = SystemConfig()
    cfg.model.params = {"max_iter": 4}
    cfg.budget.bootstrap_timeout_s = cfg.budget.action_timeout_s = None
    spec = DatasetSpec("holdout", TaskType.REGRESSION, ["f0"])
    system = SelfEvolvingSystem(spec, cfg)
    train = (np.zeros((40, 1), dtype=np.float32), np.zeros(40))
    gate = (np.full((20, 1), 2, dtype=np.float32), np.zeros(20))
    current = Batch(0, np.ones((20, 1), dtype=np.float32), np.ones(20), ["f0"])
    observed = []

    monkeypatch.setattr(loop, "isolated_call", lambda function, *args, **kwargs: function(*args))
    monkeypatch.setattr(system, "_allowed_actions", lambda *args: ["noop", "finetune"])
    monkeypatch.setattr(system.policy, "decide", lambda *args, **kwargs:
                        (ActionPlan("finetune"), PolicyTrace("rule")))

    def execute(action, ctx):
        observed.append(set(ctx.train_X[:, 0]))
        return ExecutionResult(_OneRegressor(), "fixed candidate", 0.0)

    monkeypatch.setattr(system.executor, "execute", execute)
    result = system.run(iter([current]), bootstrap_data=train, gate_data=gate)

    assert observed == [{0.0}]
    assert result.summary["n_promotions"] == 0
    assert system.anchor.version == "base-0"
    validation = result.decisions[0]["plan"]["validation"]
    assert validation["holdout_rows"] == 20
    assert validation["holdout_accept"] is False
    assert "独立设备验证拒绝" in result.decisions[0]["reason"]
