import pytest

from selftrain.actions import ActionContext, ActionExecutor
from selftrain.config import GateConfig, ModelConfig, SystemConfig
from selftrain.data import SyntheticConfig, SyntheticStream
from selftrain.harness import PolicyTrace
from selftrain.loop import SelfEvolvingSystem
from selftrain.types import ActionPlan
from selftrain.types import TaskType
from selftrain import loop


@pytest.mark.parametrize('action', ['finetune', 'reweight', 'retrain', 'switch_model'])
def test_model_seed_reaches_initial_and_candidate_estimators(action):
    cfg = SystemConfig()
    cfg.model.random_state = 123
    cfg.model.params = {'max_iter': 4}
    cfg.budget.bootstrap_timeout_s = None
    stream = SyntheticStream(SyntheticConfig(n_batches=2, batch_size=40, n_features=4))
    batches = list(stream)
    system = SelfEvolvingSystem(stream.spec, cfg)
    system.bootstrap(batches[0].X, batches[0].y)
    assert system.anchor.estimator.random_state == 123
    assert system.static.estimator.random_state == 123
    ctx = ActionContext(stream.spec, 1, batches[0].X, batches[0].y,
        batches[0].X, batches[0].y, system.anchor, system.registry,
        system.best_version, cfg.model)
    candidate = ActionExecutor(cfg.model).execute(action, ctx).model
    assert candidate.estimator.random_state == 123


def test_explicit_estimator_seed_has_documented_priority():
    cfg = ModelConfig(random_state=123, params={'random_state': 99})
    assert cfg.training_params()['random_state'] == 99
    assert cfg.training_params(alternate=True)['random_state'] == 99


@pytest.mark.parametrize('values', [{'val_batches': 0}, {'val_batches': 1.5},
                                    {'purge_batches': -1}, {'purge_batches': True}])
def test_invalid_window_config_rejected(values):
    with pytest.raises(ValueError):
        GateConfig(**values)


@pytest.mark.parametrize('task', list(TaskType))
def test_validation_and_purge_batches_never_enter_candidate_training(monkeypatch, task):
    cfg = SystemConfig()
    cfg.gate.val_batches = 2
    cfg.gate.purge_batches = 1
    cfg.budget.bootstrap_timeout_s = cfg.budget.action_timeout_s = None
    cfg.model.params = {'max_iter': 4}
    stream = SyntheticStream(SyntheticConfig(task=task, n_batches=6, batch_size=40, n_features=4))
    batches = list(stream)
    for batch in batches:
        batch.X[:, 0] = batch.index
    system = SelfEvolvingSystem(stream.spec, cfg)
    original_execute = system.executor.execute
    observed = []
    validation_inputs = []
    original_score = loop.score_model
    def score(model, task, X, y):
        if len(X) == 80:
            validation_inputs.append(set(X[:, 0]))
        return original_score(model, task, X, y)
    monkeypatch.setattr(loop, 'score_model', score)
    def execute(action, ctx):
        observed.append((ctx.batch_index, set(ctx.train_X[:, 0]), set(ctx.recent_X[:, 0])))
        return original_execute(action, ctx)
    monkeypatch.setattr(system.executor, 'execute', execute)
    monkeypatch.setattr(system.policy, 'decide', lambda *a, **kw:
        (ActionPlan('finetune' if 'finetune' in kw['allowed'] else 'noop'), PolicyTrace('rule')))
    monkeypatch.setattr(system, '_allowed_actions', lambda *a: ['noop', 'finetune'])
    result = system.run(iter(batches))
    assert [row[0] for row in observed] == [3, 4, 5]
    for k, training, recent in observed:
        assert training == set(range(k - 2))
        assert recent == {k - 3}
    assert all(not d['plan']['validation']['ready'] for d in result.decisions[:2])
    assert all(d['plan']['validation']['validation_rows'] == 80 for d in result.decisions[2:])
    assert len(system.scores) == 5  # 暖机期间仍输出、评估当前批。
    assert validation_inputs == [pair for k in (3, 4, 5) for pair in [{k - 1, k}] * 3]
