import numpy as np
import pytest

from selftrain.checkpoint import CheckpointStore, atomic_write
from selftrain.config import SystemConfig
from selftrain.data import SyntheticConfig, SyntheticStream
from selftrain.loop import SelfEvolvingSystem
from selftrain.sink import PredictionSink


def setup_run():
    cfg = SystemConfig()
    cfg.budget.bootstrap_timeout_s = None
    cfg.budget.action_timeout_s = None
    stream = SyntheticStream(SyntheticConfig(n_batches=6, batch_size=50, n_features=4, seed=17))
    return cfg, stream.spec, list(stream)


def test_resume_after_prediction_before_commit(tmp_path):
    cfg, spec, batches = setup_run()
    reference = {}
    full = SelfEvolvingSystem(spec, cfg, emit=lambda k, p: reference.update({k: p.copy()}))
    full.run(iter(batches))
    sink = PredictionSink(str(tmp_path / 'preds'))
    checkpoint = tmp_path / 'state.bin'
    def interrupted(k, p):
        sink(k, p)
        if k == 3:
            raise RuntimeError('simulated interruption')
    system = SelfEvolvingSystem(spec, cfg, emit=interrupted)
    with pytest.raises(RuntimeError, match='interruption'):
        system.run(iter(batches), checkpoint_path=checkpoint, checkpoint_outputs=sink.snapshot)
    recovered_sink = PredictionSink(str(tmp_path / 'preds'), resume=True)
    recovered = SelfEvolvingSystem(spec, cfg, emit=recovered_sink)
    result = recovered.run(iter(batches), checkpoint_path=checkpoint, resume=True,
        checkpoint_outputs=recovered_sink.snapshot, resume_validator=recovered_sink.validate_completed)
    assert result.summary['n_batches'] == 5
    assert recovered.budget.iterations == 5
    assert recovered.initialization == system.initialization
    assert len(recovered_sink.snapshot()) == 5
    for k, predictions in reference.items():
        actual = np.loadtxt(tmp_path / 'preds' / f'pred_{k:04d}.csv', delimiter=',', skiprows=1)[:, 1]
        np.testing.assert_array_equal(actual, predictions)


def test_resume_rejects_changed_prefix_and_missing_batches(tmp_path):
    cfg, spec, batches = setup_run()
    path = tmp_path / 'state.bin'
    SelfEvolvingSystem(spec, cfg).run(iter(batches[:3]), checkpoint_path=path)
    with pytest.raises(ValueError, match='缺少'):
        SelfEvolvingSystem(spec, cfg).run(iter(batches[:2]), checkpoint_path=path, resume=True)
    batches[0].X[0, 0] += 1
    with pytest.raises(ValueError, match='发生变化'):
        SelfEvolvingSystem(spec, cfg).run(iter(batches), checkpoint_path=path, resume=True)


def test_checkpoint_rejects_corruption_and_config_change(tmp_path):
    cfg, spec, batches = setup_run()
    path = tmp_path / 'state.bin'
    SelfEvolvingSystem(spec, cfg).run(iter(batches[:2]), checkpoint_path=path)
    cfg.seed += 1
    with pytest.raises(ValueError, match='不一致'):
        SelfEvolvingSystem(spec, cfg).run(iter(batches), checkpoint_path=path, resume=True)
    cfg.seed -= 1
    content = path.read_bytes()
    path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(ValueError, match='损坏'):
        SelfEvolvingSystem(spec, cfg).run(iter(batches), checkpoint_path=path, resume=True)


def test_checkpoint_excludes_key_and_restores_bandit_state(tmp_path):
    cfg, spec, batches = setup_run()
    cfg.harness.api_key = 'test-secret-never-serialize'
    path = tmp_path / 'state.bin'
    original = SelfEvolvingSystem(spec, cfg, policy_mode='bandit')
    original.run(iter(batches[:3]), checkpoint_path=path)
    assert b'test-secret-never-serialize' not in path.read_bytes()
    restored = SelfEvolvingSystem(spec, cfg, policy_mode='bandit')
    CheckpointStore(path, restored).load(restored)
    assert restored.playbook.stats() == original.playbook.stats()
    assert restored.policy.playbook is restored.playbook
    assert restored.budget.compute_seconds == original.budget.compute_seconds


def test_atomic_failure_keeps_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'state.bin'
    path.write_bytes(b'previous')
    def fail(*args):
        raise OSError('interrupted replace')
    monkeypatch.setattr('selftrain.checkpoint.os.replace', fail)
    with pytest.raises(OSError):
        atomic_write(path, b'next')
    assert path.read_bytes() == b'previous'
    assert list(tmp_path.glob('*.tmp')) == []


def test_prediction_replay_rejects_conflict_and_missing_output(tmp_path):
    sink = PredictionSink(str(tmp_path))
    sink(1, np.array([0, 1]))
    expected = sink.snapshot()
    resumed = PredictionSink(str(tmp_path), resume=True)
    resumed(1, np.array([0, 1]))
    assert len(resumed.snapshot()) == 1
    with pytest.raises(ValueError, match='拒绝覆盖'):
        resumed(1, np.array([1, 1]))
    with pytest.raises(ValueError, match='缺少'):
        resumed.validate_completed([1, 2], expected)
    (tmp_path / 'pred_0001.csv').write_text('broken')
    with pytest.raises(ValueError, match='不一致'):
        PredictionSink(str(tmp_path), resume=True)
