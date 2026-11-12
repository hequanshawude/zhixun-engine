import json
import pandas as pd
import pytest

from selftrain.config import SystemConfig, LogConfig
from selftrain.data import SyntheticConfig, SyntheticStream, CSVBatchStream
from selftrain.experiment import ExperimentLogger
from selftrain.loop import SelfEvolvingSystem
from selftrain.types import TaskType
from selftrain.cli import main


def test_multibatch_bootstrap():
    stream = SyntheticStream(SyntheticConfig(n_batches=3, batch_size=30, n_features=4))
    system = SelfEvolvingSystem(stream.spec)
    system.run(iter(stream), bootstrap_from_stream=2)
    assert system.static.trained_on == 60
    assert len(system.bootstrap_X) == 60


def test_regression_best_uses_minimum():
    stream = SyntheticStream(SyntheticConfig(task=TaskType.REGRESSION, n_batches=5, batch_size=40, n_features=4))
    system = SelfEvolvingSystem(stream.spec)
    system.run(iter(stream))
    assert system.best_primary == min(s['system'] for s in system.scores)


def test_missing_features_rejected(tmp_path):
    stream = SyntheticStream(SyntheticConfig(n_features=4))
    pd.DataFrame({'wrong': [1], 'y': [0]}).to_csv(tmp_path / 'batch_000.csv', index=False)
    with pytest.raises(ValueError, match='特征列'):
        list(CSVBatchStream(str(tmp_path), stream.spec))


def test_config_secret_redacted(tmp_path):
    cfg = SystemConfig()
    cfg.harness.api_key = 'secret-test-key'
    logger = ExperimentLogger(LogConfig(backend='jsonl', out_dir=str(tmp_path)))
    logger.start(cfg.to_dict(), {})
    assert 'secret-test-key' not in (logger.out_dir / 'meta.json').read_text()
    assert cfg.harness.api_key == 'secret-test-key'


def test_unlabeled_cli_outputs_every_batch(tmp_path):
    batches = list(SyntheticStream(SyntheticConfig(n_batches=3, batch_size=30, n_features=4)))
    train = pd.DataFrame(batches[0].X, columns=batches[0].feature_names)
    train['y'] = batches[0].y
    train.to_csv(tmp_path / 'train.csv', index=False)
    data = tmp_path / 'data'
    data.mkdir()
    for k, b in enumerate(batches[1:]):
        pd.DataFrame(b.X, columns=b.feature_names).to_csv(data / f'batch_{k:03d}.csv', index=False)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'log': {'backend': 'jsonl', 'out_dir': str(tmp_path / 'runs')}}))
    out = tmp_path / 'preds'
    assert main(['run', '--data-dir', str(data), '--train-csv', str(tmp_path / 'train.csv'),
                 '--unlabeled', '--pred-dir', str(out), '--config', str(config), '--quiet']) == 0
    assert len(list(out.glob('pred_*.csv'))) == 2
