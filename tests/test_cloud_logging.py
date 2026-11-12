import json
from types import SimpleNamespace

import pytest

from selftrain import experiment
from selftrain.config import LogConfig, SystemConfig


def test_cloud_records_include_redacted_config_decisions_and_summary(tmp_path, monkeypatch):
    calls = []
    config = {}
    run = SimpleNamespace(mode="online", url="https://example.test/run", path="u/p/r", id="r",
                          config=config, log=lambda data, **kw: calls.append(data),
                          finish=lambda: calls.append("finished"))
    params = {}

    def init(**kwargs):
        params.update(kwargs)
        return run

    monkeypatch.setattr(experiment, "_HAS_SWAN", True)
    monkeypatch.setattr(experiment, "swanlab", SimpleNamespace(init=init, Text=lambda text: text))
    logger = experiment.ExperimentLogger(LogConfig(backend="swanlab", out_dir=str(tmp_path), swanlab_mode="cloud"))
    cfg = SystemConfig()
    cfg.harness.api_key = "secret-must-not-upload"
    logger.start(cfg.to_dict(), {"name": "synthetic"})
    logger.log_decision({"batch": 1, "plan": {"action": "noop"}})
    logger.log(1, {"system_primary": 0.7})
    logger.finish({"n_batches": 1, "mean_system": 0.7})
    assert params["mode"] == "online"
    assert "secret-must-not-upload" not in json.dumps(config)
    assert config["dataset"]["name"] == "synthetic"
    assert any("decision" in x for x in calls if isinstance(x, dict))
    assert any("artifacts/summary.json" in x for x in calls if isinstance(x, dict))
    assert calls[-1] == "finished"
    assert json.loads((logger.out_dir / "cloud.json").read_text())["url"] == run.url


def test_explicit_cloud_does_not_silently_fall_back(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise OSError("unreachable")

    monkeypatch.setattr(experiment, "_HAS_SWAN", True)
    monkeypatch.setattr(experiment, "swanlab", SimpleNamespace(init=fail))
    with pytest.raises(RuntimeError, match="初始化失败"):
        experiment.ExperimentLogger(LogConfig(backend="swanlab", out_dir=str(tmp_path)))


def test_sync_failure_preserves_local_record(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("upload failed")

    run = SimpleNamespace(mode="online", url="https://example.test/run", path="u/p/r", id="r", log=fail)
    monkeypatch.setattr(experiment, "_HAS_SWAN", True)
    monkeypatch.setattr(experiment, "swanlab", SimpleNamespace(init=lambda **kw: run))
    logger = experiment.ExperimentLogger(LogConfig(backend="swanlab", out_dir=str(tmp_path)))
    with pytest.raises(RuntimeError, match="同步失败"):
        logger.log(1, {"system_primary": 0.7})
    assert json.loads((logger.out_dir / "records.jsonl").read_text())["system_primary"] == 0.7
