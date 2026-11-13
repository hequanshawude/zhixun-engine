"""CPU 续训驱动、运行时补丁与 Net2Net 加宽的测试。

这些是平台实际执行的代码路径，必须有覆盖。部分用例依赖随包的 CPU 模型 A，
缺失时自动跳过（交付 ZIP 不含数据）。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from selftrain.cpu import (
    LR_TABLE,
    TablePolicy,
    _prepare_package,
    load_package,
    run_cpu_packages,
    run_cpu_rounds,
    _stability,
    LLMPolicy,
    CpuPackage,
)
from selftrain.patcher import OLD_LOOP, OLD_REPORT, ensure_best_epoch_patch
from selftrain import cli, cpu
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PKG_A = ROOT / "data" / "raw" / "cpu" / "a"

needs_pkg = pytest.mark.skipif(not PKG_A.exists(), reason="需要随包 CPU 模型 A")


# --------------------------------------------------------------------------
# 不依赖数据的用例
# --------------------------------------------------------------------------


def test_load_package_rejects_missing(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_package(tmp_path)
    assert "缺少必需内容" in str(exc.value)


def test_table_policy_cycles_and_is_deterministic():
    policy = TablePolicy()
    picks = [policy.choose({"round": r})[0] for r in range(1, len(LR_TABLE) * 2 + 1)]
    assert picks[: len(LR_TABLE)] == LR_TABLE
    assert picks[len(LR_TABLE):] == LR_TABLE          # 循环
    assert TablePolicy().choose({"round": 1}) == policy.choose({"round": 1})


def test_stability_reports_worst_candidate_in_correct_direction():
    ledger = [{"round": i, "value": v, "decision": "reject"}
              for i, v in enumerate([0.7, 0.6, 0.8], 1)]
    pkg = SimpleNamespace(higher_is_better=True)
    assert _stability(ledger, pkg, 0.7)["worst_candidate"] == 0.6
    pkg.higher_is_better = False
    assert _stability(ledger, pkg, 0.7)["worst_candidate"] == 0.8


def test_llm_policy_records_real_decision_source():
    class Client:
        def chat(self, messages):
            return '{"learning_rate": 0.003, "epochs": 5, "reason": "ok"}', SimpleNamespace(total=7)
    policy = LLMPolicy(SimpleNamespace(credential_source="env"), client=Client())
    assert policy.choose({"round": 1})[:2] == (0.003, 5)
    assert policy.last_source == "llm" and policy.tokens == 7

    class Broken:
        def chat(self, messages):
            raise RuntimeError("offline")
    policy.client = Broken()
    assert policy.choose({"round": 2})[0] in LR_TABLE
    assert policy.last_source == "table_fallback"

    strict_policy = LLMPolicy(SimpleNamespace(credential_source="env"), client=Broken(), strict=True)
    with pytest.raises(RuntimeError, match="offline"):
        strict_policy.choose({"round": 2})


def test_cpu_multi_package_creates_separate_cloud_records(monkeypatch, tmp_path):
    created = []

    class Logger:
        def __init__(self, config, run_name):
            self.run_name = run_name
            created.append(self)
        def start(self, config, spec):
            self.config = config

    def fake_run(root, out, rounds, policy_factory, logger_factory, widen_to):
        a, b = tmp_path / "a", tmp_path / "b"
        first, second = logger_factory(a), logger_factory(b)
        assert first is not second
        assert first.run_name == "cpu-a" and second.run_name == "cpu-b"
        assert policy_factory(a) is not policy_factory(b)
        return {"failed": [], "results": {"a": {}, "b": {}}}

    monkeypatch.setattr(cli, "ExperimentLogger", Logger)
    monkeypatch.setattr(cpu, "run_cpu_packages", fake_run)
    args = SimpleNamespace(package=None, packages_root=str(tmp_path), out=str(tmp_path / "out"),
                           rounds=10, driver="table", model="Qwen/Qwen3.8-27B",
                           widen="0", cloud=True)
    assert cli.cmd_cpu(args) == 0
    assert len(created) == 2


def test_patcher_is_idempotent(tmp_path):
    train = tmp_path / "train.py"
    train.write_text("prefix\n" + OLD_LOOP + "    pass\n" + OLD_REPORT + "\nsuffix\n",
                     encoding="utf-8")
    assert ensure_best_epoch_patch(train) == "patched"
    once = train.read_text(encoding="utf-8")
    assert "selected_epoch" in once
    assert ensure_best_epoch_patch(train) == "already"     # 再打一次应当跳过
    assert train.read_text(encoding="utf-8") == once


def test_patcher_skips_unknown_source(tmp_path):
    train = tmp_path / "train.py"
    train.write_text("print('unrelated')\n", encoding="utf-8")
    assert ensure_best_epoch_patch(train) == "skipped"
    assert train.read_text(encoding="utf-8") == "print('unrelated')\n"


# --------------------------------------------------------------------------
# 依赖随包模型的用例
# --------------------------------------------------------------------------


@needs_pkg
def test_prepare_package_patches_copy(tmp_path):
    pkg = tmp_path / "pkg"
    shutil.copytree(PKG_A, pkg)
    loaded = load_package(pkg)
    work = _prepare_package(loaded, tmp_path / "out")
    assert work.root != loaded.root
    assert (work.root / "public" / "train.jsonl").exists()
    patched = (work.root / "baselines" / "train.py").read_text(encoding="utf-8")
    assert "selected_epoch" in patched
    # 原包不能被就地修改
    assert work.root != pkg


@needs_pkg
def test_run_rounds_end_to_end(tmp_path):
    pkg = tmp_path / "pkg"
    shutil.copytree(PKG_A, pkg)
    out = tmp_path / "out"

    summary = run_cpu_rounds(pkg, out, rounds=2, policy=TablePolicy(), widen_to=None)

    assert (out / "final-checkpoint.npz").exists()
    ledger = [json.loads(l) for l in (out / "rounds-ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(ledger) == 2
    assert {r["decision"] for r in ledger} <= {"accept", "reject"}
    assert summary["rounds"] == 2
    assert "start_metric" in summary
    assert summary["stability"]["monotone_deployment"] is True
    # 部署性能不得低于起点
    if summary["metric"] == "macro_f1":
        assert summary["final"] >= summary["start_metric"] - 1e-9
    else:
        assert summary["final"] <= summary["start_metric"] + 1e-9


@needs_pkg
def test_run_rounds_with_widen(tmp_path):
    pkg = tmp_path / "pkg"
    shutil.copytree(PKG_A, pkg)
    out = tmp_path / "out"
    summary = run_cpu_rounds(pkg, out, rounds=1, policy=TablePolicy(), widen_to=48)
    assert (out / "_widened" / "checkpoint.npz").exists()
    assert (out / "_widened" / "profile.json").exists()
    profile = json.loads((out / "_widened" / "profile.json").read_text(encoding="utf-8"))
    assert profile["hidden"] == 48
    assert summary["rounds"] == 1


@needs_pkg
def test_widen_preserves_function(tmp_path):
    import sys

    sys.path.insert(0, str(PKG_A))
    from baselines.model import Model

    from selftrain.widen import widen

    src = tmp_path / "src.npz"
    shutil.copy2(PKG_A / "checkpoint.npz", src)
    dst = tmp_path / "wide.npz"
    info = widen(PKG_A, src, 48, dst, None)

    assert info["old_hidden"] == 32
    assert info["new_hidden"] == 48
    assert info["forward_drift_ratio"] < 0.02

    widened = Model.load(dst)
    assert widened.meta["hidden"] == 48
    probe = np.random.default_rng(0).normal(0, 1, size=(64, len(widened.meta["paths"])))
    before = Model.load(src).predict(probe)
    after = widened.predict(probe)
    scale = max(float(np.max(np.abs(before))), 1e-9)
    assert np.max(np.abs(before - after)) / scale < 0.02


@needs_pkg
def test_widen_rejects_narrower(tmp_path):
    from selftrain.widen import widen

    src = tmp_path / "src.npz"
    shutil.copy2(PKG_A / "checkpoint.npz", src)
    with pytest.raises(ValueError):
        widen(PKG_A, src, 16, tmp_path / "x.npz", None)


@needs_pkg
def test_run_packages_isolates_failure(tmp_path):
    good = tmp_path / "root" / "a"
    bad = tmp_path / "root" / "broken"
    shutil.copytree(PKG_A, good)
    bad.mkdir(parents=True)
    (bad / "training.json").write_text(json.dumps({
        "profile": "A", "task": "classification", "validation_after": {"macro_f1": 0.5}}),
        encoding="utf-8")
    # bad 缺 baselines/public/checkpoint，应当失败但不影响 good

    summary = run_cpu_packages(tmp_path / "root", tmp_path / "out", rounds=1,
                               widen_to=None)
    assert summary["packages"] == 2
    assert summary["failed"] == ["broken"]
    assert "error" in summary["results"]["broken"]
    assert "final" in summary["results"]["a"]
