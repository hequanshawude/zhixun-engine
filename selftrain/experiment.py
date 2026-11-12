"""实验记录。

比赛里"记录规范且完整 = 满分"，这是最不该丢的分。所以这里做三件事：
    1. 双写：SwanLab 云端 + 本地 jsonl（断网/没装 swanlab 也不能丢记录）
    2. 每条记录带 run_id / batch / 配置哈希 / 数据版本，可回溯
    3. 结束时导出 summary.json + decisions.jsonl，直接能贴进报告
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import LogConfig

try:
    import swanlab  # type: ignore
    _HAS_SWAN = True
except Exception:  # pragma: no cover
    swanlab = None  # type: ignore
    _HAS_SWAN = False


def _config_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _git_commit() -> Optional[str]:
    try:
        head = Path(__file__).resolve().parent.parent / ".git" / "HEAD"
        if head.exists():
            return head.read_text().strip()[:40]
    except Exception:
        pass
    return None


class ExperimentLogger:
    """统一记录器。``backend='auto'`` 时装了 swanlab 就用，否则纯本地。"""

    def __init__(self, cfg: LogConfig, run_name: str = "run"):
        self.cfg = cfg
        self.run_name = run_name
        self.run_id = f"{run_name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.out_dir = Path(cfg.out_dir) / self.run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._records: List[Dict[str, Any]] = []
        self._decisions: List[Dict[str, Any]] = []
        self._swan = False
        self._swan_run = None
        self._last_step = 0
        self.cloud_url = None
        self.backend = cfg.backend

        if cfg.backend in ("auto", "swanlab") and _HAS_SWAN:
            try:
                self._swan_run = swanlab.init(
                    project=cfg.project,
                    workspace=cfg.workspace,
                    experiment_name=self.run_id,
                    mode="online" if cfg.swanlab_mode == "cloud" else cfg.swanlab_mode,
                    on_online_error="raise",
                    log_dir=str(self.out_dir / "swanlab"),
                    config={},
                )
                if self._swan_run.mode == "online":
                    self.cloud_url = self._swan_run.url
                    self._write_json("cloud.json", {
                        "url": self.cloud_url, "path": self._swan_run.path,
                        "id": self._swan_run.id,
                    })
                self._swan = True
                self.backend = "swanlab"
            except Exception as exc:  # pragma: no cover
                if cfg.backend == "swanlab":
                    raise RuntimeError("SwanLab 初始化失败，尚未完成云端连接") from exc
                print(f"[logger] swanlab 初始化失败，降级为本地记录: {exc}")
                self.backend = "jsonl"
        elif cfg.backend == "swanlab" and not _HAS_SWAN:
            raise RuntimeError("未安装 swanlab，无法使用云端记录")

    # ---------- 生命周期 ----------
    def start(self, config: Dict[str, Any], spec: Dict[str, Any]) -> None:
        config = json.loads(json.dumps(config))
        if "harness" in config:
            config["harness"]["api_key"] = "[REDACTED]"
        meta = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "config_hash": _config_hash(config),
            "config": config,
            "dataset": spec,
        }
        self._write_json("meta.json", meta)
        if self._swan:
            try:
                self._swan_run.config.update({
                    "run_id": self.run_id, "config_hash": meta["config_hash"],
                    "system": config, "dataset": spec,
                })
            except Exception as exc:
                self._cloud_error(exc)

    def log(self, step: int, metrics: Dict[str, Any], tags: Optional[List[str]] = None) -> None:
        rec = {"step": step, "t": round(time.time(), 3), **metrics}
        if tags:
            rec["tags"] = tags
        self._records.append(rec)
        self._last_step = max(self._last_step, step)
        # 先保存本地，再尝试云端同步。
        with (self.out_dir / "records.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        if self._swan:
            try:
                self._swan_run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)
            except Exception as exc:
                self._cloud_error(exc)

    def log_decision(self, decision: Dict[str, Any]) -> None:
        self._decisions.append(decision)
        with (self.out_dir / "decisions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(decision, ensure_ascii=False, default=str) + "\n")
        if self._swan:
            try:
                self._swan_run.log({"decision": swanlab.Text(json.dumps(decision, ensure_ascii=False))},
                                   step=decision["batch"])
            except Exception as exc:
                self._cloud_error(exc)

    def log_artifact(self, name: str, payload: Dict[str, Any]) -> None:
        self._write_json(name, payload)
        if self._swan:
            try:
                self._swan_run.log({f"artifacts/{name}": swanlab.Text(json.dumps(payload, ensure_ascii=False))},
                                   step=self._last_step + 1)
            except Exception as exc:
                self._cloud_error(exc)

    def finish(self, summary: Dict[str, Any]) -> None:
        summary = {**summary, "run_id": self.run_id, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.log_artifact("summary.json", summary)
        if self._swan:
            try:
                self._swan_run.log({f"summary/{k}": v for k, v in summary.items()
                                   if isinstance(v, (int, float))}, step=self._last_step + 1)
                self._swan_run.finish()
            except Exception as exc:
                self._cloud_error(exc)

    def abort(self, error: str, initialization=None) -> None:
        payload = {'error': error, 'initialization': initialization,
                   'run_id': self.run_id, 'status': 'failed'}
        self._write_json('failure.json', payload)
        if self._swan:
            try:
                self._swan_run.log({'failure': swanlab.Text(json.dumps(payload, ensure_ascii=False))},
                                   step=self._last_step + 1)
            finally:
                self._swan_run.finish(state='crashed', error=error)

    def _cloud_error(self, exc: Exception) -> None:
        if self.cfg.backend == "swanlab":
            raise RuntimeError("SwanLab 同步失败，本地记录已保留") from exc
        print(f"[logger] SwanLab 同步失败，本地记录已保留: {exc}")

    # ---------- 内部 ----------
    def _write_json(self, name: str, payload: Dict[str, Any]) -> None:
        (self.out_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    @property
    def records(self) -> List[Dict[str, Any]]:
        return self._records


class NullLogger:
    """测试用空记录器。"""

    def __init__(self, *a, **k):
        self.records = []
        self.run_id = "null"

    def start(self, *a, **k): ...
    def log(self, *a, **k): ...
    def log_decision(self, *a, **k): ...
    def log_artifact(self, *a, **k): ...
    def finish(self, *a, **k): ...
