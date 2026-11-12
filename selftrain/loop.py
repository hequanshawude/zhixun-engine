"""主循环：一个 batch 一次"检测 → 归因 → 决策 → 实验 → 闸门 → 记账"。

关键协议（决定评分是否有效）：
    用 ≤ k-1 的数据训练候选，用第 k 批做验证，避免信息泄漏；
    第 k 批的预测仍然由旧锚点给出，所以"边跑边学"不会作弊。
"""

from __future__ import annotations

import time
from dataclasses import replace
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

from .actions import ACTION_NAMES, ActionContext, ActionExecutor, estimate_cost, plausible_actions
from .attribution import attribute, model_importance
from .budget import Ledger
from .config import SystemConfig
from .drift import LabeledDriftDetector, UnlabeledDriftDetector
from .evaluation import (
    PromotionGate,
    primary_metric,
    primary_value,
    score_model,
)
from .experiment import NullLogger
from .harness import PolicyRouter, build_state_card, validate_plan
from .models import ModelBundle, fit_model
from .playbook import Playbook, context_key
from .types import Batch, DatasetSpec, Decision, DriftKind, Fingerprint
from .worker import ExecutionTimeout, isolated_call
from .checkpoint import CheckpointStore, data_hash


def _fit_initial_models(cfg, spec, X, y):
    anchor = fit_model(cfg.default_kind, spec.task, X, y, params=cfg.training_params(),
                       version='base-0', note='bootstrap')
    static = fit_model(cfg.default_kind, spec.task, X, y, params=cfg.training_params(),
                       version='static', note='静态基线（永不更新）')
    return anchor, static


@dataclass
class RunResult:
    summary: Dict[str, Any]
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[tuple[int, np.ndarray]] = field(default_factory=list)


class SelfEvolvingSystem:
    def __init__(
        self,
        spec: DatasetSpec,
        cfg: Optional[SystemConfig] = None,
        logger: Optional[Any] = None,
        policy_mode: str = "rule",
        emit: Optional[Callable[[int, np.ndarray], None]] = None,
    ):
        self.spec = spec
        self.cfg = cfg or SystemConfig()
        self.logger = logger or NullLogger()
        self.emit = emit

        self.drift_u = UnlabeledDriftDetector(self.cfg.drift)
        self.drift_l = LabeledDriftDetector(self.cfg.drift)
        self.gate = PromotionGate(self.cfg.gate)
        self.budget = Ledger(self.cfg.budget)
        self.executor = ActionExecutor(self.cfg.model)
        self.playbook = Playbook(
            ACTION_NAMES, path=self.cfg.playbook_path, explore=1.0
        )
        self.policy = PolicyRouter(self.cfg.harness, self.playbook, model=policy_mode)

        self.registry: Dict[str, ModelBundle] = {}
        self.anchor: Optional[ModelBundle] = None
        self.static: Optional[ModelBundle] = None
        self.best_version: str = ""

        self.history_X: List[np.ndarray] = []
        self.history_y: List[np.ndarray] = []
        self.bootstrap_X: Optional[np.ndarray] = None
        self.bootstrap_y: Optional[np.ndarray] = None
        self.bootstrap_prior: Optional[float] = None
        self.gate_X: Optional[np.ndarray] = None
        self.gate_y: Optional[np.ndarray] = None

        self.decisions: List[Dict[str, Any]] = []
        self.state_log: List[Dict[str, Any]] = []
        self.scores: List[Dict[str, Any]] = []      # 每个 batch 上 system vs static 的主指标
        self.best_primary: float = -np.inf
        self.initialization = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def bootstrap(self, X: np.ndarray, y: np.ndarray) -> None:
        """用官方训练集初始化：同时得到"初始锚点"和"永不更新的静态基线"。"""
        self.bootstrap_X = np.asarray(X, dtype=np.float32)
        self.bootstrap_y = np.asarray(y)
        self.bootstrap_prior = float(np.mean(y))

        started = time.perf_counter()
        status = 'failed'
        try:
            self.drift_u.fit(self.bootstrap_X, self.spec.feature_names)
            anchor, static = isolated_call(
                _fit_initial_models, self.cfg.model, self.spec, self.bootstrap_X, self.bootstrap_y,
                timeout_s=self.cfg.budget.bootstrap_timeout_s,
            )
            status = 'completed'
        finally:
            seconds = time.perf_counter() - started
            self.budget.record_initialization(seconds)
            self.initialization = {'status': status, 'compute_s': seconds,
                                   'training_rows': len(self.bootstrap_X)}
        self.anchor = anchor
        self.static = static
        self.registry[anchor.version] = anchor
        self.registry[static.version] = static
        self.best_version = anchor.version

        self.history_X = [self.bootstrap_X]
        self.history_y = [self.bootstrap_y]

        # 注意：不要用 bootstrap 自评来初始化 best_primary（训练集自评虚高）。
        # 留 -inf，等第一批带标签的流数据来建立真实参照。
        self.best_primary = -np.inf

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self, stream: Iterator[Batch], bootstrap_from_stream: int = 1,
            bootstrap_data=None, checkpoint_path=None, resume=False, resume_validator=None,
            checkpoint_outputs=None, gate_data=None) -> RunResult:
        if resume and not checkpoint_path:
            raise ValueError('恢复运行必须指定 checkpoint_path')
        if checkpoint_path and self.policy.mode == 'llm':
            raise ValueError('当前检查点仅支持 rule/bandit；LLM 调用重放与计费需另行对接')
        if gate_data is not None:
            if checkpoint_path:
                raise ValueError('独立闸门集暂不支持检查点恢复')
            X, y = gate_data
            X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
            if (X.ndim != 2 or X.shape[1] != self.spec.n_features or len(X) == 0
                    or y.ndim != 1 or len(y) != len(X) or not np.isfinite(X).all()
                    or not np.isfinite(y).all()):
                raise ValueError('独立闸门集维度、标签或数值无效')
            self.gate_X, self.gate_y = X, y
        store = CheckpointStore(checkpoint_path, self, bootstrap_data) if checkpoint_path else None
        if store and store.path.exists() and not resume:
            raise ValueError('检查点已存在；请指定恢复或使用新的检查点路径')
        lineage = store.load(self) if resume else None
        if resume and resume_validator:
            resume_validator([d['batch'] for d in self.decisions], lineage['outputs'])
        if self.anchor is None and bootstrap_from_stream < 1:
            raise ValueError("bootstrap_from_stream 必须至少为 1")
        self.logger.start(
            config=self.cfg.to_dict(),
            spec={**self.spec.to_dict(), "primary_metric": primary_metric(self.spec.task)},
        )
        if self.gate_X is not None:
            self.logger.log_artifact('gate-holdout.json', {
                'rows': len(self.gate_X), 'sha256': data_hash(self.gate_X, self.gate_y),
                'usage': 'candidate promotion only; never added to training history',
            })
        if lineage is not None:
            self.logger.log_artifact('resume.json', {k: v for k, v in lineage.items() if k != 'outputs'})
        if bootstrap_data is not None and not resume:
            self.bootstrap(*bootstrap_data)
        if self.initialization is not None:
            self.logger.log_artifact('initialization.json', self.initialization)

        # 若外部已经 bootstrap()，就不再消耗流里的批次
        already = self.anchor is not None
        n_boot = 0
        boot_batches = []
        consumed = list(store.consumed) if store else []
        replay_count = len(consumed) if resume else 0
        seen = 0
        last_index = None
        if store and self.anchor is not None and not resume:
            store.save(self, checkpoint_outputs() if checkpoint_outputs else None)
        for batch in stream:
            if last_index is not None and batch.index <= last_index:
                raise ValueError('批次编号必须严格递增')
            last_index = batch.index
            fingerprint = data_hash(batch.X, batch.y, batch.index) if store else None
            if seen < replay_count:
                expected = consumed[seen]
                if batch.index != expected['index'] or fingerprint != expected['hash']:
                    raise ValueError(f'已处理的第 {batch.index} 批数据发生变化，拒绝恢复')
                seen += 1
                continue
            seen += 1
            if store:
                consumed.append({'index': batch.index, 'hash': fingerprint})
            if not already and n_boot < bootstrap_from_stream:
                if batch.y is None:
                    raise ValueError("bootstrap 阶段必须有标签")
                boot_batches.append(batch)
                n_boot += 1
                if n_boot == bootstrap_from_stream:
                    self.bootstrap(np.vstack([b.X for b in boot_batches]),
                                   np.concatenate([b.y for b in boot_batches]))
                    self.logger.log_artifact('initialization.json', self.initialization)
                self.logger.log(batch.index, {"phase": 1, "bootstrapped": 1})
                if store and self.anchor is not None:
                    store.consumed = consumed
                    store.save(self, checkpoint_outputs() if checkpoint_outputs else None)
                continue

            self.step(batch)
            if store:
                store.consumed = consumed
                store.save(self, checkpoint_outputs() if checkpoint_outputs else None)

        if seen < replay_count:
            raise ValueError('恢复数据缺少已处理的批次，拒绝完成')
        if self.anchor is None:
            raise ValueError("数据不足，无法完成初始化")
        return self._finish()

    def step(self, batch: Batch) -> Optional[Dict[str, Any]]:
        assert self.anchor is not None, "必须先 bootstrap"
        k = batch.index
        t_start = time.perf_counter()

        # ---------- 1. 先用旧锚点出预测（不能等学完再预测） ----------
        preds = self.anchor.predict(batch.X)
        if self.emit is not None:
            self.emit(k, preds)

        has_label = batch.y is not None
        metrics_now: Dict[str, float] = {}
        static_now: Dict[str, float] = {}
        primary_now = float("nan")
        primary_static = float("nan")

        if has_label:
            metrics_now = score_model(self.anchor, self.spec.task, batch.X, batch.y)
            static_now = score_model(self.static, self.spec.task, batch.X, batch.y)
            primary_now = primary_value(metrics_now, self.spec.task)
            primary_static = primary_value(static_now, self.spec.task)
            self.drift_l.update(k, primary_now)

        # ---------- 2. 漂移检测 ----------
        sig = self.drift_u.score(batch.X)
        prior_now = float(np.mean(batch.y)) if has_label else None
        fp = attribute(
            self.spec,
            sig,
            perf_now=primary_now if has_label else float("nan"),
            perf_ref=self.best_primary,
            prior_now=prior_now,
            prior_ref=self.bootstrap_prior,
            importance=self._importance(),
        )
        if not has_label:
            fp = Fingerprint(
                kind=DriftKind.UNKNOWN if sig.severity > 0.2 else DriftKind.NONE,
                severity=sig.severity,
                performance_drop=0.0,
                top_features=sig.top_features,
                evidence=sig.summary(),
            )

        # ---------- 3. 决策 ----------
        past_validation = self.cfg.gate.val_batches - 1
        train_count = len(self.history_X) - past_validation - self.cfg.gate.purge_batches
        validation_ready = has_label and train_count >= 1
        train_X = self.history_X[:train_count] if validation_ready else []
        train_y = self.history_y[:train_count] if validation_ready else []
        validation_X = self.history_X[-past_validation:] if past_validation else []
        validation_y = self.history_y[-past_validation:] if past_validation else []
        gate_anchor_metrics, gate_static_metrics = metrics_now, static_now
        val_X, val_y = batch.X, batch.y
        if validation_ready and past_validation:
            val_X = np.vstack(validation_X + [batch.X])
            val_y = np.concatenate(validation_y + [batch.y])
            gate_anchor_metrics = score_model(self.anchor, self.spec.task, val_X, val_y)
            gate_static_metrics = score_model(self.static, self.spec.task, val_X, val_y)
        batch_weights = None
        if self.cfg.model.finetune_recency == "batch" and train_X:
            batch_weights = _batch_recency_weights(
                train_X, self.cfg.model.finetune_decay, self.cfg.model.finetune_floor
            )
        ctx = ActionContext(
            spec=self.spec,
            batch_index=k,
            train_X=np.vstack(train_X) if train_X else np.empty((0, self.spec.n_features)),
            train_y=np.concatenate(train_y) if train_y else np.empty(0),
            recent_X=train_X[-1] if train_X else np.empty((0, self.spec.n_features)),
            recent_y=train_y[-1] if train_y else np.empty(0),
            anchor=self.anchor,
            registry=self.registry,
            best_version=self.best_version,
            cfg=self.cfg.model,
            rng_seed=self.cfg.seed,
            batch_weights=batch_weights,
        )
        allowed = self._allowed_actions(ctx, fp)
        if not validation_ready:
            allowed = ["noop"]
        state = build_state_card(
            fp,
            metrics_now or {},
            self.decisions,
            self.budget.remaining(),
            k,
            self.spec.task.value,
            best_available=self.best_version not in ("", self.anchor.version),
            allowed=allowed,
        )
        plan, trace = self.policy.decide(
            fp, ctx, state, allowed=allowed, best_available=(self.best_version not in ("", self.anchor.version))
        )
        plan = validate_plan(plan, allowed)
        # 记录真实的 (状态卡, 动作) 对：训练策略模型的数据就从这里来，
        # 只用真实回放产生的状态，不用凭空捏造的场景。
        self.state_log.append({"batch": k, "state": state, "action": plan.action,
                               "source": trace.source, "dataset": self.spec.name})

        # ---------- 4. 执行实验 ----------
        est = estimate_cost(plan.action, ctx)
        est.llm_tokens = trace.tokens
        cand: ModelBundle = self.anchor
        exec_note = ""
        timed_out = False
        execution_failed = False
        if has_label and plan.action != "noop":
            try:
                remaining = self.budget.remaining()['compute_s'] - (time.perf_counter() - t_start)
                if remaining <= 0:
                    raise RuntimeError('本批前置计算后预算已耗尽，跳过候选')
                limit = self.cfg.budget.action_timeout_s
                res = isolated_call(self.executor.execute, plan.action, ctx,
                                    timeout_s=min(limit, remaining) if limit is not None else None)
                cand, exec_note = res.model, res.note
                est.compute_seconds = res.seconds
            except Exception as exc:
                execution_failed = True
                timed_out = isinstance(exc, ExecutionTimeout)
                exec_note = f"执行失败: {exc}"
                self.budget.block(exec_note)
                cand = self.anchor

        # ---------- 5. 闸门 ----------
        cand_metrics: Dict[str, float] = {}
        cand_primary = float("nan")
        promoted = False          # 候选是否通过闸门
        gate_gain = 0.0
        reason = "无标签，跳过评估"
        holdout_result = None

        if has_label:
            if plan.action == "noop":
                reason = "策略选择不动作" if validation_ready else "历史不足以满足验证窗口与隔离间隔"
            elif execution_failed:
                reason = '候选执行超时' if timed_out else '候选执行失败'
            else:
                cand_metrics = score_model(cand, self.spec.task, val_X, val_y)
                current_candidate = score_model(cand, self.spec.task, batch.X, batch.y) if past_validation else cand_metrics
                cand_primary = primary_value(current_candidate, self.spec.task)
                result = self.gate.check(
                    self.spec.task,
                    gate_anchor_metrics,
                    cand_metrics,
                    gate_static_metrics,
                )
                promoted = result.accept
                gate_gain = result.delta
                reason = result.reason
                if self.gate_X is not None:
                    holdout_result = self.gate.check(
                        self.spec.task,
                        score_model(self.anchor, self.spec.task, self.gate_X, self.gate_y),
                        score_model(cand, self.spec.task, self.gate_X, self.gate_y),
                        score_model(self.static, self.spec.task, self.gate_X, self.gate_y),
                    )
                    if promoted and not holdout_result.accept:
                        promoted = False
                        gate_gain = 0.0
                        reason += f"；独立设备验证拒绝：{holdout_result.reason}"

        # ---------- 6. 更新状态 ----------
        anchor_before = self.anchor.version
        if promoted:
            # 注册表中的历史版本不可原地改名（回滚可能直接返回历史对象）。
            cand = replace(cand)
            cand.version = f"v{k}-{plan.action}"
            cand.parent = anchor_before
            self.anchor = cand
            self.registry[cand.version] = cand
        # 候选失败只拒绝候选；不能据此替换正在服务的锚点。
        # 当前锚点低于静态基线时，回退到本批已验证更好的静态模型。
        sign = 1.0 if primary_metric_is_higher(self.spec.task) else -1.0
        if (has_label and not promoted and self.cfg.gate.rollback_on_breach
                and np.isfinite(primary_static) and np.isfinite(primary_now)
                and sign * (primary_static - primary_now) > self.cfg.gate.max_drop):
            self.anchor = self.static
            reason += "（已回滚到本批表现更好的静态基线）"

        # 维护"历史最佳"参照：归属到"实际跑出这个成绩的模型"，即预测时的锚点
        if has_label and np.isfinite(primary_now):
            if not np.isfinite(self.best_primary) or sign * (primary_now - self.best_primary) > 0:
                self.best_primary = primary_now
                self.best_version = anchor_before

        # ---------- 7. 记账 ----------
        gain = 0.0
        if has_label and np.isfinite(cand_primary) and np.isfinite(primary_now):
            sign = 1.0 if primary_metric_is_higher(self.spec.task) else -1.0
            gain = gate_gain if promoted else 0.0
        rec = self.budget.spend(
            k, plan.action, est, gain=gain,
            compute_seconds=time.perf_counter() - t_start,
        )

        # ---------- 8. 策略层记忆 ----------
        key = context_key(fp, self.spec.task.value)
        reward = self._reward(gain, plan.action, promoted, fp, rec)
        self.playbook.update(key, plan.action, reward)

        # ---------- 9. 记录 ----------
        if has_label:
            self.history_X.append(batch.X)
            self.history_y.append(batch.y)
            self._trim_history()
            self.scores.append(
                {
                    "batch": k,
                    "system": primary_now,
                    "static": primary_static,
                    "candidate": cand_primary,
                }
            )

        dec = Decision(
            batch_index=k,
            fingerprint=fp.to_dict(),
            plan={**plan.to_dict(), "trace": trace.source, "exec": exec_note,
                  "fallback_reason": trace.fallback_reason,
                  "unknown_llm_usage_calls": trace.unknown_usage_calls,
                  "validation": {'ready': validation_ready, 'val_batches': self.cfg.gate.val_batches,
                                 'purge_batches': self.cfg.gate.purge_batches,
                                 'train_rows': len(ctx.train_X),
                                 'validation_rows': len(val_X) if validation_ready else 0,
                                 'holdout_rows': len(self.gate_X) if self.gate_X is not None else 0,
                                 'holdout_accept': holdout_result.accept if holdout_result is not None else None,
                                 'holdout_delta': holdout_result.delta if holdout_result is not None else None},
                  "timed_out": timed_out, "execution_failed": execution_failed},
            promoted=promoted,
            reason=reason,
            anchor_before=anchor_before,
            anchor_after=self.anchor.version,
            metrics_before=gate_anchor_metrics,
            metrics_candidate=cand_metrics,
            static_metrics=gate_static_metrics,
            gain=gain,
        )
        self.decisions.append(dec.to_dict())
        self.logger.log_decision(dec.to_dict())
        self.logger.log(
            k,
            {
                "system_primary": primary_now if np.isfinite(primary_now) else -1,
                "static_primary": primary_static if np.isfinite(primary_static) else -1,
                "candidate_primary": cand_primary if np.isfinite(cand_primary) else -1,
                "drift_severity": sig.severity,
                "drift_kind": fp.kind.value,
                "promoted": int(promoted),
                "cost": rec.cost_units,
                "tokens": rec.llm_tokens,
                "execution_timeout": int(timed_out),
                "execution_failed": int(execution_failed),
                "unknown_llm_usage_calls": trace.unknown_usage_calls,
            },
            tags=[plan.action, trace.source],
        )
        return dec.to_dict()

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _reward(
        self,
        gain: float,
        action: str,
        promoted: bool,
        fp: Fingerprint,
        rec,
    ) -> float:
        """策略层的学习信号。

        朴素的"成功给正、失败给负"会让 bandit 收敛到永远 noop——因为 noop 既不
        费钱也不会失败。所以这里显式地给"不作为"计价：漂移严重时不动，就是失分。
        """
        cost_weight = 0.002                      # 1 秒算力 ≈ 0.002 收益，与 min_gain 同量级
        if promoted:
            r = gain if np.isfinite(gain) else 0.0
        elif action == "noop":
            # 不作为的代价 = 已经发生的性能下滑
            r = -0.5 * max(0.0, float(fp.performance_drop))
        else:
            r = -0.002                           # 尝试失败的小惩罚
        return float(r - cost_weight * rec.cost_units)

    def _allowed_actions(self, ctx: ActionContext, fp: Fingerprint) -> List[str]:
        """预算可行 ∩ 语义可用。动作掩码让策略层不可能在无关方向上烧钱。"""
        has_best = self.best_version != "" and self.best_version != self.anchor.version
        plausible = set(plausible_actions(fp.kind.value, has_best))
        out = []
        for a in ACTION_NAMES:
            if a not in plausible:
                continue
            ok, _ = self.budget.afford(estimate_cost(a, ctx))
            if ok:
                out.append(a)
        return out or ["noop"]

    def _importance(self) -> Optional[Dict[str, float]]:
        if self.anchor is None or self.bootstrap_X is None:
            return None
        try:
            return model_importance(self.anchor, self.bootstrap_X, self.bootstrap_y)
        except Exception:
            return None

    def _trim_history(self) -> None:
        cap = 40_000
        total = sum(len(x) for x in self.history_X)
        while total > cap and len(self.history_X) > 3:
            total -= len(self.history_X[0])
            self.history_X.pop(0)
            self.history_y.pop(0)

    def _finish(self) -> RunResult:
        scored = [s for s in self.scores if np.isfinite(s["system"])]
        mean_sys = float(np.mean([s["system"] for s in scored])) if scored else float("nan")
        mean_sta = float(np.mean([s["static"] for s in scored])) if scored else float("nan")
        higher = primary_metric_is_higher(self.spec.task)
        gain = (mean_sys - mean_sta) if higher else (mean_sta - mean_sys)

        summary = {
            "dataset": self.spec.name,
            "task": self.spec.task.value,
            "n_batches": len(self.decisions),
            "n_promotions": int(sum(1 for d in self.decisions if d["promoted"])),
            "n_timeouts": sum(d['plan'].get('timed_out', False) for d in self.decisions),
            "n_execution_failures": sum(d['plan'].get('execution_failed', False) for d in self.decisions),
            "unknown_llm_usage_calls": sum(d['plan'].get('unknown_llm_usage_calls', 0) for d in self.decisions),
            "mean_system": mean_sys,
            "mean_static": mean_sta,
            "gain_vs_static": gain,
            "budget": self.budget.summary(),
            "playbook": self.playbook.stats(),
            "policy_sources": _count_sources(self.decisions),
            "final_anchor": self.anchor.version if self.anchor else None,
            "best_version": self.best_version,
        }
        self.logger.log_artifact("playbook.json", {"table": self.playbook.stats()})
        self.logger.log_artifact('costs.json', {
            'initialization': self.initialization,
            'batches': [record.to_dict() for record in self.budget.records],
            'total': self.budget.summary(),
        })
        self.logger.finish(summary)
        if self.cfg.playbook_path:
            self.playbook.save(self.cfg.playbook_path)
        return RunResult(summary=summary, decisions=self.decisions)


def _batch_recency_weights(batches, decay: float = 0.5, floor: float = 0.15) -> np.ndarray:
    """按批次到达顺序给逐行权重：越新的批次权重越高，批内一致。

    真实数据的 CSV 常常按设备分组排列，行序与时间无关；这里用批次边界
    作为唯一可靠的时间信号。
    """
    m = len(batches)
    if m == 0:
        return np.empty(0)
    parts = []
    for i, X in enumerate(batches):
        age = m - 1 - i
        w = max(float(floor), float(decay) ** age)
        parts.append(np.full(len(X), w, dtype=float))
    return np.concatenate(parts)


def primary_metric_is_higher(task) -> bool:
    from .evaluation import higher_is_better

    return higher_is_better(primary_metric(task))


def _count_sources(decisions: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in decisions:
        s = (d.get("plan") or {}).get("trace", "?")
        out[s] = out.get(s, 0) + 1
    return out


