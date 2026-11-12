"""Harness：用大模型做"策略层"决策，但绝不让它成为单点故障。

按配置选择服务模型；调用失败或输出非法时降级到规则。
bandit 作为显式实验选项，不自动替代故障降级路径。
正式平台的服务与时限仍需 SDK 验证。
"""

from __future__ import annotations

import json
import http.client
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .actions import ACTION_NAMES, CATALOG, ActionContext, catalog_for_prompt
from .config import HarnessConfig
from .playbook import Playbook
from .types import ActionPlan, Fingerprint

try:
    from .types import DriftKind
except Exception:  # pragma: no cover
    DriftKind = None  # type: ignore


# --------------------------------------------------------------------------
# LLM 客户端（urllib，零依赖）
# --------------------------------------------------------------------------


class LLMError(RuntimeError):
    def __init__(self, message, usage=None, http_status=None, provider_code=None):
        super().__init__(message)
        self.usage = usage or LLMUsage()
        self.http_status = http_status
        self.provider_code = provider_code


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    unknown_calls: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """chat/completions 兼容客户端；PyTRiO 实际接口待官方 SDK 确认。"""

    def __init__(self, cfg: HarnessConfig, model: Optional[str] = None):
        self.cfg = cfg
        self.model = model or cfg.model
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, messages: List[Dict[str, str]]) -> Tuple[str, LLMUsage]:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        try:
            endpoint = urllib.parse.urlsplit(url)
            if endpoint.scheme not in ('http', 'https') or not endpoint.netloc:
                raise ValueError('invalid endpoint')
        except ValueError as exc:
            raise LLMError('模型服务地址无效') from exc
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        last_exc: Optional[Exception] = None
        unknown_calls = 0
        deadline = time.monotonic() + self.cfg.timeout_s

        for attempt in range(self.cfg.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                last_exc = TimeoutError('request time budget exhausted')
                break
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.cfg.api_key}",
                },
                method="POST",
            )
            try:
                self.calls += 1
                with urllib.request.urlopen(req, timeout=remaining) as resp:
                    raw = resp.read()
            except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
                last_exc = exc
                unknown_calls += 1
                if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408, 429) and exc.code < 500:
                    break
                if attempt < self.cfg.retries:
                    time.sleep(max(0, min(0.4 * (attempt + 1), deadline - time.monotonic())))
                continue
            # 已收到响应后先记用量，再验证内容；结构错误不重复付费请求。
            u = LLMUsage(unknown_calls=unknown_calls + 1)
            try:
                body = json.loads(raw.decode('utf-8'))
                usage = body.get('usage') if isinstance(body, dict) else None
                if (isinstance(usage, dict)
                        and all(type(usage.get(k)) is int and usage[k] >= 0
                                for k in ('prompt_tokens', 'completion_tokens'))):
                    u.prompt_tokens = usage['prompt_tokens']
                    u.completion_tokens = usage['completion_tokens']
                    u.unknown_calls = unknown_calls
                self.prompt_tokens += u.prompt_tokens
                self.completion_tokens += u.completion_tokens
                text = body['choices'][0]['message']['content']
                if not isinstance(text, str):
                    raise ValueError('content must be text')
            except (ValueError, KeyError, IndexError, TypeError, UnicodeError) as exc:
                raise LLMError(f'LLM 响应结构无效（{type(exc).__name__}）', u) from exc
            return text, u
        status = last_exc.code if isinstance(last_exc, urllib.error.HTTPError) else None
        provider_code = None
        if status is not None:
            try:
                raw_error = last_exc.read(2048)
                if status == 503 and raw_error.strip() == b'no available server':
                    provider_code = 'no_available_server'
                error = json.loads(raw_error).get('error', {})
                if isinstance(error, dict) and isinstance(error.get('code'), str):
                    provider_code = error['code']
            except (ValueError, UnicodeError, OSError, AttributeError):
                pass
        raise LLMError(f"LLM 请求失败（{type(last_exc).__name__}）",
                       LLMUsage(unknown_calls=unknown_calls), status, provider_code)


class PyTrioSamplingClient:
    """通过官方采样接口调用 PyTRiO 基模或训练权重。"""

    def __init__(self, cfg: HarnessConfig, model: str):
        self.cfg = cfg
        self.model = model
        self._sampler = None
        self._tokenizer = None

    def chat(self, messages: List[Dict[str, str]]) -> Tuple[str, LLMUsage]:
        usage = LLMUsage()
        try:
            import pytrio as trio

            if self._sampler is None:
                if self.cfg.base_url.rstrip('/') != 'https://pytrio.com/api/openai/v1':
                    raise ValueError('PyTRiO 权重只允许通过官方服务调用')
                trio.configure(api_key=self.cfg.api_key, base_url='https://pytrio.com',
                               request_timeout_seconds=self.cfg.timeout_s,
                               retry_max_attempts=1,
                               control_retry_timeout_seconds=self.cfg.timeout_s)
                if self.model.startswith('trio://'):
                    self._sampler = trio.ServiceClient().create_sampling_client(
                        base_model=self.cfg.pytrio_base_model, model_path=self.model)
                else:
                    self._sampler = trio.ServiceClient().create_sampling_client(
                        base_model=self.model)
                self._tokenizer = self._sampler.get_tokenizer()
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            tokens = self._tokenizer.encode(prompt, add_special_tokens=False)
            usage.prompt_tokens = len(tokens)
            usage.unknown_calls = 1
            response = self._sampler.sample(
                prompt=trio.ModelInput.from_ints(tokens), num_samples=1,
                sampling_params=trio.SamplingParams(
                    max_tokens=self.cfg.max_tokens, seed=42,
                    temperature=self.cfg.temperature if self.cfg.temperature > 0 else 0.1),
            ).result()
            sequence = response.sequences[0]
            usage.completion_tokens = len(sequence.tokens)
            usage.unknown_calls = 0
            answer = sequence.text.strip()
            eos = self._tokenizer.eos_token
            if eos and answer.endswith(eos):
                answer = answer[:-len(eos)].strip()
            return answer, usage
        except Exception as exc:
            raise LLMError(f'PyTRiO 权重推理失败（{type(exc).__name__}）', usage) from None


# --------------------------------------------------------------------------
# 状态卡：给模型的上下文必须小而准（token 就是钱）
# --------------------------------------------------------------------------


def build_state_card(
    fp: Fingerprint,
    anchor_metrics: Dict[str, float],
    history: List[Dict[str, Any]],
    budget_remaining: Dict[str, float],
    batch_index: int,
    task: str,
    best_available: bool,
    allowed: Optional[List[str]] = None,
) -> Dict[str, Any]:
    recent = history[-4:] if history else []
    return {
        "batch": batch_index,
        "task": task,
        "drift": {
            "kind": fp.kind.value,
            "severity": round(fp.severity, 3),
            "perf_drop": round(fp.performance_drop, 4),
            "prior_shift": round(fp.prior_shift, 4),
            "top_features": fp.top_features[:3],
            "evidence": fp.evidence,
        },
        "anchor_metrics": {k: round(v, 4) for k, v in anchor_metrics.items() if np.isfinite(v)},
        "recent_actions": [
            {"a": (h.get("plan") or {}).get("action"), "gain": h.get("gain"),
             "ok": h.get("promoted")}
            for h in recent
        ],
        "allowed": list(allowed) if allowed else [],
        "budget_left": {
            "compute_s": round(budget_remaining.get("compute_s", 0), 1),
            "tokens": int(budget_remaining.get("tokens", 0)),
        },
        "best_available": best_available,
    }


SYSTEM_PROMPT = """你是工业模型自进化系统的策略决策器。系统维护一个"锚点模型"，在数据漂移流上逐批运行。

你的任务：根据漂移指纹和预算，从动作清单中选择**一个**动作。

决策原则（按优先级）：
1. 能不动就不动。漂移不等于要更新，成本高会直接拉低客观分。
2. 能用便宜动作解决就不要用贵的：calibrate < reweight < finetune < retrain < switch_model。
3. 协变量漂移(covariate) → reweight 或 finetune；概念漂移(concept) → finetune 或 retrain；
   标签漂移(label) → calibrate 通常最优。
4. 若历史最近几次更新都失败，不要重复同一动作。
5. 预算不足时选更便宜的动作。
6. **只能从状态的 allowed 字段里选动作**；不在列表里的一律无效，会被系统丢弃。

只输出 JSON，不要解释、不要 markdown：
{"action":"<动作名>","params":{},"reason":"<20字以内>"}
"""


def build_messages(state: Dict[str, Any]) -> List[Dict[str, str]]:
    actions = catalog_for_prompt()
    user = (
        "动作清单：\n"
        + json.dumps(actions, ensure_ascii=False, separators=(",", ":"))
        + "\n\n当前状态：\n"
        + json.dumps(state, ensure_ascii=False, separators=(",", ":"), default=str)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """解析一个完整对象，支持代码围栏和前导说明，不截断嵌套参数。"""
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines[-1].strip() != '```':
            return None
        text = '\n'.join(lines[1:-1]).strip()
    start = 0 if text.startswith(('[', '"', '{')) else text.find('{')
    if start < 0:
        return None
    try:
        obj, end = json.JSONDecoder().raw_decode(text, start)
    except ValueError:
        return None
    if text[end:].strip() or not isinstance(obj, dict) or not isinstance(obj.get('action'), str):
        return None
    return obj


# --------------------------------------------------------------------------
# 策略
# --------------------------------------------------------------------------


class RulePolicy:
    """确定性规则引擎。没有 GPU、没有网络时也 100% 可用。"""

    def decide(self, fp: Fingerprint, ctx: ActionContext, *, best_available: bool) -> ActionPlan:
        from .types import DriftKind as DK

        if fp.kind == DK.NONE:
            return ActionPlan("noop", {}, "漂移不显著", "rule")

        if fp.kind == DK.LABEL:
            return ActionPlan("calibrate", {}, "标签漂移，调边界最划算", "rule")

        if fp.kind == DK.COVARIATE:
            if fp.severity > 0.6:
                return ActionPlan("finetune", {}, f"强协变量漂移 {fp.severity:.2f}", "rule")
            return ActionPlan("reweight", {}, f"协变量漂移 {fp.severity:.2f}，重加权即可", "rule")

        if fp.kind == DK.CONCEPT:
            if fp.performance_drop > 0.15:
                return ActionPlan("retrain", {}, f"概念漂移且掉点 {fp.performance_drop:.2f}", "rule")
            return ActionPlan("finetune", {}, "概念漂移，先做增量", "rule")

        # MIXED / UNKNOWN
        if best_available and fp.performance_drop > 0.25:
            return ActionPlan("rollback", {}, "掉点过大，先止损", "rule")
        return ActionPlan("finetune", {}, "混合漂移，增量微调", "rule")


@dataclass
class PolicyTrace:
    source: str
    tokens: int = 0
    fallback_reason: str = ""
    unknown_usage_calls: int = 0


class PolicyRouter:
    """把 LLM / bandit / 规则串成一条可信的决策链。"""

    def __init__(
        self,
        harness_cfg: HarnessConfig,
        playbook: Playbook,
        model: str = "rule",
    ):
        self.cfg = harness_cfg
        self.playbook = playbook
        self.rule = RulePolicy()
        self.mode = model                  # llm / bandit / rule
        self._cache: Dict[str, Tuple[Dict[str, Any], int]] = {}
        self._fail_streak = 0
        self._disabled = False             # 熔断：连续失败后停用 LLM
        self.max_fail_streak = 3
        self.big: Optional[LLMClient | PyTrioSamplingClient] = None
        self.small: Optional[LLMClient | PyTrioSamplingClient] = None
        if harness_cfg.enabled and model == "llm":
            def make_client(model_id):
                if harness_cfg.credential_source == 'pytrio':
                    return PyTrioSamplingClient(harness_cfg, model_id)
                return LLMClient(harness_cfg, model_id)
            self.big = make_client(harness_cfg.model)
            self.small = (self.big if harness_cfg.small_model == harness_cfg.model
                          else make_client(harness_cfg.small_model))

    # ---------- 主入口 ----------
    def decide(
        self,
        fp: Fingerprint,
        ctx: ActionContext,
        state: Dict[str, Any],
        *,
        allowed: List[str],
        best_available: bool,
    ) -> Tuple[ActionPlan, PolicyTrace]:
        if allowed == ["noop"]:
            return ActionPlan("noop", {}, "无可执行更新", "rule"), PolicyTrace("rule")
        spent_tokens = 0
        unknown_usage_calls = 0
        # 1) LLM
        if self.mode == "llm" and self.big is not None and not self._disabled:
            plan, trace = self._try_llm(state, allowed, fp)
            if plan is not None:
                self._fail_streak = 0
                return plan, trace
            fallback = trace.fallback_reason
            spent_tokens = trace.tokens
            unknown_usage_calls = trace.unknown_usage_calls
            # 熔断：服务不可达时不要每批都付超时的代价
            self._fail_streak += 1
            if self._fail_streak >= self.max_fail_streak:
                self._disabled = True
                fallback += f"（熔断：连续{self._fail_streak}次失败，后续改用本地策略）"
        else:
            fallback = "llm 已熔断" if self._disabled else ""

        # 2) bandit：首次使用规则先验，后续从允许的动作中执行 UCB。
        if self.mode == "bandit":
            from .playbook import context_key

            key = context_key(fp, ctx.spec.task.value)
            prior = self.rule.decide(fp, ctx, best_available=best_available).action
            if prior not in allowed:
                prior = "noop"

            # 无显著漂移时保持；止损回滚仍由规则决定，不随机探索历史版本。
            if fp.kind.value == "none" or prior == "rollback":
                return (ActionPlan(prior, {}, f"playbook({key}) 规则保护", "rule"),
                        PolicyTrace("bandit", spent_tokens, fallback))
            candidates = sorted((a for a in allowed if a != "rollback"),
                                key=lambda a: CATALOG[a].order)
            stats = self.playbook.table.get(key, {})
            if not any(stats[a].pulls for a in candidates if a in stats):
                action, source, reason = prior, "rule", "首次使用先验"
            else:
                action = self.playbook.select(key, allowed=candidates)
                source = "bandit"
                reason = "探索未试动作" if action not in stats or stats[action].pulls == 0 else "UCB 选择"
            return (
                ActionPlan(action, {}, f"playbook({key}) {reason}", source),
                PolicyTrace("bandit", spent_tokens, fallback),
            )

        # 3) 规则
        plan = self.rule.decide(fp, ctx, best_available=best_available)
        if plan.action not in allowed:
            plan = ActionPlan("noop", {}, "规则动作被预算/掩码禁止", "rule")
        return plan, PolicyTrace("rule", spent_tokens, fallback, unknown_usage_calls)

    # ---------- LLM 分支 ----------
    def _try_llm(
        self, state: Dict[str, Any], allowed: List[str], fp: Fingerprint
    ) -> Tuple[Optional[ActionPlan], PolicyTrace]:
        key = json.dumps(state, sort_keys=True, default=str)
        if self.cfg.cache and key in self._cache:
            cached, used = self._cache[key]
            if cached.get("action") in allowed:
                return (
                    ActionPlan(
                        cached["action"], cached.get("params", {}), cached.get("reason", ""), "llm-cache"
                    ),
                    PolicyTrace("llm-cache", 0),
                )

        if state.get('budget_left', {}).get('tokens', 0) <= 0:
            return None, PolicyTrace('llm', 0, 'Token 预算已耗尽，跳过调用')
        messages = build_messages(state)
        if self.cfg.credential_source == 'pytrio':
            # 预留输入的 UTF-8 字节上界和最多输出量，避免一次请求跨过策略预算。
            reserve = (len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))
                       + self.cfg.max_tokens + 128)
            if state.get('budget_left', {}).get('tokens', 0) < reserve:
                return None, PolicyTrace('llm', 0, '剩余 Token 不足以覆盖下一次请求')
        # 按配置选择服务模型；实际授权 ID 待官方确认。
        escalate = fp.kind.value in set(self.cfg.escalate_on) or fp.severity > 0.5
        client = self.big if escalate else (self.small or self.big)
        if client is None:
            return None, PolicyTrace("llm", 0, "no client")

        try:
            text, usage = client.chat(messages)
        except LLMError as exc:
            return None, PolicyTrace("llm", exc.usage.total, str(exc)[:80], exc.usage.unknown_calls)

        obj = parse_action(text)
        if obj is None:
            return None, PolicyTrace("llm", usage.total, "输出不是完整决策 JSON", usage.unknown_calls)

        act = str(obj.get("action", ""))
        if act not in ACTION_NAMES or act not in allowed:
            return None, PolicyTrace("llm", usage.total, "动作非法/越权", usage.unknown_calls)

        params = obj.get('params', {})
        if not isinstance(params, dict) or params:
            return None, PolicyTrace('llm', usage.total, '当前动作不接受自定义参数', usage.unknown_calls)
        plan = ActionPlan(act, params, str(obj.get("reason", ""))[:60], "llm")
        if self.cfg.cache:
            self._cache[key] = ({'action': act, 'params': {}, 'reason': plan.rationale}, usage.total)
        return plan, PolicyTrace("llm", usage.total, unknown_usage_calls=usage.unknown_calls)


# --------------------------------------------------------------------------


def validate_plan(plan: ActionPlan, allowed: List[str]) -> ActionPlan:
    """最后一道闸：不管动作从哪来，都要过这里。"""
    if plan.action not in CATALOG:
        return ActionPlan("noop", {}, f"未知动作 {plan.action}", "validator")
    if plan.action not in allowed:
        return ActionPlan("noop", {}, f"预算不允许 {plan.action}", "validator")
    # 参数白名单
    ok_params = {"threshold", "bias", "seed"}
    plan.params = {k: v for k, v in plan.params.items() if k in ok_params}
    return plan
