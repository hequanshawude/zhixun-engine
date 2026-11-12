"""策略层记忆：contextual bandit playbook。

这就是"策略层自改进(RSI)"的可验证载体：
    第 k 轮把 (漂移指纹 → 动作 → 实际增益) 写进表里，
    第 k+1 轮面对同类指纹时，选择就已经比第 1 轮更靠谱。

答辩时可以直接画"playbook 策略命中率 / 平均增益"随轮次的曲线。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


from .types import Fingerprint


def context_key(fp: Fingerprint, task: str = "") -> str:
    """把连续的指纹离散成有限个上下文。

    太细 → 每个桶样本太少，学不动；太粗 → 策略没有区分度。
    这里用 4 类漂移 × 3 档严重度 = 最多 12 个桶。
    """
    if fp.severity < 0.2:
        sev = "lo"
    elif fp.severity < 0.5:
        sev = "mid"
    else:
        sev = "hi"
    return f"{fp.kind.value}|{sev}" + (f"|{task}" if task else "")


@dataclass
class ArmStat:
    pulls: int = 0
    reward_sum: float = 0.0
    best: float = -1e9

    @property
    def mean(self) -> float:
        return self.reward_sum / self.pulls if self.pulls else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"pulls": self.pulls, "reward_sum": round(self.reward_sum, 5),
                "mean": round(self.mean, 5), "best": round(self.best, 5)}


class Playbook:
    """UCB1 多臂老虎机，每个上下文一组臂。"""

    def __init__(self, actions: List[str], path: Optional[str] = None, explore: float = 1.0):
        self.actions = list(actions)
        self.explore = explore
        self.table: Dict[str, Dict[str, ArmStat]] = {}
        self.history: List[Dict[str, Any]] = []
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self.load(self.path)

    # ---------- 选择 ----------
    def _arms(self, key: str) -> Dict[str, ArmStat]:
        if key not in self.table:
            self.table[key] = {a: ArmStat() for a in self.actions}
        return self.table[key]

    def select(self, key: str, allowed: Optional[List[str]] = None) -> str:
        arms = self._arms(key)
        cands = [a for a in (self.actions if allowed is None else allowed) if a in arms]
        if not cands:
            return "noop"
        total = sum(arms[a].pulls for a in cands)
        if total == 0:
            # 全部没试过：按成本序先试最便宜的（省预算）
            return cands[0]
        best_a, best_score = cands[0], -1e18
        for a in cands:
            st = arms[a]
            if st.pulls == 0:
                return a  # 先探索未试过的臂
            score = st.mean + self.explore * math.sqrt(2.0 * math.log(total) / st.pulls)
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    # ---------- 更新 ----------
    def update(self, key: str, action: str, reward: float) -> None:
        arms = self._arms(key)
        st = arms.setdefault(action, ArmStat())
        st.pulls += 1
        st.reward_sum += float(reward)
        st.best = max(st.best, float(reward))
        self.history.append({"key": key, "action": action, "reward": round(float(reward), 6)})

    # ---------- 统计 ----------
    def stats(self) -> Dict[str, Any]:
        out = {}
        for k, arms in self.table.items():
            out[k] = {a: st.to_dict() for a, st in arms.items() if st.pulls > 0}
        return out

    def best_action(self, key: str) -> Optional[str]:
        arms = self._arms(key)
        used = {a: st for a, st in arms.items() if st.pulls > 0}
        if not used:
            return None
        return max(used.items(), key=lambda kv: kv[1].mean)[0]

    # ---------- 持久化 ----------
    def save(self, path: Optional[str] = None) -> None:
        p = Path(path or self.path or "playbook.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "actions": self.actions,
            "explore": self.explore,
            "table": {k: {a: st.to_dict() for a, st in arms.items()} for k, arms in self.table.items()},
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.actions = data.get("actions", self.actions)
        self.explore = data.get("explore", self.explore)
        self.table = {}
        for k, arms in (data.get("table") or {}).items():
            self.table[k] = {
                a: ArmStat(
                    pulls=int(v.get("pulls", 0)),
                    reward_sum=float(v.get("reward_sum", 0.0)),
                    best=float(v.get("best", -1e9)),
                )
                for a, v in arms.items()
            }
