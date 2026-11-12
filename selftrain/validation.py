"""配对验证的通用统计工具。

"随机切分 → 同切分下系统与静态基线配对比较 → 跨切分汇总"这套流程，
XJTU-SY 和 C-MAPSS 都要用，所以放在库里而不是脚本里。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np


def bootstrap_ci(values: np.ndarray, n: int = 4000, alpha: float = 0.05):
    """配对差值的自助法置信区间。"""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def sign_test(wins: int, losses: int) -> float:
    """双尾符号检验（二项分布，p=0.5），不依赖 scipy。"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def summarize(rows: List[Dict[str, Any]], policy: str, label: str = "") -> Dict[str, Any]:
    """``rows`` 里每条记录需要 ``system``、``static``、``seed``。

    差值定义为 ``static - system``：主指标越低越好的任务（MAE/RMSE）里正数表示系统更好；
    分类任务请在调用方先把方向翻过来。
    """
    deltas = np.array([r["static"] - r["system"] for r in rows], dtype=float)
    wins = int((deltas > 0).sum())
    losses = int((deltas < 0).sum())
    ties = int((deltas == 0).sum())
    lo, hi = bootstrap_ci(deltas)

    per_seed: Dict[int, List[float]] = {}
    for r in rows:
        per_seed.setdefault(r["seed"], []).append(r["static"] - r["system"])
    seed_means = np.array([np.mean(v) for v in per_seed.values()])

    return {
        "dataset": label,
        "policy": policy,
        "n_batches": len(rows),
        "n_seeds": len(per_seed),
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "ci95": [lo, hi],
        "wins": wins, "losses": losses, "ties": ties,
        "win_rate": wins / max(wins + losses, 1),
        "sign_test_p": sign_test(wins, losses),
        "seed_mean_min": float(seed_means.min()),
        "seed_mean_max": float(seed_means.max()),
        "mean_static": float(np.mean([r["static"] for r in rows])),
        "mean_system": float(np.mean([r["system"] for r in rows])),
    }


def print_summary(stats: Dict[str, Any], metric: str = "MAE") -> None:
    print("\n" + "─" * 62)
    print(f"  数据集          : {stats.get('dataset') or '-'}")
    print(f"  策略            : {stats['policy']}")
    print(f"  切分数 × 批次数  : {stats['n_seeds']} × {stats['n_batches'] // max(stats['n_seeds'], 1)}")
    print(f"  静态 / 系统 {metric:<4}: {stats['mean_static']:.2f} / {stats['mean_system']:.2f}")
    print(f"  配对差值均值     : {stats['mean_delta']:+.2f} {metric}（正=系统更好）")
    print(f"  95% 置信区间     : [{stats['ci95'][0]:+.2f}, {stats['ci95'][1]:+.2f}]")
    print(f"  胜/负/平         : {stats['wins']}/{stats['losses']}/{stats['ties']}"
          f"  胜率 {stats['win_rate']:.1%}")
    print(f"  符号检验 p       : {stats['sign_test_p']:.4f}")
    print(f"  各种子均值范围   : [{stats['seed_mean_min']:+.2f}, {stats['seed_mean_max']:+.2f}]")
    print("─" * 62)
