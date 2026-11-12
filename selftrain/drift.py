"""漂移检测。

分两层，这是成本控制的关键：
    无标签侧（每批都能算，便宜）：PSI + KS + MMD  → 早预警，会误报
    有标签侧（拿到标签后，准）  ：Page-Hinkley 变点检测 + 窗口均值检验

策略层只在两层同时报警时才愿意花大钱。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import DriftConfig
from .types import DriftSignal


# --------------------------------------------------------------------------
# 基础统计量（不依赖 scipy）
# --------------------------------------------------------------------------


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """两样本 Kolmogorov-Smirnov。返回 (D, p_value)。

    p 值用渐近近似，n,m 较小时给保守估计——对我们的用途足够了。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n, m = len(a), len(b)
    if n < 5 or m < 5:
        return 0.0, 1.0

    a = np.sort(a)
    b = np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / n
    cdf_b = np.searchsorted(b, grid, side="right") / m
    d = float(np.max(np.abs(cdf_a - cdf_b)))

    en = np.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam <= 0:
        return d, 1.0
    k = np.arange(1, 101)
    terms = 2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * (k**2) * lam**2))
    p = float(np.clip(terms, 0.0, 1.0))
    return d, p


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index。业界经验阈值：<0.1 稳定，0.1~0.25 中等，>0.25 显著。"""
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 20 or len(cur) < 20:
        return 0.0

    qs = np.quantile(ref, np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf

    r = np.histogram(ref, qs)[0].astype(float) / len(ref)
    c = np.histogram(cur, qs)[0].astype(float) / len(cur)
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def mmd_rbf(ref: np.ndarray, cur: np.ndarray, subsample: int = 300) -> float:
    """RBF 核 MMD（有偏估计，中位数启发式带宽）。用于捕捉"联合分布"层面的漂移。"""
    rng = np.random.default_rng(0)
    a = _sub(ref, subsample, rng)
    b = _sub(cur, subsample, rng)
    if len(a) < 10 or len(b) < 10:
        return 0.0

    # 标准化，避免量纲主导带宽
    mu = np.vstack([a, b]).mean(0, keepdims=True)
    sd = np.vstack([a, b]).std(0, keepdims=True) + 1e-9
    a = (a - mu) / sd
    b = (b - mu) / sd

    z = np.vstack([a, b])
    d2 = _sq_dists(z)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / max(med, 1e-6)

    kaa = np.exp(-gamma * _sq_dists(a))
    kbb = np.exp(-gamma * _sq_dists(b))
    kab = np.exp(-gamma * _sq_dists(a, b))
    return float(max(0.0, kaa.mean() + kbb.mean() - 2 * kab.mean()))


def _sub(x: np.ndarray, n: int, rng) -> np.ndarray:
    if len(x) <= n:
        return x
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx]


def _sq_dists(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    if b is None:
        b = a
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    d = aa + bb - 2.0 * a @ b.T
    return np.maximum(d, 0.0)


def total_variation(p: np.ndarray, q: np.ndarray, n_bins: int = 10) -> float:
    """两个一维分布的经验 TV 距离（分类问题里衡量先验漂移）。"""
    p = p[np.isfinite(p)]
    q = q[np.isfinite(q)]
    if len(p) < 5 or len(q) < 5:
        return 0.0
    lo = min(p.min(), q.min())
    hi = max(p.max(), q.max())
    if hi - lo < 1e-9:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    hp = np.histogram(p, edges)[0] / len(p)
    hq = np.histogram(q, edges)[0] / len(q)
    return float(0.5 * np.abs(hp - hq).sum())


# --------------------------------------------------------------------------
# 变点检测
# --------------------------------------------------------------------------


class PageHinkley:
    """Page-Hinkley 变点检测，对"错误率突然上升"很敏感，成本 O(1)。"""

    def __init__(self, delta: float = 0.005, threshold: float = 0.05, min_obs: int = 4):
        self.delta = delta
        self.threshold = threshold
        self.min_obs = min_obs
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.max_cum = -np.inf
        self.alarms = 0

    def update(self, value: float) -> bool:
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.cum += value - self.mean - self.delta
        self.max_cum = max(self.max_cum, self.cum)
        fired = self.n >= self.min_obs and (self.max_cum - self.cum) > self.threshold
        if fired:
            self.alarms += 1
            self.reset()
        return fired

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.max_cum = -np.inf


class WindowMeanTest:
    """两窗口均值检验（ADWIN 的轻量替代）：Hoeffding 界。"""

    def __init__(self, window: int = 5, alpha: float = 0.05):
        self.window = window
        self.alpha = alpha
        self.values: List[float] = []

    def update(self, value: float) -> bool:
        self.values.append(float(value))
        if len(self.values) > 2 * self.window:
            self.values = self.values[-self.window * 2:]
        if len(self.values) < 2 * self.window:
            return False
        old = np.array(self.values[: self.window])
        new = np.array(self.values[self.window:])
        n = len(old)
        bound = np.sqrt(np.log(2.0 / self.alpha) / (2.0 * n))
        return bool(abs(new.mean() - old.mean()) > bound)


# --------------------------------------------------------------------------
# 检测器
# --------------------------------------------------------------------------


class UnlabeledDriftDetector:
    """以第一批为参照，逐批计算特征侧漂移。"""

    def __init__(self, cfg: DriftConfig):
        self.cfg = cfg
        self.ref: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self._mmd_scale: float = 1e-6

    def fit(self, X: np.ndarray, feature_names: Sequence[str]) -> None:
        self.ref = np.asarray(X, dtype=float)
        self.feature_names = list(feature_names)
        # 参照集自比的 MMD 作为噪声底
        if len(self.ref) > 40:
            half = len(self.ref) // 2
            self._mmd_scale = max(
                mmd_rbf(self.ref[:half], self.ref[half:], self.cfg.mmd_subsample), 1e-4
            )

    def score(self, X: np.ndarray) -> DriftSignal:
        if self.ref is None:
            raise RuntimeError("请先调用 fit()")
        X = np.asarray(X, dtype=float)

        ps: Dict[str, float] = {}
        ks: Dict[str, float] = {}
        for j, name in enumerate(self.feature_names):
            r = self.ref[:, j]
            c = X[:, j]
            ps[name] = psi(r, c)
            _, p = ks_2samp(r, c)
            ks[name] = p

        frac = float(np.mean([v > self.cfg.psi_threshold for v in ps.values()])) if ps else 0.0
        max_psi = max(ps.values()) if ps else 0.0
        mmd = mmd_rbf(self.ref, X, self.cfg.mmd_subsample)

        psi_sev = min(1.0, max_psi / (3.0 * self.cfg.psi_threshold + 1e-9))
        mmd_sev = min(1.0, mmd / (3.0 * self._mmd_scale + 1e-9))
        # 三路加权：漂移特征占比 45% + 最强单特征 35% + 联合分布 20%
        severity = float(np.clip(0.45 * frac + 0.35 * psi_sev + 0.20 * mmd_sev, 0.0, 1.0))

        top = sorted(ps.items(), key=lambda kv: kv[1], reverse=True)[: self.cfg.n_top_features]
        return DriftSignal(
            severity=severity,
            per_feature_psi=ps,
            per_feature_ks=ks,
            top_features=[k for k, _ in top],
            frac_drifted=frac,
            mmd=mmd,
            reference_size=int(len(self.ref)),
        )


class LabeledDriftDetector:
    """基于每批指标序列的变点检测。"""

    def __init__(self, cfg: DriftConfig):
        self.cfg = cfg
        self.ph = PageHinkley(delta=cfg.ph_delta, threshold=cfg.ph_lambda)
        self.window = WindowMeanTest(window=cfg.window)
        self.series: List[float] = []
        self.alarms: List[int] = []

    def update(self, batch_index: int, value: float) -> bool:
        self.series.append(float(value))
        a = self.ph.update(value)
        b = self.window.update(value)
        if a or b:
            self.alarms.append(batch_index)
        return a or b

    @property
    def latest_alarm(self) -> Optional[int]:
        return self.alarms[-1] if self.alarms else None
