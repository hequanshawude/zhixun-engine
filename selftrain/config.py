"""配置：一个 dataclass 树，支持从 yaml/json 覆盖。

设计原则：所有"可调超参"都在这里，方便做策略层实验与 ablation。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DriftConfig:
    # 无标签侧
    psi_threshold: float = 0.20          # 单特征 PSI 超过即视为漂移
    ks_alpha: float = 0.01               # KS 显著性水平
    mmd_subsample: int = 300
    n_top_features: int = 10
    # 有标签侧
    ph_delta: float = 0.005              # Page-Hinkley 容忍度
    ph_lambda: float = 0.05              # Page-Hinkley 检测阈值
    window: int = 5                      # 最近 N 个 batch 的滚动统计
    # 综合严重度归一化
    severity_scale: float = 0.6


@dataclass
class ModelConfig:
    default_kind: str = "hgb"            # hgb / logreg / ridge / lightgbm
    alt_kind: str = "logreg"             # switch_model 时用
    params: Dict[str, Any] = field(default_factory=dict)
    random_state: int = 42
    # finetune 的"新旧"权重来源：
    #   row   —— 按拼接后的行序衰减（旧行为；仅当行序=时间序时成立）
    #   batch —— 按批次到达顺序衰减（正确的时间语义，真实数据推荐）
    finetune_recency: str = "row"
    finetune_decay: float = 0.5          # 每回退一批，权重乘这个系数
    finetune_floor: float = 0.15         # 最老批次的权重下限

    def training_params(self, alternate: bool = False) -> Dict[str, Any]:
        params = {} if alternate else dict(self.params)
        params['random_state'] = self.params.get('random_state', self.random_state)
        return params


@dataclass
class GateConfig:
    # 晋级闸门
    min_gain: float = 0.002              # 必须比锚点高出的主指标绝对值
    max_drop: float = 0.02               # 超过这个退化直接回滚
    # 相对阈值：按锚点指标量级缩放，0 表示不启用。
    # 回归任务（RUL 的 MAE 可达数百）必须靠这两个，否则绝对阈值形同虚设。
    min_gain_rel: float = 0.0
    max_drop_rel: float = 0.0
    static_guard: float = 0.0            # 不得低于静态基线的容差
    rollback_on_breach: bool = True
    # 评估
    val_batches: int = 1
    purge_batches: int = 0

    def __post_init__(self):
        if type(self.val_batches) is not int or self.val_batches < 1:
            raise ValueError('val_batches 必须为正整数')
        if type(self.purge_batches) is not int or self.purge_batches < 0:
            raise ValueError('purge_batches 必须为非负整数')


@dataclass
class BudgetConfig:
    max_compute_seconds: float = 3600.0
    max_llm_tokens: int = 200_000
    max_label_queries: int = 500
    max_iterations: int = 200
    bootstrap_timeout_s: Optional[float] = 120.0  # 首次模型训练；含工作进程启动
    action_timeout_s: Optional[float] = 60.0      # 单次候选训练；null 显式关闭隔离
    # 归一化权重：1 秒算力 = 1 单位；1 token = ?; 1 次标注 = ?
    token_cost: float = 1.0 / 2000.0
    label_cost: float = 2.0


@dataclass
class HarnessConfig:
    enabled: bool = False
    credential_source: str = "env"      # env / pytrio（官方客户端登录状态）
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "qwen3.8-27b"
    small_model: str = "qwen3.8-27b"     # 实际服务模型 ID 须由平台确认
    pytrio_base_model: str = "Qwen/Qwen3.5-4B"  # trio:// 权重所属的基模
    escalate_on: List[str] = field(default_factory=lambda: ["concept", "mixed"])
    temperature: float = 0.0
    max_tokens: int = 384
    timeout_s: float = 30.0
    retries: int = 2
    cache: bool = True


@dataclass
class LogConfig:
    backend: str = "auto"                # auto / swanlab / jsonl / none
    project: str = "zhixun-engine"
    workspace: Optional[str] = None
    experiment: str = "run"
    out_dir: str = "runs"
    swanlab_mode: str = "online"         # online / offline / local / disabled


@dataclass
class SystemConfig:
    seed: int = 42
    drift: DriftConfig = field(default_factory=DriftConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    log: LogConfig = field(default_factory=LogConfig)
    playbook_path: Optional[str] = None

    # ---------- 序列化 ----------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemConfig":
        return _from_dict(cls, data or {})

    @classmethod
    def load(cls, path: Optional[str]) -> "SystemConfig":
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml  # 局部导入：没有 yaml 也能跑 JSON
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)


def _from_dict(cls, data: Dict[str, Any]):
    """递归地把 dict 还原成 dataclass，忽略未知字段。"""
    if not is_dataclass(cls):
        return data
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) or (isinstance(f.type, str) and f.type in _REGISTRY):
            sub = _REGISTRY.get(f.type if isinstance(f.type, str) else f.type.__name__)
            if sub is not None and isinstance(value, dict):
                kwargs[f.name] = _from_dict(sub, value)
                continue
        kwargs[f.name] = value
    return cls(**kwargs)


_REGISTRY = {}
for _name in ("DriftConfig", "ModelConfig", "GateConfig", "BudgetConfig", "HarnessConfig", "LogConfig"):
    _REGISTRY[_name] = globals()[_name]


def dump_config(cfg: SystemConfig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def apply_env_overrides(cfg: SystemConfig) -> SystemConfig:
    """用环境变量覆盖敏感项，便于镜像里注入平台 token。"""
    if os.getenv("HARNESS_BASE_URL"):
        cfg.harness.base_url = os.environ["HARNESS_BASE_URL"]
    if os.getenv("HARNESS_API_KEY"):
        cfg.harness.api_key = os.environ["HARNESS_API_KEY"]
    if os.getenv("HARNESS_MODEL"):
        cfg.harness.model = os.environ["HARNESS_MODEL"]
    if cfg.harness.credential_source == "pytrio":
        from .pytrio_access import configure_credentials
        configure_credentials(cfg.harness)
    if os.getenv("SWANLAB_API_KEY"):
        os.environ.setdefault("SWANLAB_API_KEY", os.environ["SWANLAB_API_KEY"])
    return cfg
