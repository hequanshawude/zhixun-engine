"""一键运行：环境自检 → 基线回放 → 配对验证 →（可选）27B 决策，全部记录到 SwanLab。

用法（在项目根目录）：

    run-all.cmd                          # 完整跑一遍，上云
    run-all.cmd --dry-run                # 只打印要执行的命令
    run-all.cmd --seeds 10 --skip-preflight
    run-all.cmd --local                  # 只在本地记录，不联网
    run-all.cmd --with-llm               # 额外跑 PyTRiO 27B（消耗额度）

设计取舍：
    - 默认全部上云，符合"实验必须留云端记录"的要求；
    - 27B 默认不跑：每次约 7k token，容易被误触消耗额度，必须显式 --with-llm；
    - preflight 默认跑（含 81 项测试），但可用 --skip-preflight 跳过；
    - 任何一步失败都不会中断后续步骤，最后统一汇总并返回非零退出码。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
# 只认"实验"链接：swanlab 会先打印项目页 https://swanlab.cn/@user/project，
# 直接 \S+ 会把项目页当成实验结果抓走。
URL_RE = re.compile(r"https://swanlab\.cn/\S+/runs/\S+")


def _recent_cloud_url(since: float) -> str:
    """兜底：某些入口只在 cloud.json 里落 URL，不在 stdout 打印。"""
    newest, newest_mtime = "", 0.0
    runs = ROOT / "runs"
    if not runs.exists():
        return ""
    for path in runs.rglob("cloud.json"):
        try:
            mtime = path.stat().st_mtime
            if mtime < since - 5 or mtime <= newest_mtime:
                continue
            url = json.loads(path.read_text(encoding="utf-8")).get("url") or ""
            if "/runs/" in url:
                newest, newest_mtime = url, mtime
        except Exception:
            continue
    return newest


@dataclass
class Step:
    title: str
    cmd: list[str]
    ok: bool = False
    url: str = ""
    seconds: float = 0.0
    detail: str = ""
    skipped: bool = False


def _banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70, flush=True)


def _width(s: str) -> int:
    """终端显示宽度：CJK 字符占两格，直接 len() 会把表格排歪。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _width(s))


def _tail(text: str, n: int = 12) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def run_step(title: str, cmd: list[str], dry: bool = False,
             timeout: int = 7200) -> Step:
    step = Step(title=title, cmd=cmd)
    _banner(title)
    print("  $ " + " ".join(cmd), flush=True)
    if dry:
        step.skipped = True
        step.detail = "dry-run"
        return step

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        step.seconds = time.time() - t0
        step.detail = f"超时（>{timeout}s）"
        print(f"  ✗ {step.detail}", flush=True)
        return step

    step.seconds = time.time() - t0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    step.ok = proc.returncode == 0
    match = URL_RE.search(out)
    step.url = match.group(0).rstrip('.,;)') if match else ""
    if not step.url and step.ok:
        step.url = _recent_cloud_url(t0)

    if step.ok:
        suffix = f"  →  {step.url}" if step.url else ""
        print(f"  ✓ 完成，用时 {step.seconds:.0f}s{suffix}", flush=True)
    else:
        step.detail = f"退出码 {proc.returncode}"
        print(f"  ✗ 失败（{step.detail}），输出尾部：", flush=True)
        print(_tail(out), flush=True)
    return step


# --------------------------------------------------------------------------
# 环境检查
# --------------------------------------------------------------------------


def check_env(need_llm: bool, cloud: bool) -> list[str]:
    problems: list[str] = []
    if not (ROOT / ".venv" / "Scripts" / "python.exe").exists():
        problems.append("找不到 .venv\\Scripts\\python.exe，请先创建虚拟环境")
    if not (ROOT / "data" / "processed").exists():
        problems.append("找不到 data\\processed，请先准备数据集")
    for name, rel in (("XJTU-SY", "data/processed/xjtu-sy/train.csv"),
                      ("C-MAPSS", "data/processed/cmapss-fd001-holdout/train.csv")):
        if not (ROOT / rel).exists():
            problems.append(f"{name} 处理后数据缺失：{rel}")
    if cloud and not (Path.home() / ".swanlab" / ".netrc").exists():
        problems.append("未登录 SwanLab（找不到 ~/.swanlab/.netrc），请先运行 login-cloud.cmd")
    if need_llm:
        if not (ROOT / "configs" / "xjtu-sy-27b-cloud.yaml").exists():
            problems.append("缺少 27B 配置 configs/xjtu-sy-27b-cloud.yaml")
        if not (Path.home() / ".pytrio" / "config.toml").exists():
            problems.append("未登录 PyTRiO（找不到 ~/.pytrio/config.toml），请先运行 login-pytrio.cmd")
    return problems


# --------------------------------------------------------------------------
# 各实验的命令
# --------------------------------------------------------------------------


def xjtu_canonical(policy: str, config: str) -> list[str]:
    return [PY, "-m", "selftrain", "run",
            "--data-dir", "data/processed/xjtu-sy/stream",
            "--train-csv", "data/processed/xjtu-sy/train.csv",
            "--target", "y", "--task", "regression",
            "--policy", policy, "--config", config,
            "--name", f"xjtu-sy-{policy}", "--quiet"]


def cmapss_canonical(policy: str, config: str) -> list[str]:
    return [PY, "-m", "selftrain", "run",
            "--data-dir", "data/processed/cmapss-fd001/stream",
            "--train-csv", "data/processed/cmapss-fd001-holdout/train.csv",
            "--gate-csv", "data/processed/cmapss-fd001-holdout/gate.csv",
            "--target", "y", "--task", "regression",
            "--policy", policy, "--config", config,
            "--name", f"fd001-{policy}", "--quiet"]


def main() -> int:
    ap = argparse.ArgumentParser(description="智训引擎一键运行（默认全部上云）")
    ap.add_argument("--seeds", type=int, default=30, help="配对验证的随机切分个数")
    ap.add_argument("--local", action="store_true", help="只写本地记录，不连 SwanLab")
    ap.add_argument("--with-llm", action="store_true", help="额外运行 PyTRiO Qwen3.8-27B（消耗额度）")
    ap.add_argument("--skip-preflight", action="store_true", help="跳过 81 项测试与六组合成流演练")
    ap.add_argument("--skip-validate", action="store_true", help="跳过配对验证（只跑基线）")
    ap.add_argument("--datasets", default="xjtu,cmapss", help="逗号分隔：xjtu,cmapss")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    args = ap.parse_args()

    cloud = not args.local
    wanted = {s.strip().lower() for s in args.datasets.split(",") if s.strip()}

    _banner("环境检查")
    problems = check_env(args.with_llm, cloud)
    if problems:
        for p in problems:
            print(f"  ✗ {p}")
        if not args.dry_run:
            print("\n请先解决上述问题再运行。")
            return 2
    else:
        print("  ✓ 虚拟环境、处理后数据、平台登录均就绪")

    # 上云时用 *-cloud.yaml，本地时用 *-rule.yaml
    suffix = "cloud" if cloud else "rule"
    xjtu_conf = f"configs/xjtu-sy-{suffix}.yaml"
    cmapss_conf = f"configs/cmapss-fd001-holdout-{suffix}.yaml"

    steps: list[Step] = []

    if not args.skip_preflight:
        steps.append(run_step("① 自检：81 项测试 + 六组合成流演练",
                              [PY, "scripts/preflight.py"], dry=args.dry_run))
    else:
        steps.append(Step("① 自检", [], skipped=True, detail="--skip-preflight"))

    if "xjtu" in wanted:
        steps.append(run_step("② XJTU-SY 规范化基线（规则策略）",
                              xjtu_canonical("rule", xjtu_conf), dry=args.dry_run))
    if "cmapss" in wanted:
        steps.append(run_step("③ C-MAPSS FD001 规范化基线（规则策略）",
                              cmapss_canonical("rule", cmapss_conf), dry=args.dry_run))

    if not args.skip_validate:
        if "xjtu" in wanted:
            steps.append(run_step(
                f"④ XJTU-SY 配对验证（{args.seeds} 个随机切分）",
                [PY, "scripts/xjtu_validate.py", "--seeds", str(args.seeds),
                 "--policy", "rule", "--config", xjtu_conf] + (["--cloud"] if cloud else []),
                dry=args.dry_run))
        if "cmapss" in wanted:
            steps.append(run_step(
                f"⑤ C-MAPSS FD001 配对验证（{args.seeds} 个随机切分）",
                [PY, "scripts/cmapss_validate.py", "--seeds", str(args.seeds),
                 "--policy", "rule", "--config", cmapss_conf] + (["--cloud"] if cloud else []),
                dry=args.dry_run))

    if args.with_llm:
        if "xjtu" in wanted:
            steps.append(run_step(
                "⑥ XJTU-SY + PyTRiO Qwen3.8-27B 决策",
                xjtu_canonical("llm", "configs/xjtu-sy-27b-cloud.yaml"), dry=args.dry_run))
        if "cmapss" in wanted:
            steps.append(run_step(
                "⑦ C-MAPSS FD001 + PyTRiO Qwen3.8-27B 决策",
                cmapss_canonical("llm", "configs/cmapss-fd001-holdout-27b.yaml"), dry=args.dry_run))

    # ---------------- 汇总 ----------------
    _banner("运行汇总")
    print(f"{_pad('步骤', 42)}{_pad('状态', 8)}{'用时':>8}   云端记录")
    print("-" * 100)
    for s in steps:
        status = "跳过" if s.skipped else ("成功" if s.ok else "失败")
        print(f"{_pad(s.title, 42)}{_pad(status, 8)}{s.seconds:>7.0f}s   {s.url or s.detail}")

    failed = [s for s in steps if not s.ok and not s.skipped]
    out_dir = ROOT / "out" / "run-all"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (out_dir / f"summary-{stamp}.json").write_text(
            json.dumps({"cloud": cloud, "seeds": args.seeds,
                        "with_llm": args.with_llm, "steps": [asdict(s) for s in steps]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n汇总已写入：{out_dir / f'summary-{stamp}.json'}")

    if failed:
        print(f"\n{len(failed)} 个步骤失败，见上方输出。")
        return 1
    if not args.dry_run:
        print("\n全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
