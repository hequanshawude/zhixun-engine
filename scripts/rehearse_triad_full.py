"""Run the three-group final-checkpoint candidate through its default Docker CMD."""
import json
import os
import subprocess
from pathlib import Path

from pytrio.config.config import auth_state, configuration

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "out/triad-full-27b-20261114"
if TARGET.exists():
    raise FileExistsError(TARGET)
if configuration.base_url.rstrip("/") != "https://pytrio.com" or not auth_state.api_key:
    raise RuntimeError("PyTRiO login is unavailable")
TARGET.mkdir(parents=True)
env = dict(os.environ)
env["PYTRIO_API_KEY"] = auth_state.api_key
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"
command = [
    "docker", "run", "--rm", "--network", "bridge", "--cpus", "4", "--memory", "2g",
    "-e", "PYTRIO_API_KEY", "-e", "PYTHONIOENCODING", "-e", "PYTHONUTF8",
    "-v", f"{(ROOT / 'data/raw/cpu').resolve()}:/data:ro",
    "-v", f"{TARGET.resolve()}:/out",
    "zhixun-engine:triad-27b-candidate-20261114",
]
with (TARGET / "rehearsal.log").open("w", encoding="utf-8") as log:
    proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
summary_path = TARGET / "all-summary.json"
if not summary_path.exists():
    raise RuntimeError(f"Container exit {proc.returncode}; no all-summary.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
for profile, item in summary["results"].items():
    print(profile, "score", item.get("final"), "Qwen rounds", item.get("llm_rounds"),
          "tokens", item.get("tokens"), "error", item.get("error"), flush=True)
if proc.returncode or summary["failed"] or summary["packages"] != 3:
    raise RuntimeError(f"Full run failed: {summary['failed']}")
if any(item.get("rounds") != 10 or item.get("llm_rounds") != 10 or item.get("tokens", 0) <= 0
       for item in summary["results"].values()):
    raise RuntimeError("Some Qwen-driven rounds were missing")
