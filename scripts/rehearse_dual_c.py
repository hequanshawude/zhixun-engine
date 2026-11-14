"""Run both C branches from the official starting package in ten paired rounds."""
import json
import os
import subprocess
from pathlib import Path

from pytrio.config.config import auth_state, configuration


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "out/dual-c-27b-full-fixed-20261114"
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
    "-v", f"{(ROOT / 'data/raw/cpu/c').resolve()}:/data:ro",
    "-v", f"{TARGET.resolve()}:/out",
    "zhixun-engine:dual-c-27b-candidate-20261114",
]
with (TARGET / "rehearsal.log").open("w", encoding="utf-8") as log:
    proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
summary_path = TARGET / "data/summary.json"
if not summary_path.exists():
    raise RuntimeError(f"Container exit {proc.returncode}; no summary.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
print("C", "rmse", summary.get("final"), "Qwen rounds", summary.get("llm_rounds"),
      "reference rounds", summary.get("dual", {}).get("reference_rounds"),
      "tokens", summary.get("tokens"), flush=True)
if proc.returncode or summary.get("llm_rounds") != 10 or summary.get("tokens", 0) <= 0:
    raise RuntimeError("Dual C run did not complete")
