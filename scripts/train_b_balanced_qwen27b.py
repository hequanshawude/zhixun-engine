"""Experimental 10-round B continuation with train-only device balancing."""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package", type=Path, default=root / "data/raw/cpu/b")
parser.add_argument("--source", type=Path, required=True,
                    help="A B checkpoint widened from the official starting checkpoint")
parser.add_argument("--output", type=Path, required=True,
                    help="A new directory for ten rounds and the final checkpoint")
args = parser.parse_args()
pkg = args.package.resolve()
sys.path.insert(0, str(pkg))
from baselines.model import Model
from baselines.data import read_dataset
from selftrain.config import HarnessConfig, LogConfig
from selftrain.cpu import LLMPolicy, LR_TABLE, EPOCH_CHOICES
from selftrain.experiment import ExperimentLogger
from selftrain.pytrio_access import configure_credentials

source = args.source.resolve()
output = args.output.resolve()
if output.exists():
    raise FileExistsError(output)
if Model.load(source).meta.get("profile") != "B":
    raise ValueError("B checkpoint required")
output.mkdir(parents=True)
train = read_dataset(pkg / "public/train.jsonl", "regression", [], Model.load(source).meta["paths"])
val = read_dataset(pkg / "public/validation.jsonl", "regression", [], train["paths"])
tgroups = np.asarray([s.rsplit("_s", 1)[0] for s in train["ids"]])
vgroups = np.asarray([s.rsplit("_s", 1)[0] for s in val["ids"]])
rng = np.random.default_rng(20261114)
indices = np.concatenate([rng.choice(np.flatnonzero(tgroups == g), size=800, replace=True)
                          for g in np.unique(tgroups)])

def assess(model):
    err = model.predict(val["x"]) - val["y"]
    per = {g: float(np.sqrt(np.mean(err[vgroups == g] ** 2))) for g in np.unique(vgroups)}
    return {"pooled_rmse": float(np.sqrt(np.mean(err ** 2))),
            "device_mean_rmse": float(np.mean(list(per.values()))), "per_device": per}

cfg = HarnessConfig(credential_source="pytrio", base_url="https://pytrio.com/api/openai/v1",
                    model="Qwen/Qwen3.8-27B", small_model="Qwen/Qwen3.8-27B",
                    pytrio_base_model="Qwen/Qwen3.8-27B", max_tokens=128,
                    timeout_s=90, retries=0)
configure_credentials(cfg)
policy = LLMPolicy(cfg, strict=True)
logger = ExperimentLogger(LogConfig(backend="swanlab", workspace="hequan",
                                    project="zhixunyinqing", out_dir=str(output / "swanlab"),
                                    swanlab_mode="online"), run_name="b-balanced-qwen27b")
logger.start({"model": cfg.model, "rounds": 10, "balanced_rows_per_device": 800,
              "starting_checkpoint": str(source.relative_to(root))},
             {"data": "official public B train/validation", "gate": "pooled and device-mean RMSE"})

accepted = source
best = assess(Model.load(accepted))
history = []
try:
    for r in range(1, 11):
        card = {"dataset": "B", "metric": "rmse", "higher_is_better": False,
                "round": r, "rounds_left": 11-r,
                "best_so_far": round(best["pooled_rmse"], 4),
                "device_mean_rmse": round(best["device_mean_rmse"], 4),
                "validation_device_counts": {g: int(np.sum(vgroups == g)) for g in np.unique(vgroups)},
                "training_sampling": "800 rows per bearing, training labels only",
                "history": history[-5:], "allowed_learning_rate": LR_TABLE,
                "allowed_epochs": EPOCH_CHOICES}
        lr, epochs, reason = policy.choose(card)
        model = Model.load(accepted)
        model.meta["learning_rate"] = lr
        candidate = None
        candidate_metric = None
        epoch_records = []
        for ep in range(1, epochs + 1):
            model.fit(train["x"][indices], train["y"][indices], 1)
            metric = assess(model)
            epoch_records.append({"epoch": ep, **metric})
            if metric["pooled_rmse"] < best["pooled_rmse"] and metric["device_mean_rmse"] < best["device_mean_rmse"]:
                if candidate_metric is None or metric["pooled_rmse"] < candidate_metric["pooled_rmse"]:
                    candidate, candidate_metric = copy.deepcopy(model), metric
        round_dir = output / f"round-{r:02}"
        round_dir.mkdir(exist_ok=True)
        accepted_this_round = candidate is not None
        if accepted_this_round:
            next_path = round_dir / "accepted.npz"
            candidate.save(next_path)
            accepted = next_path
            best = candidate_metric
        record = {"round": r, "learning_rate": lr, "epochs": epochs,
                  "reason": reason, "accepted": accepted_this_round,
                  "best": best, "epoch_records": epoch_records,
                  "tokens_cumulative": policy.tokens}
        (round_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        history.append({"r": r, "lr": lr, "ep": epochs,
                        "v": round(best["pooled_rmse"], 4), "ok": accepted_this_round})
        logger.log(r, {"B_pooled_rmse": best["pooled_rmse"],
                       "B_device_mean_rmse": best["device_mean_rmse"],
                       "accepted": int(accepted_this_round), "tokens": policy.tokens})
        logger.log_decision({"batch": r, "learning_rate": lr, "epochs": epochs,
                             "reason": reason, "accepted": accepted_this_round,
                             "best": best, "tokens_cumulative": policy.tokens})
        print(f"round {r:02}: lr={lr} ep={epochs} accepted={accepted_this_round} "
              f"pooled={best['pooled_rmse']:.3f} device={best['device_mean_rmse']:.3f} "
              f"tokens={policy.tokens}", flush=True)
    final = output / "final-checkpoint.npz"
    shutil.copyfile(accepted, final)
    summary = {"profile": "B", "rounds": 10, "llm_rounds": 10, "tokens": policy.tokens,
               "start": assess(Model.load(source)), "final": best,
               "final_checkpoint": str(final), "cloud_url": logger.cloud_url}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.finish({"pooled_rmse": best["pooled_rmse"],
                   "device_mean_rmse": best["device_mean_rmse"], "tokens": policy.tokens})
    print("final", json.dumps(summary, ensure_ascii=False), flush=True)
except Exception as exc:
    logger.abort(f"{type(exc).__name__}: {exc}")
    raise
