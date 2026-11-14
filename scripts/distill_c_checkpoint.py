"""Fit a C checkpoint readout to a train-only teacher after ten CPU rounds.

The teacher and student must share the same parent checkpoint. Validation only
gates the finished checkpoint; it is never used in the linear solve.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selftrain.config import LogConfig
from selftrain.experiment import ExperimentLogger


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(y, pred):
    error = pred - y
    return {"rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloud", action="store_true")
    args = parser.parse_args()
    package, checkpoint = args.package.resolve(), args.checkpoint.resolve()
    correction, output = args.correction.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    sys.path.insert(0, str(package))
    sys.path.insert(0, str(ROOT / "scripts"))
    from baselines.data import read_dataset
    from baselines.model import Model
    from rul_tree_correction import CorrectedModel

    base = Model.load(checkpoint)
    if base.meta["task"] != "regression" or base.meta.get("profile") != "C":
        raise ValueError("C regression checkpoint required")
    teacher = CorrectedModel(base, checkpoint, correction)
    train = read_dataset(package / "public/train.jsonl", "regression", [], base.meta["paths"])
    validation = read_dataset(package / "public/validation.jsonl", "regression", [], base.meta["paths"])
    hidden = np.tanh(base.normalized(train["x"]) @ base.params["w1"] + base.params["b1"])
    design = np.column_stack((hidden, np.ones(len(hidden))))
    target = (teacher.predict(train["x"]) - base.meta["target_mean"]) / base.meta["target_scale"]
    weights = np.linalg.solve(design.T @ design, design.T @ target)
    student = Model.load(checkpoint)
    student.params["w2"][:, 0] = weights[:-1]
    student.params["b2"][0] = weights[-1]
    student.meta["post_training"] = {"method": "train_only_teacher_readout",
                                     "parent_sha256": sha(checkpoint),
                                     "teacher_sha256": sha(correction)}
    student.validate()
    original = base.predict(validation["x"])
    revised = student.predict(validation["x"])
    groups = np.array([item.split("_")[0] for item in validation["ids"]])
    group_scores = {name: {"before": score(validation["y"][groups == name], original[groups == name]),
                           "after": score(validation["y"][groups == name], revised[groups == name])}
                    for name in np.unique(groups)}
    before, after = score(validation["y"], original), score(validation["y"], revised)
    accepted = after["rmse"] < before["rmse"] and all(
        row["after"]["rmse"] < row["before"]["rmse"] for row in group_scores.values())
    result = {"status": "accepted" if accepted else "rejected",
              "profile": "C", "train_rows": len(train["y"]),
              "validation_rows": len(validation["y"]),
              "parent_checkpoint_sha256": sha(checkpoint), "teacher_sha256": sha(correction),
              "before": before, "after": after, "groups": group_scores,
              "parent_epochs": base.meta["epochs"], "parent_steps": base.meta["steps"]}
    if accepted:
        artifact = output / "checkpoint.npz"
        student.save(artifact)
        result["checkpoint_sha256"] = sha(artifact)
    logger = None
    if args.cloud:
        logger = ExperimentLogger(LogConfig(backend="swanlab", workspace="hequan",
                                            project="zhixunyinqing", out_dir=str(output / "swanlab"),
                                            swanlab_mode="online"), run_name="c-readout-distill")
        logger.start({"method": "train_only_teacher_readout",
                      "parent_checkpoint_sha256": result["parent_checkpoint_sha256"]},
                     {"source": "official public C train/validation", "train_rows": len(train["y"])})
        logger.log(1, {"C_rmse_before": before["rmse"], "C_rmse_after": after["rmse"],
                       "accepted": int(accepted)})
        logger.log_decision({"batch": 1, "method": "train_only_teacher_readout",
                             "accepted": accepted, "groups": group_scores})
        result["cloud_url"] = logger.cloud_url
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if logger:
        logger.finish({"C_rmse_before": before["rmse"], "C_rmse_after": after["rmse"],
                       "accepted": int(accepted)})
    print(json.dumps(result, ensure_ascii=False))
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
