"""Fit the validation-gated correction and export a standalone JSONL predictor."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


def fit_correction(tag, base, checkpoint, train, destination):
    x, y = train["x"], train["y"]
    if tag == "A":
        from gas_pair_gate import GasPairGate, save_confirmed

        names = base.meta["classes"]
        features = base.normalized(x)[:, 1::8]
        pos = features[y == names.index("Acetaldehyde")]
        neg = features[y == names.index("Ethanol")]
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        save_confirmed(destination, pos, neg, digest, "coverage")
        return GasPairGate(checkpoint, destination)

    from rul_tree_correction import CorrectedModel, save_forest

    if tag == "B":
        mode, blend = "direct", 0.5
        forest = ExtraTreesRegressor(n_estimators=32, max_depth=9, min_samples_leaf=12,
                                     max_features=0.8, random_state=42, n_jobs=2)
    else:
        mode, blend = "residual", 0.65
        rng = np.random.default_rng(112)
        x = np.vstack([x, x + rng.normal(size=x.shape) * base.scale * 0.005,
                       x + rng.normal(size=x.shape) * base.scale * 0.015])
        y = np.tile(y, 3) - base.predict(x)
        forest = ExtraTreesRegressor(n_estimators=32, max_depth=14, min_samples_leaf=45,
                                     max_features=0.85, random_state=37, n_jobs=3)
    scale = base.meta["target_scale"]
    forest.fit(x, y / scale)
    save_forest(destination, forest, checkpoint, mode, blend, scale)
    return CorrectedModel(base, checkpoint, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("A", "B", "C"), required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package.resolve()))
    from baselines.model import Model
    from baselines.data import read_dataset
    from baselines.train import metrics

    bundle = args.out / "predictor"
    bundle.mkdir(exist_ok=True)
    checkpoint = bundle / "checkpoint.npz"
    shutil.copyfile(args.out / "final-checkpoint.npz", checkpoint)
    base = Model.load(checkpoint)
    def dataset(split):
        return read_dataset(args.package / "public" / (split + ".jsonl"),
                            base.meta["task"], base.meta["classes"], base.meta["paths"])
    destination = bundle / "correction.npz"
    candidate = fit_correction(args.profile, base, checkpoint, dataset("train"), destination)
    validation = dataset("validation")
    baseline, corrected = metrics(base, validation), metrics(candidate, validation)
    metric = "macro_f1" if args.profile == "A" else "rmse"
    delta = corrected[metric] - baseline[metric]
    accepted = delta > 0 if args.profile == "A" else delta < 0
    if not accepted:
        destination.unlink()
    source = Path(__file__).resolve().parent
    for name in ("predict_scheme.py", "gas_pair_gate.py", "rul_tree_correction.py"):
        shutil.copyfile(source / name, bundle / name)
    for name in ("__init__.py", "data.py", "model.py", "predict.py"):
        target = bundle / "baselines" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(args.package / "baselines" / name, target)
    spec = dict(profile=args.profile, metric=metric, baseline=baseline, candidate=corrected,
                accepted=accepted, selected=corrected if accepted else baseline)
    (bundle / "selection.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
