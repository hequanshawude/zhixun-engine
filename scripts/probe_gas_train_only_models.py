"""Compare A train-only corrections against the current checkpoint/gate."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/raw/cpu/a"))
from baselines.data import read_dataset
from baselines.model import Model

checkpoint = ROOT / "out/triad-full-27b-20261114/a/predictor/checkpoint.npz"
model = Model.load(checkpoint)
train = read_dataset(ROOT / "data/raw/cpu/a/public/train.jsonl", "classification", model.meta["classes"], model.meta["paths"])
val = read_dataset(ROOT / "data/raw/cpu/a/public/validation.jsonl", "classification", model.meta["classes"], model.meta["paths"])
x_train = model.normalized(train["x"])
x_val = model.normalized(val["x"])
score = lambda p: float(np.mean([2 * np.sum((p == k) & (val["y"] == k)) /
                                   (np.sum(p == k) + np.sum(val["y"] == k))
                                   for k in range(len(model.meta["classes"]))]))
print("sizes", len(train["y"]), len(val["y"]), "classes", model.meta["classes"], flush=True)
print("base", score(model.predict(val["x"]).argmax(1)), flush=True)
for classifier in (
    ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=2),
    ExtraTreesClassifier(n_estimators=100, max_depth=20, min_samples_leaf=1, max_features=0.8, random_state=42, n_jobs=2),
    RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=2),
    KNeighborsClassifier(n_neighbors=3, weights="distance"),
    KNeighborsClassifier(n_neighbors=7, weights="distance"),
):
    classifier.fit(x_train, train["y"])
    probability = classifier.predict_proba(x_val)
    estimate = probability.argmax(1)
    print(classifier, score(estimate), "accuracy", float(np.mean(estimate == val["y"])), flush=True)
