"""Read only explicitly supplied public training/validation data."""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

MAX_BYTES = 128 * 1024 * 1024
FORBIDDEN_INPUTS = {"truerul", "normalizedrul", "remainingusefullife", "rul", "totalsnapshots"}


def sha256(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def json_value(raw):
    def reject(_):
        raise ValueError("nonfinite JSON number")
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject)


def flatten(inputs):
    paths, values = [], []
    def visit(value, path, depth):
        if depth > 16:
            raise ValueError("input nesting exceeds limit")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str) or re.sub(r"[^a-z0-9]", "", key.lower()) in FORBIDDEN_INPUTS:
                    raise ValueError("target-derived input field is not allowed")
                visit(value[key], path + [("key", key)], depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [("index", index)], depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(float(value))
        else:
            raise ValueError("model inputs must contain finite numeric features only")
    visit(inputs, [], 0)
    if not values or len(values) > 4096:
        raise ValueError("expected 1..4096 numeric features")
    return paths, values


def read_dataset(path, task, classes=None, expected_paths=None):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_BYTES:
        raise ValueError("dataset must be a regular JSONL file no larger than 128 MiB")
    raw = path.read_bytes()
    ids, features, targets = [], [], []
    seen = set()
    paths = expected_paths
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json_value(line)
        if not isinstance(row, dict) or not {"id", "inputs", "answer"} <= set(row):
            raise ValueError(f"line {number}: id, inputs and answer are required")
        identity = row["id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError(f"line {number}: sample IDs must be nonempty and unique")
        current_paths, values = flatten(row["inputs"])
        if paths is None:
            paths = current_paths
        if current_paths != paths:
            raise ValueError(f"line {number}: feature schema changed")
        answer = row["answer"]
        if task == "classification":
            if not isinstance(answer, str) or not classes or answer not in classes:
                raise ValueError(f"line {number}: label is outside the configured class list")
            target = classes.index(answer)
        elif task == "regression":
            if isinstance(answer, bool) or not isinstance(answer, (float, int)) or not math.isfinite(answer) or answer < 0:
                raise ValueError(f"line {number}: RUL must be a finite nonnegative number")
            target = float(answer)
        else:
            raise ValueError("unsupported task")
        ids.append(identity); seen.add(identity); features.append(values); targets.append(target)
    if not ids:
        raise ValueError("dataset is empty")
    return {"ids": ids, "x": np.asarray(features, dtype=np.float64),
            "y": np.asarray(targets, dtype=np.int64 if task == "classification" else np.float64),
            "paths": paths, "sha256": sha256(raw)}
