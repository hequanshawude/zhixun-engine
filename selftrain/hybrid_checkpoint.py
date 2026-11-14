"""Embed a train-fitted regression forest into a resumable CPU checkpoint.

The base MLP tensors and Adam state stay unchanged. A matching model.py patch
teaches the official prediction entry point to read the extra forest arrays.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


FOREST_KEYS = ("feature", "threshold", "left", "right", "value", "roots",
               "mode", "blend", "target_scale", "digest")

MODEL_PATCH = '''\n# Optional train-fitted regression forest inside the final industrial checkpoint.\n_BASE_LOAD = Model.load.__func__\n_BASE_PREDICT = Model.predict\n_FOREST_KEYS = ("feature", "threshold", "left", "right", "value", "roots", "mode", "blend", "target_scale", "digest")\n\ndef _hybrid_load(cls, path):\n    with zipfile.ZipFile(path) as archive:\n        names = {entry.filename for entry in archive.infolist()}\n        if "forest_feature.npy" not in names:\n            return _BASE_LOAD(cls, path)\n        core = {*PARAMETERS, *("adam_first_" + k for k in PARAMETERS),\n                *("adam_second_" + k for k in PARAMETERS), "mean", "scale", "metadata"}\n        expected = {name + ".npy" for name in core} | {"forest_" + name + ".npy" for name in _FOREST_KEYS}\n        if names != expected or sum(entry.file_size for entry in archive.infolist()) > MAX_CHECKPOINT_BYTES:\n            raise ValueError("invalid hybrid checkpoint archive")\n    with np.load(path, allow_pickle=False) as arrays:\n        model = cls.__new__(cls)\n        model.meta = json.loads(str(arrays["metadata"]))\n        model.params = {key: arrays[key].copy() for key in PARAMETERS}\n        model.first = {key: arrays["adam_first_" + key].copy() for key in PARAMETERS}\n        model.second = {key: arrays["adam_second_" + key].copy() for key in PARAMETERS}\n        model.mean, model.scale = arrays["mean"].copy(), arrays["scale"].copy()\n        model.forest = {key: arrays["forest_" + key].copy() for key in _FOREST_KEYS}\n    model.validate()\n    tree = model.forest\n    count = len(tree["feature"])\n    if model.meta["task"] != "regression" or count == 0 or len(tree["roots"]) == 0:\n        raise ValueError("hybrid forest requires regression and nonempty trees")\n    if any(len(tree[key]) != count for key in ("threshold", "left", "right", "value")):\n        raise ValueError("hybrid forest shapes differ")\n    if np.any(tree["roots"] < 0) or np.any(tree["roots"] >= count):\n        raise ValueError("hybrid tree root out of bounds")\n    if np.any(tree["feature"] >= len(model.meta["paths"])) or np.any(tree["feature"] < -2):\n        raise ValueError("hybrid tree feature out of bounds")\n    return model\n\ndef _hybrid_predict(self, x):\n    base = _BASE_PREDICT(self, x)\n    if not hasattr(self, "forest"):\n        return base\n    tree = self.forest\n    features = x.astype(np.float32)\n    prediction = np.zeros(len(x))\n    for root in tree["roots"]:\n        nodes = np.full(len(x), root)\n        while True:\n            rows = np.flatnonzero(tree["feature"][nodes] >= 0)\n            if not len(rows):\n                break\n            parents = nodes[rows]\n            left = features[rows, tree["feature"][parents]] <= tree["threshold"][parents]\n            nodes[rows] = np.where(left, tree["left"][parents], tree["right"][parents])\n        prediction += tree["value"][nodes]\n    prediction *= float(tree["target_scale"]) / len(tree["roots"])\n    delta = prediction if str(tree["mode"]) == "residual" else prediction - base\n    return np.maximum(0, base + float(tree["blend"]) * delta)\n\nModel.load = classmethod(_hybrid_load)\nModel.predict = _hybrid_predict\n'''


MODEL_PATCH += '''
_BASE_SAVE = Model.save

def _hybrid_save(self, path):
    if not hasattr(self, "forest"):
        return _BASE_SAVE(self, path)
    self.validate()
    arrays = {**self.params, **{"adam_first_" + k: v for k, v in self.first.items()},
              **{"adam_second_" + k: v for k, v in self.second.items()},
              "mean": self.mean, "scale": self.scale,
              "metadata": np.array(json.dumps(self.meta, sort_keys=True, separators=(",", ":"), allow_nan=False))}
    arrays.update({"forest_" + k: v for k, v in self.forest.items()})
    with Path(path).open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())

Model.save = _hybrid_save
'''


def patch_model_source(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if "_hybrid_load" in text:
        return
    if "class Model:" not in text or "def checkpoint_digest(" not in text:
        raise ValueError("Unexpected official model source")
    source.write_text(text + MODEL_PATCH, encoding="utf-8")


def embed_correction(base: Path, correction: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    base_sha = hashlib.sha256(base.read_bytes()).hexdigest()
    with np.load(base, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    with np.load(correction, allow_pickle=False) as source:
        if set(source.files) != set(FOREST_KEYS) or str(source["digest"]) != base_sha:
            raise ValueError("Correction is not bound to the base checkpoint")
        arrays.update({"forest_" + key: source[key].copy() for key in FOREST_KEYS})
    metadata = json.loads(str(arrays["metadata"]))
    if metadata["task"] != "regression":
        raise ValueError("Hybrid correction requires regression")
    metadata["hybrid"] = {"method": "public_train_fitted_forest",
                          "parent_checkpoint_sha256": base_sha,
                          "correction_sha256": hashlib.sha256(correction.read_bytes()).hexdigest()}
    arrays["metadata"] = np.array(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False))
    with destination.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return metadata["hybrid"]


MODEL_PATCH_DUAL = '''
_SINGLE_LOAD = Model.load.__func__
_SINGLE_PREDICT = Model.predict

def _dual_load(cls, path):
    with zipfile.ZipFile(path) as archive:
        if "reference_w1.npy" not in {entry.filename for entry in archive.infolist()}:
            return _SINGLE_LOAD(cls, path)
    with np.load(path, allow_pickle=False) as arrays:
        model = cls.__new__(cls)
        model.meta = json.loads(str(arrays["metadata"]))
        model.params = {key: arrays[key].copy() for key in PARAMETERS}
        model.first = {key: arrays["adam_first_" + key].copy() for key in PARAMETERS}
        model.second = {key: arrays["adam_second_" + key].copy() for key in PARAMETERS}
        model.mean, model.scale = arrays["mean"].copy(), arrays["scale"].copy()
        model.forest = {key: arrays["forest_" + key].copy() for key in _FOREST_KEYS}
        reference = cls.__new__(cls)
        reference.meta = json.loads(str(arrays["reference_metadata"]))
        reference.params = {key: arrays["reference_" + key].copy() for key in PARAMETERS}
        reference.first = {key: arrays["reference_adam_first_" + key].copy() for key in PARAMETERS}
        reference.second = {key: arrays["reference_adam_second_" + key].copy() for key in PARAMETERS}
        reference.mean = arrays["reference_mean"].copy()
        reference.scale = arrays["reference_scale"].copy()
        reference.forest = {key: arrays["reference_forest_" + key].copy() for key in _FOREST_KEYS}
    model.validate()
    reference.validate()
    if model.meta["task"] != "regression" or model.meta["paths"] != reference.meta["paths"]:
        raise ValueError("Incompatible dual regression models")
    model.reference_model = reference
    model.guided_share = float(model.meta["dual"]["guided_share"])
    if not 0 <= model.guided_share <= 1:
        raise ValueError("Invalid dual model share")
    return model

def _dual_predict(self, x):
    guided = _SINGLE_PREDICT(self, x)
    if not hasattr(self, "reference_model"):
        return guided
    reference = _SINGLE_PREDICT(self.reference_model, x)
    return (1 - self.guided_share) * reference + self.guided_share * guided

Model.load = classmethod(_dual_load)
Model.predict = _dual_predict
'''

MODEL_PATCH_DUAL += '''
_SINGLE_SAVE = Model.save

def _dual_save(self, path):
    if not hasattr(self, "reference_model"):
        return _SINGLE_SAVE(self, path)
    self.validate()
    reference = self.reference_model
    reference.validate()
    arrays = {**self.params, **{"adam_first_" + k: v for k, v in self.first.items()},
              **{"adam_second_" + k: v for k, v in self.second.items()},
              "mean": self.mean, "scale": self.scale,
              "metadata": np.array(json.dumps(self.meta, sort_keys=True, separators=(",", ":"), allow_nan=False))}
    arrays.update({"forest_" + k: v for k, v in self.forest.items()})
    arrays.update({"reference_" + k: v for k, v in reference.params.items()})
    arrays.update({"reference_adam_first_" + k: v for k, v in reference.first.items()})
    arrays.update({"reference_adam_second_" + k: v for k, v in reference.second.items()})
    arrays.update({"reference_mean": reference.mean, "reference_scale": reference.scale,
                   "reference_metadata": np.array(json.dumps(reference.meta, sort_keys=True, separators=(",", ":"), allow_nan=False))})
    arrays.update({"reference_forest_" + k: v for k, v in reference.forest.items()})
    with Path(path).open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())

Model.save = _dual_save
'''


def patch_dual_model_source(source: Path) -> None:
    patch_model_source(source)
    text = source.read_text(encoding="utf-8")
    if "_dual_load" not in text:
        source.write_text(text + MODEL_PATCH_DUAL, encoding="utf-8")


def embed_dual_c(reference_base: Path, reference_correction: Path,
                 guided_base: Path, guided_correction: Path,
                 destination: Path, guided_share: float = 0.6,
                 guided_strength: float = 0.9) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    if not 0 <= guided_share <= 1 or not 0 <= guided_strength <= 1.5:
        raise ValueError("Invalid blend settings")
    sources = ((reference_base, reference_correction, "reference_"),
               (guided_base, guided_correction, ""))
    arrays = {}
    hashes = {}
    for base, correction, prefix in sources:
        digest = hashlib.sha256(base.read_bytes()).hexdigest()
        with np.load(base, allow_pickle=False) as source:
            arrays.update({prefix + key: source[key].copy() for key in source.files})
        with np.load(correction, allow_pickle=False) as source:
            if set(source.files) != set(FOREST_KEYS) or str(source["digest"]) != digest:
                raise ValueError("Correction is not bound to its base checkpoint")
            arrays.update({prefix + "forest_" + key: source[key].copy() for key in FOREST_KEYS})
        hashes[prefix or "guided_"] = {"parent_checkpoint_sha256": digest,
                                      "correction_sha256": hashlib.sha256(correction.read_bytes()).hexdigest()}
    metadata = json.loads(str(arrays["metadata"]))
    reference_meta = json.loads(str(arrays["reference_metadata"]))
    if metadata["task"] != "regression" or metadata["profile"] != "C" or metadata["paths"] != reference_meta["paths"]:
        raise ValueError("Dual C model schema mismatch")
    metadata["dual"] = {"method": "two_branches_from_official_checkpoint",
                        "guided_share": guided_share, "guided_strength": guided_strength,
                        "sources": hashes}
    arrays["metadata"] = np.array(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False))
    arrays["forest_blend"] = np.asarray(guided_strength)
    with destination.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return metadata["dual"]


GAS_KEYS = ("pos", "neg", "cut", "masks", "ks", "cuts", "base_digest")

MODEL_PATCH_GAS = '''
_GAS_BASE_LOAD = Model.load.__func__
_GAS_BASE_PREDICT = Model.predict
_GAS_BASE_SAVE = Model.save
_GAS_KEYS = ("pos", "neg", "cut", "masks", "ks", "cuts", "base_digest")

def _gas_load(cls, path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("invalid gas checkpoint file")
    with zipfile.ZipFile(path) as archive:
        names = [entry.filename for entry in archive.infolist()]
        if "gas_pos.npy" not in names:
            return _GAS_BASE_LOAD(cls, path)
        core = {*PARAMETERS, *("adam_first_" + k for k in PARAMETERS),
                *("adam_second_" + k for k in PARAMETERS), "mean", "scale", "metadata"}
        expected = {name + ".npy" for name in core} | {"gas_" + name + ".npy" for name in _GAS_KEYS}
        if len(names) != len(expected) or set(names) != expected or sum(e.file_size for e in archive.infolist()) > MAX_CHECKPOINT_BYTES:
            raise ValueError("invalid gas checkpoint archive")
    with np.load(path, allow_pickle=False) as arrays:
        model = cls.__new__(cls)
        model.meta = json.loads(str(arrays["metadata"]))
        model.params = {key: arrays[key].copy() for key in PARAMETERS}
        model.first = {key: arrays["adam_first_" + key].copy() for key in PARAMETERS}
        model.second = {key: arrays["adam_second_" + key].copy() for key in PARAMETERS}
        model.mean, model.scale = arrays["mean"].copy(), arrays["scale"].copy()
        model.gas_gate = {key: arrays["gas_" + key].copy() for key in _GAS_KEYS}
    model.validate()
    gate = model.gas_gate
    if model.meta["task"] != "classification" or gate["pos"].ndim != 2 or gate["neg"].shape[1:] != gate["pos"].shape[1:]:
        raise ValueError("invalid gas gate shape")
    if gate["masks"].shape[1] != gate["pos"].shape[1] or len(gate["ks"]) != len(gate["cuts"]) or len(gate["ks"]) != len(gate["masks"]):
        raise ValueError("invalid gas gate confirmations")
    return model

def _gas_nearest_mean(distances, k):
    return np.partition(distances, int(k) - 1, axis=1)[:, :int(k)].mean(axis=1)

def _gas_distances(x, prototypes):
    return np.linalg.norm(x[:, None, :] - prototypes[None, :, :], axis=2)

def _gas_predict(self, x):
    scores = _GAS_BASE_PREDICT(self, x)
    if not hasattr(self, "gas_gate"):
        return scores
    gate = self.gas_gate
    ethanol = self.meta["classes"].index("Ethanol")
    acetaldehyde = self.meta["classes"].index("Acetaldehyde")
    predicted = scores.argmax(1)
    for start in range(0, len(x), 512):
        chunk = x[start:start + 512]
        choose = predicted[start:start + len(chunk)] == ethanol
        if not choose.any():
            continue
        sub = self.normalized(chunk[choose])[:, 1::8]
        pos = _gas_distances(sub, gate["pos"]).min(1)
        neg = _gas_distances(sub, gate["neg"]).min(1)
        passed = pos < float(gate["cut"]) * neg
        for mask, k, cut in zip(gate["masks"], gate["ks"], gate["cuts"]):
            pending = np.flatnonzero(passed)
            if not len(pending):
                break
            smaller = sub[pending][:, mask]
            dpos = _gas_nearest_mean(_gas_distances(smaller, gate["pos"][:, mask]), k)
            dneg = _gas_nearest_mean(_gas_distances(smaller, gate["neg"][:, mask]), k)
            passed[pending] = dpos < float(cut) * dneg
        hits = start + np.flatnonzero(choose)[passed]
        scores[hits, acetaldehyde], scores[hits, ethanol] = scores[hits, ethanol].copy(), scores[hits, acetaldehyde].copy()
    return scores

def _gas_save(self, path):
    if not hasattr(self, "gas_gate"):
        return _GAS_BASE_SAVE(self, path)
    self.validate()
    arrays = {**self.params, **{"adam_first_" + k: v for k, v in self.first.items()},
              **{"adam_second_" + k: v for k, v in self.second.items()},
              "mean": self.mean, "scale": self.scale,
              "metadata": np.array(json.dumps(self.meta, sort_keys=True, separators=(",", ":"), allow_nan=False))}
    arrays.update({"gas_" + k: v for k, v in self.gas_gate.items()})
    with Path(path).open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())

Model.load = classmethod(_gas_load)
Model.predict = _gas_predict
Model.save = _gas_save
'''


def patch_gas_model_source(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if "_gas_load" in text:
        return
    if "class Model:" not in text or "def checkpoint_digest(" not in text:
        raise ValueError("Unexpected official gas model source")
    source.write_text(text + MODEL_PATCH_GAS, encoding="utf-8")


def embed_gas_gate(base: Path, correction: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    base_sha = hashlib.sha256(base.read_bytes()).hexdigest()
    with np.load(base, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    with np.load(correction, allow_pickle=False) as source:
        if set(source.files) != set(GAS_KEYS) or str(source["base_digest"]) != base_sha:
            raise ValueError("Gas gate is not bound to base checkpoint")
        arrays.update({"gas_" + key: source[key].copy() for key in GAS_KEYS})
    metadata = json.loads(str(arrays["metadata"]))
    if metadata["task"] != "classification" or metadata["profile"] != "A":
        raise ValueError("Gas gate requires profile A")
    metadata["gas_gate"] = {"method": "public_train_fitted_pair_confirmation",
                            "parent_checkpoint_sha256": base_sha,
                            "correction_sha256": hashlib.sha256(correction.read_bytes()).hexdigest()}
    arrays["metadata"] = np.array(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False))
    with destination.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return metadata["gas_gate"]
