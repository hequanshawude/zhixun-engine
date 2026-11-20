"""Train-only A branch with resumable optimizer state and native prediction export.

Candidate layout: checkpoint.npz plus training_state/{base,branch,teacher,
migration}.npz and state.json. The prediction projection MUST NOT be resumed
with Model.fit. prepare() validates its paired state before further training.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
from baselines.model import Model

VERSION = "a-pair-stateful-projection-v1"


def sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(model, path):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    model.save(temp)
    temp.replace(path)


def _json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf8")


def _signature(train):
    h = hashlib.sha256()
    for key in ("x", "y"):
        h.update(np.ascontiguousarray(train[key]).tobytes())
    h.update(json.dumps({k: train[k] for k in ("ids", "paths")}, sort_keys=True).encode())
    return "sha256:" + h.hexdigest()


def _distance(x, p):
    return np.maximum(0., np.sum(x*x, 1)[:, None] + np.sum(p*p, 1)[None, :] - 2*x@p.T)


def _teacher(x, pos, neg):
    dp, dn = _distance(x, pos).min(1), _distance(x, neg).min(1)
    return np.clip(np.log((dn+.01)/(dp+.01)), -6, 6)


def _synthetic(x, seed, count):
    rng = np.random.default_rng(seed)
    p, q = x[rng.integers(len(x), size=count)], x[rng.integers(len(x), size=count)]
    alpha = rng.uniform(-.5, 1.5, (count, 1))
    z = alpha*p+(1-alpha)*q
    z += rng.normal(size=z.shape)*rng.choice([.05, .2, .5, 1.], (count, 1))
    return np.clip(z, -10, 10)


def _migration(base, branch, indices):
    model = copy.deepcopy(base)
    old, width = base.meta["hidden"], base.meta["hidden"]+branch.meta["hidden"]
    if width != 128 or old != 32:
        raise ValueError("Expected original A 32-unit parent")
    for bank in (model.params, model.first, model.second):
        bank["w1"] = np.pad(bank["w1"], ((0, 0), (0, 96)))
        bank["b1"] = np.pad(bank["b1"], (0, 96))
        bank["w2"] = np.pad(bank["w2"], ((0, 96), (0, 0)))
    model.params["w1"][np.ix_(indices, np.arange(old, width))] = branch.params["w1"]
    model.params["b1"][old:] = branch.params["b1"]
    model.meta.update(hidden=128, parameter_count=sum(v.size for v in model.params.values()))
    model.meta["a_stateful_projection"] = {"version": VERSION, "stage": "zero_output_migration"}
    model.validate()
    return model


def prepare(parent_path, train_set, state_dir):
    """Create candidate state from the actual parent artifact; no validation input."""
    parent_path, state_dir = Path(parent_path), Path(state_dir)
    parent = Model.load(parent_path)
    parent_digest = sha(parent_path)
    signature = _signature(train_set)
    source_state = parent_path.parent / "training_state"
    if state_dir.exists():
        raise FileExistsError(state_dir)
    if source_state.is_dir():
        info = json.loads((source_state / "state.json").read_text(encoding="utf8"))
        if info["version"] != VERSION or info["last_projection_sha256"] != parent_digest:
            raise ValueError("Parent projection and training state do not match")
        if info["train_signature"] != signature:
            raise ValueError("Training data changed")
        for name, digest in info["files"].items():
            if sha(source_state / name) != digest:
                raise ValueError("State digest mismatch: " + name)
        shutil.copytree(source_state, state_dir)
        info["parent_checkpoint_sha256"] = parent_digest
        _json(state_dir / "state.json", info)
        probe = state_dir / "resume-projection-check.npz"
        export(state_dir, probe)
        rebuilt = Model.load(probe)
        if rebuilt.meta != parent.meta or not np.array_equal(rebuilt.mean, parent.mean) or not np.array_equal(rebuilt.scale, parent.scale):
            raise ValueError("Saved training state does not reconstruct parent metadata")
        for bank in ('params', 'first', 'second'):
            for key in parent.params:
                if not np.array_equal(getattr(rebuilt, bank)[key], getattr(parent, bank)[key]):
                    raise ValueError("Saved training state does not reconstruct parent tensors")
        # ZIP OS metadata differs between Windows and Linux; the host still verifies
        # the actual parent bytes, while reconstruction compares every model value.
        info['last_projection_sha256'] = parent_digest
        _json(state_dir / 'state.json', info)
        probe.unlink()
        return state_dir
    if parent.meta["hidden"] != 32 or parent.meta.get("a_stateful_projection"):
        raise ValueError("Cannot initialize from a projection without its training state")
    state_dir.mkdir(parents=True)
    indices = np.arange(1, 128, 8)
    e, a = (parent.meta["classes"].index(s) for s in ("Ethanol", "Acetaldehyde"))
    mask = np.isin(train_set["y"], [e, a])
    x = parent.normalized(train_set["x"])[mask][:, indices]
    y = (train_set["y"][mask] == a)
    if not y.any() or y.all():
        raise ValueError("Need public training examples for both pair classes")
    sx, tx = _synthetic(x, 20261117, 12000), _synthetic(x, 20261118, 3000)
    sy, ty = _teacher(sx, x[y], x[~y]), _teacher(tx, x[y], x[~y])
    branch = Model("regression", [parent.meta["paths"][i] for i in indices], [], sx, sy,
                   hidden=96, seed=44, learning_rate=.003, batch_size=512)
    branch.mean[:] = 0.
    branch.scale[:] = 1.
    migration = _migration(parent, branch, indices)
    drift = float(np.max(np.abs(migration.predict(train_set["x"])-parent.predict(train_set["x"]))))
    if drift > 1e-12:
        raise AssertionError("Function-preserving migration failed")
    shutil.copy2(parent_path, state_dir / "base.npz")
    _save(branch, state_dir / "branch.npz")
    _save(migration, state_dir / "migration.npz")
    np.savez(state_dir / "teacher.npz", sx=sx, sy=sy, tx=tx, ty=ty)
    info = {"version": VERSION, "original_parent_sha256": parent_digest,
            "parent_checkpoint_sha256": parent_digest, "last_projection_sha256": parent_digest,
            "train_signature": signature, "train_raw_sha256": train_set.get("sha256"),
            "indices": indices.tolist(), "gain": 2., "blend": .5, "branch_seed": 44,
            "branch_epochs": 0, "branch_steps": 0, "base_steps": parent.meta["steps"],
            "migration_max_abs_difference": drift,
            "files": {name: sha(state_dir / name) for name in ("base.npz", "branch.npz", "teacher.npz", "migration.npz")}}
    _json(state_dir / "state.json", info)
    return state_dir


def train_round(state_dir, epochs=90, learning_rate=.003):
    """Perform the requested REAL branch epochs; preserve branch Adam and RNG epochs."""
    begin = time.perf_counter()
    state_dir = Path(state_dir)
    info = json.loads((state_dir / "state.json").read_text(encoding="utf8"))
    for name, digest in info["files"].items():
        if sha(state_dir / name) != digest:
            raise ValueError("State digest mismatch: " + name)
    branch = Model.load(state_dir / "branch.npz")
    before_steps, before_epochs = branch.meta["steps"], branch.meta["epochs"]
    if before_steps != info["branch_steps"] or before_epochs != info["branch_epochs"]:
        raise ValueError("Branch clock mismatch")
    with np.load(state_dir / "teacher.npz", allow_pickle=False) as teacher:
        sx, sy, tx, ty = (teacher[k] for k in ("sx", "sy", "tx", "ty"))
    branch.meta["learning_rate"] = float(learning_rate)
    losses = branch.fit(sx, sy, int(epochs))
    raw = branch.forward(tx)[1][:, 0]*branch.meta["target_scale"]+branch.meta["target_mean"]
    mse = float(np.mean((raw-ty)**2))
    _save(branch, state_dir / "branch.npz")
    info.update(branch_epochs=branch.meta["epochs"], branch_steps=branch.meta["steps"],
                learning_rate=learning_rate, synthetic_holdout_mse=mse)
    info["files"]["branch.npz"] = sha(state_dir / "branch.npz")
    _json(state_dir / "state.json", info)
    return {"method": VERSION, "optimizer": "Adam in separate regression branch coordinates",
            "parent_sha256": info["parent_checkpoint_sha256"],
            "initial_branch_steps": before_steps, "final_branch_steps": branch.meta["steps"],
            "initial_branch_epochs": before_epochs, "final_branch_epochs": branch.meta["epochs"],
            "actual_epochs": epochs, "synthetic_samples": len(sx), "learning_rate": learning_rate,
            "synthetic_holdout_mse": mse, "loss_first": losses[0], "loss_last": losses[-1],
            "wall_seconds": time.perf_counter()-begin, "base_optimizer_unchanged": True,
            "prediction_checkpoint_is_projection_only": True}


def export(state_dir, checkpoint_path):
    """Export native predictor, with explicit optimizer-projection limitation."""
    state_dir, checkpoint_path = Path(state_dir), Path(checkpoint_path)
    info = json.loads((state_dir / "state.json").read_text(encoding="utf8"))
    base, branch = Model.load(state_dir / "base.npz"), Model.load(state_dir / "branch.npz")
    if info["branch_epochs"] == 0:
        model = _migration(base, branch, info["indices"])
    else:
        model = _migration(base, branch, info["indices"])
        e, a = (base.meta["classes"].index(s) for s in ("Ethanol", "Acetaldehyde"))
        gain, blend = info["gain"], info["blend"]
        for arr in (model.params["w2"], model.params["b2"]):
            average = (arr[..., e]+arr[..., a])/2
            arr[..., e] = (1-blend)*arr[..., e]+blend*average
            arr[..., a] = (1-blend)*arr[..., a]+blend*average
        weight = branch.params["w2"][:, 0]*branch.meta["target_scale"]*gain*blend/2
        bias = (branch.params["b2"][0]*branch.meta["target_scale"]+branch.meta["target_mean"])*gain*blend/2
        model.params["w2"][32:, e], model.params["w2"][32:, a] = -weight, weight
        model.params["b2"][e] -= bias
        model.params["b2"][a] += bias
        # A projected predictor has no equivalent single Adam trajectory.
        # The true old and branch moments remain in training_state/*.npz.
        model.first = {k: np.zeros_like(v) for k, v in model.params.items()}
        model.second = {k: np.zeros_like(v) for k, v in model.params.items()}
    model.meta.update(optimizer_state_valid_for_resume=False, resume_requires_training_state=True)
    model.meta["a_stateful_projection"] = {
        "version": VERSION, "stage": "prediction_projection", "base_sha256": info["files"]["base.npz"],
        "branch_sha256": info["files"]["branch.npz"], "branch_epochs": info["branch_epochs"],
        "branch_steps": info["branch_steps"], "base_steps": info["base_steps"],
        "total_training_updates": info["base_steps"]+info["branch_steps"],
        "gain": info["gain"], "blend": info["blend"]}
    model.validate()
    model.save(checkpoint_path)
    info["last_projection_sha256"] = sha(checkpoint_path)
    _json(state_dir / "state.json", info)
    return model
