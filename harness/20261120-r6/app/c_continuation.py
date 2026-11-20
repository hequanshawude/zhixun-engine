"""C condition-residual continuation, native inference plus explicit latent Adam state.

Only public train determines the condition transform, anchors and noise samples.
Original hidden parameters and Adam moments are frozen; original output Adam
continues its own clock. New coordinates have a separate Adam clock starting 0.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def _save(path, arrays):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in sorted(arrays.items()):
            buf = io.BytesIO(); np.lib.format.write_array(buf, value, allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            archive.writestr(info, buf.getvalue())


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _projection(parent, x):
    si = np.array([i for i, p in enumerate(parent.meta["paths"]) if "settings" in p])
    se = np.array([i for i, p in enumerate(parent.meta["paths"]) if "sensors" in p])
    if len(si) != 3 or len(se) != 21:
        raise ValueError("C requires 3 settings and 21 sensors")
    settings = x[:, si]
    centers = [settings[np.argmin(np.sum(settings * settings, axis=1))]]
    for _ in range(5):
        distance = np.min(np.sum((settings[:, None] - np.array(centers)) ** 2, axis=2), axis=1)
        centers.append(settings[np.argmax(distance)])
    centers = np.array(centers)
    for _ in range(30):
        groups = np.argmin(np.sum((settings[:, None] - centers) ** 2, axis=2), axis=1)
        fresh = np.array([settings[groups == g].mean(0) for g in range(6)])
        if np.max(np.abs(fresh - centers)) < 1e-10:
            break
        centers = fresh
    all_centers = np.array([x[groups == g].mean(0) for g in range(6)])
    cond = np.r_[si, se[[0, 4, 17, 18]]]
    co = np.linalg.lstsq(np.column_stack([all_centers[:, cond], np.ones(6)]),
                         all_centers[:, se], rcond=1e-8)[0]
    transform = np.eye(x.shape[1]); transform[np.ix_(cond, se)] -= co[:-1]
    offset = np.zeros(x.shape[1]); offset[se] = -co[-1]
    scale = np.maximum((x @ transform + offset).std(0), .001)
    return transform / scale, offset / scale


def _fold(model, state):
    model.params["w1"][:, 32:] = state["transform"] @ state["w"]
    model.params["b1"][32:] = state["offset"] @ state["w"] + state["b"]


def prepare(parent, train_set, state_dir):
    """Return model and state. A 128-unit parent requires matching saved state."""
    model = copy.deepcopy(parent)
    state_path = Path(state_dir) / "c_latent_adam.npz"
    if parent.meta["hidden"] == 128:
        info = parent.meta["c_continuation"]
        if info["train_sha256"] != train_set["sha256"] or _digest(state_path) != info["state_sha256"]:
            raise ValueError("C parent latent optimizer state or training projection differs")
        with np.load(state_path, allow_pickle=False) as stored:
            state = {key: stored[key].copy() for key in stored.files}
        before = model.params["w1"].copy(), model.params["b1"].copy()
        _fold(model, state)
        if not np.array_equal(before[0], model.params["w1"]) or not np.array_equal(before[1], model.params["b1"]):
            raise ValueError("C latent state does not project to supplied checkpoint")
        return model, state
    if parent.meta["hidden"] != 32:
        raise ValueError("C migration requires official width 32")
    x = parent.normalized(train_set["x"])
    transform, offset = _projection(parent, x)
    rng = np.random.default_rng(42)
    w = rng.normal(size=(x.shape[1], 96)) / np.sqrt(x.shape[1]) * .25
    anchors = x[rng.integers(len(x), size=96)] @ transform + offset
    b = -np.sum(anchors * w.T, axis=1)
    state = {"transform": transform, "offset": offset, "w": w, "b": b,
             "mw": np.zeros_like(w), "vw": np.zeros_like(w),
             "mb": np.zeros_like(b), "vb": np.zeros_like(b),
             "sample_indices": np.random.default_rng(20261118).choice(len(x), min(30000, len(x)), replace=False)}
    model.params["w1"] = np.column_stack([parent.params["w1"], transform @ w])
    model.params["b1"] = np.r_[parent.params["b1"], offset @ w + b]
    model.params["w2"] = np.vstack([parent.params["w2"], np.zeros((96, 1))])
    for bank in ("first", "second"):
        source = getattr(parent, bank); dest = getattr(model, bank)
        dest["w1"] = np.column_stack([source["w1"], np.zeros((x.shape[1], 96))])
        dest["b1"] = np.r_[source["b1"], np.zeros(96)]
        dest["w2"] = np.vstack([source["w2"], np.zeros((96, 1))])
    model.meta.update(hidden=128, parameter_count=sum(v.size for v in model.params.values()))
    model.meta["c_continuation"] = {"method": "train_only_condition_finite_noise_adam",
        "branch_steps": 0, "output_steps": parent.meta["steps"], "training_epochs": 0,
        "seed": 20261118, "train_sha256": train_set["sha256"],
        "resume_requires_latent_state": True, "native_model_fit_resume_supported": False,
        "frozen": ["original_w1", "original_b1", "original_hidden_adam_moments"]}
    model.validate()
    return model, state


def _adam(value, first, second, gradient, step, lr):
    first *= .9; first += .1 * gradient
    second *= .999; second += .001 * gradient ** 2
    value -= lr * (first / (1 - .9 ** step)) / (np.sqrt(second / (1 - .999 ** step)) + 1e-8)


def train_round(model, train_set, state_dir, epochs=10, learning_rate=.003, noise_weight=.1, state=None):
    """Continue the actual parent state; save complete latent state for next round."""
    if state is None:
        model, state = prepare(model, train_set, state_dir)
    if not (1 <= epochs <= 100 and 0 < learning_rate <= .03 and 0 < noise_weight < 1):
        raise ValueError("Training configuration outside reviewed bounds")
    info = model.meta["c_continuation"]
    indices = state["sample_indices"]
    x = model.normalized(train_set["x"])[indices]
    target = (train_set["y"][indices] - model.meta["target_mean"]) / model.meta["target_scale"]
    losses = []
    batch_size = 512
    for _ in range(epochs):
        rng = np.random.default_rng(np.random.SeedSequence([info["seed"], info["training_epochs"]]))
        order = rng.permutation(len(x)); total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]; clean = x[batch]; n = len(batch)
            noisy = np.clip(clean + rng.normal(size=clean.shape) * rng.choice([.001, .003, .01, .03], (n, 1)), -10, 10)
            xx = np.vstack([clean, noisy]); yy = np.r_[target[batch], target[batch]]
            weights = np.r_[np.full(n, 1 - noise_weight), np.full(n, noise_weight)] / n
            z = xx @ state["transform"] + state["offset"]
            old_h = np.tanh(xx @ model.params["w1"][:, :32] + model.params["b1"][:32])
            new_h = np.tanh(z @ state["w"] + state["b"])
            pred = old_h @ model.params["w2"][:32, 0] + new_h @ model.params["w2"][32:, 0] + model.params["b2"][0]
            error = pred - yy; delta = 2 * weights * error
            total += n * float(weights @ (error * error))
            dh = delta[:, None] * model.params["w2"][32:, 0][None, :] * (1 - new_h ** 2)
            gw = z.T @ dh + 2e-5 * state["w"]; gb = dh.sum(0)
            go = np.r_[old_h.T @ delta, new_h.T @ delta][:, None] + 2e-5 * model.params["w2"]
            gbo = np.array([delta.sum()])
            norm = np.sqrt(np.sum(gw * gw) + np.sum(gb * gb) + np.sum(go * go) + np.sum(gbo * gbo))
            factor = min(1., 5 / max(norm, 1e-12)); gw *= factor; gb *= factor; go *= factor; gbo *= factor
            info["branch_steps"] += 1; info["output_steps"] += 1; model.meta["steps"] += 1
            _adam(state["w"], state["mw"], state["vw"], gw, info["branch_steps"], learning_rate)
            _adam(state["b"], state["mb"], state["vb"], gb, info["branch_steps"], learning_rate)
            _adam(model.params["w2"][:32], model.first["w2"][:32], model.second["w2"][:32], go[:32], info["output_steps"], learning_rate)
            _adam(model.params["w2"][32:], model.first["w2"][32:], model.second["w2"][32:], go[32:], info["branch_steps"], learning_rate)
            _adam(model.params["b2"], model.first["b2"], model.second["b2"], gbo, info["output_steps"], learning_rate)
        info["training_epochs"] += 1; model.meta["epochs"] += 1
        losses.append(total / len(x))
    _fold(model, state)
    model.meta["learning_rate"] = learning_rate
    info.update(noise_weight=noise_weight, last_epochs=epochs, batch_size=batch_size)
    folder = Path(state_dir); folder.mkdir(parents=True, exist_ok=True)
    _save(folder / "c_latent_adam.npz", state)
    info["state_sha256"] = _digest(folder / "c_latent_adam.npz")
    model.validate()
    return losses
