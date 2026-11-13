"""Create a local, input-only C predictor bundle from verified checkpoints."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out/27b-blend-c-20261113-share60-strength90")
    parser.add_argument("--guided-share", type=float, default=0.6)
    parser.add_argument("--guided-correction-strength", type=float, default=0.9)
    args = parser.parse_args()
    if not 0 <= args.guided_share <= 1 or not 0 <= args.guided_correction_strength <= 1.5:
        raise ValueError("Invalid blend weights")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing candidate: {out}")
    out.mkdir(parents=True)

    sources = {
        "reference": ROOT / "out/bc-audit-20261113/docker-c/predictor",
        "guided": ROOT / "out/local-27b-strict-20261113/c-utf8/predictor",
    }
    hashes = {}
    for name, source in sources.items():
        target = out / name
        target.mkdir()
        for filename in ("checkpoint.npz", "correction.npz"):
            src = source / filename
            if not src.is_file():
                raise FileNotFoundError(src)
            dst = target / filename
            shutil.copyfile(src, dst)
            hashes[f"{name}/{filename}"] = digest(dst)

    for filename in ("__init__.py", "data.py", "model.py"):
        target = out / "baselines" / filename
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "data/raw/cpu/c/baselines" / filename, target)
    for filename in ("predict_blend.py", "rul_tree_correction.py"):
        shutil.copyfile(ROOT / "scripts" / filename, out / filename)
    spec = {
        "profile": "C", "task": "regression", "guided_model": "Qwen/Qwen3.8-27B",
        "guided_share": args.guided_share,
        "guided_correction_strength": args.guided_correction_strength,
        "sha256": hashes,
    }
    (out / "manifest.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
