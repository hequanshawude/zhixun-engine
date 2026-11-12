"""对比检查点在训练集与验证集上的指标，判断过拟合程度。"""

import sys
from pathlib import Path

ROOT = Path(r"D:\智训引擎")
PKG = ROOT / "data" / "raw" / "cpu"


def measure(profile: str, ckpt: Path):
    tag = profile.lower()
    pkg = PKG / tag
    sys.path.insert(0, str(pkg))
    for name in [m for m in list(sys.modules) if m.startswith("baselines")]:
        del sys.modules[name]
    from baselines.data import read_dataset
    from baselines.model import Model
    from baselines.train import metrics

    model = Model.load(ckpt)
    out = {}
    for split in ("train", "validation"):
        data = read_dataset(pkg / f"public/{split}.jsonl", model.meta["task"],
                            classes=model.meta["classes"],
                            expected_paths=model.meta["paths"])
        out[split] = metrics(model, data)
    return out


if __name__ == "__main__":
    for item in sys.argv[1:]:
        profile, _, path = item.partition(":")
        m = measure(profile, Path(path))
        gap = ""
        if profile == "A":
            gap = f"  F1 差距 {m['train']['macro_f1'] - m['validation']['macro_f1']:+.4f}"
        else:
            gap = f"  RMSE 差距 {m['validation']['rmse'] - m['train']['rmse']:+.2f}"
        print(f"{profile}: train={m['train']}  val={m['validation']}{gap}")
