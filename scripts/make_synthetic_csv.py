"""把合成漂移流转成 CSV 批次，用来验证 `selftrain run` 这条真实数据路径。

用法：
    python scripts/make_synthetic_csv.py --out data/synthetic --scenario mixed --batches 12
    python -m selftrain run --data-dir data/synthetic --target y
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selftrain.data import SyntheticConfig, SyntheticStream  # noqa: E402
from selftrain.types import TaskType  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--scenario", default="mixed")
    p.add_argument("--task", default="binary")
    p.add_argument("--batches", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=300)
    p.add_argument("--features", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stream = SyntheticStream(
        SyntheticConfig(
            task=TaskType(args.task),
            n_batches=args.batches,
            batch_size=args.batch_size,
            n_features=args.features,
            scenario=args.scenario,
            seed=args.seed,
        )
    )
    for batch in stream:
        df = pd.DataFrame(batch.X, columns=batch.feature_names)
        df["y"] = batch.y
        path = out / f"batch_{batch.index:03d}.csv"
        df.to_csv(path, index=False)
        print(f"  {path}  shape={df.shape}")

    print(f"\n共 {args.batches} 批 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
