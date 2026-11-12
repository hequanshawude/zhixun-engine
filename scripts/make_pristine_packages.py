"""把本地已打补丁的包还原成"平台原版"，用于验证镜像内的自动补丁路径。"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, r"D:\智训引擎")
from selftrain.patcher import NEW_LOOP, NEW_REPORT, OLD_LOOP, OLD_REPORT

ROOT = Path(r"D:\智训引擎")
for tag in ("a", "b", "c"):
    src = ROOT / "data/raw/cpu" / tag
    dst = ROOT / "out" / f"pristine-{tag}"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    train_py = dst / "baselines" / "train.py"
    text = train_py.read_text(encoding="utf-8")
    if NEW_LOOP in text and NEW_REPORT in text:
        text = text.replace(NEW_LOOP, OLD_LOOP).replace(NEW_REPORT, OLD_REPORT)
        train_py.write_text(text, encoding="utf-8")
        print(tag, "已还原为原版")
    else:
        print(tag, "无法还原（锚点不匹配）")
