"""给模型包的 model.py 加 L2 权重衰减，并让衰减系数随检查点元信息传递。

为什么这么做：
    A 的 train/val F1 差距 0.255，B 的 RMSE 差距 423——两个都在背训练集，而训练代码里
    没有任何正则项。加 weight decay 是最直接的干预。

为什么放在 meta 里：
    训练器只从 profile 读学习率/批大小，平台不会给我们传新参数。把 `weight_decay`
    写进检查点 meta，续训时自动继承，不需要改 CLI 协议，也不用平台配合。
"""

from pathlib import Path

OLD = """                for name, gradient in gradients.items():
                    gradient *= min(1.0, 5.0 / max(norm, 1e-12))
                    self.first[name] = .9 * self.first[name] + .1 * gradient"""

NEW = """                for name, gradient in gradients.items():
                    gradient *= min(1.0, 5.0 / max(norm, 1e-12))
                    if weight_decay:
                        gradient = gradient + weight_decay * self.params[name]
                    self.first[name] = .9 * self.first[name] + .1 * gradient"""

ANCHOR = """        epoch_losses = []
        for _ in range(epochs):"""

ANCHOR_NEW = """        epoch_losses = []
        weight_decay = float(self.meta.get("weight_decay", 0.0) or 0.0)
        for _ in range(epochs):"""

for tag in ("a", "b", "c"):
    path = Path("data/raw/cpu") / tag / "baselines" / "model.py"
    text = path.read_text(encoding="utf-8")
    if "weight_decay" in text:
        print(tag, "已打过补丁")
        continue
    if OLD not in text or ANCHOR not in text:
        print(tag, "锚点不匹配，跳过")
        continue
    text = text.replace(OLD, NEW).replace(ANCHOR, ANCHOR_NEW)
    path.write_text(text, encoding="utf-8")
    print(tag, "补丁完成")
