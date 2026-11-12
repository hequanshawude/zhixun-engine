"""给三个包的 baselines/train.py 打补丁：按公开验证集保留最优 epoch 的检查点。

原训练器只在循环结束后保存最后一个 epoch，而实测末轮常常不是最优（B 的 epoch19
RMSE 647 好于 epoch20 的 667）。补丁不改变网络、优化器或数据边界，只是把"选哪个
epoch 存档"从"最后一个"改成"验证集最好的那个"，并把选择结果写进 training.json。
"""

from pathlib import Path

OLD_LOOP = '''    for _ in range(count):
        loss = model.fit(train["x"], train["y"], 1)[0]
        records.append({"epoch": model.meta["epochs"], "step": model.meta["steps"], "loss": loss,
                        "validation": metrics(model, validation)})
'''

NEW_LOOP = '''    task_kind = config["task"]

    def _score(record):
        # 分类看 macro-F1（越高越好），回归看 RMSE（越低越好）
        metric = record["validation"]
        return metric["macro_f1"] if task_kind == "classification" else -metric["rmse"]

    best_record, best_score = None, None
    with tempfile.TemporaryDirectory(prefix=".best-") as best_dir:
        best_path = Path(best_dir) / "best.npz"
        for _ in range(count):
            loss = model.fit(train["x"], train["y"], 1)[0]
            record = {"epoch": model.meta["epochs"], "step": model.meta["steps"], "loss": loss,
                      "validation": metrics(model, validation)}
            records.append(record)
            score = _score(record)
            if best_score is None or score > best_score:
                best_score, best_record = score, record
                if best_path.exists():
                    best_path.unlink()
                model.save(best_path)
        model = Model.load(best_path)
'''

OLD_REPORT = '''"validation_before": validation_before, "validation_after": records[-1]["validation"],'''
NEW_REPORT = '''"validation_before": validation_before, "validation_after": best_record["validation"],
                  "selected_epoch": best_record["epoch"], "selected_step": best_record["step"],'''

for tag in ("a", "b", "c"):
    path = Path("data/raw/cpu") / tag / "baselines" / "train.py"
    text = path.read_text(encoding="utf-8")
    if "selected_epoch" in text:
        print(tag, "已打过补丁，跳过")
        continue
    if OLD_LOOP not in text or OLD_REPORT not in text:
        print(tag, "锚点不匹配，跳过")
        continue
    text = text.replace(OLD_LOOP, NEW_LOOP).replace(OLD_REPORT, NEW_REPORT)
    path.write_text(text, encoding="utf-8")
    print(tag, "补丁完成")
