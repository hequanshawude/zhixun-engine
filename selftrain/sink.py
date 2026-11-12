"""预测落盘：评测要求"先输出当前批结果，才能拿下一批"，所以必须是流式写出。

隐藏流评测时通常把每批预测写到固定目录，由平台读取。这里做成可配置。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from .checkpoint import atomic_write


class PredictionSink:
    """把每批预测写成 `batch_XXX.csv`，并维护一份索引。"""

    def __init__(self, out_dir: str, task: str = "binary", resume: bool = False):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.resume = resume
        self._index = []
        index_path = self.dir / 'index.json'
        if resume and index_path.exists():
            self._index = json.loads(index_path.read_text(encoding='utf-8'))
            if len({row['batch'] for row in self._index}) != len(self._index):
                raise ValueError('预测索引含重复批次')
            for row in self._index:
                expected_name = f"pred_{row['batch']:04d}.csv"
                if row['file'] != expected_name:
                    raise ValueError('预测索引文件名不合法')
                content = (self.dir / expected_name).read_bytes()
                if hashlib.sha256(content).hexdigest() != row.get('sha256'):
                    raise ValueError('预测文件与索引不一致，拒绝恢复')

    def snapshot(self):
        return [dict(row) for row in self._index]

    def validate_completed(self, batches, expected=None):
        available = {row['batch'] for row in self._index}
        if not set(batches).issubset(available):
            raise ValueError('恢复目录缺少已完成批次的预测')
        if expected is None:
            raise ValueError('检查点未绑定预测索引，无法验证恢复输出')
        indexed = {row['batch']: row for row in self._index}
        if any(indexed.get(row['batch']) != row for row in expected):
            raise ValueError('预测索引与检查点不一致，拒绝恢复')
        if len(available - set(batches)) > 1:
            raise ValueError('预测目录超出检查点多个批次，拒绝恢复')

    def __call__(self, batch_index: int, preds: np.ndarray) -> None:
        preds = np.asarray(preds).ravel()
        path = self.dir / f"pred_{batch_index:04d}.csv"
        output = io.StringIO(newline='')
        w = csv.writer(output)
        w.writerow(['index', 'pred'])
        for i, p in enumerate(preds):
            w.writerow([i, _fmt(p, self.task)])
        content = output.getvalue().encode('utf-8')
        if self.resume and path.exists() and path.read_bytes() != content:
            raise ValueError(f'第 {batch_index} 批重放预测与现有文件不同，拒绝覆盖')
        atomic_write(path, content)
        entry = {'batch': batch_index, 'n': int(len(preds)), 'file': path.name,
                 'sha256': hashlib.sha256(content).hexdigest()}
        self._index = [row for row in self._index if row['batch'] != batch_index] + [entry]
        self._index.sort(key=lambda row: row['batch'])
        atomic_write(self.dir / 'index.json', json.dumps(self._index, ensure_ascii=False, indent=2).encode('utf-8'))


def _fmt(p, task: str):
    if task == "regression":
        return f"{float(p):.6f}"
    return int(round(float(p)))


def null_sink(batch_index: int, preds: np.ndarray) -> None:  # pragma: no cover
    return None
