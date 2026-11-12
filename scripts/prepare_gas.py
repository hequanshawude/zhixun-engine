"""将 UCI Gas 官方十个 libsvm 批次转换为项目 CSV 流。"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file


def main():
    source = Path('data/raw/gas/gas.zip')
    target = Path('data/processed/gas')
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    features = [f'f{i:03d}' for i in range(1, 129)]
    with zipfile.ZipFile(source) as z:
        if z.testzip() is not None:
            raise ValueError('Gas 原始压缩包损坏')
        for number in range(1, 11):
            with z.open(f'Dataset/batch{number}.dat') as handle:
                X, y = load_svmlight_file(handle, n_features=128)
            if not np.isfinite(X.data).all() or not np.isin(y, [1, 2, 3, 4, 5, 6]).all():
                raise ValueError(f'第 {number} 批含非法特征或标签')
            frame = pd.DataFrame(X.toarray(), columns=features)
            frame.insert(0, 'y', y.astype(int) - 1)
            path = target / f'batch_{number:03d}.csv'
            frame.to_csv(path, index=False, float_format='%.9g')
            rows.append({'batch': number, 'rows': len(y),
                         'labels': [int(value) for value in np.unique(y)]})
    if sum(item['rows'] for item in rows) != 13910:
        raise ValueError('Gas 样本总数与官方说明不符')
    (target / 'manifest.json').write_text(json.dumps({
        'source': 'https://archive.ics.uci.edu/dataset/224/gas',
        'label_mapping': 'official 1..6 -> csv 0..5',
        'feature_count': 128,
        'batches': rows,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Gas ready: {sum(item["rows"] for item in rows)} rows, 10 ordered batches')


if __name__ == '__main__':
    main()
