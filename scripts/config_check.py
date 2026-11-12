"""非默认随机种子、验证窗口和隔离间隔的三任务演练。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selftrain.cli import _run
from selftrain.config import SystemConfig
from selftrain.data import SyntheticConfig, SyntheticStream
from selftrain.types import TaskType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'out' / ('config-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    out.mkdir(parents=True)
    print(str(out), flush=True)
    rows = []
    for task in TaskType:
        cfg = SystemConfig.load(str(ROOT / 'configs/cloud.yaml')) if args.cloud else SystemConfig()
        cfg.log.backend = 'swanlab' if args.cloud else 'jsonl'
        cfg.log.out_dir = str(out / 'runs')
        cfg.playbook_path = None
        cfg.model.random_state = 123
        cfg.gate.val_batches, cfg.gate.purge_batches = 2, 1
        stream = SyntheticStream(SyntheticConfig(task=task, n_batches=6, batch_size=60,
                                                 n_features=4, seed=314))
        system, result, logger = _run(cfg, stream.spec, iter(stream), 'rule', task.value,
            quiet=True, pred_dir=str(out / task.value))
        assert system.static.estimator.random_state == 123
        assert result.summary['n_batches'] == 5
        assert all(d['plan']['action'] == 'noop' for d in result.decisions[:2])
        assert all(d['plan']['validation']['validation_rows'] == 120 for d in result.decisions[2:])
        row = {'task': task.value, 'passed': True, 'url': logger.cloud_url,
               'summary': result.summary, 'log_dir': str(logger.out_dir)}
        if args.cloud:
            import swanlab
            remote = swanlab.Api().run(json.loads((logger.out_dir / 'cloud.json').read_text())['path'])
            assert remote.state == 'FINISHED'
            config = remote.profile['config']['system']['value']
            assert config['model']['random_state'] == 123
            assert config['gate']['val_batches'] == 2 and config['gate']['purge_batches'] == 1
            row['cloud_verified'] = True
        rows.append(row)
        (out / 'report.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        print(task.value + ': passed', flush=True)


if __name__ == '__main__':
    main()
