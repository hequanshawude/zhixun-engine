"""验证三类任务、候选超时和首次训练失败的记录，可选上传云端。"""
import argparse
import hashlib
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

import numpy as np
from selftrain.config import SystemConfig
from selftrain.data import SyntheticConfig, SyntheticStream
from selftrain.experiment import ExperimentLogger
from selftrain.loop import SelfEvolvingSystem
from selftrain.types import TaskType
from selftrain.worker import ExecutionTimeout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', action='store_true')
    args = parser.parse_args()
    directory = ROOT / 'out' / ('runtime-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    print(str(directory), flush=True)
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / 'selftrain').glob('*.py')}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows = []
    cases = [(t.value, t) for t in TaskType] + [('candidate-timeout', TaskType.BINARY),
                                               ('initialization-timeout', TaskType.BINARY)]
    api = None
    if args.cloud:
        import swanlab
        api = swanlab.Api()
    for name, task in cases:
        cfg = SystemConfig.load(str(ROOT / 'configs/cloud.yaml')) if args.cloud else SystemConfig()
        cfg.log.backend = 'swanlab' if args.cloud else 'jsonl'
        cfg.log.out_dir = str(directory / 'runs')
        cfg.playbook_path = None
        if name == 'candidate-timeout':
            cfg.budget.action_timeout_s = 0.001
        if name == 'initialization-timeout':
            cfg.budget.bootstrap_timeout_s = 0.001
        stream = SyntheticStream(SyntheticConfig(task=task, n_batches=6, batch_size=80,
            n_features=6, scenario='mixed', drift_start=1, seed=2718))
        logger = ExperimentLogger(cfg.log, run_name=name)
        emitted = []
        def emit(k, predictions):
            assert len(predictions) == 80 and np.isfinite(predictions).all()
            emitted.append(k)
        system = SelfEvolvingSystem(stream.spec, cfg, logger=logger, emit=emit)
        row = {'case': name, 'log_dir': str(logger.out_dir), 'url': logger.cloud_url}
        try:
            result = system.run(iter(stream))
            assert name != 'initialization-timeout', '预期首次训练超时'
            assert emitted == [1, 2, 3, 4, 5]
            assert result.summary['budget']['initialization_s'] > 0
            assert result.summary['budget']['iterations'] == 5
            if name == 'candidate-timeout':
                assert result.summary['n_timeouts'] > 0
                assert result.summary['n_promotions'] == 0
            else:
                assert result.summary['n_execution_failures'] == 0
            row.update(passed=True, summary=result.summary, expected_cloud_state='FINISHED')
        except ExecutionTimeout as exc:
            logger.abort(str(exc), system.initialization)
            assert name == 'initialization-timeout'
            assert not emitted
            failure = json.loads((logger.out_dir / 'failure.json').read_text(encoding='utf-8'))
            assert failure['initialization']['compute_s'] > 0
            row.update(passed=True, expected_cloud_state='CRASHED')
        if api:
            cloud = json.loads((logger.out_dir / 'cloud.json').read_text(encoding='utf-8'))
            key = 'failure' if name == 'initialization-timeout' else 'artifacts/costs.json'
            # 上传结束后媒体索引可能稍晚可查询，有限重试，不重跑实验。
            for attempt in range(8):
                remote = api.run(cloud['path'])
                media = remote.medias(keys=[key], all=True)
                count = sum(len(x.get('metrics', [])) for x in media['list'])
                if remote.state == row['expected_cloud_state'] and count == 1:
                    break
                time.sleep(2)
            assert remote.state == row['expected_cloud_state'], remote.state
            assert count == 1, f'云端附件未就绪：{key}'
            row['cloud_verified'] = True
        rows.append(row)
        (directory / 'report.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{name}: passed', flush=True)


if __name__ == '__main__':
    main()
