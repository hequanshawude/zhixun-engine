"""用三个独立进程验证中断、恢复和连续运行的预测一致性。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['partial', 'resume', 'reference'])
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--cloud', action='store_true')
    args = parser.parse_args()
    if args.stage:
        cfg = SystemConfig.load(str(ROOT / 'configs/cloud.yaml')) if args.cloud else SystemConfig()
        cfg.log.backend = 'swanlab' if args.cloud else 'jsonl'
        cfg.log.out_dir = str(args.directory / 'runs' / args.stage)
        cfg.playbook_path = None
        stream = SyntheticStream(SyntheticConfig(n_batches=6, batch_size=60, n_features=4, seed=37))
        def interrupted():
            for batch in stream:
                if batch.index == 3:
                    raise RuntimeError('planned interruption')
                yield batch
        try:
            _run(cfg, stream.spec, interrupted() if args.stage == 'partial' else iter(stream),
                 'rule', args.stage, quiet=True,
                 pred_dir=str(args.directory / ('reference' if args.stage == 'reference' else 'predictions')),
                 checkpoint=str(args.directory / 'state.bin') if args.stage != 'reference' else None,
                 resume=args.stage == 'resume')
        except RuntimeError as exc:
            if args.stage == 'partial' and str(exc) == 'planned interruption':
                return 42
            raise
        return 0
    directory = ROOT / 'out' / ('checkpoint-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    for stage in ('partial', 'resume', 'reference'):
        command = [sys.executable, str(Path(__file__).resolve()), '--stage', stage,
                   '--directory', str(directory)] + (['--cloud'] if args.cloud else [])
        result = subprocess.run(command, capture_output=True, timeout=180)
        (directory / (stage + '.log')).write_bytes(result.stdout + result.stderr)
        assert result.returncode == (42 if stage == 'partial' else 0), (stage, result.returncode)
        print(stage + ': passed', flush=True)
    index = json.loads((directory / 'predictions/index.json').read_text(encoding='utf-8'))
    assert [row['batch'] for row in index] == [1, 2, 3, 4, 5]
    for row in index:
        assert (directory / 'predictions' / row['file']).read_bytes() == (directory / 'reference' / row['file']).read_bytes()
    reports = {}
    for stage in ('resume', 'reference'):
        file = next((directory / 'runs' / stage).glob('*/summary.json'))
        reports[stage] = json.loads(file.read_text(encoding='utf-8'))
    assert reports['resume']['mean_system'] == reports['reference']['mean_system']
    assert reports['resume']['budget']['iterations'] == 5
    lineage = json.loads(next((directory / 'runs/resume').glob('*/resume.json')).read_text(encoding='utf-8'))
    assert lineage['completed_batches'] == 2
    report = {'passed': True, 'identical_prediction_batches': 5,
              'resume_iterations': 5, 'lineage': lineage, 'cloud_runs': {}}
    if args.cloud:
        import swanlab
        api = swanlab.Api()
        for stage in ('partial', 'resume', 'reference'):
            cloud = json.loads(next((directory / 'runs' / stage).glob('*/cloud.json')).read_text(encoding='utf-8'))
            remote = api.run(cloud['path'])
            assert remote.state == ('CRASHED' if stage == 'partial' else 'FINISHED')
            report['cloud_runs'][stage] = {'url': remote.url, 'state': remote.state}
    (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(directory), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
