"""离线准备检查；合成数据结果不代表官方成绩。"""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selftrain.config import SystemConfig
from selftrain.data import SyntheticConfig, SyntheticStream
from selftrain.experiment import ExperimentLogger
from selftrain.loop import SelfEvolvingSystem
from selftrain.types import TaskType


def main():
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    cloud = '--cloud' in sys.argv
    out = ROOT / 'out' / 'preflight'
    out.mkdir(parents=True, exist_ok=True)
    check = subprocess.run([sys.executable, '-m', 'pytest', '-o', 'addopts=', '-q',
                            '--basetemp=' + str(out / 'pytest-tmp')], cwd=ROOT,
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
    (out / 'tests.txt').write_text(check.stdout + check.stderr, encoding='utf-8')
    runs = []
    for task in TaskType:
        for policy in ('rule', 'bandit'):
            cfg = SystemConfig()
            cfg.log.backend = 'swanlab' if cloud else 'jsonl'
            cfg.log.out_dir = str(out / 'runs')
            if cloud:
                cfg.log.workspace = 'hequan'
                cfg.log.project = 'zhixunyinqing'
                cfg.log.swanlab_mode = 'online'
            stream = SyntheticStream(SyntheticConfig(task=task, n_batches=8,
                batch_size=100, n_features=8, scenario='mixed', seed=42))
            logger = ExperimentLogger(cfg.log, run_name=f'{task.value}-{policy}')
            predictions = []
            system = SelfEvolvingSystem(stream.spec, cfg, logger=logger,
                policy_mode=policy, emit=lambda k, p: predictions.append((k, len(p))))
            result = system.run(iter(stream))
            runs.append({'task': task.value, 'policy': policy,
                'completed': len(predictions) == 7 and all(n == 100 for _, n in predictions),
                'gain_vs_local_static': result.summary['gain_vs_static'],
                'log_dir': str(logger.out_dir)})
    files = [p for folder in ('selftrain', 'configs', 'tests', 'scripts')
             for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(files)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True,
                            text=True, encoding='utf-8', errors='replace')
    (out / 'environment.txt').write_text(platform.platform() + '\n' + sys.version + '\n' + freeze.stdout, encoding='utf-8')
    report = {'tests_passed': check.returncode == 0, 'synthetic_runs': runs,
              'official_ready': False, 'pending': ['官方 SDK 与隐藏测试调用协议',
              '正式环境的 PyTRiO 凭据注入和模型访问方式',
              '正式环境的 SwanLab 云端归档或回传规范',
              '官方资源限制下镜像验收和静态基线核对', '演示视频录制']}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if check.returncode == 0 and all(r['completed'] for r in runs) else 1


if __name__ == '__main__':
    raise SystemExit(main())

