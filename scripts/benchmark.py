"""固定矩阵对照：静态基线、规则、策略记忆；保留所有结果。"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import uuid

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from selftrain.config import SystemConfig
from selftrain.data import SyntheticConfig, SyntheticStream
from selftrain.experiment import ExperimentLogger
from selftrain.loop import SelfEvolvingSystem
from selftrain.sink import PredictionSink

SCENARIOS = ('none', 'covariate', 'concept', 'label', 'sudden', 'mixed')
LABELS = {'none': '无漂移', 'covariate': '输入分布变化', 'concept': '概念漂移',
          'label': '类别比例变化', 'sudden': '突变', 'mixed': '混合漂移'}
SEEDS = (7, 42, 2026)


class BenchmarkLogger(ExperimentLogger):
    def __init__(self, cfg, name, metadata):
        self.metadata = metadata
        super().__init__(cfg, run_name=name)

    def start(self, config, spec):
        super().start({**config, 'benchmark': self.metadata}, spec)


def render_report(rows, directory, seeds=SEEDS):
    good = [r for r in rows if r['status'] == 'completed']
    lines = ['# 多场景对照实验报告', '',
        '范围：二分类合成数据，6 种场景 × 3 个数据种子 × 2 种更新策略，共 36 次运行。',
        '每次 14 批 × 300 条 × 16 维，首批训练，后 13 批先预测再学习。静态模型在每次运行中同步评估。',
        '同一场景与种子共用相同数据；规则和策略记忆使用相同模型、预算上限，每次清空策略记忆。',
        f'模型训练随机种子固定为 42；数据种子为 {list(seeds)}。每次最多 200 轮、3600 秒预算估算，LLM 关闭。',
        '指标为逐批 macro F1 的均值，越大越好；下表为 3 个种子的平均值，非官方分数。',
        '类别比例变化通过修改生成器截距实现，同时改变条件概率，不能视为严格的纯标签漂移。', '',
        f'完成 {len(good)}/{len(rows)} 次已执行运行。', '',
        '| 场景 | 策略 | 静态 F1 | 系统 F1（均值±样本标准差） | 平均增益 | 最差增益 | 胜/平/负 | 平均本地成本秒 |',
        '|---|---|---:|---:|---:|---:|---|---:|']
    for scene in SCENARIOS:
        for policy in ('rule', 'bandit'):
            selected = [r for r in good if r['scenario'] == scene and r['policy'] == policy]
            if not selected:
                continue
            scores = [r['summary']['mean_system'] for r in selected]
            gains = [r['summary']['gain_vs_static'] for r in selected]
            baseline = statistics.mean(r['summary']['mean_static'] for r in selected)
            cost = statistics.mean(r['summary']['budget']['compute_s'] for r in selected)
            wins = sum(g > 1e-9 for g in gains)
            ties = sum(abs(g) <= 1e-9 for g in gains)
            sd = statistics.stdev(scores) if len(scores) > 1 else 0
            lines.append(f'| {LABELS[scene]} | {policy} | {baseline:.4f} | {statistics.mean(scores):.4f} ± {sd:.4f} | {statistics.mean(gains):+.4f} | {min(gains):+.4f} | {wins}/{ties}/{len(gains)-wins-ties} | {cost:.2f} |')
    lines += ['', '## 策略差异', '', '| 场景 | 种子 | bandit 相对 rule 的 F1 差值 | 不同动作批数 |', '|---|---:|---:|---:|']
    for scene in SCENARIOS:
        for seed in seeds:
            pair = {r['policy']: r for r in good if r['scenario'] == scene and r['seed'] == seed}
            if len(pair) != 2:
                continue
            a, b = pair['rule'], pair['bandit']
            assert abs(a['summary']['mean_static'] - b['summary']['mean_static']) < 1e-12
            delta = b['summary']['mean_system'] - a['summary']['mean_system']
            different = sum(x != y for x, y in zip(a['actions'], b['actions']))
            lines.append(f'| {LABELS[scene]} | {seed} | {delta:+.4f} | {different} |')
    lines += ['', '## 单次记录', '', '| 场景 | 种子 | 策略 | 状态 | 云端 |', '|---|---:|---|---|---|']
    for r in rows:
        link = f"[查看实验]({r['url']})" if r.get('url') else '本地记录'
        lines.append(f"| {LABELS[r['scenario']]} | {r['seed']} | {r['policy']} | {r['status']} | {link} |")
    lines += ['', '## 解释边界', '',
        '- 所有失败和负增益结果均保留。三个种子只用于初步稳定性检查，不能证明普遍有效。',
        '- 本地成本包含初始化与主循环墙钟耗时，不包含全部云端同步开销；不是官方 GPU/Token 计费或独立静态运行成本。',
        '- bandit 是带规则先验的策略记忆；若结果与规则相同，不能据此声称策略自改进已产生收益。',
        '- 本轮先固定代码完成评估，不在这些种子上反复调参；后续修改应使用新的保留种子检验。',
        '- 本报告只覆盖二分类。官方数据、指标与 SDK 发布后必须重新验证。', '']
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', action='store_true')
    parser.add_argument('--seeds', type=int, nargs=3, default=SEEDS,
                        help='固定的三个数据种子，修改策略后应使用未调参的新种子')
    args = parser.parse_args()
    if len(set(args.seeds)) != 3:
        parser.error('三个种子必须不同')
    directory = ROOT / 'out' / ('benchmark-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / 'selftrain').glob('*.py')}
    manifest['scripts/benchmark.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (directory / 'protocol.json').write_text(json.dumps({
        'scenarios': SCENARIOS, 'seeds': args.seeds, 'policies': ['rule', 'bandit'],
        'model_seed': 42, 'max_iterations': 200, 'max_compute_seconds': 3600,
    }, indent=2), encoding='utf-8')
    rows = []
    print(f'报告目录: {directory}', flush=True)
    for scene in SCENARIOS:
        for seed in args.seeds:
            generator = SyntheticConfig(scenario=scene, seed=seed)
            stream = SyntheticStream(generator)
            batches = list(stream)
            data_hash = hashlib.sha256(b''.join(b.X.tobytes() + b.y.tobytes() for b in batches)).hexdigest()
            for policy in ('rule', 'bandit'):
                cfg = SystemConfig.load(str(ROOT / 'configs/cloud.yaml')) if args.cloud else SystemConfig()
                cfg.log.backend = 'swanlab' if args.cloud else 'jsonl'
                cfg.log.out_dir = str(directory / 'runs')
                cfg.playbook_path = None
                cfg.seed = seed
                name = f'{scene}-seed{seed}-{policy}'
                row = {'scenario': scene, 'seed': seed, 'policy': policy, 'data_sha256': data_hash}
                logger = None
                started = time.perf_counter()
                try:
                    logger = BenchmarkLogger(cfg.log, name, {'generator': asdict(generator), 'policy': policy, 'data_sha256': data_hash})
                    sink = PredictionSink(str(logger.out_dir / 'predictions'))
                    emitted = []
                    def emit(k, pred):
                        assert len(pred) == generator.batch_size and np.isfinite(pred).all()
                        sink(k, pred)
                        emitted.append(k)
                    system = SelfEvolvingSystem(stream.spec, cfg, logger=logger, policy_mode=policy, emit=emit)
                    result = system.run(iter(batches))
                    assert emitted == list(range(1, generator.n_batches))
                    row.update(status='completed', summary=result.summary,
                               actions=[d['plan']['action'] for d in result.decisions],
                               scores=system.scores, url=logger.cloud_url, log_dir=str(logger.out_dir))
                except Exception as exc:
                    row.update(status='failed', error=str(exc), url=logger.cloud_url if logger else None)
                    if logger and logger._swan_run and logger._swan_run.alive:
                        logger._swan_run.finish(state='crashed', error=str(exc))
                row['wall_seconds'] = time.perf_counter() - started
                rows.append(row)
                (directory / 'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f"[{len(rows)}/36] {name}: {row['status']} gain={row.get('summary',{}).get('gain_vs_static')}", flush=True)
    render_report(rows, directory, args.seeds)
    print(f'完成，报告: {directory / "report.md"}', flush=True)
    return int(any(r['status'] != 'completed' for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
