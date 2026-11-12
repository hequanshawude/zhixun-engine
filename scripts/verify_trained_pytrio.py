"""用 PyTRiO 已保存权重调用一条留出样本，记录可复现的链路验证。"""
import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selftrain.config import LogConfig, SystemConfig, apply_env_overrides
from selftrain.experiment import ExperimentLogger
from selftrain.harness import LLMClient, PyTrioSamplingClient, parse_action

ROOT = Path('out/pytrio-training')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdk', action='store_true', help='use PyTRiO sampling API')
    args = parser.parse_args()
    saved = json.loads((ROOT / 'result.json').read_text(encoding='utf-8'))
    row = json.loads((ROOT / 'holdout.jsonl').read_text(encoding='utf-8').splitlines()[0])
    cfg = apply_env_overrides(SystemConfig.load('configs/pytrio.yaml'))
    cfg.harness.model = saved['model_path']
    cfg.harness.max_tokens = 512
    cfg.harness.timeout_s = 90
    logger = ExperimentLogger(LogConfig(backend='swanlab', workspace='hequan',
        project='zhixunyinqing', out_dir='runs/pytrio-training', swanlab_mode='online'),
        run_name='pytrio-trained-inference-check')
    logger.start({'base_model': saved['base_model'], 'model_path': saved['model_path'],
                  'max_tokens': cfg.harness.max_tokens, 'inference_api': 'sdk' if args.sdk else 'openai'},
                 {'source': 'synthetic-rule-decisions', 'holdout_index': row['index']})
    report = {'status': 'failed', 'model_path': saved['model_path'],
              'training_cloud_url': saved['cloud_url'], 'check_cloud_url': logger.cloud_url,
              'inference_api': 'sdk' if args.sdk else 'openai'}
    try:
        if args.sdk:
            print('Sampling trained weights via PyTRiO SDK', flush=True)
            cfg.harness.max_tokens = 128
            answer, usage = PyTrioSamplingClient(cfg.harness, saved['model_path']).chat(row['messages'])
            prompt_tokens, completion_tokens, unknown_calls = (
                usage.prompt_tokens, usage.completion_tokens, usage.unknown_calls)
        else:
            answer, usage = LLMClient(cfg.harness).chat(row['messages'])
            prompt_tokens, completion_tokens, unknown_calls = (
                usage.prompt_tokens, usage.completion_tokens, usage.unknown_calls)
        report.update(response=answer, prompt_tokens=prompt_tokens,
                      completion_tokens=completion_tokens, unknown_calls=unknown_calls)
        parsed = parse_action(answer)
        response_action = parsed['action'] if parsed else None
        teacher_action = json.loads(row['answer'])['action']
        report.update(status='decision_verified' if response_action else
                      ('inference_only' if answer.strip() else 'empty_response'),
                      response_action=response_action, teacher_action=teacher_action,
                      matches_teacher=response_action == teacher_action)
        logger.log(1, {'prompt_tokens': prompt_tokens,
                       'completion_tokens': completion_tokens,
                       'nonempty_response': int(bool(answer.strip())),
                       'matches_teacher': int(response_action == teacher_action)})
        logger.finish({'status': report['status']})
    except Exception as exc:
        report.update(error_type=type(exc).__name__,
                      error_message=str(exc), cause_type=type(exc.__cause__).__name__ if exc.__cause__ else None,
                      http_status=getattr(exc, 'http_status', None),
                      provider_code=getattr(exc, 'provider_code', None))
        logger.abort(type(exc).__name__)
    (ROOT / 'verification.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Verification: {report['status']}; response action: {report.get('response_action')}; "
          f"teacher action: {report.get('teacher_action')}")
    return 0 if report['status'] == 'decision_verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
