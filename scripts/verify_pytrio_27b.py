"""验证赛题指定 27B 模型能输出 Harness 可解析的决策，并记录云端。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selftrain.config import LogConfig, SystemConfig, apply_env_overrides
from selftrain.experiment import ExperimentLogger
from selftrain.harness import LLMClient, PyTrioSamplingClient, parse_action


def main():
    use_sdk = '--sdk' in sys.argv
    row = json.loads(Path('out/pytrio-training/holdout.jsonl').read_text(
        encoding='utf-8').splitlines()[0])
    cfg = apply_env_overrides(SystemConfig.load('configs/pytrio.yaml'))
    cfg.harness.max_tokens = 128 if use_sdk else 512
    cfg.harness.timeout_s = 90
    logger = ExperimentLogger(LogConfig(backend='swanlab', workspace='hequan',
        project='zhixunyinqing', out_dir='runs/pytrio-27b', swanlab_mode='online'),
        run_name='pytrio-27b-decision-check')
    logger.start({'model': cfg.harness.model, 'max_tokens': cfg.harness.max_tokens,
                  'inference_api': 'sdk' if use_sdk else 'openai'},
                 {'source': 'synthetic-rule-decisions', 'holdout_index': row['index']})
    report = {'status': 'failed', 'model': cfg.harness.model,
              'inference_api': 'sdk' if use_sdk else 'openai',
              'cloud_url': logger.cloud_url, 'holdout_index': row['index']}
    try:
        client = (PyTrioSamplingClient(cfg.harness, cfg.harness.model)
                  if use_sdk else LLMClient(cfg.harness))
        answer, usage = client.chat(row['messages'])
        parsed = parse_action(answer)
        expected = json.loads(row['answer'])['action']
        report.update(status='decision_verified' if parsed else 'invalid_decision',
                      response=answer, response_action=parsed.get('action') if parsed else None,
                      teacher_action=expected,
                      prompt_tokens=usage.prompt_tokens,
                      completion_tokens=usage.completion_tokens,
                      unknown_calls=usage.unknown_calls)
        logger.log(1, {'valid_decision': int(bool(parsed)),
                       'matches_teacher': int(bool(parsed and parsed.get('action') == expected)),
                       'prompt_tokens': usage.prompt_tokens,
                       'completion_tokens': usage.completion_tokens})
        logger.finish({'valid_decision': int(bool(parsed))})
    except Exception as exc:
        report.update(error_type=type(exc).__name__,
                      error_message=str(exc),
                      http_status=getattr(exc, 'http_status', None),
                      provider_code=getattr(exc, 'provider_code', None))
        logger.abort(type(exc).__name__)
    output = Path('out/pytrio-27b')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'verification.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"27B strategy check: {report['status']}; "
          f"action={report.get('response_action')}; "
          f"tokens={report.get('prompt_tokens', 0) + report.get('completion_tokens', 0)}")
    return 0 if report['status'] == 'decision_verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
