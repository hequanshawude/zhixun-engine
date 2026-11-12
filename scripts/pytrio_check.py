"""默认仅验证账号模型列表；--smoke 明确启用一次最多 64 输出 token 的调用。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selftrain.config import SystemConfig, apply_env_overrides
from selftrain.pytrio_access import MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    report = {'status': 'failed', 'inference_requested': args.smoke, 'model': MODEL}
    try:
        cfg = apply_env_overrides(SystemConfig.load('configs/pytrio.yaml'))
        import pytrio
        pytrio.configure(api_key=cfg.harness.api_key, base_url='https://pytrio.com',
                         request_timeout_seconds=20, retry_max_attempts=1,
                         control_retry_timeout_seconds=20)
        client = pytrio.ServiceClient()
        models = client.get_supported_models()
        report['supported_models'] = models
        if MODEL not in models:
            raise ValueError('账号模型列表暂未包含赛题指定的 Qwen/Qwen3.8-27B')
        report['status'] = 'connected'
        if args.smoke:
            from selftrain.harness import LLMClient
            cfg.harness.max_tokens = 64
            answer, usage = LLMClient(cfg.harness).chat([
                {'role': 'user', 'content': 'Reply with the single word OK.'}])
            if not answer.strip():
                raise ValueError('模型返回空文本，尚未验证可用输出')
            report.update(status='inference_verified', usage={
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'unknown_calls': usage.unknown_calls}, response_chars=len(answer))
        print('PyTRiO connection verified. Model: ' + MODEL)
    except Exception as exc:
        # 不输出第三方异常内容或响应，避免潜在凭据泄漏。
        model_list_verified = report.get('status') == 'connected'
        report.update(status='failed', error_type=type(exc).__name__)
        if getattr(exc, 'http_status', None) is not None:
            report['http_status'] = exc.http_status
        if getattr(exc, 'provider_code', None):
            report['provider_code'] = exc.provider_code
        if model_list_verified:
            report['model_list_verified'] = True
        if getattr(exc, 'provider_code', None) == 'insufficient_quota':
            report.update(status='quota_blocked', provider_code='insufficient_quota')
            print('PyTRiO model access verified; inference blocked by account quota.')
        elif getattr(exc, 'provider_code', None) == 'no_available_server':
            report.update(status='model_server_unavailable')
            print('PyTRiO model access verified; no server is available for this model.')
        else:
            print('PyTRiO check failed: ' + type(exc).__name__ + '. Check login/network/model access.')
    output = Path('out/pytrio-setup')
    output.mkdir(parents=True, exist_ok=True)
    name = 'smoke.json' if args.smoke else 'connection.json'
    (output / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if report['status'] in ('connected', 'inference_verified') else 1


if __name__ == '__main__':
    raise SystemExit(main())
