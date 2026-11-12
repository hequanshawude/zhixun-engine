"""仅在本机回环地址模拟模型服务，验证 HTTP 接入和降级，不调用真实模型。"""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selftrain.actions import ACTION_NAMES
from selftrain.config import HarnessConfig
from selftrain.harness import PolicyRouter
from selftrain.playbook import Playbook
from selftrain.types import DriftKind, Fingerprint, TaskType


class Handler(BaseHTTPRequestHandler):
    counts = {}

    def log_message(self, *args):
        pass

    def do_POST(self):
        case = self.path.split('/')[1]
        self.counts[case] = self.counts.get(case, 0) + 1
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert request['stream'] is False and request['model'] == 'local-test'
        if case == 'unauthorized':
            self.send_response(401)
            self.end_headers()
            return
        body = {'choices': [{'message': {'content':
            '```json\n{"action":"finetune","params":{},"reason":"local mock"}\n```'}}],
            'usage': {'prompt_tokens': 11, 'completion_tokens': 7}}
        if case == 'invalid-json':
            body['choices'][0]['message']['content'] = '{broken'
        if case == 'broken-envelope':
            body['choices'] = []
        if case == 'missing-usage':
            del body['usage']
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    directory = ROOT / 'out' / ('harness-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    rows = []
    try:
        for case in ('valid', 'invalid-json', 'broken-envelope', 'missing-usage', 'unauthorized', 'zero-budget'):
            cfg = HarnessConfig(enabled=True, model='local-test', small_model='local-test',
                api_key='local-test', base_url=f'http://127.0.0.1:{server.server_port}/{case}', timeout_s=3)
            router = PolicyRouter(cfg, Playbook(ACTION_NAMES), model='llm')
            ctx = SimpleNamespace(spec=SimpleNamespace(task=TaskType.BINARY))
            fp = Fingerprint(DriftKind.CONCEPT, 0.5, 0.1)
            plan, trace = router.decide(fp, ctx,
                {'budget_left': {'tokens': 0 if case == 'zero-budget' else 1000}},
                allowed=['noop', 'finetune'], best_available=False)
            expected_source = 'llm' if case in ('valid', 'missing-usage') else 'rule'
            assert trace.source == expected_source
            assert trace.tokens == (18 if case in ('valid', 'invalid-json', 'broken-envelope') else 0)
            assert trace.unknown_usage_calls == int(case in ('missing-usage', 'unauthorized'))
            assert Handler.counts.get(case, 0) == (0 if case == 'zero-budget' else 1)
            rows.append({'case': case, 'passed': True, 'source': trace.source,
                'known_tokens': trace.tokens, 'unknown_usage_calls': trace.unknown_usage_calls,
                'http_calls': Handler.counts.get(case, 0), 'reason': trace.fallback_reason})
            print(f'{case}: passed', flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / 'selftrain').glob('*.py')}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (directory / 'report.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(directory))


if __name__ == '__main__':
    main()
