"""读回云端曲线，与本地结果逐点核对；不创建实验。"""
import argparse
import json
import math
from pathlib import Path

import swanlab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = json.loads((args.directory / 'results.json').read_text(encoding='utf-8'))
    assert len(rows) == 36, f'Expected 36 runs, got {len(rows)}'
    api = swanlab.Api()
    checks = []
    for row in rows:
        check = {k: row[k] for k in ('scenario', 'seed', 'policy')}
        try:
            assert row['status'] == 'completed', row.get('error')
            cloud = json.loads((Path(row['log_dir']) / 'cloud.json').read_text(encoding='utf-8'))
            remote = api.run(cloud['path'])
            assert remote.state == 'FINISHED', remote.state
            curves = remote.metrics(keys=['system_primary', 'static_primary', 'summary/gain_vs_static'])
            curves = {x['key']: x['metrics'] for x in curves['list']}
            for key, local_key in [('system_primary', 'system'), ('static_primary', 'static')]:
                observed = {x['index']: x['data'] for x in curves[key]}
                assert len(observed) == 13
                for score in row['scores']:
                    assert math.isclose(observed[score['batch']], score[local_key], abs_tol=1e-10)
            assert math.isclose(curves['summary/gain_vs_static'][0]['data'],
                                row['summary']['gain_vs_static'], abs_tol=1e-10)
            cfg = remote.profile['config']
            assert cfg['system']['value']['harness']['api_key'] == '[REDACTED]'
            assert cfg['system']['value']['benchmark']['data_sha256'] == row['data_sha256']
            media = remote.medias(keys=['decision'], all=True)
            count = sum(len(x.get('metrics', [])) for x in media['list'])
            assert count == 13, count
            check.update(verified=True, url=remote.url, decision_count=count)
        except Exception as exc:
            check.update(verified=False, error=str(exc))
        checks.append(check)
        (args.directory / 'cloud-verification.json').write_text(
            json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"[{len(checks)}/36] {check['scenario']} {check['seed']} {check['policy']}: {check['verified']}", flush=True)
    return int(not all(c['verified'] for c in checks))


if __name__ == '__main__':
    raise SystemExit(main())
