"""按白名单冻结准备版，保留文件哈希并逐项读回校验 ZIP。"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    name = 'zhixun-preparation-' + datetime.now().strftime('%Y%m%d-%H%M%S')
    files = set()
    for folder in ('selftrain', 'scripts', 'tests'):
        files.update((ROOT / folder).glob('*.py'))
    files.update((ROOT / 'docs').glob('*.md'))
    files.update((ROOT / 'configs').glob('*.yaml'))
    files.update(ROOT.glob('*.cmd'))
    files.update(ROOT.glob('requirements*.txt'))
    files.update(ROOT / p for p in ('README.md', 'pyproject.toml', 'Dockerfile', 'Dockerfile.local', '.dockerignore'))
    if (ROOT / '智讯引擎赛题.pdf').exists():
        files.add(ROOT / '智讯引擎赛题.pdf')
    names = {'report.json', 'report.md', 'manifest.json', 'protocol.json',
             'cloud-verification.json', 'results.json', 'image.json',
             'validation-rule.json', 'validation-bandit.json', 'final.json'}
    for folder in (ROOT / 'out').iterdir():
        if folder.is_file() and folder.name.startswith('tests-') and folder.suffix == '.xml':
            files.add(folder)
        elif folder.is_dir() and folder.name.startswith(('benchmark-', 'runtime-', 'checkpoint-', 'harness-', 'config-', 'docker-', 'xjtu-validation', 'cmapss-validation', 'cpu-models')):
            files.update(p for p in folder.rglob('*') if p.is_file() and p.name in names
                         and 'runs' not in p.relative_to(folder).parts)
    for evidence_name in ('result.json', 'verification.json', 'adapter-manifest.json'):
        evidence = ROOT / 'out' / 'pytrio-training' / evidence_name
        if evidence.exists():
            files.add(evidence)
    for evidence in (ROOT / 'out' / 'pytrio-27b' / 'verification.json',
                     ROOT / 'out' / 'pytrio-setup' / 'smoke.json',
                     ROOT / 'out' / 'preflight' / 'report.json',
                     ROOT / 'out' / 'preflight' / 'tests.txt',
                     ROOT / 'runs' / 'gas-pytrio-27b' / 'run-gas-20261110-210954-f5c615bb' / 'summary.json',
                     ROOT / 'runs' / 'gas-pytrio-trained' / 'run-gas-20261110-155828-a9a83d90' / 'summary.json'):
        if evidence.exists():
            files.add(evidence)
    # CPU 正式路径的公开验证摘要、逐轮决策及云端链接；不包含检查点、数据或凭据。
    cpu_evidence = ROOT / 'out' / 'cpu-qwen-swanlab-full-20261111'
    for evidence in (cpu_evidence / 'all-summary.json',
                     ROOT / 'out' / 'docker-cpu-online-audit' / 'summary.json',
                     ROOT / 'out' / 'docker-cpu-cloud-audit' / 'summary.json'):
        if evidence.exists():
            files.add(evidence)
    for profile in ('a', 'b', 'c'):
        for evidence_name in ('summary.json', 'rounds-ledger.jsonl'):
            evidence = cpu_evidence / profile / evidence_name
            if evidence.exists():
                files.add(evidence)
    for cloud in (cpu_evidence / 'swanlab').glob('cpu-*/cloud.json'):
        files.add(cloud)
    for cloud in (ROOT / 'out' / 'docker-cpu-online-audit' / 'swanlab').glob('cpu-*/cloud.json'):
        files.add(cloud)
    # Keep both positive and negative public-data results without bundling data.
    for family in ('cmapss-fd001-27b', 'cmapss-fd001-holdout',
                   'xjtu-sy-27b', 'xjtu-sy-cloud', 'cmapss-cloud'):
        for run_dir in (ROOT / 'runs' / family).glob('run-stream-*'):
            for evidence_name in ('summary.json', 'cloud.json'):
                evidence = run_dir / evidence_name
                if evidence.exists():
                    files.add(evidence)
    payloads = {}
    for path in sorted(files):
        if path.is_symlink():
            raise ValueError('拒绝打包符号链接：' + str(path))
        data = path.read_bytes()
        if path.parent.name == 'configs':
            import yaml
            config = yaml.safe_load(data) or {}
            key = config.get('harness', {}).get('api_key')
            if key not in (None, '', 'EMPTY', '[REDACTED]'):
                raise ValueError('配置含非占位密钥，拒绝打包：' + path.name)
        payloads[path.relative_to(ROOT).as_posix()] = data
    manifest = {p: hashlib.sha256(data).hexdigest() for p, data in payloads.items()}
    payloads['RELEASE-MANIFEST.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    payloads['RELEASE-NOTES.txt'] = (
        '智训引擎：平台开放前准备版\n'
        '先读 docs/交付包说明.md 和 docs/上线准备清单.md。\n'
        '不包含账号凭据、训练数据、恢复检查点、Python 环境或 Docker 镜像二进制。\n'
        'out 中保留历史验证证据；报告中的原本机绝对路径属于历史记录。\n'
        '这不是已通过官方平台验收的正式提交包。\n'
    ).encode('utf-8')
    destination = ROOT / 'dist' / (name + '.zip')
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in payloads.items():
            archive.writestr(path, data)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert all(hashlib.sha256(archive.read(p)).hexdigest() == digest for p, digest in manifest.items())
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.sha256').write_text(digest + '  ' + destination.name + '\n', encoding='utf-8')
    print(json.dumps({'archive': str(destination), 'files': len(payloads),
                      'bytes': destination.stat().st_size, 'sha256': digest}, ensure_ascii=False))


if __name__ == '__main__':
    main()
