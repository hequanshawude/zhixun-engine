"""按已验证的 PyTRiO 权重路径下载 LoRA 适配器并记录完整性信息。"""
import hashlib
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path('out/pytrio-training')
MAX_BYTES = 250_000_000


def main():
    import pytrio

    saved = json.loads((ROOT / 'result.json').read_text(encoding='utf-8'))
    pytrio.configure(request_timeout_seconds=30, retry_max_attempts=1,
                     control_retry_timeout_seconds=30)
    rest = pytrio.ServiceClient().create_rest_client()
    checkpoints = rest.list_user_checkpoints(limit=100).result(timeout=45).checkpoints
    matching = [item for item in checkpoints if item.path == saved['model_path']]
    if len(matching) != 1:
        raise RuntimeError('未找到唯一匹配的 PyTRiO 训练权重')
    checkpoint = matching[0]
    if checkpoint.size_bytes <= 0 or checkpoint.size_bytes > MAX_BYTES:
        raise RuntimeError('权重体积超出本次小规模下载上限')

    # 当前 PyTRiO SDK 0.2.9 实测返回 TAR，虽然网页版文档示例称 ZIP。
    archive = ROOT / f'{checkpoint.checkpoint_id}.tar'
    if not archive.exists():
        rest.download_checkpoint(checkpoint.checkpoint_id, archive, resume=True).result(timeout=240)
    if archive.stat().st_size > MAX_BYTES:
        raise RuntimeError('下载文件超出上限')
    with tarfile.open(archive, 'r:*') as packed:
        entries = packed.getmembers()
        members = [entry.name for entry in entries]
        if not all(entry.isfile() and not PurePosixPath(entry.name).is_absolute()
                   and '..' not in PurePosixPath(entry.name).parts for entry in entries):
            raise RuntimeError('LoRA 压缩包存在不安全路径')
        if not any(name.endswith('adapter_model.safetensors') for name in members):
            raise RuntimeError('压缩包缺少 LoRA adapter 权重')
        if not any(name.endswith('adapter_config.json') for name in members):
            raise RuntimeError('压缩包缺少 LoRA 配置')
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    manifest = {
        'checkpoint_id': checkpoint.checkpoint_id,
        'model_path': saved['model_path'],
        'base_model': saved['base_model'],
        'archive': str(archive),
        'size_bytes': archive.stat().st_size,
        'sha256': digest.hexdigest(),
        'members': members,
    }
    (ROOT / 'adapter-manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"LoRA adapter saved: {archive}; {manifest['size_bytes']} bytes; "
          f"sha256={manifest['sha256']}")


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'LoRA download failed: {type(exc).__name__}', file=sys.stderr)
        raise SystemExit(1)
