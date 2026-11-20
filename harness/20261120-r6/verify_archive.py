"""Audit a docker-save archive without extracting its filesystem."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def sha(stream):
    digest = hashlib.sha256()
    for data in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(data)
    return digest.hexdigest()


def verify(archive_path, source_dir, official_dir, expected_tag=None):
    expected_tag = expected_tag or 'zhixun-engine:sdk104-r6-20261120'
    expected_app = {p.relative_to(source_dir).as_posix(): p.read_bytes()
                    for p in source_dir.rglob('*.py')}
    expected_sdk = {p.name: p.read_bytes() for p in official_dir.glob('*.py')}
    app, sdk = {}, {}
    sdk_root = 'usr/local/lib/python3.10/site-packages/contest_harness/'
    dependency_dirs = set()
    with archive_path.open('rb') as raw:
        digest = sha(raw)
    with tarfile.open(archive_path, 'r:') as archive:
        images = json.load(archive.extractfile('manifest.json'))
        manifests = json.load(archive.extractfile('index.json'))['manifests']
        assert len(images) == len(manifests) == 1
        image, entry = images[0], manifests[0]
        assert image['RepoTags'] == [expected_tag]
        manifest_path = entry['digest'].replace(':', '/', 1)
        manifest = json.load(archive.extractfile('blobs/' + manifest_path))
        assert image['Config'] == 'blobs/' + manifest['config']['digest'].replace(':', '/', 1)
        assert image['Layers'] == ['blobs/' + layer['digest'].replace(':', '/', 1)
                                   for layer in manifest['layers']]
        for path in [image['Config'], 'blobs/' + manifest_path, *image['Layers']]:
            with archive.extractfile(path) as blob:
                assert sha(blob) == path.rsplit('/', 1)[1], path
        config = json.load(archive.extractfile(image['Config']))
        assert (config['os'], config['architecture']) == ('linux', 'amd64')
        assert config['config']['Entrypoint'] == ['python3', '/app/harness.py']
        assert not any(key in item.upper() for item in config['config'].get('Env', ())
                       for key in ('API_KEY=', 'AUTH_TOKEN=', 'PASSWORD=', 'SECRET='))
        for layer_path in image['Layers']:
            with tarfile.open(fileobj=archive.extractfile(layer_path), mode='r|*') as layer:
                for member in layer:
                    name = member.name.removeprefix('./').lstrip('/')
                    assert name.split('/')[0] not in ('data', 'model', 'previous', 'state', 'wheels'), name
                    if member.isdir():
                        if name.endswith('.dist-info'):
                            dependency_dirs.add(name.rsplit('/', 1)[-1])
                        continue
                    if name.startswith('app/'):
                        assert member.isfile(), name
                        app[name.removeprefix('app/')] = layer.extractfile(member).read()
                    elif name.startswith(sdk_root):
                        assert member.isfile(), name
                        sdk[name.removeprefix(sdk_root)] = layer.extractfile(member).read()
                    elif member.isfile() and '/site-packages/numpy/' not in name:
                        basename = name.rsplit('/', 1)[-1].lower()
                        assert not basename.endswith(('.npz', '.safetensors', '.onnx', '.ckpt', '.pt')), name
                        assert basename not in ('.env', '.netrc', 'id_rsa', '.git-credentials'), name
        assert app == expected_app, 'archive source differs from audited build source'
        assert sdk == expected_sdk, 'installed SDK source differs from official release'
        assert 'numpy-2.2.6.dist-info' in dependency_dirs
        assert {x for x in dependency_dirs if x.startswith('contest_harness-')} == {'contest_harness-1.0.4.dist-info'}
    return {
        'archive': archive_path.name,
        'bytes': archive_path.stat().st_size,
        'sha256': digest,
        'image': expected_tag,
        'manifest': entry['digest'],
        'platform': 'linux/amd64',
        'app_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in sorted(app.items())},
        'sdk_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in sorted(sdk.items())},
    }


if __name__ == '__main__':
    paths = [Path(path) for path in sys.argv[1:4]]
    values = verify(*paths, sys.argv[4] if len(sys.argv) > 4 else None)
    print(json.dumps(values, indent=2))
