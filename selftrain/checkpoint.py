"""仅加载本程序生成的可信本地检查点；模型状态使用 pickle，禁止接收外部文件。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

import numpy as np


STATE_FIELDS = (
    'drift_u', 'drift_l', 'budget', 'playbook', 'registry', 'anchor', 'static',
    'best_version', 'history_X', 'history_y', 'bootstrap_X', 'bootstrap_y',
    'bootstrap_prior', 'decisions', 'scores', 'best_primary', 'initialization',
)


def data_hash(X, y, index=None):
    digest = hashlib.sha256(str(index).encode())
    for value in (X, y):
        if value is None:
            digest.update(b'none')
            continue
        array = np.asarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        if array.dtype.hasobject:
            digest.update(json.dumps(array.tolist(), ensure_ascii=False).encode('utf-8'))
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CheckpointStore:
    def __init__(self, path, system, bootstrap_data=None):
        self.path = Path(path)
        config = system.cfg.to_dict()
        config.pop('log')
        config.pop('playbook_path')
        config['harness'].pop('api_key')
        sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path(__file__).parent.glob('*.py')}
        versions = {}
        for package in ('numpy', 'scikit-learn', 'lightgbm'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        self.compatibility = {
            'format': 1, 'python': sys.version.split()[0], 'platform': sys.platform,
            'versions': versions, 'sources': sources, 'spec': asdict(system.spec),
            'config_hash': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            'policy': system.policy.mode,
            'external_training_hash': data_hash(*bootstrap_data) if bootstrap_data is not None else None,
        }
        self.consumed = []

    def save(self, system, outputs=None):
        state = {name: getattr(system, name) for name in STATE_FIELDS}
        # 不序列化 logger、emit、配置或 LLMClient，避免保存凭据和不可恢复的句柄。
        state['policy_state'] = {name: getattr(system.policy, name)
                                 for name in ('_cache', '_fail_streak', '_disabled')}
        state['consumed'] = self.consumed
        state['parent_run_id'] = system.logger.run_id
        state['outputs'] = outputs
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        header = {'compatibility': self.compatibility,
                  'payload_sha256': hashlib.sha256(payload).hexdigest()}
        atomic_write(self.path, json.dumps(header, sort_keys=True).encode() + b'\n' + payload)

    def load(self, system):
        header_raw, payload = self.path.read_bytes().split(b'\n', 1)
        header = json.loads(header_raw)
        if header.get('compatibility') != self.compatibility:
            raise ValueError('检查点与当前代码、依赖、任务、训练数据或配置不一致，拒绝恢复')
        if hashlib.sha256(payload).hexdigest() != header.get('payload_sha256'):
            raise ValueError('检查点损坏，拒绝恢复')
        state = pickle.loads(payload)
        for name in STATE_FIELDS:
            setattr(system, name, state[name])
        system.playbook.path = Path(system.cfg.playbook_path) if system.cfg.playbook_path else None
        system.policy.playbook = system.playbook
        for name, value in state['policy_state'].items():
            setattr(system.policy, name, value)
        self.consumed = state['consumed']
        return {'parent_run_id': state['parent_run_id'],
                'completed_batches': len(system.decisions),
                'outputs': state['outputs'],
                'last_consumed_index': self.consumed[-1]['index'] if self.consumed else None}
