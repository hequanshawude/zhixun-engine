"""在可终止的独立进程中执行本项目函数；pickle 文件仅用于本机内部通信。"""
from __future__ import annotations

import math
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class ExecutionTimeout(TimeoutError):
    pass


def isolated_call(function, *args, timeout_s):
    if timeout_s is None:
        return function(*args)
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ExecutionTimeout('执行时间额度已耗尽')
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='selftrain-worker-') as directory:
        request = Path(directory) / 'request.pkl'
        response = Path(directory) / 'response.pkl'
        with request.open('wb') as output:
            pickle.dump((function, args), output, protocol=pickle.HIGHEST_PROTOCOL)
        remaining = timeout_s - (time.perf_counter() - started)
        if remaining <= 0:
            raise ExecutionTimeout('准备训练数据时执行时间额度已耗尽')
        try:
            process = subprocess.run(
                [sys.executable, '-m', 'selftrain.worker', str(request), str(response)],
                cwd=Path(__file__).resolve().parent.parent,
                timeout=remaining, capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
            )
        except subprocess.TimeoutExpired as exc:
            # subprocess.run 已终止并等待这个工作进程，退出后清理临时文件。
            raise ExecutionTimeout(f'训练超过 {timeout_s:.2f} 秒，工作进程已终止') from exc
        if process.returncode != 0 or not response.exists():
            raise RuntimeError(f'训练工作进程异常退出（代码 {process.returncode}）')
        with response.open('rb') as source:
            ok, value = pickle.load(source)
        if not ok:
            raise RuntimeError(f'训练工作进程失败：{value}')
        return value


def main():
    request, response = map(Path, sys.argv[1:])
    try:
        with request.open('rb') as source:
            function, args = pickle.load(source)
        result = (True, function(*args))
    except Exception as exc:
        result = (False, type(exc).__name__ + ': ' + str(exc))
    with response.open('wb') as output:
        pickle.dump(result, output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
