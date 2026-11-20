# 智训引擎 r7 本地构建

构建上下文的 `app/` 包含 Linux/amd64 归档中的全部 12 个运行源码文件。镜像通过 `python3 /app/harness.py` 提供赛事的 `describe`、`train` 和 `predict` 入口。Qwen/Qwen3.8-27B 由赛事宿主代理调用。镜像和本目录的源码不需要密钥。

r7 相对 r6 只改「不整场失败」这一条：

- 模型回执正文不可用（补问一次后仍不可用）时，改用已审阅的兜底决策（A：`epochs 90`、`lr 0.003`；B：`lr 0.003`、`epochs 5`、`edits []`；C：`epochs 10`、`lr 0.003`；均 `improve_primary`），并在 `training.json`、`finish` 的 `configuration` 写 `model_fallback: true`，stderr 打印原因。
- 轮内训练/导出/校验异常时，改为选择父检查点完成本轮（`configuration.degraded: true`，`strategy: degrade_to_parent`，带异常类型与详情），整场继续。
- 宿主 RPC 结果未知（`RPC_OUTCOME_UNKNOWN`）仍按官方要求中止本轮，不做兜底。

## 依赖

Dockerfile 固定 Python 3.10 Slim 基础镜像的摘要。把下列两个 wheel 放在本目录的 `wheels/` 中。

| 安装包 | 版本 | 本次构建所用 SHA-256 |
| --- | --- | --- |
| contest_harness-1.0.4-py3-none-any.whl | 1.0.4 | `f8ee56c6d3117f2c759936d64ef5e8908ef10363945da075abd4aee6723fda11` |
| numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl | 2.2.6 | `fc7b73d02efb0e18c000e9ad8b83480dfcd5dfd11065997ed4c6747470ae8915` |

SDK wheel 仍由[赛事官网 SDK 发行包](https://rsi-index.115.zone/downloads/contest-harness-sdk-1.0.4.tar.gz)构建。发行包 SHA-256 为 `7afd282d6b5fd889d64d0515a67d169806970e36200689b18d2040026167a4af`。二次构建 wheel 时，压缩包元数据可能影响 wheel 摘要，可逐文件核对安装后的 SDK 源码。NumPy wheel 来自 [PyPI 官方发行页](https://pypi.org/project/numpy/2.2.6/#files)。

在已缓存基础镜像、两个 wheel 均已备齐的 Docker 环境中，从本目录运行下列命令。

```sh
docker build --pull=false --network none --provenance=false --platform linux/amd64 -t zhixun-engine:sdk104-r7-20261120 .
docker run --rm --network none --read-only zhixun-engine:sdk104-r7-20261120 describe
docker save --platform linux/amd64 -o zhixun-engine-sdk104-r7-20261120.tar zhixun-engine:sdk104-r7-20261120
```

归档 `zhixun-engine-sdk104-r7-20261120.tar` 的 SHA-256 为 `3c73a72a7277b8c46c4d8e239ee29c5ceec5fa9e1217da85a20809b94da56e43`，61,884,928 字节，OCI manifest `sha256:e0a14cd3a91ef7e2c6a5c92a3f3fc053734bb6f58001d21c95dc62b7509176eb`。重新构建可能得到不同的归档摘要，运行文件见 `IMAGE-SOURCE.sha256`。训练和预测可用公开 CPU 数据包和本地宿主回放。实际评测由赛事宿主提供父检查点、模型回执和验证分数。

## 验证记录

[本地复跑与逐轮记录](logs/README.md)和[SwanLab 记录](https://swanlab.cn/@hequan/zhixunyinqing/runs/x93cs9sv)均对应本版镜像，使用模拟模型回执。
