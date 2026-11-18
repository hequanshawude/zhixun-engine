# 2026-11-18 智训引擎参赛镜像源码

镜像 `zhixun-engine:ac-stateful-v2-local-20261118` 包含 A、B、C 三组，基于赛事 SDK 1.0.2 和 `contest.harness.workspace.v2` 协议。容器通过宿主模型网关调用 `Qwen/Qwen3.8-27B`，本地训练和保存 CPU 模型。

`sdk_app_v2/` 是镜像最终 `/app` 中的源码。A、C 使用支持恢复训练的新方案，B 沿用此前验证的训练流程。`app/` 是构建 SDK 基础层时使用的源码，最终层会覆盖其中更新的文件。`IMAGE-SOURCE.sha256` 可用于核对最终源码。

| 组别 | 本地公开验证结果 | 云端实验 |
| --- | ---: | --- |
| A | Macro-F1 0.884841 | [SwanLab](https://swanlab.cn/@hequan/zhixunyinqing/runs/n2xbt11k) |
| B | RMSE 397.174829 | [SwanLab](https://swanlab.cn/@hequan/zhixunyinqing/runs/1jj8a8d1) |
| C | RMSE 46.640542 | [SwanLab](https://swanlab.cn/@hequan/zhixunyinqing/runs/yngkrc5t) |

三组实验各记录了 10 次真实 Qwen 调用。这些分数来自本地公开验证。正式成绩和运行时限以赛事平台评测为准。A、C 续训时需同时保留 `checkpoint.npz` 和对应训练状态。

## 构建与校验

在本目录执行。原构建的前置镜像 `zhixun-engine:groupguard-27b-local-20261116` 仅用于复制 NumPy 2.2.6 的运行文件。该前置镜像的构建配方见 [11 月 16 日版本](https://github.com/hequanshawude/zhixun-engine/blob/6ab6934/Dockerfile.groupguard-27b-20261116)，它需要先在同一提交的仓库根目录用 `docker build -t zhixun-engine:preflight .` 构建基础镜像。另从赛事工作台获取官方 SDK 1.0.2，用 `python -m pip wheel --no-deps --wheel-dir wheels <解压后的 SDK 根目录>` 生成 `wheels/contest_harness-1.0.2-py3-none-any.whl`。

```sh
docker build --platform linux/amd64 -f Dockerfile.sdk-base -t zhixun-engine:sdk-1.0.2-local-20261118 .
docker build --platform linux/amd64 -t zhixun-engine:ac-stateful-v2-local-20261118 .
docker run --rm --network none zhixun-engine:ac-stateful-v2-local-20261118 describe
docker save -o zhixun-engine-ac-stateful-sdk-20261118.tar zhixun-engine:ac-stateful-v2-local-20261118
```

已核对的本地归档文件为 `zhixun-engine-ac-stateful-sdk-20261118.tar`，大小 64,629,760 字节，SHA-256 为 `1311c1dc2d0545ab0e0c834facb789e1936a6aced2acd93302aec0986ec50aea`。重新构建的归档会受基础镜像和构建环境影响，可用 `describe` 和源码校验值核对内容。
