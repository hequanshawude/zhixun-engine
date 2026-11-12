# 本地演练镜像；官方资源、镜像大小与模型打包约束待 SDK 发布确认。
FROM python:3.10-slim

# --- 运行期环境 ---
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app

# --- 依赖层（先装依赖，利用缓存）---
# 若现场无法联网，改用离线 wheels：
#   pip download -r requirements.txt -d wheels/
#   然后 build 时加 COPY wheels ./wheels 并用 --find-links ./wheels
COPY requirements.txt requirements-pytrio.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-pytrio.txt

# LightGBM 的 Linux 动态库依赖 OpenMP 运行库。
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN python -c "import numpy, pandas, sklearn, lightgbm, yaml, swanlab, pytrio"
# 两种 PyTRiO 策略模型的 tokenizer 构建时缓存，运行时无需访问模型仓库。
RUN python -c "from pytrio.model.tokenizer import get_tokenizer; [get_tokenizer(name) for name in ('Qwen/Qwen3.5-4B', 'Qwen/Qwen3.8-27B')]"

# --- 代码层 ---
COPY selftrain ./selftrain
COPY configs ./configs

# 评测数据与输出挂载点
RUN mkdir -p /data /out /app/runs
VOLUME ["/data", "/out"]

# 默认入口：赛规核心任务——从平台检查点续训 CPU 工业模型，跑满 10 轮。
#   /data 下可以放一个模型包，也可以放多个（每个子目录含 baselines/、public/、checkpoint.npz、training.json）；\n#   --widen auto 会按 profile 先做 Net2Net 加宽（A→64、B/C→128），这是本地验证过的主要收益来源
#   大模型不可用时自动降级为轮换表策略，保证无人干预也能跑完。
# 另一条入口（Harness 逐批预测）用法不变：
#   docker run ... run --data-dir /data --pred-dir /out --config /app/configs/offline.yaml
ENTRYPOINT ["python", "-m", "selftrain"]
CMD ["cpu", "--packages-root", "/data", "--out", "/out", "--rounds", "10", "--driver", "llm", "--widen", "auto"]
