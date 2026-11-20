FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BASELINE_CPU_THREADS=1

RUN --mount=type=bind,source=wheels,target=/wheels \
    python -m pip install --no-index --no-deps --no-compile --no-cache-dir --find-links=/wheels \
        contest-harness==1.0.4 numpy==2.2.6

WORKDIR /app
COPY app/ /app/
ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]
