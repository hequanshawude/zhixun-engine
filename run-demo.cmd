@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -m selftrain demo --config configs/local.yaml --pred-dir out/preds
pause
