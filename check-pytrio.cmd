@echo off
chcp 65001 >nul
title PyTRiO Model Check
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Checking Qwen/Qwen3.8-27B with one short request...
".venv\Scripts\python.exe" scripts\pytrio_check.py --smoke
if errorlevel 1 (
  echo The model is still unavailable. See out\pytrio-setup\smoke.json.
) else (
  echo Model inference is working. The Gas comparison can now be repeated.
)
pause
