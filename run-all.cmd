@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv\Scripts\python.exe not found.
  echo         Run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
  exit /b 2
)

".venv\Scripts\python.exe" "scripts\run_all.py" %*
exit /b %ERRORLEVEL%
