@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -m selftrain demo --config configs/cloud.yaml --scenario mixed --batches 14 --seed 42 --pred-dir out/cloud-preds
if errorlevel 1 goto failed
echo.
echo 云端演练已结束，请打开上方“云端实验”链接查看结果。
pause
exit /b 0
:failed
echo.
echo 云端演练未完成，请保留错误提示以便排查。
pause
exit /b 1
