@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo 即将运行 36 次合成数据对照，保存到 SwanLab 私有项目；需要保持联网。
".venv\Scripts\python.exe" scripts/benchmark.py --cloud
if errorlevel 1 goto failed
echo 报告已生成，路径见上方输出。
pause
exit /b 0
:failed
echo 部分实验失败，请查看输出及报告中的失败记录。
pause
exit /b 1
