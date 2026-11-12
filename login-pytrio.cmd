@echo off
chcp 65001 >nul
title PyTRiO Login
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo 请从 https://pytrio.com/dashboard 复制 API Key。
echo 在下方粘贴并回车。输入不会显示，请不要把密钥发送到聊天中。
".venv\Scripts\trio.exe" login --base-url https://pytrio.com
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\pytrio_check.py
if errorlevel 1 goto failed
echo PyTRiO 连接验证成功，可以关闭此窗口。
pause
exit /b 0
:failed
echo 登录或连接验证未成功，请保留窗口以便排查。
pause
exit /b 1
