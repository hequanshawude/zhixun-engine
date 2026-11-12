@echo off
chcp 65001 >nul
title SwanLab Cloud Login
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo 请先在 SwanLab 网页登录，在账号设置中复制 API Key。
echo 然后在下方提示处粘贴并按回车，密钥不会明文显示。
echo.
".venv\Scripts\swanlab.exe" login
if errorlevel 1 goto failed
".venv\Scripts\swanlab.exe" verify
if errorlevel 1 goto failed
echo.
echo 云端账号验证成功，可以关闭此窗口。
pause
exit /b 0
:failed
echo.
echo 登录或验证未成功，请保留错误提示以便排查。不要发送 API Key。
pause
exit /b 1
