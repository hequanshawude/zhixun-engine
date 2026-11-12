$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$dockerBin = 'C:\Program Files\Docker\Docker\resources\bin'
if (Test-Path $dockerBin) { $env:Path = "$dockerBin;$env:Path" }
# 仅当前构建进程使用电脑现有的本地代理，不修改系统网络设置。
$proxy = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings'
if ($proxy.ProxyEnable -eq 1 -and $proxy.ProxyServer -match '^127\.0\.0\.1:\d+$') {
    $env:HTTP_PROXY = "http://$($proxy.ProxyServer)"
    $env:HTTPS_PROXY = $env:HTTP_PROXY
}
docker build -t zhixun-engine:preflight .
exit $LASTEXITCODE
