$Host.UI.RawUI.WindowTitle = 'PyTRiO Login'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = 'utf-8'
Write-Host 'Copy your API Key from https://pytrio.com/dashboard'
Write-Host 'Paste below and press Enter. The key will remain hidden.'
& '.\.venv\Scripts\trio.exe' login --base-url https://pytrio.com
if ($LASTEXITCODE -eq 0) {
    & '.\.venv\Scripts\python.exe' scripts/pytrio_check.py
}
Write-Host 'You can leave this window open. Do not send your API Key in chat.'
