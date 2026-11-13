param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('b', 'c')]
    [string]$Profile,
    [ValidateRange(1, 10)]
    [int]$Rounds = 10,
    [switch]$NoCloud
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$package = Join-Path $projectRoot "data\raw\cpu\$Profile"
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment missing: $python" }
if (-not (Test-Path -LiteralPath $package)) { throw "Official CPU package missing: $package" }

$runName = '{0}-{1}-{2}' -f $Profile, (Get-Date -Format 'yyyyMMdd-HHmmss'), ([guid]::NewGuid().ToString('N').Substring(0, 8))
$out = Join-Path $projectRoot "out\local-27b\$runName"
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:ZHIXUN_SCHEME_V2 = '1'
$runArgs = @('-m', 'selftrain', 'cpu', '--package', $package, '--out', $out,
          '--rounds', [string]$Rounds, '--driver', 'llm', '--require-llm',
          '--model', 'trio://run_gxpt6e6iux26/sampler_weights/zhixun-cpu-policy-27b-v1',
          '--base-model', 'Qwen/Qwen3.8-27B', '--widen', 'auto')
if (-not $NoCloud) { $runArgs += '--cloud' }

Push-Location $projectRoot
try {
    & $python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "27B local run failed, exit code $LASTEXITCODE; output: $out" }
    $summary = Get-Content -LiteralPath (Join-Path $out 'summary.json') -Raw | ConvertFrom-Json
    if ($summary.llm_rounds -ne $Rounds -or $summary.tokens -le 0) {
        throw "27B decisions not verified; output: $out"
    }
    Write-Output "Verified 27B decisions: $($summary.llm_rounds)/$Rounds; tokens: $($summary.tokens); output: $out"
}
finally {
    Pop-Location
}
