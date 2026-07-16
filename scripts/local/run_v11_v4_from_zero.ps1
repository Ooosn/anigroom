[CmdletBinding()]
param(
    [string]$RunId = "",
    [string]$DataRoot = "D:\petsgaussianhair_v11_repro\data\neuralfur_work\whiteTiger_processed\roaringwalk",
    [ValidateRange(1, 32)]
    [double]$GpuMemoryLimitGb = 25,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"

function Convert-ToBashPath([string]$PathValue) {
    return $PathValue.Replace("\", "/")
}

$projectRoot = "D:\petsgaussianhair"
$python = "D:\Users\namew\miniconda3\envs\mygs\python.exe"
$condaEnv = Split-Path -Parent $python
$condaHook = "D:\Users\namew\miniconda3\shell\condabin\conda-hook.ps1"
$bash = "C:\Program Files\Git\bin\bash.exe"
$runner = Join-Path $projectRoot "scripts\server\run_v11_v4_from_zero.sh"
$meshPath = Join-Path $projectRoot "data_sources\neuralfur_official_results\whiteTiger\furless_reshaped.obj"

foreach ($required in @($projectRoot, $python, $condaHook, $bash, $runner, $DataRoot, $meshPath)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required local v11-v4 path not found: $required"
    }
}

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = Get-Date -Format "yyyyMMddHHmmss"
}
if ($RunId -notmatch '^\d{14}$') {
    throw "RunId must be a 14-digit timestamp"
}

$runRoot = Join-Path $projectRoot ("outputs\" + $RunId)
$logRoot = Join-Path $projectRoot ("logs\" + $RunId)
New-Item -ItemType Directory -Force -Path $runRoot, $logRoot | Out-Null

& $condaHook
conda activate mygs
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "MSVC cl.exe is unavailable after activating mygs"
}

$env:PATH = "$condaEnv;$condaEnv\Scripts;$condaEnv\Library\bin;$env:PATH"
$env:PYTHONPATH = "$projectRoot;$env:PYTHONPATH"
$env:PROJECT_ROOT = Convert-ToBashPath $projectRoot
$env:PYTHON = Convert-ToBashPath $python
$env:DATA_ROOT = Convert-ToBashPath $DataRoot
$env:MESH_PATH = Convert-ToBashPath $meshPath
$env:RUN_ID = $RunId
$env:RUN_ROOT = Convert-ToBashPath $runRoot
$env:LOG_ROOT = Convert-ToBashPath $logRoot
$env:GPU_MEMORY_LIMIT_GB = [string]$GpuMemoryLimitGb
$env:VERIFY_ONLY = if ($VerifyOnly) { "1" } else { "0" }

Write-Host "[v11-v4-local] run_id=$RunId"
Write-Host "[v11-v4-local] gpu_memory_limit_gb=$GpuMemoryLimitGb"
Write-Host "[v11-v4-local] data_root=$DataRoot"
Write-Host "[v11-v4-local] run_root=$runRoot"

& $bash (Convert-ToBashPath $runner)
exit $LASTEXITCODE
