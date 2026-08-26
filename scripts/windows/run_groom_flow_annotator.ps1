[CmdletBinding()]
param(
    [string]$InputDir = 'D:\RTS\datasets\panda_r068_input_v1\images',
    [string]$OutputDir = 'D:\RTS\datasets\panda_r068_manual_flow'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = 'D:\Users\namew\miniconda3\envs\mygs\python.exe'
$tool = Join-Path $projectRoot 'tools\groom_flow_annotator.py'

foreach ($path in @($python, $tool)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing required file: $path"
    }
}
if (-not (Test-Path -LiteralPath $InputDir -PathType Container)) {
    throw "Input image folder does not exist: $InputDir"
}
if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) {
    [System.IO.Directory]::CreateDirectory($OutputDir) | Out-Null
}

$arguments = [string[]]@(
    '-B', $tool,
    '--input-dir', (Resolve-Path -LiteralPath $InputDir).Path,
    '--output-dir', (Resolve-Path -LiteralPath $OutputDir).Path
)
& $python @arguments
$exitCode = [int]$LASTEXITCODE
exit $exitCode
