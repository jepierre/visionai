param(
    [ValidateSet('cu126', 'cu128')]
    [string]$CudaWheel = 'cu126'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot '.venv'

if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

$python = Join-Path $venvPath 'Scripts\python.exe'
& $python -m pip install --upgrade pip

# Driver 566.26 is below the cu128 reference project's stated driver target.
# cu126 is therefore the default; only choose cu128 after upgrading the driver.
& $python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaWheel"
& $python -m pip install -r (Join-Path $projectRoot 'requirements-phase0.txt')

Write-Host "Environment created: $venvPath"
Write-Host "Next: copy .env.example to .env, authenticate with 'huggingface-cli login', then run scripts\validate_environment.py."
