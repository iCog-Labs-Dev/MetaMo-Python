$ErrorActionPreference = "Stop"
$VenvDir = ".venv"

# check if python exists
$pythonExists = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonExists) {
    Write-Host "Error: python not found. Install Python 3 first."
    exit 1
}

# check if we are in the right folder
if (-not (Test-Path "requirements.txt")) {
    Write-Host "Error: no requirements.txt found in $(Get-Location)."
    exit 1
}

# create the venv if it doesn't exist
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment in $VenvDir ..."
    python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create the virtual environment."
        exit 1
    }
} else {
    Write-Host "'$VenvDir' already exists, creation skipped."
}

# activate the venv
$activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
. $activateScript

Write-Host "Upgrading pip ..."
python -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt ..."
pip install -r requirements.txt

Write-Host "Setup complete."
Write-Host "Activate with: .\$VenvDir\Scripts\Activate.ps1"
Write-Host "Run with: python usecase\simulation\main.py"