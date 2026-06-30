$ErrorActionPreference = "Stop"
$VenvDir = ".venv"

# Make sure python exists first
$pythonExists = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonExists) {
    Write-Host "Error: python not found. Install Python 3 first."
    exit 1
}

# Make sure we're in the right folder
if (-not (Test-Path "requirements.txt")) {
    Write-Host "Error: no requirements.txt found in $(Get-Location)."
    exit 1
}

# Create the venv if it doesn't exist
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

# Activate the venv
$activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
. $activateScript

Write-Host "Upgrading pip ..."
python -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt ..."
pip install -r requirements.txt

# Create .env if .env.example exists
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "Created .env from .env.example. Add your GEMINI_API_KEY to it."
    } else {
        Write-Host "No .env.example found, skipping .env creation."
    }
} else {
    Write-Host ".env already exists, leaving it as is."
}

Write-Host "Setup complete."
Write-Host "Activate with: .\$VenvDir\Scripts\Activate.ps1"
Write-Host "Run with: python simulation\main.py"