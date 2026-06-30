#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

# Make sure python exists first
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found. Install Python 3 first."
  exit 1
fi

# Make sure we're in the right folder
if [[ ! -f "requirements.txt" ]]; then
  echo "Error: no requirements.txt found in $(pwd)."
  exit 1
fi

# Create the venv if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment in $VENV_DIR ..."
  if ! python3 -m venv "$VENV_DIR" 2>/tmp/venv_error.log; then
    if grep -qi "ensurepip" /tmp/venv_error.log; then
      echo "The 'venv' module isn't installed for python3."
      echo "On Ubuntu/Debian, run: sudo apt update && sudo apt install python3-venv"
    else
      cat /tmp/venv_error.log
    fi
    rm -f /tmp/venv_error.log
    exit 1
  fi
  rm -f /tmp/venv_error.log
else
  echo "'$VENV_DIR' already exists, creation skipped."
fi

# Activate the venv
source "$VENV_DIR/bin/activate"

echo "Upgrading pip ..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt ..."
pip install -r requirements.txt

# Create .env if .env.example exists
if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    cp ".env.example" ".env"
    echo "Created .env from .env.example. Add your GEMINI_API_KEY to it."
  else
    echo "No .env.example found, skipping .env creation."
  fi
else
  echo ".env already exists, leaving it as is."
fi

echo "Setup complete."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run with: python simulation/main.py"