# setup.ps1 - sets up Python virtual environment

# Create virtual environment
python -m venv venv
Write-Host "Virtual environment 'venv' created."

# Instructions to activate
Write-Host "Activate the virtual environment with:"
Write-Host ".\venv\Scripts\Activate.ps1"

# Install packages if requirements.txt exists
if (Test-Path "requirements.txt") {
    Write-Host "Installing dependencies from requirements.txt..."
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    Write-Host "Dependencies installed!"
} else {
    Write-Host "No requirements.txt found yet. You can add packages later."
}

Write-Host "Setup complete!"
