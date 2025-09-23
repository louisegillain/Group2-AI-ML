#!/bin/bash
# setup.sh, sets up Python virtual environment

# Create virtual environmen
python3 -m venv venv
echo "Virtual environment 'venv' created."

# Instructions to activate
echo "Activate the virtual environment with:"
echo "source venv/bin/activate"
echo "If you have an error, see SETUP.md"

# Install packages if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    source venv/bin/activate
    pip install -r requirements.txt
    echo "Dependencies installed!"
else
    echo "No requirements.txt found yet. You can add packages later."
fi

echo "Setup complete!"
