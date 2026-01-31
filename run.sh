#!/bin/bash

# Define environment directory
VENV_DIR="venv"

# check if python3 exists
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH."
    exit 1
fi

# 1. Create Virtual Environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        exit 1
    fi
    FRESH_INSTALL=true
else
    echo "Virtual environment already exists."
    FRESH_INSTALL=false
fi

# 2. Activate Virtual Environment
source "$VENV_DIR/bin/activate"

# 3. Install Dependencies
if [ "$FRESH_INSTALL" = true ] || [ -f "requirements.txt" ]; then
    echo "Installing/Updating dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies."
        exit 1
    fi
fi

# 4. Run the Application
echo "Starting Gaming PC Builder..."
streamlit run Allocator/streamlit_app.py
