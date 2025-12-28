#!/bin/bash
# Raspberry Pi 5 Setup Script
# Run this on your Raspberry Pi 5 after transferring the project files

set -e  # Exit on error

echo "=========================================="
echo "Raspberry Pi 5 Attendance System Setup"
echo "=========================================="

# Check if running on Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "Warning: This doesn't appear to be a Raspberry Pi"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo ""
echo "[1/6] Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo ""
echo "[2/6] Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    python3-picamera2 \
    sqlite3 \
    git

# Create virtual environment
echo ""
echo "[3/6] Creating Python virtual environment..."
if [ -d "venv_pi" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv_pi
    echo "✓ Virtual environment created"
fi

# Activate venv and install Python packages
echo ""
echo "[4/6] Installing Python packages..."
source venv_pi/bin/activate

pip install --upgrade pip
pip install \
    insightface \
    onnxruntime \
    numpy \
    opencv-python-headless \
    pandas \
    PyYAML \
    tqdm

echo "✓ Python packages installed"

# Create directories
echo ""
echo "[5/6] Creating project directories..."
mkdir -p databases
mkdir -p outputs
mkdir -p logs

# Test installation
echo ""
echo "[6/6] Testing installation..."
python3 << 'EOF'
try:
    import insightface
    import cv2
    import numpy as np
    print("✓ All imports successful!")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)
EOF

# Print summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Register students:"
echo "   source venv_pi/bin/activate"
echo "   python scripts/register_student.py --name \"Student Name\" --id \"STU001\" --webcam"
echo ""
echo "2. Run attendance system:"
echo "   python scripts/pi_attendance.py --db databases/students.db"
echo ""
echo "3. For headless operation:"
echo "   python scripts/pi_attendance.py --no-display"
echo ""
echo "See PI5_DEPLOYMENT.md for detailed instructions!"
echo "=========================================="
