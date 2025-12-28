#!/bin/bash

# RunPod Setup Script for YOLO Face Detection Training
# This script sets up the environment on RunPod

echo "========================================="
echo "RunPod Environment Setup"
echo "========================================="

# Update pip
echo "📦 Updating pip..."
pip install --upgrade pip

# Install ultralytics and dependencies
echo "📦 Installing ultralytics (YOLOv8)..."
pip install ultralytics

# Verify installation
echo "✅ Verifying installation..."
python3 -c "from ultralytics import YOLO; print('✅ Ultralytics installed successfully!')"

# Check for GPU
echo "🖥️  Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Upload your dataset to /workspace/datasets/"
echo "2. Upload your config file to /workspace/config/"
echo "3. Run: python3 /workspace/scripts/train_runpod.py"
echo ""
