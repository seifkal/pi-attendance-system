#!/usr/bin/env python3
"""
YOLO Face Detection Training Script - RunPod Version
For Smart Class Camera - Bachelor Thesis Project

This script trains a YOLOv8 model for face detection on RunPod.
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO


def train(args):
    """Train the YOLO face detection model."""
    
    # Load the base model
    model_variant = f"yolov8{args.model}.pt"
    print(f"\n{'='*60}")
    print(f"Loading base model: {model_variant}")
    print(f"{'='*60}\n")
    model = YOLO(model_variant)
    
    # Training configuration
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'patience': args.patience,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'plots': True,
        'val': True,
    }
    
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # Start training
    print("🚀 Starting training...")
    results = model.train(**train_args)
    
    # Print results
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 Best model saved to: {results.save_dir}/weights/best.pt")
    print(f"📁 Last model saved to: {results.save_dir}/weights/last.pt")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for face detection on RunPod",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset
    parser.add_argument(
        '--data', 
        type=str, 
        default='/workspace/config/data_runpod.yaml',
        help='Path to dataset configuration YAML'
    )
    
    # Model
    parser.add_argument(
        '--model', 
        type=str, 
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLOv8 model variant (n=nano, s=small, m=medium, l=large, x=xlarge)'
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='0', help='GPU device (0 for first GPU)')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loader workers')
    
    # Output
    parser.add_argument('--project', type=str, default='/workspace/runs/train', help='Project save directory')
    parser.add_argument('--name', type=str, default='widerface_yolov8n', help='Experiment name')
    
    args = parser.parse_args()
    
    # Train
    train(args)


if __name__ == "__main__":
    main()
