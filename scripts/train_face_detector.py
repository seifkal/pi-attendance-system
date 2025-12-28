#!/usr/bin/env python3
"""
YOLO Face Detection Training Script
For Smart Class Camera - Bachelor Thesis Project

This script trains a YOLOv8 model for face detection using the WIDER Face dataset.
The trained model will be used to detect and crop faces for ArcFace recognition.
"""

import argparse
import os
from pathlib import Path

def check_dependencies():
    """Check if ultralytics package is installed."""
    try:
        from ultralytics import YOLO
        return True
    except ImportError:
        print("=" * 60)
        print("ERROR: ultralytics package not found!")
        print("Please install it using:")
        print("  pip install ultralytics")
        print("=" * 60)
        return False


def validate_dataset(data_yaml: str):
    """Validate dataset configuration and structure."""
    import yaml
    
    if not os.path.exists(data_yaml):
        print(f"ERROR: Dataset config not found: {data_yaml}")
        return False
    
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = Path(config['path'])
    train_path = base_path / config['train']
    val_path = base_path / config['val']
    
    print("Validating dataset...")
    print(f"  Base path: {base_path}")
    print(f"  Train path: {train_path}")
    print(f"  Val path: {val_path}")
    
    if not train_path.exists():
        print(f"ERROR: Training images not found at {train_path}")
        return False
    
    if not val_path.exists():
        print(f"ERROR: Validation images not found at {val_path}")
        return False
    
    train_count = len(list(train_path.glob("*.jpg")))
    val_count = len(list(val_path.glob("*.jpg")))
    
    print(f"  Training images: {train_count}")
    print(f"  Validation images: {val_count}")
    print("Dataset validation: PASSED ✓")
    
    return True


def train(args):
    """Train the YOLO face detection model."""
    from ultralytics import YOLO
    
    # Load the base model
    model_variant = f"yolov8{args.model}.pt"
    print(f"\nLoading base model: {model_variant}")
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
    }
    
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")
    
    # Start training
    print("Starting training...")
    results = model.train(**train_args)
    
    # Print results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    print(f"Last model saved to: {results.save_dir}/weights/last.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for face detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset
    parser.add_argument(
        '--data', 
        type=str, 
        default='/Users/saifqal/Project/config/data.yaml',
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
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='', help='Device (cuda:0, cpu, or empty for auto)')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loader workers')
    
    # Output
    parser.add_argument('--project', type=str, default='/Users/saifqal/Project/models', help='Project save directory')
    parser.add_argument('--name', type=str, default='face_detector', help='Experiment name')
    
    # Modes
    parser.add_argument('--validate-only', action='store_true', help='Only validate dataset, do not train')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Validate dataset
    if not validate_dataset(args.data):
        return
    
    if args.validate_only:
        print("\nValidation complete. Use without --validate-only to start training.")
        return
    
    # Train
    train(args)


if __name__ == "__main__":
    main()
