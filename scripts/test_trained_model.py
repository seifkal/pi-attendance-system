#!/usr/bin/env python3
"""
Test your trained YOLOv8 face detection model
"""

from ultralytics import YOLO
from pathlib import Path

def main():
    # Path to your trained model
    model_path = '/Users/saifqal/Project/models/best.pt'
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found at: {model_path}")
        print("Please download it first using:")
        print("scp -P 12572 root@103.196.86.84:/workspace/runs/train/widerface_yolov8n/weights/best.pt /Users/saifqal/Project/models/")
        return
    
    print("="*60)
    print("🚀 Loading trained model...")
    print("="*60)
    
    # Load your trained model
    model = YOLO(model_path)
    
    print(f"✅ Model loaded successfully!")
    print(f"📁 Model path: {model_path}")
    
    # Validate on validation set
    print("\n" + "="*60)
    print("📊 Running validation on test set...")
    print("="*60)
    
    results = model.val(
        data='/Users/saifqal/Project/config/data.yaml',
        imgsz=640,
        batch=16,
        device='',  # Auto-detect (use GPU if available)
    )
    
    # Print results
    print("\n" + "="*60)
    print("📈 VALIDATION RESULTS")
    print("="*60)
    print(f"mAP50:     {results.box.map50:.3f} ({results.box.map50*100:.1f}%)")
    print(f"mAP50-95:  {results.box.map:.3f} ({results.box.map*100:.1f}%)")
    print(f"Precision: {results.box.mp:.3f} ({results.box.mp*100:.1f}%)")
    print(f"Recall:    {results.box.mr:.3f} ({results.box.mr*100:.1f}%)")
    print("="*60)
    
    # Test on a single image
    print("\n" + "="*60)
    print("🖼️  Testing on a sample image...")
    print("="*60)
    
    # Find a test image
    test_image = list(Path('/Users/saifqal/Project/datasets/wider_YOLO/widerface_yolo_fixed/val/images').glob('*.jpg'))[0]
    
    print(f"Test image: {test_image.name}")
    
    # Run inference
    detect_results = model(str(test_image))
    
    # Show results
    num_faces = len(detect_results[0].boxes)
    print(f"✅ Detected {num_faces} face(s)")
    
    # Save result
    output_path = Path('/Users/saifqal/Project/outputs/test_detection.jpg')
    output_path.parent.mkdir(exist_ok=True)
    
    detect_results[0].save(str(output_path))
    print(f"📁 Result saved to: {output_path}")
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the saved image at outputs/test_detection.jpg")
    print("2. Use this model in your detect_and_crop.py script")
    print("3. Integrate with ArcFace for face recognition")

if __name__ == "__main__":
    main()
