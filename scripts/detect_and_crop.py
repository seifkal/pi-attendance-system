#!/usr/bin/env python3
"""
Face Detection and Cropping Script
For Smart Class Camera - Bachelor Thesis Project

This script detects faces using the trained YOLO model and crops them
for subsequent face recognition using ArcFace.
"""

import argparse
import cv2
import os
from pathlib import Path
from datetime import datetime


def check_dependencies():
    """Check if required packages are installed."""
    try:
        from ultralytics import YOLO
        import cv2
        return True
    except ImportError as e:
        print("=" * 60)
        print(f"ERROR: Missing dependency - {e}")
        print("Please install required packages:")
        print("  pip install ultralytics opencv-python")
        print("=" * 60)
        return False


def load_model(model_path: str):
    """Load the trained YOLO model."""
    from ultralytics import YOLO
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return None
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    return model


def crop_face(image, box, padding: float = 0.2):
    """
    Crop a face from the image with optional padding.
    
    Args:
        image: Input image (numpy array)
        box: Bounding box [x1, y1, x2, y2]
        padding: Padding ratio to add around the face (0.2 = 20%)
    
    Returns:
        Cropped face image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    # Calculate box dimensions
    box_w = x2 - x1
    box_h = y2 - y1
    
    # Add padding
    pad_w = int(box_w * padding)
    pad_h = int(box_h * padding)
    
    # Expand box with padding, ensuring bounds
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    # Crop
    cropped = image[y1:y2, x1:x2]
    
    return cropped


def detect_and_crop(
    model,
    source: str,
    output_dir: str,
    conf_threshold: float = 0.5,
    padding: float = 0.2,
    save_annotated: bool = True,
    resize_face: tuple = None
):
    """
    Detect faces in images and save cropped faces.
    
    Args:
        model: Loaded YOLO model
        source: Input image path, directory, or video
        output_dir: Output directory for cropped faces
        conf_threshold: Confidence threshold for detections
        padding: Padding ratio around detected faces
        save_annotated: Whether to save annotated images with bounding boxes
        resize_face: Optional tuple (width, height) to resize cropped faces
    """
    output_path = Path(output_dir)
    faces_dir = output_path / "cropped_faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    
    if save_annotated:
        annotated_dir = output_path / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
    
    # Run detection
    print(f"\nRunning face detection on: {source}")
    print(f"Confidence threshold: {conf_threshold}")
    
    results = model.predict(
        source=source,
        conf=conf_threshold,
        save=False,
        verbose=False
    )
    
    total_faces = 0
    
    for result in results:
        # Get image info
        img_path = Path(result.path)
        img_name = img_path.stem
        
        # Load original image for cropping
        img = cv2.imread(str(result.path))
        if img is None:
            print(f"Warning: Could not load image {result.path}")
            continue
        
        boxes = result.boxes
        num_faces = len(boxes)
        
        if num_faces > 0:
            print(f"  {img_name}: {num_faces} face(s) detected")
        
        # Process each detection
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Crop face
            cropped = crop_face(img, xyxy, padding=padding)
            
            # Optionally resize
            if resize_face is not None:
                cropped = cv2.resize(cropped, resize_face)
            
            # Save cropped face
            face_filename = f"{img_name}_face{i:03d}_conf{conf:.2f}.jpg"
            face_path = faces_dir / face_filename
            cv2.imwrite(str(face_path), cropped)
            
            total_faces += 1
        
        # Save annotated image
        if save_annotated and num_faces > 0:
            annotated_img = result.plot()
            annotated_path = annotated_dir / f"{img_name}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated_img)
    
    print("\n" + "=" * 60)
    print("DETECTION COMPLETE!")
    print("=" * 60)
    print(f"Total faces detected: {total_faces}")
    print(f"Cropped faces saved to: {faces_dir}")
    if save_annotated:
        print(f"Annotated images saved to: {annotated_dir}")
    
    return total_faces


def process_video(
    model,
    video_path: str,
    output_dir: str,
    conf_threshold: float = 0.5,
    padding: float = 0.2,
    frame_skip: int = 10,
    resize_face: tuple = None
):
    """
    Process video and extract faces from frames.
    
    Args:
        model: Loaded YOLO model
        video_path: Path to input video
        output_dir: Output directory
        conf_threshold: Confidence threshold
        padding: Padding ratio around faces
        frame_skip: Process every Nth frame
        resize_face: Optional resize dimensions for faces
    """
    output_path = Path(output_dir)
    faces_dir = output_path / "cropped_faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nProcessing video: {video_path}")
    print(f"  FPS: {fps}, Total frames: {total_frames}")
    print(f"  Processing every {frame_skip} frames")
    
    frame_num = 0
    total_faces = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % frame_skip == 0:
            # Run detection on frame
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    # Crop face
                    cropped = crop_face(frame, xyxy, padding=padding)
                    
                    if resize_face:
                        cropped = cv2.resize(cropped, resize_face)
                    
                    # Save with timestamp
                    timestamp = frame_num / fps
                    face_filename = f"frame{frame_num:06d}_t{timestamp:.2f}s_face{i:03d}.jpg"
                    cv2.imwrite(str(faces_dir / face_filename), cropped)
                    total_faces += 1
        
        frame_num += 1
        
        # Progress update
        if frame_num % 100 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames, {total_faces} faces found")
    
    cap.release()
    
    print("\n" + "=" * 60)
    print("VIDEO PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Frames processed: {total_frames // frame_skip}")
    print(f"Total faces extracted: {total_faces}")
    print(f"Saved to: {faces_dir}")
    
    return total_faces


def main():
    parser = argparse.ArgumentParser(
        description="Detect and crop faces using trained YOLO model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Input source: image path, directory, or video file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='/Users/saifqal/Project/models/face_detector/weights/best.pt',
        help='Path to trained YOLO model'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='/Users/saifqal/Project/outputs',
        help='Output directory for cropped faces'
    )
    
    # Detection parameters
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--padding', type=float, default=0.2, help='Padding ratio around faces')
    parser.add_argument('--no-annotated', action='store_true', help='Do not save annotated images')
    
    # Face preprocessing
    parser.add_argument(
        '--resize',
        type=int,
        nargs=2,
        default=None,
        metavar=('WIDTH', 'HEIGHT'),
        help='Resize cropped faces to specific dimensions (for ArcFace: 112 112)'
    )
    
    # Video options
    parser.add_argument('--frame-skip', type=int, default=10, help='Process every Nth frame for video')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return
    
    # Determine if source is video
    source_path = Path(args.source)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    resize_tuple = tuple(args.resize) if args.resize else None
    
    if source_path.suffix.lower() in video_extensions:
        # Process video
        process_video(
            model=model,
            video_path=args.source,
            output_dir=args.output,
            conf_threshold=args.conf,
            padding=args.padding,
            frame_skip=args.frame_skip,
            resize_face=resize_tuple
        )
    else:
        # Process image(s)
        detect_and_crop(
            model=model,
            source=args.source,
            output_dir=args.output,
            conf_threshold=args.conf,
            padding=args.padding,
            save_annotated=not args.no_annotated,
            resize_face=resize_tuple
        )


if __name__ == "__main__":
    main()
