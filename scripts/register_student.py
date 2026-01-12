#!/usr/bin/env python3
"""
Student Registration Script
For Smart Class Camera - Bachelor Thesis Project

Register students into the face recognition database.
Supports multiple input modes: directory of images, webcam capture.
"""

import argparse
import cv2
import os
import sys
from pathlib import Path
from typing import List
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from face_recognition import FaceRecognizer
from student_database import StudentDatabase
from detect_and_crop import crop_face, load_model as load_yolo

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def register_from_images(
    student_id: str,
    name: str,
    image_paths: List[str],
    face_recognizer: FaceRecognizer,
    database: StudentDatabase,
    yolo_model=None,
    min_confidence: float = 0.7,
    notes: str = None
) -> bool:
    """
    Register a student from a list of images.
    
    Args:
        student_id: Unique student ID
        name: Student name
        image_paths: List of image file paths
        face_recognizer: FaceRecognizer instance
        database: StudentDatabase instance
        yolo_model: Optional YOLO model for face detection
        min_confidence: Minimum face detection confidence
        notes: Optional notes about the student
    
    Returns:
        True if registration successful
    """
    embeddings = []
    successful_images = 0
    
    logger.info(f"Processing {len(image_paths)} images for {name}...")
    
    for img_path in image_paths:
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"  ✗ Could not load: {img_path}")
                continue
            
            # Option 1: Use YOLO for face detection (more robust)
            if yolo_model is not None:
                results = yolo_model.predict(img, conf=min_confidence, verbose=False)
                
                if len(results) == 0 or len(results[0].boxes) == 0:
                    logger.warning(f"  ✗ No face detected: {Path(img_path).name}")
                    continue
                
                # Get first/best detection
                box = results[0].boxes[0]
                xyxy = box.xyxy[0].cpu().numpy()
                
                # Crop face
                face_crop = crop_face(img, xyxy, padding=0.2)
                
                # Extract embedding
                embedding = face_recognizer.extract_embedding_from_crop(face_crop)
                
            # Option 2: Use InsightFace's built-in detection
            else:
                try:
                    embedding = face_recognizer.extract_embedding(img)
                except ValueError as e:
                    logger.warning(f"  ✗ {e}: {Path(img_path).name}")
                    continue
            
            embeddings.append(embedding)
            successful_images += 1
            logger.info(f"  ✓ Processed: {Path(img_path).name}")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {img_path}: {e}")
            continue
    
    # Check if we have enough embeddings
    if len(embeddings) < 2:
        logger.error(f"✗ Insufficient valid images. Need at least 2, got {len(embeddings)}")
        return False
    
    logger.info(f"Successfully extracted {len(embeddings)} embeddings from {successful_images} images")
    
    # Register student
    success = database.add_student(
        student_id=student_id,
        name=name,
        embeddings=embeddings,
        notes=notes
    )
    
    if success:
        logger.info(f"✓ Student registered: {name} ({student_id})")
    else:
        logger.error(f"✗ Failed to register student in database")
    
    return success


def register_from_webcam(
    student_id: str,
    name: str,
    face_recognizer: FaceRecognizer,
    database: StudentDatabase,
    num_samples: int = 5,
    notes: str = None
) -> bool:
    """
    Register a student using webcam capture.
    
    Args:
        student_id: Unique student ID
        name: Student name
        face_recognizer: FaceRecognizer instance
        database: StudentDatabase instance
        num_samples: Number of face samples to capture
        notes: Optional notes
    
    Returns:
        True if registration successful
    """
    logger.info(f"Starting webcam registration for {name}...")
    logger.info(f"Will capture {num_samples} samples")
    logger.info("Press SPACE to capture, Q to quit\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return False
    
    embeddings = []
    captured = 0
    
    try:
        while captured < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display frame
            display = frame.copy()
            
            # Try to detect face for visual feedback
            try:
                _, det_info = face_recognizer.extract_embedding(frame, return_detection=True)
                bbox = det_info['bbox'].astype(int)
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(display, "Face detected - Press SPACE", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except ValueError:
                cv2.putText(display, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(display, f"Captured: {captured}/{num_samples}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Student Registration', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                try:
                    embedding = face_recognizer.extract_embedding(frame)
                    embeddings.append(embedding)
                    captured += 1
                    logger.info(f"✓ Captured sample {captured}/{num_samples}")
                except ValueError as e:
                    logger.warning(f"✗ {e}")
            
            elif key == ord('q'):  # Q to quit
                logger.info("Registration cancelled by user")
                break
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    if len(embeddings) < 2:
        logger.error(f"✗ Insufficient samples captured. Need at least 2, got {len(embeddings)}")
        return False
    
    # Register student
    success = database.add_student(
        student_id=student_id,
        name=name,
        embeddings=embeddings,
        notes=notes
    )
    
    if success:
        logger.info(f"✓ Student registered: {name} ({student_id}) with {len(embeddings)} samples")
    
    return success


def register_from_webcam_headless(
    student_id: str,
    name: str,
    face_recognizer: FaceRecognizer,
    database: StudentDatabase,
    num_samples: int = 5,
    notes: str = None
) -> bool:
    """
    Register a student using webcam capture in HEADLESS mode (no display).
    Manually triggers capture when user presses Enter.
    """
    import time
    
    logger.info(f"Starting HEADLESS webcam registration for {name}...")
    logger.info(f"Will capture {num_samples} samples.")
    logger.info("Please sit in front of the camera.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return False
    
    # Set buffer size to 1 if possible to reduce lag, though flushing manually is safer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    embeddings = []
    captured = 0
    
    try:
        print("\n" + "="*40)
        print("  INSTRUCTIONS  ")
        print("  1. Look at the camera")
        print("  2. Press ENTER to take a photo")
        print("  3. Repeat for different angles")
        print("="*40 + "\n")
        
        while captured < num_samples:
            sys.stdout.write(f"\n[{captured + 1}/{num_samples}] Press ENTER to capture... ")
            sys.stdout.flush()
            input() # Wait for Enter
            
            print("Capturing...")
            
            # Flush buffer to get a fresh frame (cameras buffer old frames)
            for _ in range(5):
                cap.read()
                
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read from camera")
                break
                
            # Try to detect face
            try:
                embedding = face_recognizer.extract_embedding(frame)
                embeddings.append(embedding)
                captured += 1
                print(f"✓ Capture successful! ({captured}/{num_samples})")
                    
            except ValueError:
                print("✗ No face detected! Try again.")
                print("  (Make sure you are well-lit and facing the camera)")
            
    finally:
        cap.release()
    
    if len(embeddings) < 2:
        logger.error(f"\n✗ Insufficient samples captured. Need at least 2, got {len(embeddings)}")
        return False
    
    # Register student
    success = database.add_student(
        student_id=student_id,
        name=name,
        embeddings=embeddings,
        notes=notes
    )
    
    if success:
        print(f"\n✓ Student registered: {name} ({student_id}) with {len(embeddings)} samples")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Register students for face recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--name', type=str, required=True, help='Student full name')
    parser.add_argument('--id', type=str, required=True, help='Unique student ID')
    
    # Input source
    parser.add_argument('--source', type=str, help='Input: directory of images or image file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for registration')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of webcam samples to capture')
    parser.add_argument('--no-display', action='store_true', help='Headless mode (no video display) - auto capture')
    
    # Database
    parser.add_argument('--db', type=str, default='databases/students.db', help='Database path')
    
    # Models
    parser.add_argument('--arcface-model', type=str, default='buffalo_s', 
                       help='ArcFace model (buffalo_s, buffalo_m, buffalo_l)')
    parser.add_argument('--yolo-model', type=str, 
                       default='/Users/saifqal/Project/models/face_detector/weights/best.pt',
                       help='Path to YOLO face detection model (optional)')
    parser.add_argument('--no-yolo', action='store_true', 
                       help='Use InsightFace detection instead of YOLO')
    
    # Options
    parser.add_argument('--notes', type=str, help='Optional notes about the student')
    parser.add_argument('--min-confidence', type=float, default=0.7, 
                       help='Minimum face detection confidence')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.webcam and not args.source:
        parser.error("Must specify either --source or --webcam")
    
    if args.webcam and args.source:
        parser.error("Cannot use both --source and --webcam")
    
    print("=" * 60)
    print("STUDENT REGISTRATION")
    print("=" * 60)
    print(f"Student: {args.name}")
    print(f"ID: {args.id}")
    print(f"Database: {args.db}")
    print(f"ArcFace Model: {args.arcface_model}")
    print("=" * 60)
    
    # Initialize face recognizer
    logger.info("Loading ArcFace model...")
    # Headless mode requires specific environment on Pi, assumed handled by user setup
    face_recognizer = FaceRecognizer(model_name=args.arcface_model, device='cpu')
    
    # Load YOLO model if needed
    yolo_model = None
    if not args.no_yolo and not args.webcam:
        if os.path.exists(args.yolo_model):
            logger.info("Loading YOLO face detection model...")
            yolo_model = load_yolo(args.yolo_model)
        else:
            logger.warning(f"YOLO model not found: {args.yolo_model}")
            logger.warning("Using InsightFace detection instead")
    
    # Initialize database
    logger.info("Connecting to database...")
    database = StudentDatabase(args.db)
    
    # Check if student already exists
    existing = database.get_student_info(args.id)
    if existing:
        logger.error(f"✗ Student ID {args.id} already exists: {existing['name']}")
        logger.info("Use a different ID or remove the existing student first")
        database.close()
        return
    
    # Register student
    if args.webcam:
        if args.no_display:
            success = register_from_webcam_headless(
                student_id=args.id,
                name=args.name,
                face_recognizer=face_recognizer,
                database=database,
                num_samples=args.num_samples,
                notes=args.notes
            )
        else:
            success = register_from_webcam(
                student_id=args.id,
                name=args.name,
                face_recognizer=face_recognizer,
                database=database,
                num_samples=args.num_samples,
                notes=args.notes
            )
    else:
        # Collect image paths
        source_path = Path(args.source)
        if source_path.is_file():
            image_paths = [source_path]
        elif source_path.is_dir():
            extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_paths = [p for p in source_path.iterdir() 
                          if p.suffix.lower() in extensions]
        else:
            logger.error(f"Invalid source path: {args.source}")
            database.close()
            return
        
        if not image_paths:
            logger.error("No images found in source")
            database.close()
            return
        
        success = register_from_images(
            student_id=args.id,
            name=args.name,
            image_paths=image_paths,
            face_recognizer=face_recognizer,
            database=database,
            yolo_model=yolo_model,
            min_confidence=args.min_confidence,
            notes=args.notes
        )
    
    # Cleanup
    database.close()
    
    print("=" * 60)
    if success:
        print("✓ REGISTRATION SUCCESSFUL")
    else:
        print("✗ REGISTRATION FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
