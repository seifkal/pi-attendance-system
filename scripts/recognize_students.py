#!/usr/bin/env python3
"""
Student Recognition and Attendance Tracking
For Smart Class Camera - Bachelor Thesis Project

Recognize students from images/video and automatically mark attendance.
Supports session mode with periodic attendance checks and attention tracking.
"""

import argparse
import cv2
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from face_recognition import FaceRecognizer
from student_database import StudentDatabase
from detect_and_crop import crop_face, load_model as load_yolo
from activity_monitor import ActivityMonitor, get_attention_color
from session_manager import SessionManager, SessionConfig

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def recognize_faces_in_image(
    image: np.ndarray,
    yolo_model,
    face_recognizer: FaceRecognizer,
    database: StudentDatabase,
    threshold: float = 0.55,
    mark_attendance: bool = True,
    min_det_conf: float = 0.5
) -> Tuple[np.ndarray, List[dict]]:
    """
    Recognize all faces in an image.
    
    Args:
        image: Input image (BGR)
        yolo_model: YOLO face detection model
        face_recognizer: FaceRecognizer instance
        database: StudentDatabase instance
        threshold: Recognition similarity threshold
        mark_attendance: Whether to mark attendance in database
        min_det_conf: Minimum detection confidence
    
    Returns:
        (annotated_image, recognitions)
        recognitions = [{'bbox': [x1,y1,x2,y2], 'student_id': str, 'name': str, 'confidence': float}, ...]
    """
    annotated = image.copy()
    recognitions = []
    
    # Detect faces
    results = yolo_model.predict(image, conf=min_det_conf, verbose=False)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        return annotated, recognitions
    
    # Process each detected face
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        det_conf = float(box.conf[0])
        
        # Crop face
        face_crop = crop_face(image, xyxy, padding=0.2)
        
        try:
            # Extract embedding and landmarks
            embedding, landmarks = face_recognizer.extract_embedding_from_crop(
                face_crop, return_landmarks=True
            )
            
            # Find match in database
            match = database.find_match(embedding, threshold=threshold)
            
            if match:
                student_id, name, similarity = match
                
                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Green box for recognized faces
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Label with name and confidence
                label = f"{name} ({similarity:.2f})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                recognitions.append({
                    'bbox': xyxy.tolist(),
                    'student_id': student_id,
                    'name': name,
                    'confidence': similarity,
                    'det_confidence': det_conf,
                    'landmarks': landmarks
                })
                
                # Mark attendance
                if mark_attendance:
                    database.mark_attendance(student_id, confidence=similarity)
                    logger.info(f"  ✓ Recognized: {name} (ID: {student_id}, confidence: {similarity:.3f})")
            
            else:
                # Draw red box for unknown faces
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, "Unknown", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                logger.info(f"  ? Unknown face detected (det_conf: {det_conf:.3f})")
        
        except ValueError as e:
            logger.warning(f"  ✗ Could not extract embedding: {e}")
            continue
    
    return annotated, recognitions


def process_image(
    image_path: str,
    yolo_model,
    face_recognizer: FaceRecognizer,
    database: StudentDatabase,
    output_dir: Optional[str] = None,
    **kwargs
):
    """Process a single image."""
    logger.info(f"Processing: {Path(image_path).name}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return
    
    # Recognize faces
    annotated, recognitions = recognize_faces_in_image(
        image, yolo_model, face_recognizer, database, **kwargs
    )
    
    # Save annotated image
    if output_dir:
        output_path = Path(output_dir) / f"recognized_{Path(image_path).name}"
        cv2.imwrite(str(output_path), annotated)
        logger.info(f"  Saved: {output_path}")
    
    return annotated, recognitions


def process_video(
    video_path: str,
    yolo_model,
    face_recognizer: FaceRecognizer,
    database: StudentDatabase,
    output_dir: Optional[str] = None,
    frame_skip: int = 5,
    display: bool = True,
    **kwargs
):
    """Process a video file."""
    logger.info(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"  FPS: {fps}, Total frames: {frame_count}")
    logger.info(f"  Processing every {frame_skip} frames")
    
    # Setup output video writer if needed
    writer = None
    if output_dir:
        output_path = Path(output_dir) / f"recognized_{Path(video_path).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_num = 0
    processed_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every N frames
            if frame_num % frame_skip == 0:
                annotated, recognitions = recognize_faces_in_image(
                    frame, yolo_model, face_recognizer, database,
                    mark_attendance=False,  # Don't mark attendance for every frame
                    **kwargs
                )
                processed_frames += 1
                
                if display:
                    cv2.imshow('Recognition', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if writer:
                    writer.write(annotated)
            
            frame_num += 1
            
            # Progress
            if frame_num % 100 == 0:
                logger.info(f"  Processed {frame_num}/{frame_count} frames")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    
    logger.info(f"✓ Video processing complete: {processed_frames} frames processed")


def process_webcam(
    yolo_model,
    face_recognizer: FaceRecognizer,
    database: StudentDatabase,
    mark_attendance: bool = True,
    **kwargs
):
    """Process webcam stream with live recognition."""
    logger.info("Starting webcam recognition...")
    logger.info("Press 'Q' to quit, 'A' to toggle auto-attendance\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return
    
    # Track recent recognitions to avoid duplicate attendance
    recent_recognitions = {}  # {student_id: timestamp}
    attendance_cooldown = 5.0  # seconds
    
    frame_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            current_time = datetime.now()
            
            # Determine if we should mark attendance
            should_mark = mark_attendance
            
            annotated, recognitions = recognize_faces_in_image(
                frame, yolo_model, face_recognizer, database,
                mark_attendance=False,  # Manual control
                **kwargs
            )
            
            # Handle attendance marking with cooldown
            if should_mark:
                for rec in recognitions:
                    student_id = rec['student_id']
                    
                    # Check cooldown
                    if student_id in recent_recognitions:
                        elapsed = (current_time - recent_recognitions[student_id]).total_seconds()
                        if elapsed < attendance_cooldown:
                            continue
                    
                    # Mark attendance
                    database.mark_attendance(student_id, confidence=rec['confidence'])
                    recent_recognitions[student_id] = current_time
                    logger.info(f"✓ Attendance: {rec['name']}")
            
            # Display info
            frame_count += 1
            elapsed = (current_time - start_time).total_seconds()
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"Auto-Attendance: {'ON' if mark_attendance else 'OFF'}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Student Recognition', annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                mark_attendance = not mark_attendance
                logger.info(f"Auto-attendance: {'ON' if mark_attendance else 'OFF'}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    logger.info(f"✓ Webcam session ended. Average FPS: {fps:.1f}")


def process_webcam_session(
    yolo_model,
    face_recognizer: FaceRecognizer,
    database: StudentDatabase,
    session_name: str = "Class Session",
    duration_minutes: int = 90,
    attendance_interval: int = 15,
    **kwargs
):
    """
    Process webcam with session mode - tracks attendance and attention.
    
    Args:
        session_name: Name for the session
        duration_minutes: Session duration in minutes
        attendance_interval: Minutes between attendance checks
    """
    # Get registered students
    students = database.get_all_students()
    if not students:
        logger.error("No students registered!")
        return
    
    # Create session config
    config = SessionConfig(
        duration_minutes=duration_minutes,
        attendance_interval_minutes=attendance_interval,
        session_name=session_name
    )
    
    # Create managers
    session_manager = SessionManager(students, config)
    activity_monitors = {}  # {student_id: ActivityMonitor}
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return
    
    # Start session
    session_manager.start_session(session_name)
    
    frame_count = 0
    start_time = datetime.now()
    
    try:
        while session_manager.is_active:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame (extract threshold, don't pass mark_attendance via kwargs)
            threshold = kwargs.get('threshold', 0.55)
            min_det_conf = kwargs.get('min_det_conf', 0.5)
            annotated, recognitions = recognize_faces_in_image(
                frame, yolo_model, face_recognizer, database,
                threshold=threshold,
                mark_attendance=False,
                min_det_conf=min_det_conf
            )
            
            # Process each recognized student
            for rec in recognitions:
                student_id = rec['student_id']
                
                # Record presence
                session_manager.record_student_seen(student_id)
                
                # Get or create activity monitor for this student
                if student_id not in activity_monitors:
                    activity_monitors[student_id] = ActivityMonitor()
                
                # Extract landmarks if available
                landmarks = rec.get('landmarks')
                if landmarks is not None:
                    attention = activity_monitors[student_id].analyze(landmarks)
                    session_manager.record_attention(student_id, attention.score)
                    
                    # Color-code based on attention
                    color = get_attention_color(attention.state)
                    x1, y1, x2, y2 = map(int, rec['bbox'])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    
                    # Show attention status
                    attn_label = f"{attention.state} ({attention.score:.0%})"
                    cv2.putText(annotated, attn_label, (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display session info
            frame_count += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            status = session_manager.get_current_status()
            
            # Session info overlay
            cv2.putText(annotated, f"SESSION: {status['session_name']}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, f"Time: {status['elapsed_minutes']:.0f}/{config.duration_minutes} min", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, f"Present: {status['students_present']}/{status['total_students']}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated, f"Next check in: {status['next_check_in']:.1f} min", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Session Mode', annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("\nSession stopped by user.")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Stop session and generate report
        if session_manager.is_active:
            session_manager.stop_session()


def main():
    parser = argparse.ArgumentParser(
        description="Recognize students and track attendance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input source
    parser.add_argument('--source', type=str, help='Input: image, directory, or video file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for live recognition')
    
    # Database and models
    parser.add_argument('--db', type=str, default='databases/students.db', help='Student database path')
    parser.add_argument('--arcface-model', type=str, default='buffalo_s', help='ArcFace model')
    parser.add_argument('--yolo-model', type=str,
                       default='/Users/saifqal/Project/models/runpod_results/widerface_yolov8n/weights/best.pt',
                       help='YOLO face detection model')
    
    # Recognition parameters
    parser.add_argument('--threshold', type=float, default=0.55, help='Recognition threshold')
    parser.add_argument('--det-conf', type=float, default=0.5, help='Face detection confidence')
    
    # Output
    parser.add_argument('--output', type=str, default='outputs/recognition', help='Output directory')
    parser.add_argument('--no-save', action='store_true', help='Do not save annotated images')
    parser.add_argument('--mark-attendance', action='store_true', default=True, help='Mark attendance')
    
    # Video options
    parser.add_argument('--frame-skip', type=int, default=5, help='Process every Nth frame for video')
    parser.add_argument('--no-display', action='store_true', help='Do not display output')
    
    # Session mode options
    parser.add_argument('--session', action='store_true', help='Enable session mode with attendance checks')
    parser.add_argument('--session-name', type=str, default='Class Session', help='Session name')
    parser.add_argument('--duration', type=int, default=90, help='Session duration in minutes')
    parser.add_argument('--interval', type=int, default=15, help='Attendance check interval in minutes')
    
    args = parser.parse_args()
    
    if not args.webcam and not args.source:
        parser.error("Must specify either --source or --webcam")
    
    print("=" * 60)
    print("STUDENT RECOGNITION & ATTENDANCE")
    print("=" * 60)
    print(f"Database: {args.db}")
    print(f"Recognition threshold: {args.threshold}")
    print(f"Mark attendance: {args.mark_attendance}")
    print("=" * 60)
    
    # Load models
    logger.info("Loading models...")
    yolo_model = load_yolo(args.yolo_model)
    face_recognizer = FaceRecognizer(model_name=args.arcface_model, device='cpu')
    
    # Load database
    logger.info("Loading student database...")
    database = StudentDatabase(args.db)
    
    students = database.get_all_students()
    logger.info(f"✓ Database loaded: {len(students)} registered students")
    
    if len(students) == 0:
        logger.warning("No students registered! Register students first using register_student.py")
        database.close()
        return
    
    # Prepare output directory
    output_dir = None if args.no_save else args.output
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process input
    kwargs = {
        'threshold': args.threshold,
        'mark_attendance': args.mark_attendance,
        'min_det_conf': args.det_conf
    }
    
    try:
        if args.webcam:
            if args.session:
                # Session mode with attendance tracking
                process_webcam_session(
                    yolo_model, face_recognizer, database,
                    session_name=args.session_name,
                    duration_minutes=args.duration,
                    attendance_interval=args.interval,
                    **kwargs
                )
            else:
                # Regular webcam mode
                process_webcam(yolo_model, face_recognizer, database, **kwargs)
        
        else:
            source_path = Path(args.source)
            
            if source_path.is_file():
                # Check if video
                video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
                if source_path.suffix.lower() in video_exts:
                    process_video(
                        str(source_path), yolo_model, face_recognizer, database,
                        output_dir=output_dir,
                        frame_skip=args.frame_skip,
                        display=not args.no_display,
                        **kwargs
                    )
                else:
                    # Single image
                    process_image(str(source_path), yolo_model, face_recognizer, database,
                                output_dir=output_dir, **kwargs)
            
            elif source_path.is_dir():
                # Directory of images
                image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
                images = [p for p in source_path.iterdir() if p.suffix.lower() in image_exts]
                
                logger.info(f"Found {len(images)} images in directory")
                
                for img_path in images:
                    process_image(str(img_path), yolo_model, face_recognizer, database,
                                output_dir=output_dir, **kwargs)
            
            else:
                logger.error(f"Invalid source: {args.source}")
    
    finally:
        database.close()
    
    print("=" * 60)
    print("✓ RECOGNITION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
