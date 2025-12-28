#!/usr/bin/env python3
"""
Interactive Student Registration
For Smart Class Camera - Bachelor Thesis Project

Simple terminal interface to register new students.
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def get_input(prompt: str, default=None, required=False):
    """Get user input with optional default value."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        while True:
            user_input = input(f"{prompt}: ").strip()
            if user_input or not required:
                return user_input
            print("  This field is required.")


def main():
    print("\n" + "=" * 50)
    print("   👤 STUDENT REGISTRATION")
    print("=" * 50)
    print("\nEnter student details:\n")
    
    # Student info
    student_id = get_input("Student ID (e.g., STU001)", required=True)
    name = get_input("Full name", required=True)
    notes = get_input("Notes (optional)", default="")
    
    # Registration method
    print("\nRegistration method:")
    print("  1. Webcam capture (recommended)")
    print("  2. Image folder")
    method = get_input("Select method", default="1")
    
    if method == "1":
        # Webcam registration
        num_samples = get_input("Number of face samples to capture", default="5")
        try:
            num_samples = int(num_samples)
        except:
            num_samples = 5
        
        source_path = None
        use_webcam = True
    else:
        # Image folder registration
        source_path = get_input("Path to image folder", required=True)
        if not os.path.exists(source_path):
            print(f"\n❌ Path not found: {source_path}")
            return
        
        num_samples = 5
        use_webcam = False
    
    # Database path
    db_path = get_input("Database path", default="databases/students.db")
    
    # Confirmation
    print("\n" + "-" * 50)
    print("REGISTRATION DETAILS:")
    print("-" * 50)
    print(f"  Student ID:  {student_id}")
    print(f"  Name:        {name}")
    print(f"  Notes:       {notes if notes else '(none)'}")
    print(f"  Method:      {'Webcam' if use_webcam else 'Images'}")
    if use_webcam:
        print(f"  Samples:     {num_samples}")
    else:
        print(f"  Source:      {source_path}")
    print(f"  Database:    {db_path}")
    print("-" * 50)
    
    confirm = input("\nProceed with registration? (y/n) [y]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Registration cancelled.")
        return
    
    print("\n🔄 Loading models...\n")
    
    # Import modules
    from face_recognition import FaceRecognizer
    from student_database import StudentDatabase
    from register_student import register_from_webcam, register_from_images
    
    # Initialize
    face_recognizer = FaceRecognizer(model_name='buffalo_s', device='cpu')
    database = StudentDatabase(db_path)
    
    # Check if student exists
    existing = database.get_student_info(student_id)
    if existing:
        print(f"\n❌ Student ID '{student_id}' already exists: {existing['name']}")
        overwrite = input("Overwrite existing student? (y/n) [n]: ").strip().lower()
        if overwrite != 'y':
            database.close()
            return
        # Remove existing student
        database.remove_student(student_id)
        print(f"✓ Removed existing student: {existing['name']}")
    
    # Register
    print("\n" + "=" * 50)
    
    if use_webcam:
        print("📷 Starting webcam capture...")
        print("   Press SPACE to capture a sample")
        print("   Press Q to finish early")
        print("=" * 50 + "\n")
        
        success = register_from_webcam(
            student_id=student_id,
            name=name,
            face_recognizer=face_recognizer,
            database=database,
            num_samples=num_samples,
            notes=notes if notes else None
        )
    else:
        # Load YOLO for image processing
        yolo_model = None
        yolo_path = Path(__file__).parent.parent / "models/runpod_results/widerface_yolov8n/weights/best.pt"
        if yolo_path.exists():
            from detect_and_crop import load_model as load_yolo
            print("Loading YOLO face detector...")
            yolo_model = load_yolo(str(yolo_path))
        
        # Get image list
        source = Path(source_path)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        if source.is_file():
            image_paths = [source]
        else:
            image_paths = [p for p in source.iterdir() if p.suffix.lower() in extensions]
        
        print(f"Found {len(image_paths)} images")
        print("=" * 50 + "\n")
        
        success = register_from_images(
            student_id=student_id,
            name=name,
            image_paths=[str(p) for p in image_paths],
            face_recognizer=face_recognizer,
            database=database,
            yolo_model=yolo_model,
            notes=notes if notes else None
        )
    
    # Cleanup
    database.close()
    
    print("\n" + "=" * 50)
    if success:
        print(f"✅ SUCCESS: {name} ({student_id}) registered!")
    else:
        print("❌ REGISTRATION FAILED")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nRegistration cancelled.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
