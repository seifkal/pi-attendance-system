#!/usr/bin/env python3
"""
Pi Camera Attendance System
Optimized for Raspberry Pi 5

Simple, fast attendance tracking using InsightFace detection + ArcFace recognition.
Supports session mode with periodic attendance checks and attention tracking.
"""

import cv2
import sys
import time
from pathlib import Path
from datetime import datetime

# server (database, website,API) imports
import requests  # Allows sending data to our website
import json      # Helps format the data

# The address of the computer running the server
# I used my cloudflare tunnel here thats why it says turks21.uk
SERVER_URL = "https://attendance.turks21.uk/api/upload_report"

sys.path.append(str(Path(__file__).parent))

from face_recognition import FaceRecognizer
from student_database import StudentDatabase
from activity_monitor import ActivityMonitor, get_attention_color
from session_manager import SessionManager, SessionConfig
from detect_and_crop import crop_face, load_model as load_yolo
from ultralytics import YOLO

# Try to import picamera2 for Pi Camera Module
try:
    from picamera2 import Picamera2
    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False
    print("picamera2 not available, will use USB webcam")


class AttendanceSystem:
    """Optimized attendance system for Raspberry Pi 5"""
    
    def __init__(
        self,
        database_path: str = "databases/students.db",
        recognition_threshold: float = 0.55,
        use_pi_camera: bool = True,
        process_every_n_frames: int = 3  # Process every 3rd frame for speed
    ):
        """Initialize the attendance system"""
        print("=" * 60)
        print("RASPBERRY PI 5 ATTENDANCE SYSTEM")
        print("=" * 60)
        
        # Load models
        print("Loading ArcFace model (buffalo_s - optimized for Pi)...")
        self.recognizer = FaceRecognizer(model_name='buffalo_s', device='cpu')

        # Load YOLO model
        yolo_path = Path(__file__).parent.parent / 'models' / 'best.pt'
        print(f"Loading YOLO model from {yolo_path}...")
        self.yolo = YOLO(str(yolo_path))
        
        print("Loading database...")
        self.database = StudentDatabase(database_path)
        
        students = self.database.get_all_students()
        print(f"✓ Loaded {len(students)} registered students")
        for s in students:
            print(f"  - {s['name']} ({s['student_id']})")
        
        # Settings
        self.recognition_threshold = recognition_threshold
        self.process_every_n_frames = process_every_n_frames
        self.use_pi_camera = use_pi_camera and PI_CAMERA_AVAILABLE
        
        # Attendance tracking
        self.recent_recognitions = {}
        self.attendance_cooldown = 300.0  # 5 minutes between attendance marks
        
        # Performance tracking
        self.frame_count = 0
        self.recognition_count = 0
        self.start_time = None
        
        # Camera
        self.camera = None

        # Store the session manager so we can access it during cleanup
        self.current_session_manager = None
        
    def initialize_camera(self, resolution=(640, 480)):
        """Initialize camera (Pi Camera or USB)"""
        print("\nInitializing camera...")
        
        if self.use_pi_camera:
            try:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": resolution, "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                print(f"✓ Pi Camera Module initialized at {resolution}")
                time.sleep(2)  # Warm up
                return True
            except Exception as e:
                print(f"✗ Failed to initialize Pi Camera: {e}")
                print("  Falling back to USB webcam...")
                self.use_pi_camera = False
        
        # USB webcam fallback
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("✗ ERROR: Could not open webcam")
            return False
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        print(f"✓ USB webcam initialized at {resolution}")
        return True
    
    def get_frame(self):
        """Get frame from camera"""
        if self.use_pi_camera:
            frame = self.camera.capture_array()
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        else:
            return self.camera.read()
    
    def process_frame(self, frame, mark_attendance=True):
        """Process a single frame for face recognition using YOLOv8 + ArcFace"""
        display = frame.copy()
        current_time = datetime.now()
        recognitions = []
        
        # 1. Detect faces using YOLOv8
        results = self.yolo.predict(frame, conf=0.5, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return display, recognitions
            
        # 2. Process each detected face
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Crop face
            face_crop = crop_face(frame, xyxy, padding=0.2)
            
            try:
                # 3. Get embedding and landmarks from crop
                embedding, landmarks = self.recognizer.extract_embedding_from_crop(
                    face_crop, return_landmarks=True
                )
                
                # 4. Find match
                match = self.database.find_match(embedding, threshold=self.recognition_threshold)
                
                if match:
                    student_id, name, confidence = match
                    
                    # Green box for recognized
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Label
                    label = f"{name} ({confidence:.2f})"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(display, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    recognitions.append({
                        'student_id': student_id,
                        'name': name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'landmarks': landmarks
                    })
                    
                    # Mark attendance (with cooldown)
                    if mark_attendance:
                        if student_id in self.recent_recognitions:
                            elapsed = (current_time - self.recent_recognitions[student_id]).total_seconds()
                            if elapsed >= self.attendance_cooldown:
                                self.database.mark_attendance(student_id, confidence=confidence)
                                self.recent_recognitions[student_id] = current_time
                                print(f"✓ ATTENDANCE: {name} (confidence: {confidence:.3f})")
                        else:
                            self.database.mark_attendance(student_id, confidence=confidence)
                            self.recent_recognitions[student_id] = current_time
                            print(f"✓ ATTENDANCE: {name} (confidence: {confidence:.3f})")
                    
                    self.recognition_count += 1
                else:
                    # Red box for unknown
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display, "Unknown", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            except ValueError:
                # Face too small or invalid in crop
                continue
        
        return display, recognitions
    
    def run(self, display_video=True, duration=None):
        """
        Run the attendance system
        
        Args:
            display_video: Show video feed (set to False for headless operation)
            duration: Run for specific duration in seconds (None = infinite)
        """
        if not self.initialize_camera():
            return
        
        print("\n" + "=" * 60)
        print("ATTENDANCE SYSTEM RUNNING")
        print("=" * 60)
        print(f"Recognition threshold: {self.recognition_threshold}")
        print(f"Processing every {self.process_every_n_frames} frames")
        print(f"Attendance cooldown: {self.attendance_cooldown}s")
        if display_video:
            print("Press 'Q' to quit")
        print("=" * 60 + "\n")
        
        self.start_time = time.time()
        last_process_frame = 0
        
        try:
            while True:
                # Check duration
                if duration and (time.time() - self.start_time) > duration:
                    print(f"\n✓ Duration limit reached ({duration}s)")
                    break
                
                ret, frame = self.get_frame()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Process every N frames
                if self.frame_count - last_process_frame >= self.process_every_n_frames:
                    display, recognitions = self.process_frame(frame)
                    last_process_frame = self.frame_count
                else:
                    display = frame
                
                # Add performance stats
                elapsed = time.time() - self.start_time
                camera_fps = self.frame_count / elapsed if elapsed > 0 else 0
                processing_fps = self.recognition_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(display, f"Camera FPS: {camera_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, f"Processing FPS: {processing_fps:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, f"Recognitions: {self.recognition_count}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display
                if display_video:
                    cv2.imshow('Pi Attendance System', display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress update (every 100 frames)
                if self.frame_count % 100 == 0:
                    print(f"  Frames: {self.frame_count}, FPS: {camera_fps:.1f}, Recognitions: {self.recognition_count}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n" + "=" * 60)
        # --- UPLOAD TO DATABASE LOGIC ---
        # Check if we have a valid session to upload
        if hasattr(self, 'current_session_manager') and self.current_session_manager:
            print("\n📊 Preparing to upload session data...")
            
            # 1. Stop the session to finalize stats
            self.current_session_manager.stop_session()
            
            # 2. Get the raw data
            report_data = self.current_session_manager.generate_report()
            
            # 3. Format it for the Server
            upload_payload = {
                "session_name": report_data.get('session_name', "Unknown Session"),
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "duration": report_data.get('duration_minutes', 90),
                "students": []
            }

            # Convert complex student data into the simple list the server expects
            for student in report_data.get('attendance_records', []):
                # Calculate average attention
                attn_logs = student.get('attention_logs', [])
                avg_attn = sum(attn_logs) / len(attn_logs) if attn_logs else 0.0
                
                upload_payload['students'].append({
                    "name": student['name'],
                    "status": "Present" if student['present'] else "Absent",
                    "first_seen": student.get('first_seen', "N/A"),
                    "checks": student.get('check_count', 0),
                    "attention_score": round(avg_attn * 100, 1) # Convert 0.85 to 85.0
                })

            # 4. Send to Cloudflare URL
            try:
                print(f"🚀 Sending report to {SERVER_URL}...")
                response = requests.post(SERVER_URL, json=upload_payload, timeout=10)
                
                if response.status_code == 200:
                    print("✅ SUCCESS: Report uploaded to Server!")
                else:
                    print(f"❌ FAILED: Server returned {response.status_code}")
                    print(response.text)
            except Exception as e:
                print(f"❌ CONNECTION ERROR: {e}")
        # --- UPLOAD TO DATABASE LOGIC ENDS HERE---
        
        print("SHUTTING DOWN")
        print("=" * 60)
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            camera_fps = self.frame_count / elapsed if elapsed > 0 else 0
            processing_fps = self.recognition_count / elapsed if elapsed > 0 else 0
            
            print(f"Runtime: {elapsed:.1f}s")
            print(f"Frames processed: {self.frame_count}")
            print(f"Average camera FPS: {camera_fps:.1f}")
            print(f"Average processing FPS: {processing_fps:.1f}")
            print(f"Total recognitions: {self.recognition_count}")
        
        if self.use_pi_camera and self.camera:
            self.camera.stop()
        elif self.camera:
            self.camera.release()
        
        try:
            if cv2.getWindowProperty('Pi Attendance System', cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyAllWindows()
        except:
            pass
        
        self.database.close()
        print("=" * 60)
    
    def run_session(
        self,
        session_name: str = "Class Session",
        duration_minutes: int = 90,
        attendance_interval: int = 15,
        display_video: bool = True
    ):
        """
        Run session mode with periodic attendance checks and attention tracking.
        
        Args:
            session_name: Name for the session
            duration_minutes: Session duration in minutes
            attendance_interval: Minutes between attendance checks
            display_video: Show video feed
        """
        if not self.initialize_camera():
            return
        
        # Get registered students
        students = self.database.get_all_students()
        if not students:
            print("ERROR: No students registered!")
            return
        
        # Create session config
        config = SessionConfig(
            duration_minutes=duration_minutes,
            attendance_interval_minutes=attendance_interval,
            session_name=session_name
        )
        
        # Create managers
        self.current_session_manager = SessionManager(students, config)
        activity_monitors = {}  # {student_id: ActivityMonitor}
        
        # Start session
        self.current_session_manager.start_session(session_name)
        
        self.start_time = time.time()
        last_process_frame = 0
        
        try:
            while self.current_session_manager.is_active:
                ret, frame = self.get_frame()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Process every N frames
                if self.frame_count - last_process_frame >= self.process_every_n_frames:
                    display, recognitions = self.process_frame(frame, mark_attendance=False)
                    last_process_frame = self.frame_count
                    
                    # Process each recognized student for session tracking
                    for rec in recognitions:
                        student_id = rec['student_id']
                        
                        # Record presence in session
                        self.current_session_manager.record_student_seen(student_id)
                        
                        # Activity monitoring
                        if student_id not in activity_monitors:
                            activity_monitors[student_id] = ActivityMonitor()
                        
                        landmarks = rec.get('landmarks')
                        if landmarks is not None:
                            attention = activity_monitors[student_id].analyze(landmarks)
                            self.current_session_manager.record_attention(student_id, attention.score)
                            
                            # Update display with attention color
                            color = get_attention_color(attention.state)
                            x1, y1, x2, y2 = rec['bbox']
                            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                            
                            # Show attention label
                            attn_label = f"{attention.state} ({attention.score:.0%})"
                            cv2.putText(display, attn_label, (x1, y2 + 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    display = frame
                
                # Session info overlay
                status = self.current_session_manager.get_current_status()
                elapsed = time.time() - self.start_time
                camera_fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(display, f"SESSION: {status['session_name']}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, f"Time: {status['elapsed_minutes']:.0f}/{duration_minutes} min", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, f"Present: {status['students_present']}/{status['total_students']}", (10, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"Next check: {status['next_check_in']:.1f} min", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(display, f"FPS: {camera_fps:.1f}", (10, 125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Display
                if display_video:
                    cv2.imshow('Pi Session Mode', display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nSession stopped by user.")
                        break
                
                # Progress update
                if self.frame_count % 100 == 0:
                    print(f"  Frames: {self.frame_count}, FPS: {camera_fps:.1f}")
        
        finally:
            self.cleanup()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pi 5 Attendance System")
    parser.add_argument('--db', type=str, default='databases/students.db', help='Database path')
    parser.add_argument('--threshold', type=float, default=0.55, help='Recognition threshold')
    parser.add_argument('--process-every', type=int, default=3, help='Process every N frames')
    parser.add_argument('--no-display', action='store_true', help='Headless mode (no video display)')
    parser.add_argument('--usb-camera', action='store_true', help='Force USB camera (ignore Pi Camera)')
    parser.add_argument('--duration', type=int, help='Run for N seconds then exit (regular mode)')
    
    # Session mode options
    parser.add_argument('--session', action='store_true', help='Enable session mode with attendance checks')
    parser.add_argument('--session-name', type=str, default='Class Session', help='Session name')
    parser.add_argument('--session-duration', type=int, default=90, help='Session duration in minutes')
    parser.add_argument('--interval', type=int, default=15, help='Attendance check interval in minutes')
    
    args = parser.parse_args()
    
    system = AttendanceSystem(
        database_path=args.db,
        recognition_threshold=args.threshold,
        use_pi_camera=not args.usb_camera,
        process_every_n_frames=args.process_every
    )
    
    if args.session:
        # Session mode
        system.run_session(
            session_name=args.session_name,
            duration_minutes=args.session_duration,
            attendance_interval=args.interval,
            display_video=not args.no_display
        )
    else:
        # Regular mode
        system.run(
            display_video=not args.no_display,
            duration=args.duration
        )


if __name__ == "__main__":
    main()
