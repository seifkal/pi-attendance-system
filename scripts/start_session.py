#!/usr/bin/env python3
"""
Interactive Session Launcher
For Smart Class Camera - Bachelor Thesis Project

Simple terminal interface to configure and start attendance sessions.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def get_input(prompt: str, default, input_type=str):
    """Get user input with default value."""
    try:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            return default
        return input_type(user_input)
    except ValueError:
        print(f"  Invalid input, using default: {default}")
        return default


def main():
    print("\n" + "=" * 50)
    print("   🎓 SMART CLASS ATTENDANCE SYSTEM")
    print("=" * 50)
    print("\nConfigure your session:\n")
    
    # Session configuration
    session_name = get_input("Session name", "Class Session")
    duration = get_input("Session duration (minutes)", 90, int)
    interval = get_input("Attendance check interval (minutes)", 15, int)
    process_every = get_input("Process every N frames (higher = faster)", 3, int)
    
    # Camera selection
    print("\nCamera options:")
    print("  1. Pi Camera Module (default)")
    print("  2. USB Webcam")
    camera_choice = get_input("Select camera", 1, int)
    use_usb = camera_choice == 2
    
    # Display option
    print("\nDisplay options:")
    print("  1. Show video feed (default)")
    print("  2. Headless mode (no display)")
    display_choice = get_input("Select display mode", 1, int)
    no_display = display_choice == 2
    
    # Recognition threshold
    threshold = get_input("Recognition threshold (0.0-1.0)", 0.55, float)
    
    # Confirmation
    print("\n" + "-" * 50)
    print("SESSION CONFIGURATION:")
    print("-" * 50)
    print(f"  Name:           {session_name}")
    print(f"  Duration:       {duration} minutes")
    print(f"  Check interval: {interval} minutes")
    print(f"  Process every:  {process_every} frames")
    print(f"  Camera:         {'USB Webcam' if use_usb else 'Pi Camera'}")
    print(f"  Display:        {'Headless' if no_display else 'Video feed'}")
    print(f"  Threshold:      {threshold}")
    print("-" * 50)
    
    confirm = input("\nStart session? (y/n) [y]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Session cancelled.")
        return
    
    print("\n🚀 Starting session...\n")
    
    # Import and run
    from pi_attendance import AttendanceSystem
    
    system = AttendanceSystem(
        database_path='databases/students.db',
        recognition_threshold=threshold,
        use_pi_camera=not use_usb,
        process_every_n_frames=process_every
    )
    
    system.run_session(
        session_name=session_name,
        duration_minutes=duration,
        attendance_interval=interval,
        display_video=not no_display
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSession cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
