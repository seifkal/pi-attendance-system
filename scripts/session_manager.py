#!/usr/bin/env python3
"""
Session Manager Module
For Smart Class Camera - Bachelor Thesis Project

Manages class sessions with periodic attendance checks and attention tracking.
"""

import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import logging

from session_report import SessionReport, StudentSessionData, generate_session_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Configuration for a session."""
    duration_minutes: int = 90
    attendance_interval_minutes: int = 15
    lateness_threshold_minutes: int = 5
    attention_sample_interval_seconds: float = 2.0
    output_dir: str = "outputs/sessions"
    session_name: str = "Class Session"


class SessionManager:
    """
    Manage class sessions with automatic attendance checks.
    
    Features:
    - Periodic attendance checks (every N minutes)
    - Continuous attention tracking
    - Automatic report generation at session end
    """
    
    def __init__(
        self,
        registered_students: List[Dict],
        config: Optional[SessionConfig] = None
    ):
        """
        Initialize session manager.
        
        Args:
            registered_students: List of {'student_id': str, 'name': str}
            config: Session configuration
        """
        self.config = config or SessionConfig()
        self.registered_students = registered_students
        
        # Session state
        self.report: Optional[SessionReport] = None
        self.is_active = False
        self.start_time: Optional[datetime] = None
        self.current_check_number = 0
        
        # Track currently visible students
        self.visible_students: Dict[str, datetime] = {}  # {student_id: last_seen}
        
        # Attendance check scheduling
        self._check_timer: Optional[threading.Timer] = None
        self._next_check_time: Optional[datetime] = None
        
        # Callbacks
        self.on_attendance_check: Optional[Callable] = None
        self.on_session_end: Optional[Callable] = None
    
    def start_session(self, session_name: Optional[str] = None) -> SessionReport:
        """
        Start a new session.
        
        Args:
            session_name: Optional name for the session
            
        Returns:
            SessionReport object for tracking
        """
        if self.is_active:
            logger.warning("Session already active. Stop current session first.")
            return self.report
        
        self.start_time = datetime.now()
        session_id = generate_session_id()
        
        self.report = SessionReport(
            session_id=session_id,
            session_name=session_name or self.config.session_name,
            start_time=self.start_time,
            lateness_threshold_minutes=self.config.lateness_threshold_minutes
        )
        
        # Register all students
        for student in self.registered_students:
            self.report.add_student(student['student_id'], student['name'])
        
        self.is_active = True
        self.current_check_number = 0
        self.visible_students = {}
        
        logger.info("=" * 60)
        logger.info(f"SESSION STARTED: {self.report.session_name}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Duration: {self.config.duration_minutes} minutes")
        logger.info(f"Attendance checks every {self.config.attendance_interval_minutes} minutes")
        logger.info(f"Registered students: {len(self.registered_students)}")
        logger.info("=" * 60)
        
        # Schedule first attendance check (at start)
        self._do_attendance_check()
        
        # Schedule end of session
        self._schedule_session_end()
        
        return self.report
    
    def stop_session(self) -> Optional[SessionReport]:
        """
        Stop the current session and generate report.
        
        Returns:
            Final SessionReport or None if no active session
        """
        if not self.is_active:
            logger.warning("No active session to stop.")
            return None
        
        # Cancel pending timers
        if self._check_timer:
            self._check_timer.cancel()
        
        # Finalize report
        self.report.finalize()
        self.is_active = False
        
        # Export report
        output_path = self._export_report()
        
        # Print summary
        self.report.print_summary()
        print(f"  Report saved: {output_path}")
        
        if self.on_session_end:
            self.on_session_end(self.report)
        
        return self.report
    
    def record_student_seen(self, student_id: str, timestamp: Optional[datetime] = None):
        """
        Record that a student was detected in the frame.
        
        Args:
            student_id: ID of the detected student
            timestamp: Detection time (defaults to now)
        """
        if not self.is_active or not self.report:
            return
        
        timestamp = timestamp or datetime.now()
        self.visible_students[student_id] = timestamp
        self.report.record_presence(student_id, timestamp)
    
    def record_attention(self, student_id: str, score: float):
        """
        Record attention score for a student.
        
        Args:
            student_id: Student ID
            score: Attention score 0.0 to 1.0
        """
        if not self.is_active or not self.report:
            return
        
        self.report.record_attention(student_id, score)
    
    def get_elapsed_minutes(self) -> float:
        """Get minutes elapsed since session start."""
        if not self.start_time:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds() / 60
    
    def get_remaining_minutes(self) -> float:
        """Get minutes remaining in session."""
        return max(0, self.config.duration_minutes - self.get_elapsed_minutes())
    
    def get_next_check_in(self) -> float:
        """Get minutes until next attendance check."""
        if not self._next_check_time:
            return 0.0
        delta = (self._next_check_time - datetime.now()).total_seconds() / 60
        return max(0, delta)
    
    def get_current_status(self) -> Dict:
        """Get current session status."""
        if not self.is_active:
            return {'active': False}
        
        present = sum(
            1 for s in self.report.students.values() 
            if s.status in ('present', 'late')
        )
        
        return {
            'active': True,
            'session_name': self.report.session_name,
            'elapsed_minutes': round(self.get_elapsed_minutes(), 1),
            'remaining_minutes': round(self.get_remaining_minutes(), 1),
            'next_check_in': round(self.get_next_check_in(), 1),
            'attendance_check': self.current_check_number,
            'students_present': present,
            'total_students': len(self.registered_students)
        }
    
    def _do_attendance_check(self):
        """Perform an attendance check."""
        if not self.is_active:
            return
        
        self.current_check_number += 1
        check_time = datetime.now()
        
        logger.info(f"\n📋 ATTENDANCE CHECK #{self.current_check_number} at {check_time.strftime('%H:%M')}")
        
        present_count = 0
        for student in self.registered_students:
            student_id = student['student_id']
            
            # Check if student was seen recently (within last 30 seconds)
            last_seen = self.visible_students.get(student_id)
            is_present = False
            
            if last_seen:
                seconds_since = (check_time - last_seen).total_seconds()
                is_present = seconds_since < 30
            
            self.report.record_attendance_check(student_id, is_present, self.current_check_number)
            
            if is_present:
                present_count += 1
                status = "✓"
            else:
                status = "✗"
            
            logger.info(f"  {status} {student['name']}")
        
        logger.info(f"  Present: {present_count}/{len(self.registered_students)}")
        
        if self.on_attendance_check:
            self.on_attendance_check(self.current_check_number, present_count)
        
        # Schedule next check
        self._schedule_next_check()
    
    def _schedule_next_check(self):
        """Schedule the next attendance check."""
        if not self.is_active:
            return
        
        # Check if session should end before next check
        remaining = self.get_remaining_minutes()
        interval = self.config.attendance_interval_minutes
        
        if remaining <= 0:
            return  # Session is ending
        
        if remaining < interval:
            # Last check will be at session end
            wait_seconds = remaining * 60
        else:
            wait_seconds = interval * 60
        
        self._next_check_time = datetime.now() + timedelta(seconds=wait_seconds)
        self._check_timer = threading.Timer(wait_seconds, self._do_attendance_check)
        self._check_timer.daemon = True
        self._check_timer.start()
    
    def _schedule_session_end(self):
        """Schedule automatic session end."""
        def end_session():
            if self.is_active:
                logger.info("\n⏰ Session duration complete!")
                self.stop_session()
        
        wait_seconds = self.config.duration_minutes * 60
        timer = threading.Timer(wait_seconds, end_session)
        timer.daemon = True
        timer.start()
    
    
    def generate_report(self) -> Dict:
        """
        Generate a dictionary report of the session data.
        Compatible with the format expected by pi_attendance.py for server upload.
        """
        if not self.report:
            return {}
            
        # Get data from SessionReport
        records = []
        for student in self.report.students.values():
            records.append({
                'name': student.name,
                'present': student.status in ('present', 'late'),
                'first_seen': student.first_seen.strftime('%H:%M:%S') if student.first_seen else "N/A",
                'check_count': student.checks_present,
                'attention_logs': student.attention_scores
            })
            
        return {
            'session_name': self.report.session_name,
            'duration_minutes': self.report.duration_minutes,
            'attendance_records': records,
        }

    def _export_report(self) -> str:
        """Export the session report to CSV."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"session_{self.report.session_id}.csv"
        filepath = output_dir / filename
        
        return self.report.export_csv(str(filepath))


# Test function
if __name__ == "__main__":
    print("Testing SessionManager...")
    
    # Mock students
    students = [
        {'student_id': 'STU001', 'name': 'John Doe'},
        {'student_id': 'STU002', 'name': 'Jane Smith'},
        {'student_id': 'STU003', 'name': 'Bob Wilson'},
    ]
    
    # Create manager with short session for testing
    config = SessionConfig(
        duration_minutes=1,  # 1 minute for testing
        attendance_interval_minutes=0.5,  # 30 seconds
        session_name="Test Session"
    )
    
    manager = SessionManager(students, config)
    
    # Start session
    report = manager.start_session()
    print(f"\nSession ID: {report.session_id}")
    
    # Simulate student detections
    manager.record_student_seen('STU001')
    manager.record_student_seen('STU002')
    # STU003 not seen
    
    # Simulate attention
    manager.record_attention('STU001', 0.9)
    manager.record_attention('STU002', 0.4)
    
    # Wait a bit
    time.sleep(2)
    
    # Get status
    status = manager.get_current_status()
    print(f"\nStatus: {status}")
    
    # Stop early
    manager.stop_session()
    
    print("\n✓ SessionManager test complete")
