#!/usr/bin/env python3
"""
Session Report Module
For Smart Class Camera - Bachelor Thesis Project

Generates CSV reports for class sessions with attendance and attention data.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StudentSessionData:
    """Data for a single student in a session."""
    student_id: str
    name: str
    status: str = 'absent'  # 'present', 'late', 'absent'
    first_seen: Optional[datetime] = None
    checks_present: int = 0
    total_checks: int = 0
    attention_scores: List[float] = field(default_factory=list)
    
    @property
    def attention_percentage(self) -> Optional[float]:
        """Calculate average attention percentage."""
        if not self.attention_scores:
            return None
        return (sum(self.attention_scores) / len(self.attention_scores)) * 100
    
    @property
    def is_low_attention(self) -> bool:
        """Check if attention is below threshold (50%)."""
        pct = self.attention_percentage
        return pct is not None and pct < 50.0


@dataclass  
class SessionReport:
    """Complete session report with all student data."""
    session_id: str
    session_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: int = 0
    students: Dict[str, StudentSessionData] = field(default_factory=dict)
    lateness_threshold_minutes: int = 5
    
    def add_student(self, student_id: str, name: str):
        """Register a student for this session."""
        if student_id not in self.students:
            self.students[student_id] = StudentSessionData(
                student_id=student_id,
                name=name
            )
    
    def record_presence(self, student_id: str, timestamp: datetime):
        """Record that a student was seen at a given time."""
        if student_id not in self.students:
            return
        
        student = self.students[student_id]
        
        # First time seeing this student
        if student.first_seen is None:
            student.first_seen = timestamp
            
            # Determine if late
            elapsed = (timestamp - self.start_time).total_seconds() / 60
            if elapsed > self.lateness_threshold_minutes:
                student.status = 'late'
            else:
                student.status = 'present'
    
    def record_attendance_check(self, student_id: str, is_present: bool, check_number: int):
        """Record result of an attendance check."""
        if student_id not in self.students:
            return
        
        student = self.students[student_id]
        student.total_checks = max(student.total_checks, check_number)
        if is_present:
            student.checks_present += 1
    
    def record_attention(self, student_id: str, score: float):
        """Record an attention score for a student."""
        if student_id not in self.students:
            return
        self.students[student_id].attention_scores.append(score)
    
    def finalize(self, end_time: Optional[datetime] = None):
        """Finalize the session report."""
        self.end_time = end_time or datetime.now()
        self.duration_minutes = int((self.end_time - self.start_time).total_seconds() / 60)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        present = sum(1 for s in self.students.values() if s.status == 'present')
        late = sum(1 for s in self.students.values() if s.status == 'late')
        absent = sum(1 for s in self.students.values() if s.status == 'absent')
        
        attention_scores = [
            s.attention_percentage for s in self.students.values() 
            if s.attention_percentage is not None
        ]
        avg_attention = sum(attention_scores) / len(attention_scores) if attention_scores else None
        
        return {
            'session_id': self.session_id,
            'session_name': self.session_name,
            'duration_minutes': self.duration_minutes,
            'total_students': len(self.students),
            'present': present,
            'late': late,
            'absent': absent,
            'avg_attention': avg_attention
        }
    
    def export_csv(self, filepath: str) -> str:
        """
        Export report to CSV file.
        
        Args:
            filepath: Path to save CSV file
            
        Returns:
            Absolute path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Session header
            writer.writerow(['session_id', 'session_name', 'start_time', 'end_time', 'duration_minutes'])
            writer.writerow([
                self.session_id,
                self.session_name,
                self.start_time.strftime('%Y-%m-%d %H:%M'),
                self.end_time.strftime('%Y-%m-%d %H:%M') if self.end_time else '',
                self.duration_minutes
            ])
            
            # Empty row separator
            writer.writerow([])
            
            # Student data header
            writer.writerow([
                'student_id', 'name', 'status', 'first_seen', 
                'checks_present', 'total_checks', 'attention_pct', 'low_attention'
            ])
            
            # Student rows
            for student in sorted(self.students.values(), key=lambda s: s.name):
                attention_pct = student.attention_percentage
                writer.writerow([
                    student.student_id,
                    student.name,
                    student.status,
                    student.first_seen.strftime('%H:%M') if student.first_seen else '',
                    student.checks_present,
                    student.total_checks,
                    f'{attention_pct:.1f}' if attention_pct is not None else '',
                    'true' if student.is_low_attention else 'false'
                ])
        
        logger.info(f"✓ Report exported: {filepath}")
        return str(filepath.absolute())
    
    def print_summary(self):
        """Print a summary to console."""
        summary = self.get_summary()
        
        print(f"\n✓ Session complete: {summary['session_name']}")
        print(f"  Duration: {summary['duration_minutes']} min | ", end='')
        print(f"Present: {summary['present']} | Late: {summary['late']} | Absent: {summary['absent']}")
        
        if summary['avg_attention'] is not None:
            print(f"  Avg Attention: {summary['avg_attention']:.1f}%")


def generate_session_id() -> str:
    """Generate a unique session ID based on current timestamp."""
    return f"SES_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# Test function
if __name__ == "__main__":
    print("Testing SessionReport...")
    
    # Create a test session
    report = SessionReport(
        session_id=generate_session_id(),
        session_name="CS101 - Test",
        start_time=datetime.now()
    )
    
    # Add test students
    report.add_student("STU001", "John Doe")
    report.add_student("STU002", "Jane Smith")
    report.add_student("STU003", "Bob Wilson")
    
    # Simulate presence
    from datetime import timedelta
    report.record_presence("STU001", report.start_time + timedelta(minutes=1))
    report.record_presence("STU002", report.start_time + timedelta(minutes=10))  # Late
    # STU003 never seen = absent
    
    # Simulate attention
    for _ in range(10):
        report.record_attention("STU001", 0.85)
        report.record_attention("STU002", 0.45)  # Low attention
    
    # Simulate attendance checks
    report.record_attendance_check("STU001", True, 1)
    report.record_attendance_check("STU002", True, 1)
    report.record_attendance_check("STU003", False, 1)
    
    # Finalize
    report.finalize()
    
    # Print summary
    report.print_summary()
    
    # Export CSV
    output_path = report.export_csv("outputs/sessions/test_report.csv")
    print(f"  Report saved: {output_path}")
    
    print("\n✓ SessionReport test complete")
