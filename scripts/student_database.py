#!/usr/bin/env python3
"""
Student Database Management
For Smart Class Camera - Bachelor Thesis Project

This module manages student registration, face embeddings storage,
and attendance tracking using SQLite + numpy arrays.
"""

import os
import sqlite3
import pickle
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudentDatabase:
    """
    Manage student face embeddings and attendance records.
    
    Database structure:
    - SQLite for structured data (students, attendance, activity logs)
    - Pickle/numpy for face embeddings (512-dim vectors)
    """
    
    def __init__(self, db_path: str = "databases/students.db"):
        """
        Initialize the student database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Embeddings stored separately (numpy-friendly)
        self.embeddings_path = self.db_path.parent / "embeddings.pkl"
        
        self.conn = None
        self.embeddings_cache = {}  # {student_id: [embedding1, embedding2, ...]}
        
        self._initialize_database()
        self._load_embeddings()
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        
        # Students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                enrollment_date TEXT NOT NULL,
                embedding_count INTEGER DEFAULT 0,
                notes TEXT
            )
        ''')
        
        # Attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            )
        ''')
        
        # Activity log table (attention, phone usage, etc.)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                timestamp TEXT NOT NULL,
                activity_type TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL,
                metadata TEXT,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                session_name TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_minutes INTEGER,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        # Session attendance (per-interval tracking)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                student_id TEXT NOT NULL,
                check_number INTEGER NOT NULL,
                check_time TEXT NOT NULL,
                is_present INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            )
        ''')
        
        # Attention samples
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attention_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                student_id TEXT,
                sample_time TEXT NOT NULL,
                attention_state TEXT,
                attention_score REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            )
        ''')
        
        self.conn.commit()
        logger.info(f"✓ Database initialized: {self.db_path}")
    
    def _load_embeddings(self):
        """Load embeddings from pickle file."""
        if self.embeddings_path.exists():
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
            logger.info(f"✓ Loaded embeddings for {len(self.embeddings_cache)} students")
        else:
            self.embeddings_cache = {}
            logger.info("No existing embeddings found, starting fresh")
    
    def _save_embeddings(self):
        """Save embeddings to pickle file."""
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
        logger.debug("Embeddings saved to disk")
    
    def add_student(
        self,
        student_id: str,
        name: str,
        embeddings: List[np.ndarray],
        notes: Optional[str] = None
    ) -> bool:
        """
        Register a new student with their face embeddings.
        
        Args:
            student_id: Unique student identifier
            name: Student's full name
            embeddings: List of face embeddings (512-dim numpy arrays)
            notes: Optional notes about the student
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            # Check if student already exists
            cursor.execute('SELECT student_id FROM students WHERE student_id = ?', (student_id,))
            if cursor.fetchone():
                logger.warning(f"Student {student_id} already exists. Use update_student() instead.")
                return False
            
            # Insert student record
            enrollment_date = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO students (student_id, name, enrollment_date, embedding_count, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (student_id, name, enrollment_date, len(embeddings), notes))
            
            # Store embeddings
            self.embeddings_cache[student_id] = [emb.copy() for emb in embeddings]
            self._save_embeddings()
            
            self.conn.commit()
            logger.info(f"✓ Registered student: {name} ({student_id}) with {len(embeddings)} embeddings")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            self.conn.rollback()
            return False
    
    def remove_student(self, student_id: str) -> bool:
        """
        Remove a student from the database.
        
        Args:
            student_id: Student ID to remove
            
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            
            # Delete from students table
            cursor.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
            
            # Delete from attendance
            cursor.execute('DELETE FROM attendance WHERE student_id = ?', (student_id,))
            
            # Remove embeddings
            if student_id in self.embeddings_cache:
                del self.embeddings_cache[student_id]
                self._save_embeddings()
            
            self.conn.commit()
            logger.info(f"✓ Removed student: {student_id}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            self.conn.rollback()
            return False
    
    def update_student_embeddings(
        self,
        student_id: str,
        new_embeddings: List[np.ndarray],
        replace: bool = False
    ) -> bool:
        """
        Add or replace embeddings for an existing student.
        
        Args:
            student_id: Student identifier
            new_embeddings: New embeddings to add
            replace: If True, replace all embeddings; if False, append
        
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            
            # Check if student exists
            cursor.execute('SELECT embedding_count FROM students WHERE student_id = ?', (student_id,))
            result = cursor.fetchone()
            if not result:
                logger.error(f"Student {student_id} not found")
                return False
            
            # Update embeddings
            if replace:
                self.embeddings_cache[student_id] = [emb.copy() for emb in new_embeddings]
            else:
                if student_id in self.embeddings_cache:
                    self.embeddings_cache[student_id].extend([emb.copy() for emb in new_embeddings])
                else:
                    self.embeddings_cache[student_id] = [emb.copy() for emb in new_embeddings]
            
            # Update database
            new_count = len(self.embeddings_cache[student_id])
            cursor.execute('UPDATE students SET embedding_count = ? WHERE student_id = ?', 
                          (new_count, student_id))
            
            self._save_embeddings()
            self.conn.commit()
            
            logger.info(f"✓ Updated embeddings for {student_id}: now {new_count} total")
            return True
            
        except Exception as e:
            logger.error(f"Error updating embeddings: {e}")
            return False
    
    def get_student_info(self, student_id: str) -> Optional[Dict]:
        """Get student information."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                'id': row[0],
                'student_id': row[1],
                'name': row[2],
                'enrollment_date': row[3],
                'embedding_count': row[4],
                'notes': row[5]
            }
        return None
    
    def get_all_students(self) -> List[Dict]:
        """Get all registered students."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM students')
        rows = cursor.fetchall()
        
        students = []
        for row in rows:
            students.append({
                'id': row[0],
                'student_id': row[1],
                'name': row[2],
                'enrollment_date': row[3],
                'embedding_count': row[4],
                'notes': row[5]
            })
        return students
    
    def find_match(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.55,
        use_average: bool = True
    ) -> Optional[Tuple[str, str, float]]:
        """
        Find the best matching student for a query embedding.
        
        Args:
            query_embedding: Query face embedding (512-dim)
            threshold: Minimum similarity threshold
            use_average: If True, compare against average embedding per student
        
        Returns:
            (student_id, name, similarity) if match found, None otherwise
        """
        if not self.embeddings_cache:
            logger.warning("No embeddings in database")
            return None
        
        best_match = None
        best_similarity = -1.0
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        for student_id, embeddings in self.embeddings_cache.items():
            if use_average:
                # Compute average embedding
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                
                # Cosine similarity
                similarity = float(np.dot(query_embedding, avg_embedding))
            else:
                # Maximum similarity across all embeddings
                similarities = []
                for emb in embeddings:
                    emb_norm = emb / np.linalg.norm(emb)
                    sim = float(np.dot(query_embedding, emb_norm))
                    similarities.append(sim)
                similarity = max(similarities)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = student_id
        
        # Check threshold
        if best_similarity >= threshold:
            student_info = self.get_student_info(best_match)
            return (best_match, student_info['name'], best_similarity)
        
        return None
    
    def mark_attendance(
        self,
        student_id: str,
        confidence: float,
        timestamp: Optional[str] = None
    ) -> bool:
        """
        Mark attendance for a student.
        
        Args:
            student_id: Student identifier
            confidence: Recognition confidence score
            timestamp: Optional timestamp (ISO format), defaults to now
        
        Returns:
            True if successful
        """
        try:
            if timestamp is None:
                timestamp = datetime.now().isoformat()
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO attendance (student_id, timestamp, confidence)
                VALUES (?, ?, ?)
            ''', (student_id, timestamp, confidence))
            
            self.conn.commit()
            logger.debug(f"✓ Attendance marked: {student_id} at {timestamp}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error marking attendance: {e}")
            return False
    
    def log_activity(
        self,
        activity_type: str,
        status: str,
        student_id: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> bool:
        """
        Log student activity (attention, phone usage, etc.).
        
        Args:
            activity_type: Type of activity ('attention', 'phone_usage', etc.)
            status: Activity status ('paying_attention', 'distracted', 'phone_detected', etc.)
            student_id: Optional student identifier
            confidence: Optional confidence score
            metadata: Optional JSON metadata
            timestamp: Optional timestamp
        
        Returns:
            True if successful
        """
        try:
            if timestamp is None:
                timestamp = datetime.now().isoformat()
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO activity_log (student_id, timestamp, activity_type, status, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (student_id, timestamp, activity_type, status, confidence, metadata))
            
            self.conn.commit()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error logging activity: {e}")
            return False
    
    def get_attendance_records(
        self,
        student_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get attendance records with optional filtering.
        
        Args:
            student_id: Filter by student
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
        
        Returns:
            List of attendance records
        """
        cursor = self.conn.cursor()
        
        query = 'SELECT * FROM attendance WHERE 1=1'
        params = []
        
        if student_id:
            query += ' AND student_id = ?'
            params.append(student_id)
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        
        query += ' ORDER BY timestamp DESC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        records = []
        for row in rows:
            records.append({
                'id': row[0],
                'student_id': row[1],
                'timestamp': row[2],
                'confidence': row[3]
            })
        return records
    
    def export_attendance_csv(self, output_path: str, start_date: Optional[str] = None):
        """Export attendance records to CSV."""
        import csv
        
        records = self.get_attendance_records(start_date=start_date)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['student_id', 'timestamp', 'confidence'])
            writer.writeheader()
            writer.writerows(records)
        
        logger.info(f"✓ Exported {len(records)} attendance records to {output_path}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    # Test the database
    print("Testing StudentDatabase...")
    
    db = StudentDatabase("databases/test_students.db")
    
    # Create dummy embeddings
    dummy_emb1 = np.random.randn(512).astype(np.float32)
    dummy_emb2 = np.random.randn(512).astype(np.float32)
    
    # Add a student
    success = db.add_student(
        student_id="STU001",
        name="John Doe",
        embeddings=[dummy_emb1, dummy_emb2],
        notes="Test student"
    )
    print(f"Add student: {'✓' if success else '✗'}")
    
    # Get student info
    info = db.get_student_info("STU001")
    print(f"Student info: {info}")
    
    # Mark attendance
    db.mark_attendance("STU001", confidence=0.95)
    print("✓ Attendance marked")
    
    # Get all students
    students = db.get_all_students()
    print(f"Total students: {len(students)}")
    
    db.close()
