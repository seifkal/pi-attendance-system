#!/usr/bin/env python3
"""
Activity Monitor Module
For Smart Class Camera - Bachelor Thesis Project

Head pose estimation and attention classification using facial landmarks.
Uses 5-point landmarks from InsightFace (eyes, nose, mouth corners).
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HeadPose:
    """Head pose angles in degrees."""
    yaw: float    # Left/right rotation (-90 to 90, negative = looking left)
    pitch: float  # Up/down rotation (-90 to 90, positive = looking down)
    roll: float   # Head tilt (-180 to 180)


@dataclass
class AttentionState:
    """Attention classification result."""
    state: str           # 'attentive', 'distracted', 'looking_away', 'looking_down'
    score: float         # 0.0 to 1.0 attention score
    pose: HeadPose       # Raw head pose data
    is_attentive: bool   # Convenience flag


class ActivityMonitor:
    """
    Monitor student activity using head pose estimation.
    
    Uses 5-point facial landmarks to estimate head orientation
    and classify attention state without additional models.
    """
    
    # Attention thresholds (in degrees)
    YAW_ATTENTIVE = 15.0       # Within ±15° = looking forward
    YAW_DISTRACTED = 30.0      # 15-30° = mildly distracted
    PITCH_ATTENTIVE_UP = -10.0 # Looking up slightly OK
    PITCH_ATTENTIVE_DOWN = 15.0 # Looking down slightly OK  
    PITCH_LOOKING_DOWN = 25.0  # > 25° = looking at phone/desk
    
    def __init__(
        self,
        yaw_threshold: float = 15.0,
        pitch_threshold: float = 15.0,
        reference_landmarks: Optional[np.ndarray] = None
    ):
        """
        Initialize the activity monitor.
        
        Args:
            yaw_threshold: Max yaw angle for 'attentive' state
            pitch_threshold: Max pitch angle for 'attentive' state
            reference_landmarks: Optional reference for calibration
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.reference_landmarks = reference_landmarks
        
        # For tracking attention over time
        self.attention_history = []
        self.max_history = 30  # ~1 second at 30fps
    
    def estimate_head_pose(self, landmarks: np.ndarray) -> HeadPose:
        """
        Estimate head pose from 5-point facial landmarks.
        
        InsightFace 5-point landmarks order:
        [0] Left eye center
        [1] Right eye center
        [2] Nose tip
        [3] Left mouth corner
        [4] Right mouth corner
        
        Args:
            landmarks: 5x2 numpy array of landmark coordinates
            
        Returns:
            HeadPose with yaw, pitch, roll angles in degrees
        """
        if landmarks is None or len(landmarks) < 5:
            return HeadPose(yaw=0.0, pitch=0.0, roll=0.0)
        
        # Extract key points
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        
        # Calculate eye center (midpoint between eyes)
        eye_center = (left_eye + right_eye) / 2
        
        # Calculate mouth center
        mouth_center = (left_mouth + right_mouth) / 2
        
        # Eye width (distance between eyes)
        eye_width = np.linalg.norm(right_eye - left_eye)
        
        # Avoid division by zero
        if eye_width < 1e-6:
            return HeadPose(yaw=0.0, pitch=0.0, roll=0.0)
        
        # --- YAW (left/right rotation) ---
        # Based on horizontal offset of nose from eye center
        nose_offset_x = nose[0] - eye_center[0]
        # Normalize by eye width for scale invariance
        yaw_ratio = nose_offset_x / eye_width
        # Convert to approximate degrees (empirical scaling)
        yaw = np.clip(yaw_ratio * 60.0, -90.0, 90.0)
        
        # --- PITCH (up/down rotation) ---
        # Based on vertical position of nose relative to eyes
        # When looking down, nose appears lower relative to eyes
        eye_to_nose_y = nose[1] - eye_center[1]
        eye_to_mouth_y = mouth_center[1] - eye_center[1]
        
        # Normalize by face height
        if eye_to_mouth_y > 1e-6:
            pitch_ratio = (eye_to_nose_y / eye_to_mouth_y) - 0.5
            pitch = np.clip(pitch_ratio * 80.0, -90.0, 90.0)
        else:
            pitch = 0.0
        
        # --- ROLL (head tilt) ---
        # Angle of line connecting both eyes
        eye_delta = right_eye - left_eye
        roll = np.degrees(np.arctan2(eye_delta[1], eye_delta[0]))
        
        return HeadPose(yaw=float(yaw), pitch=float(pitch), roll=float(roll))
    
    def classify_attention(self, pose: HeadPose) -> str:
        """
        Classify attention state based on head pose.
        
        Args:
            pose: HeadPose object with yaw, pitch, roll
            
        Returns:
            One of: 'attentive', 'distracted', 'looking_away', 'looking_down'
        """
        abs_yaw = abs(pose.yaw)
        
        # Check for looking down (phone/sleeping)
        if pose.pitch > self.PITCH_LOOKING_DOWN:
            return 'looking_down'
        
        # Check for looking away (turned head)
        if abs_yaw > self.YAW_DISTRACTED:
            return 'looking_away'
        
        # Check for mild distraction
        if abs_yaw > self.YAW_ATTENTIVE:
            return 'distracted'
        
        # Check pitch for attentive range
        if pose.pitch < self.PITCH_ATTENTIVE_UP or pose.pitch > self.PITCH_ATTENTIVE_DOWN:
            return 'distracted'
        
        return 'attentive'
    
    def get_attention_score(self, pose: HeadPose) -> float:
        """
        Calculate attention score from 0.0 (not paying attention) to 1.0 (fully attentive).
        
        Args:
            pose: HeadPose object
            
        Returns:
            Float between 0.0 and 1.0
        """
        # Penalize based on yaw (looking left/right)
        yaw_penalty = min(abs(pose.yaw) / 45.0, 1.0)  # Full penalty at 45°
        
        # Penalize based on pitch (looking up/down)
        pitch_deviation = max(0, pose.pitch - self.PITCH_ATTENTIVE_DOWN)
        pitch_penalty = min(pitch_deviation / 30.0, 1.0)  # Full penalty at 30° down
        
        # Combine penalties
        total_penalty = max(yaw_penalty, pitch_penalty)
        score = 1.0 - total_penalty
        
        return max(0.0, min(1.0, score))
    
    def analyze(self, landmarks: np.ndarray) -> AttentionState:
        """
        Full attention analysis from landmarks.
        
        Args:
            landmarks: 5x2 array of facial landmarks
            
        Returns:
            AttentionState with classification and score
        """
        pose = self.estimate_head_pose(landmarks)
        state = self.classify_attention(pose)
        score = self.get_attention_score(pose)
        
        # Update history
        self.attention_history.append(score)
        if len(self.attention_history) > self.max_history:
            self.attention_history.pop(0)
        
        return AttentionState(
            state=state,
            score=score,
            pose=pose,
            is_attentive=(state == 'attentive')
        )
    
    def get_average_attention(self) -> float:
        """Get average attention score from recent history."""
        if not self.attention_history:
            return 1.0
        return sum(self.attention_history) / len(self.attention_history)
    
    def reset_history(self):
        """Clear attention history."""
        self.attention_history = []


# Color coding for visualization
ATTENTION_COLORS = {
    'attentive': (0, 255, 0),      # Green
    'distracted': (0, 255, 255),    # Yellow
    'looking_away': (0, 0, 255),    # Red
    'looking_down': (0, 0, 255),    # Red
}


def get_attention_color(state: str) -> Tuple[int, int, int]:
    """Get BGR color for attention state visualization."""
    return ATTENTION_COLORS.get(state, (255, 255, 255))


# Test function
if __name__ == "__main__":
    print("Testing ActivityMonitor...")
    
    monitor = ActivityMonitor()
    
    # Simulate landmarks for different poses
    test_cases = [
        ("Looking straight", np.array([
            [100, 100],  # Left eye
            [150, 100],  # Right eye
            [125, 130],  # Nose
            [110, 160],  # Left mouth
            [140, 160],  # Right mouth
        ])),
        ("Looking left", np.array([
            [100, 100],
            [150, 100],
            [105, 130],  # Nose shifted left
            [110, 160],
            [140, 160],
        ])),
        ("Looking down", np.array([
            [100, 100],
            [150, 100],
            [125, 145],  # Nose shifted down
            [110, 160],
            [140, 160],
        ])),
    ]
    
    for name, landmarks in test_cases:
        result = monitor.analyze(landmarks)
        print(f"\n{name}:")
        print(f"  Pose: yaw={result.pose.yaw:.1f}°, pitch={result.pose.pitch:.1f}°, roll={result.pose.roll:.1f}°")
        print(f"  State: {result.state}")
        print(f"  Score: {result.score:.2f}")
    
    print("\n✓ ActivityMonitor test complete")
