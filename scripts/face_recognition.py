#!/usr/bin/env python3
"""
Face Recognition Module using ArcFace (InsightFace)
For Smart Class Camera - Bachelor Thesis Project

This module handles face embedding extraction and similarity computation
using pre-trained ArcFace models from InsightFace.
Optimized for Raspberry Pi 5 deployment.
"""

import os
import cv2
import numpy as np 
from typing import Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    ArcFace-based face recognition using InsightFace.
    
    This class provides methods to:
    - Load pre-trained ArcFace models (optimized for edge devices)
    - Extract face embeddings (512-dimensional vectors)
    - Compute similarity between embeddings
    - Handle preprocessing for optimal recognition
    """
    
    def __init__(
        self,
        model_name: str = 'buffalo_s',
        device: str = 'cpu',
        det_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize the Face Recognizer.
        
        Args:
            model_name: InsightFace model name. Options:
                - 'buffalo_s': Smaller, faster (recommended for Pi 5)
                - 'buffalo_m': Medium size
                - 'buffalo_l': Larger, more accurate (slower on Pi)
            device: Device for inference ('cpu' or 'cuda')
            det_size: Detection input size (width, height)
        """
        self.model_name = model_name
        self.device = device
        self.det_size = det_size
        self.model = None
        
        logger.info(f"Initializing FaceRecognizer with model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the InsightFace model."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            # Initialize FaceAnalysis with the specified model
            ctx_id = 0 if self.device == 'cuda' else -1  # -1 for CPU
            
            self.model = FaceAnalysis(
                name=self.model_name,
                providers=['CPUExecutionProvider'] if self.device == 'cpu' 
                         else ['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Prepare model with detection size
            self.model.prepare(ctx_id=ctx_id, det_size=self.det_size)
            
            logger.info(f"✓ Model loaded successfully on {self.device.upper()}")
            
        except ImportError:
            logger.error("InsightFace not installed. Run: pip install insightface")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def extract_embedding(
        self,
        face_image: np.ndarray,
        return_detection: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        Extract face embedding from a face image.
        
        Args:
            face_image: Input face image (BGR format, as from cv2.imread)
            return_detection: If True, also return detection info
        
        Returns:
            embedding: 512-dimensional face embedding (numpy array)
            or (embedding, detection_info) if return_detection=True
            
        Raises:
            ValueError: If no face is detected in the image
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Invalid input image")
        
        # Ensure image is in correct format (BGR, uint8)
        if face_image.dtype != np.uint8:
            face_image = (face_image * 255).astype(np.uint8)
        
        # Get face analysis
        faces = self.model.get(face_image)
        
        if len(faces) == 0:
            raise ValueError("No face detected in the image")
        
        # Use the largest face if multiple detected
        if len(faces) > 1:
            logger.warning(f"Multiple faces detected ({len(faces)}), using largest face")
            faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        
        face = faces[0]
        embedding = face.embedding
        
        # Normalize embedding (L2 normalization)
        embedding = self.normalize_embedding(embedding)
        
        if return_detection:
            detection_info = {
                'bbox': face.bbox,
                'det_score': face.det_score,
                'landmark': face.kps if hasattr(face, 'kps') else None,
                'age': face.age if hasattr(face, 'age') else None,
                'gender': face.gender if hasattr(face, 'gender') else None
            }
            return embedding, detection_info
        
        return embedding
    
    def extract_embedding_from_crop(
        self,
        cropped_face: np.ndarray,
        resize_to: Tuple[int, int] = (112, 112),
        return_landmarks: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Extract embedding from pre-cropped face image.
        Useful when face detection is done separately (e.g., YOLOv8n).
        
        Args:
            cropped_face: Pre-cropped face image (BGR)
            resize_to: Target size for ArcFace input (default: 112x112)
            return_landmarks: If True, also return 5-point facial landmarks
        
        Returns:
            embedding: 512-dimensional face embedding
            or (embedding, landmarks) if return_landmarks=True
        """
        # Ensure correct dtype
        if cropped_face.dtype != np.uint8:
            cropped_face = (cropped_face * 255).astype(np.uint8)
        
        # Pad the cropped face to make it larger for better detection
        # This helps InsightFace detect the face in the cropped region
        h, w = cropped_face.shape[:2]
        
        # Add significant padding (100% of size on each side)
        padded = cv2.copyMakeBorder(
            cropped_face,
            top=h, bottom=h,
            left=w, right=w,
            borderType=cv2.BORDER_REPLICATE
        )
        
        if return_landmarks:
            embedding, det_info = self.extract_embedding(padded, return_detection=True)
            landmarks = det_info.get('landmark')
            return embedding, landmarks
        
        # Now extract embedding from the padded image
        return self.extract_embedding(padded)

    
    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """
        L2 normalize an embedding vector.
        
        Args:
            embedding: Raw embedding vector
        
        Returns:
            Normalized embedding (unit vector)
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    @staticmethod
    def compute_similarity(
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two face embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine' or 'euclidean')
        
        Returns:
            Similarity score:
                - Cosine: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
                - Euclidean: 0.0 = identical, higher = more different
        """
        # Ensure embeddings are normalized
        embedding1 = FaceRecognizer.normalize_embedding(embedding1)
        embedding2 = FaceRecognizer.normalize_embedding(embedding2)
        
        if metric == 'cosine':
            # Cosine similarity (dot product of normalized vectors)
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        
        elif metric == 'euclidean':
            # Euclidean distance
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(distance)
        
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")
    
    def verify_faces(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float = 0.6
    ) -> Tuple[bool, float]:
        """
        Verify if two face images belong to the same person.
        
        Args:
            image1: First face image
            image2: Second face image
            threshold: Similarity threshold for verification
        
        Returns:
            (is_match, similarity_score)
        """
        embedding1 = self.extract_embedding(image1)
        embedding2 = self.extract_embedding(image2)
        
        similarity = self.compute_similarity(embedding1, embedding2)
        is_match = similarity >= threshold
        
        return is_match, similarity


def test_face_recognizer():
    """Test the FaceRecognizer with sample images."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python face_recognition.py <image1> <image2>")
        print("This will test face recognition on two images")
        return
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error: Could not load images")
        return
    
    # Initialize recognizer
    print("Initializing Face Recognizer...")
    recognizer = FaceRecognizer(model_name='buffalo_s', device='cpu')
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    try:
        emb1, info1 = recognizer.extract_embedding(img1, return_detection=True)
        emb2, info2 = recognizer.extract_embedding(img2, return_detection=True)
        
        print(f"✓ Embedding 1: shape={emb1.shape}, norm={np.linalg.norm(emb1):.4f}")
        print(f"  Detection score: {info1['det_score']:.4f}")
        
        print(f"✓ Embedding 2: shape={emb2.shape}, norm={np.linalg.norm(emb2):.4f}")
        print(f"  Detection score: {info2['det_score']:.4f}")
        
        # Compute similarity
        similarity = recognizer.compute_similarity(emb1, emb2)
        print(f"\n{'='*50}")
        print(f"Cosine Similarity: {similarity:.4f}")
        
        # Verify
        is_match, score = recognizer.verify_faces(img1, img2, threshold=0.6)
        print(f"Match (threshold=0.6): {'✓ YES' if is_match else '✗ NO'}")
        print(f"{'='*50}")
        
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_face_recognizer()
