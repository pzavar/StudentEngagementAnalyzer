import cv2
import numpy as np
from typing import List, Tuple, Dict
import os
import logging

class FaceExtractor:
    def __init__(self, 
                 min_face_size: tuple = (30, 30),
                 tracking_threshold: int = 50,
                 detection_confidence: float = 0.3):
        """
        Initialize FaceExtractor with enhanced multi-face detection

        Args:
            min_face_size: Minimum face size to detect
            tracking_threshold: Pixel threshold for face tracking
            detection_confidence: Confidence threshold for face detection
        """
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces: Dict[str, Dict] = {}  # Enhanced face tracking dictionary
        self.tracking_threshold = tracking_threshold
        self.min_face_size = min_face_size
        self.detection_confidence = detection_confidence

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_faces(self, frame: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Extract and identify faces from a frame
        Enhanced for multiple face detection and tracking

        Returns:
            List of (student_id, face_image) tuples
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Enhanced face detection parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            self.logger.debug("No faces detected in frame")
            return []

        results = []
        current_faces = set()

        for (x, y, w, h) in faces:
            face_center = (x + w//2, y + h//2)

            # Get region of interest for face
            face_roi = frame[y:y+h, x:x+w]

            # Enhanced face identification
            student_id = self._identify_face(face_center, face_roi)
            current_faces.add(student_id)

            # Update face tracking data
            self.known_faces[student_id] = {
                'center': face_center,
                'last_seen': 0,  # Frame counter
                'size': (w, h)
            }

            results.append((student_id, face_roi))

        # Log tracking information
        self.logger.debug(f"Detected {len(results)} faces in frame")

        return results

    def _identify_face(self, face_center: tuple, face_roi: np.ndarray) -> str:
        """
        Enhanced face identification with improved tracking
        """
        if not self.known_faces:
            student_id = f"student_0"
            self.known_faces[student_id] = {
                'center': face_center,
                'last_seen': 0,
                'size': face_roi.shape[:2]
            }
            return student_id

        # Find the closest known face
        min_distance = float('inf')
        closest_id = None

        for student_id, face_data in self.known_faces.items():
            distance = np.sqrt(
                (face_center[0] - face_data['center'][0])**2 +
                (face_center[1] - face_data['center'][1])**2
            )

            if distance < min_distance and distance < self.tracking_threshold:
                min_distance = distance
                closest_id = student_id

        if closest_id is not None:
            return closest_id

        # New face detected
        new_id = f"student_{len(self.known_faces)}"
        return new_id

    def reset_tracking(self):
        """Reset face tracking data between video segments"""
        self.known_faces.clear()
        self.logger.info("Face tracking data reset")