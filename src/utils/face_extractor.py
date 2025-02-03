import cv2
import numpy as np
from typing import List, Tuple
import os

class FaceExtractor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_face_centers = []
        self.known_face_ids = []
        self.tracking_threshold = 50  # pixels

    def extract_faces(self, frame: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Extract and identify faces from a frame
        Returns list of (student_id, face_image) tuples
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        results = []
        for (x, y, w, h) in faces:
            face_center = (x + w//2, y + h//2)
            student_id = self._identify_face(face_center)
            face_image = frame[y:y+h, x:x+w]
            results.append((student_id, face_image))

        return results

    def _identify_face(self, face_center: tuple) -> str:
        """Identify a face based on position tracking"""
        if not self.known_face_centers:
            student_id = f"student_{len(self.known_face_centers)}"
            self.known_face_centers.append(face_center)
            self.known_face_ids.append(student_id)
            return student_id

        # Find the closest known face center
        distances = [
            np.sqrt((center[0] - face_center[0])**2 + (center[1] - face_center[1])**2)
            for center in self.known_face_centers
        ]
        min_distance = min(distances)

        if min_distance < self.tracking_threshold:
            return self.known_face_ids[distances.index(min_distance)]

        student_id = f"student_{len(self.known_face_centers)}"
        self.known_face_centers.append(face_center)
        self.known_face_ids.append(student_id)
        return student_id