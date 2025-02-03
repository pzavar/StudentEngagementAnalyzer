import cv2
import numpy as np
from typing import Generator, Tuple, List
import tempfile
import logging

class VideoProcessor:
    def __init__(self, sampling_rate: int = 60, min_face_size: tuple = (30, 30)):
        """
        Initialize VideoProcessor with enhanced multi-face detection capabilities

        Args:
            sampling_rate: Number of seconds between frame captures
            min_face_size: Minimum face size to detect (width, height)
        """
        self.sampling_rate = sampling_rate
        self.min_face_size = min_face_size
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_video(self, video_file) -> Generator[Tuple[float, np.ndarray], None, None]:
        """
        Process video file and yield frames at specified intervals
        Enhanced to handle Zoom grid layouts with multiple participants
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            tmp_file.flush()

            cap = cv2.VideoCapture(tmp_file.name)

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = int(fps * self.sampling_rate)

            self.logger.info(f"Processing video: {fps} FPS, {frame_count} frames")
            self.logger.info(f"Sampling every {frame_interval} frames ({self.sampling_rate} seconds)")

            current_frame = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame % frame_interval == 0:
                    timestamp = current_frame / fps

                    # Convert to RGB for consistent processing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Enhance frame quality
                    frame_rgb = self._enhance_frame(frame_rgb)

                    yield timestamp, frame_rgb

                current_frame += 1

            cap.release()
            self.logger.info("Video processing completed")

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame quality for better face detection
        """
        # Normalize brightness and contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return enhanced