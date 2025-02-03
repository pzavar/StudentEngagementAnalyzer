import cv2
import numpy as np
from typing import Generator, Tuple
import tempfile

class VideoProcessor:
    def __init__(self, sampling_rate: int = 60):
        """
        Initialize VideoProcessor
        Args:
            sampling_rate: Number of seconds between frame captures
        """
        self.sampling_rate = sampling_rate

    def process_video(self, video_file) -> Generator[Tuple[float, np.ndarray], None, None]:
        """
        Process video file and yield frames at specified intervals
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            tmp_file.flush()
            
            cap = cv2.VideoCapture(tmp_file.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * self.sampling_rate)
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield timestamp, frame_rgb
                    
                frame_count += 1
                
            cap.release()
