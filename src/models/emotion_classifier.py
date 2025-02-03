import tensorflow as tf
import numpy as np
from typing import List, Tuple
import os
import streamlit as st

class EmotionClassifier:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the TFLite model for emotion classification"""
        model_path = "model.tflite"
        if not os.path.exists(model_path):
            st.error(f"Error: {model_path} not found. Please ensure the model file is present in the project root directory.")
            st.info("Please train the model first using the provided training script. See README.md for instructions.")
            return

        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        self.model = interpreter

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        if self.model is None:
            return None

        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image

    def predict_emotion(self, face_image: np.ndarray) -> Tuple[str, float]:
        """Predict emotion from face image"""
        if self.model is None:
            return "unknown", 0.0

        processed_image = self.preprocess_image(face_image)
        if processed_image is None:
            return "unknown", 0.0

        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        self.model.set_tensor(input_details[0]['index'], 
                            np.expand_dims(processed_image, axis=0))
        self.model.invoke()

        predictions = self.model.get_tensor(output_details[0]['index'])
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]

        return self.emotions[emotion_idx], confidence

    def map_to_engagement(self, emotion: str) -> str:
        """Map emotion to engagement level"""
        if emotion == "unknown":
            return "unknown"

        engagement_mapping = {
            'happy': 'highly engaged',
            'surprise': 'highly engaged',
            'neutral': 'moderately engaged',
            'sad': 'not engaged',
            'angry': 'not engaged',
            'disgust': 'not engaged',
            'fear': 'not engaged'
        }
        return engagement_mapping[emotion]