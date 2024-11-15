import cv2
import pickle
import dlib
from imutils import face_utils
import numpy as np
import os

class QuadrantPredictor:
    def __init__(self, model_path='models/look_at_quadrants_model.pkl', scaler_path='models/scaler.pkl'):
        # Load the trained quadrant model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load the scaler used during training
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file {scaler_path} not found. Please save the scaler during training.")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Initialize Dlib's face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = 'utils/shape_predictor_68_face_landmarks.dat'
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Shape predictor file {predictor_path} not found.")
        
        self.predictor = dlib.shape_predictor(predictor_path)

    def predict(self, frame):
        """Predict the quadrant based on the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Extract left and right eye regions
            left_eye = shape[36:42]
            right_eye = shape[42:48]

            # Calculate the center of the pupils
            left_eye_center = left_eye.mean(axis=0).astype("int")
            right_eye_center = right_eye.mean(axis=0).astype("int")

            # Get bounding rectangles for normalization
            left_rect = cv2.boundingRect(left_eye)
            right_rect = cv2.boundingRect(right_eye)

            # Calculate pupil position relative to eye
            left_rel_x = (left_eye_center[0] - left_rect[0]) / left_rect[2]
            left_rel_y = (left_eye_center[1] - left_rect[1]) / left_rect[3]
            right_rel_x = (right_eye_center[0] - right_rect[0]) / right_rect[2]
            right_rel_y = (right_eye_center[1] - right_rect[1]) / right_rect[3]

            # Create feature vector
            features = np.array([left_rel_x, left_rel_y, right_rel_x, right_rel_y]).reshape(1, -1)

            # Scale features (if scaler was used during training)
            features_scaled = self.scaler.transform(features)

            # Predict quadrant
            quadrant = self.model.predict(features_scaled)[0]
            quadrant_map = {0: 'top_left', 1: 'top_right', 2: 'bottom_left', 3: 'bottom_right'}

            # Return the quadrant
            return quadrant_map[quadrant], None  # Return None for the frame

        # If no face is detected, return None
        return None, None