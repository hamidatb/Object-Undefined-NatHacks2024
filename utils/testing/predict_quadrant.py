import cv2
import pickle
import dlib
from imutils import face_utils
import numpy as np
import os
import time

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

            # Return the quadrant and the frame
            return quadrant_map[quadrant], frame

        # If no face is detected, return None
        return None, frame


if __name__ == "__main__":
    # Initialize the predictor
    predictor = QuadrantPredictor()
    quadrant_map = {0: 'top_left', 1: 'top_right', 2: 'bottom_left', 3: 'bottom_right'}

    # Open the webcam
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Unable to access camera.")
        exit()

    # Timing variables for tracking quadrant focus time
    quadrant_start_time = None
    current_quadrant = None
    quadrant_times = {'top_left': 0, 'top_right': 0, 'bottom_left': 0, 'bottom_right': 0}

    # Full screen window
    cv2.namedWindow("Quadrant Prediction", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Quadrant Prediction", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Get the frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Predict the quadrant and get the annotated frame
        quadrant, annotated_frame = predictor.predict(frame)

        # Highlight the predicted quadrant with a green filled rectangle
        if quadrant is not None:
            # Track time in the current quadrant
            if quadrant != current_quadrant:
                if current_quadrant is not None and quadrant_start_time is not None:
                    elapsed_time = time.time() - quadrant_start_time
                    quadrant_times[current_quadrant] += elapsed_time
                    print(f"Time spent in {current_quadrant}: {elapsed_time:.2f} seconds")

                # Update to the new quadrant
                current_quadrant = quadrant
                quadrant_start_time = time.time()

            # Define the quadrant areas
            top_left = (0, 0, frame_width // 2, frame_height // 2)
            top_right = (frame_width // 2, 0, frame_width, frame_height // 2)
            bottom_left = (0, frame_height // 2, frame_width // 2, frame_height)
            bottom_right = (frame_width // 2, frame_height // 2, frame_width, frame_height)

            # Map quadrant names to coordinates
            quadrant_rects = {
                'top_left': top_left,
                'top_right': top_right,
                'bottom_left': bottom_left,
                'bottom_right': bottom_right
            }

            # Get the coordinates for the predicted quadrant
            x1, y1, x2, y2 = quadrant_rects[quadrant]

            # Draw a green filled rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=-1)

        # Show the annotated frame
        cv2.imshow("Quadrant Prediction", annotated_frame)

        # Quit the application when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Add the final time spent in the current quadrant
    if current_quadrant is not None and quadrant_start_time is not None:
        elapsed_time = time.time() - quadrant_start_time
        quadrant_times[current_quadrant] += elapsed_time
        print(f"Time spent in {current_quadrant}: {elapsed_time:.2f} seconds")

    # Print total time spent in each quadrant
    print("Total Time in Each Quadrant:")
    for q, t in quadrant_times.items():
        print(f"{q}: {t:.2f} seconds")

    # Release resources gracefully
    cap.release()
    cv2.destroyAllWindows()
