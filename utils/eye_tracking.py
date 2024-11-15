# eye_tracking.py
# Important: Download the shape_predictor_68_face_landmarks.dat file from here and place it in the utils/ directory. Extract the .bz2 file to get the .dat file.

import cv2
import dlib
from imutils import face_utils
import numpy as np
import threading
import time

class EyeTracker:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('utils/shape_predictor_68_face_landmarks.dat')  # Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.cap = cv2.VideoCapture(0)
        self.running = False

    def start_tracking(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[36:42]
                rightEye = shape[42:48]
                # Compute the center of mass for each eye
                leftEyeCenter = leftEye.mean(axis=0).astype("int")
                rightEyeCenter = rightEye.mean(axis=0).astype("int")
                # Draw circles on the eyes
                cv2.circle(frame, tuple(leftEyeCenter), 2, (0, 255, 0), -1)
                cv2.circle(frame, tuple(rightEyeCenter), 2, (0, 255, 0), -1)
                # Calculate gaze direction (placeholder logic)
                cv2.putText(frame, "Gaze Direction: Center", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Eye Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def stop_tracking(self):
        self.running = False
