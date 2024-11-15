# utils/capture_quadrant_images.py

import cv2
import os
import dlib
from imutils import face_utils
import json
import time
import sys
import numpy as np
from screeninfo import get_monitors

def draw_facial_landmarks(frame, shape):
    """Draw rectangles around eyes and facial landmarks."""
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    
    # Draw rectangles around eyes
    left_rect = cv2.boundingRect(left_eye)
    right_rect = cv2.boundingRect(right_eye)
    cv2.rectangle(frame, (left_rect[0], left_rect[1]), 
                  (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (right_rect[0], right_rect[1]), 
                  (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]), (0, 255, 0), 2)

    # Draw lines connecting facial landmarks
    for (x, y) in shape:
        cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

def draw_focus_circle(frame, quadrant, frame_width, frame_height):
    """Draw a neon green circle to guide the user to the correct quadrant."""
    radius = 100  # Increased radius for better visibility
    thickness = 3
    color = (0, 255, 0)  # Neon green

    if quadrant == 'top_left':
        center = (int(frame_width * 0.25), int(frame_height * 0.25))
    elif quadrant == 'top_right':
        center = (int(frame_width * 0.75), int(frame_height * 0.25))
    elif quadrant == 'bottom_left':
        center = (int(frame_width * 0.25), int(frame_height * 0.75))
    elif quadrant == 'bottom_right':
        center = (int(frame_width * 0.75), int(frame_height * 0.75))
    else:
        center = (frame_width // 2, frame_height // 2)  # Default center

    cv2.circle(frame, center, radius, color, thickness)

def capture_relative_pupil_positions(quadrant, num_samples=100, camera_index=1, output_dir='models/quadrants', interval_ms=20):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    quadrant_dir = os.path.join(output_dir, quadrant)
    os.makedirs(quadrant_dir, exist_ok=True)

    # Open the camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to access camera index {camera_index}")
        return

    # Initialize Dlib's face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor_path = 'utils/shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(predictor_path):
        print(f"Error: Shape predictor file not found at {predictor_path}")
        cap.release()
        return
    predictor = dlib.shape_predictor(predictor_path)

    # Fetch screen dimensions dynamically
    try:
        monitor = get_monitors()[0]
        screen_width, screen_height = monitor.width, monitor.height
    except Exception as e:
        print(f"Error fetching screen dimensions: {e}")
        cap.release()
        return

    window_name = "Eye Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(window_name, 0, 0)  # Position window at top-left corner

    print(f"\nPrepare to capture {num_samples} samples for {quadrant}.")
    print("Look at the circle on the screen. Capturing will start soon.")

    # Display countdown with focus circle
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            cap.release()
            cv2.destroyAllWindows()
            return

        frame_height, frame_width = frame.shape[:2]
        draw_focus_circle(frame, quadrant, frame_width, frame_height)
        cv2.putText(frame, f"Starting in {i}...", (int(frame_width*0.4), int(frame_height*0.5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit("Exited.")
            return

    # Start capturing data
    print(f"Capturing samples for {quadrant}. Press 'q' to quit early.")

    samples = []
    count = 0
    last_capture_time = time.time()

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        frame_height, frame_width = frame.shape[:2]
        draw_focus_circle(frame, quadrant, frame_width, frame_height)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Draw facial landmarks and eye boxes
            draw_facial_landmarks(frame, shape)

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

            # Save data sample
            sample = {
                "quadrant": quadrant,
                "left_eye": [left_rel_x, left_rel_y],
                "right_eye": [right_rel_x, right_rel_y]
            }
            samples.append(sample)
            count += 1

            # Annotate the frame for visualization
            cv2.rectangle(frame, (left_rect[0], left_rect[1]),
                          (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), (0, 255, 0), 2)
            cv2.rectangle(frame, (right_rect[0], right_rect[1]),
                          (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]), (0, 255, 0), 2)
            cv2.circle(frame, tuple(left_eye_center), 5, (255, 0, 0), -1)
            cv2.circle(frame, tuple(right_eye_center), 5, (255, 0, 0), -1)

            # Display progress
            cv2.putText(frame, f"Captured: {count}/{num_samples}", 
                        (int(frame_width*0.05), int(frame_height*0.05)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Check if interval_ms have passed to capture next image
            current_time = time.time()
            if (current_time - last_capture_time) * 1000 >= interval_ms:
                # Save the current frame
                file_path = os.path.join(quadrant_dir, f"{quadrant}_{count}.jpg")
                cv2.imwrite(file_path, frame)
                print(f"Saved: {file_path}")
                last_capture_time = current_time

        # Show the frame with landmarks and focus circle
        cv2.imshow(window_name, frame)

        # Handle user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting early...")
            break

    # Save samples to a JSON file
    output_json = os.path.join(output_dir, f"{quadrant}_data.json")
    try:
        with open(output_json, 'w') as f:
            json.dump(samples, f, indent=4)
        print(f"Saved {len(samples)} samples to {output_json}")
    except Exception as e:
        print(f"Error saving JSON data: {e}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    quadrants = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    for quadrant in quadrants:
        capture_relative_pupil_positions(quadrant)
        print("Switching to the next quadrant. Please prepare...")
        time.sleep(2)  # Wait 2 seconds before moving to the next quadrant
