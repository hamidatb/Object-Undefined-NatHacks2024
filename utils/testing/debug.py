import cv2
import numpy as np
from screeninfo import get_monitors

# Get screen resolution
monitor = get_monitors()[0]
screen_width, screen_height = monitor.width, monitor.height

# Create a window and remove borders
window_name = "Borderless Fullscreen"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Create a blank image matching the screen resolution
frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

# Display the borderless window
while True:
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break

cv2.destroyAllWindows()
