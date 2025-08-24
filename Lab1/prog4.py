import cv2
import numpy as np

# Create a blank black image (500x500, 3 channels for RGB)
img = np.zeros((500, 500, 3), dtype=np.uint8)

# Define rectangle start & end points
start_point = (100, 100)   # top-left corner
end_point   = (400, 400)   # bottom-right corner

# Define color (B, G, R) and thickness
color = (0, 255, 0)   # Green
thickness = 3         # -1 = filled rectangle, positive value = border thickness

# Draw the rectangle
cv2.rectangle(img, start_point, end_point, color, thickness)

# Show the image
cv2.imshow("Rectangle", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
