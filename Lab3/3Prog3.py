import cv2
import numpy as np

# Read input image
image = cv2.imread('test.jpg')   # put your image in the same folder
if image is None:
    print("Error: Image not found!")
    exit()

# Apply Box Filter (Mean filter)
box_filtered = cv2.blur(image, (7, 7))

# Apply Gaussian Filter
gaussian_filtered = cv2.GaussianBlur(image, (7, 7), 1.5)

# Show results
cv2.imshow('Original Image', image)
cv2.imshow('Box Filtered', box_filtered)
cv2.imshow('Gaussian Filtered', gaussian_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Write a program to compare box filter and guassian filter image outputs