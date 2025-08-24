# Lab3/Prog2.py
import cv2
import numpy as np

# Read image in grayscale
image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found. Place dollar.jpg in the Lab3 folder.")
    exit()

# Compute Sobel gradients
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Convert to absolute values
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

# Combine both gradients
gradient = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Show results
cv2.imshow('Original', image)
cv2.imshow('Gradient X', abs_grad_x)
cv2.imshow('Gradient Y', abs_grad_y)
cv2.imshow('Combined Gradient', gradient)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Write a program to obtain gradient of an image