import cv2
import numpy as np

def unsharp_mask(image, kernel_size=(5,5), sigma=1.0, amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    mask = cv2.subtract(image, blurred)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return sharpened

image = cv2.imread('test.jpg')
result = unsharp_mask(image, kernel_size=(5,5), sigma=1.0, amount=1.5, threshold=10)

cv2.imshow('Original', image)
cv2.imshow('Sharpened (Unsharp Mask)', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# write a program to read an image and perform unsharp masking