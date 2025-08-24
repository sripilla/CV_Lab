# Q5 - Edge Detection Algorithms
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found. Place 'test.jpg' in the same folder.")
    exit()

# 1. Canny Edge Detection
edges_canny = cv2.Canny(image, 100, 200)

# 2. Sobel Edge Detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sobel_x, sobel_y)
sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 3. Laplacian Edge Detection
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)

# Show results
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(edges_canny, cmap='gray')
plt.title("Canny Edges")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(sobel_mag, cmap='gray')
plt.title("Sobel Edges")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Edges")
plt.axis("off")

plt.tight_layout()
plt.show()
