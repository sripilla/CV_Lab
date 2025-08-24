# Prog1_ManualThreshold.py
# Create binary image using manual thresholding

import cv2
import numpy as np

def manual_threshold(image, thresh_value):
    height, width = image.shape
    binary_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if image[i, j] > thresh_value:
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0
    return binary_image

def main():
    image_path = "data/dog.jpg"   # change as needed
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise FileNotFoundError("Image not found! Place dog.jpg in data folder.")

    threshold_value = 127
    binary = manual_threshold(gray, threshold_value)

    # Show results
    cv2.imshow("Original Grayscale", gray)
    cv2.imshow("Binary Image", binary)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# write a program to create binary images using thresholding methods.