import cv2 as cv
import sys
from pathlib import Path

# Change this path to your actual image
img_path = Path("data/test1.jpg")

# Check if file exists
if not img_path.exists():
    print(f"❌ Image not found at {img_path.resolve()}")
    sys.exit(1)

# Read the image
img = cv.imread(str(img_path))

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply histogram equalization
equalized = cv.equalizeHist(gray)

# Show results
cv.imshow("Original Gray", gray)
cv.imshow("Histogram Equalized", equalized)

cv.waitKey(0)
cv.destroyAllWindows()


# Write a program to read an image and perform histogram equalization