import cv2

# Read the image
img = cv2.imread("sample.png")   # make sure sample.png is in the same folder

if img is None:
    print("❌ Could not read image. Check filename/path.")
    exit()

# Get image dimensions
(h, w) = img.shape[:2]
center = (w // 2, h // 2)   # rotation center (image middle)

# Create rotation matrix (angle=45 degrees, scale=1.0)
M = cv2.getRotationMatrix2D(center, 45, 1.0)

# Apply rotation
rotated = cv2.warpAffine(img, M, (w, h))

# Show images
cv2.imshow("Original", img)
cv2.imshow("Rotated 45 Degrees", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()
