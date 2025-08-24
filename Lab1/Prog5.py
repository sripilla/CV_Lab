import cv2

# Read the image
img = cv2.imread("sample.png")   # make sure sample.png is in the same folder

if img is None:
    print("❌ Could not read image. Check filename/path.")
    exit()

# Resize to fixed dimensions (e.g., 300x200)
resized_fixed = cv2.resize(img, (300, 200))

# Resize by scale factor (e.g., half size and double size)
resized_half = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)   # 50% smaller
resized_double = cv2.resize(img, (0, 0), fx=2.0, fy=2.0) # 200% larger

# Show results
cv2.imshow("Original", img)
cv2.imshow("Fixed 300x200", resized_fixed)
cv2.imshow("Half Size", resized_half)
cv2.imshow("Double Size", resized_double)

cv2.waitKey(0)
cv2.destroyAllWindows()
