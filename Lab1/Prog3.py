import cv2

# Read the image
img = cv2.imread("sample.png")   # make sure sample.png is in the same folder

if img is None:
    print("❌ Could not read image. Check filename/path.")
    exit()

# Choose pixel location (x=50, y=100 for example)
x, y = 50, 100

# OpenCV uses BGR format by default
(b, g, r) = img[y, x]

print(f"Pixel at (x={x}, y={y}): R={r}, G={g}, B={b}")

# Display the image with a mark on that pixel
marked = img.copy()
cv2.circle(marked, (x, y), 5, (0, 0, 255), -1)  # red dot

cv2.imshow("Image", marked)
cv2.waitKey(0)
cv2.destroyAllWindows()

# write a simple program to extracting the RGB values of a pixel