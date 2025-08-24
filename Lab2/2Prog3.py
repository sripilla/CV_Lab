import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ---- read input image ----
img_path = Path("data/test1.jpg")
image = cv2.imread(str(img_path))                    # BGR
if image is None:
    print("Error: could not load image:", img_path.resolve())
    sys.exit(1)

# ---- resizing ----
resized_image = cv2.resize(image, (400, 300), interpolation=cv2.INTER_LINEAR)  # (width, height)

# ---- cropping ----
x, y, w, h = 50, 50, 200, 150                       # top-left (x,y), width, height
cropped_image = resized_image[y:y+h, x:x+w]          # array slicing: [rows, cols] -> [y:y+h, x:x+w]

# ---- convert BGR to RGB for Matplotlib display ----
image_rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

# ---- show results ----
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Resized Image (400×300)")
plt.imshow(resized_rgb)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Cropped Image (200×150)")
plt.imshow(cropped_rgb)
plt.axis("off")

plt.tight_layout()
plt.show()
