import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# ---------- Histogram specification for a single-channel (grayscale) image ----------
def histogram_matching(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Map source intensities so its histogram matches template's (uint8, 0-255)."""
    if source.dtype != np.uint8 or template.dtype != np.uint8:
        raise ValueError("Inputs must be uint8 images (0..255).")

    oldshape = source.shape
    s = source.ravel()
    t = template.ravel()

    # Unique values and counts
    s_values, bin_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(t, return_counts=True)

    # Normalized CDFs
    s_quantiles = np.cumsum(s_counts).astype(np.float64); s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64); t_quantiles /= t_quantiles[-1]

    # For each source CDF level, find corresponding template intensity (linear interp)
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    # Map back to image shape
    matched = interp_t_values[bin_idx].reshape(oldshape).astype(np.uint8)
    return matched

# ---------- Load images (grayscale) ----------
src_path = Path("data/test1.jpg")   # INPUT image
ref_path = Path("data/test2.jpg")   # REFERENCE image

src = cv.imread(str(src_path), cv.IMREAD_GRAYSCALE)
ref = cv.imread(str(ref_path), cv.IMREAD_GRAYSCALE)

if src is None or ref is None:
    print("❌ Error: could not load input or reference image.")
    print(f"    Input path:     {src_path.resolve()}")
    print(f"    Reference path: {ref_path.resolve()}")
    sys.exit(1)

# ---------- Perform histogram specification ----------
matched = histogram_matching(src, ref)

# ---------- Show images with OpenCV (optional) ----------
# cv.imshow("Input (Gray)", src)
# cv.imshow("Reference (Gray)", ref)
# cv.imshow("Matched (Gray)", matched)
# cv.waitKey(0); cv.destroyAllWindows()

# ---------- Plot images + histograms with Matplotlib ----------
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1); plt.title("Input Image");     plt.imshow(src, cmap="gray"); plt.axis("off")
plt.subplot(3, 2, 2); plt.title("Histogram of Input")
plt.hist(src.ravel(), 256, [0, 256])

plt.subplot(3, 2, 3); plt.title("Reference Image"); plt.imshow(ref, cmap="gray"); plt.axis("off")
plt.subplot(3, 2, 4); plt.title("Histogram of Reference")
plt.hist(ref.ravel(), 256, [0, 256])

plt.subplot(3, 2, 5); plt.title("Matched Image");   plt.imshow(matched, cmap="gray"); plt.axis("off")
plt.subplot(3, 2, 6); plt.title("Histogram of Matched")
plt.hist(matched.ravel(), 256, [0, 256])

plt.tight_layout()
plt.show()

# write a program to read an input image, reference image and perform histogram specification
