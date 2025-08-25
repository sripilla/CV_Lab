"""
Lab 5 – Basic Feature Extraction: Harris & FAST Corner Detection

Usage examples
--------------
# 1) Use default params on all images inside data/ folder
python Lab5_Feature_Extraction.py --images "data/*.jpg"

# 2) Tune thresholds
python Lab5_Feature_Extraction.py --images "data/*.png" --harris_thresh 0.02 --fast_thresh 15

# 3) Disable matplotlib popups and only save results to out/
python Lab5_Feature_Extraction.py --images "data/*.jpg" --no_show

Notes
-----
- Input: any set of images matching the glob pattern passed to --images
- Output: annotated images saved to out/ as <name>_HARRIS.jpg and <name>_FAST.jpg
- Visualization: a Matplotlib figure compares original, Harris, and FAST side-by-side.

This script is **self-contained** and uses OpenCV + NumPy + Matplotlib only.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import glob
from typing import List, Tuple

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- Harris Corner Utilities -----------------------------
def harris_keypoints(
    gray: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    thresh: float = 0.01,
) -> List[cv.KeyPoint]:
    """Compute Harris response and convert strong responses to cv.KeyPoint list.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image in uint8 or float32. Will be converted to float32 as required.
    block_size : int
        Neighborhood size considered for corner detection.
    ksize : int
        Aperture parameter for the Sobel() operator (odd: 3,5,7...).
    k : float
        Harris detector free parameter (usually 0.04–0.06).
    thresh : float
        Relative threshold as a fraction of max response; points above thresh*max are kept.

    Returns
    -------
    List[cv.KeyPoint]
        List of OpenCV KeyPoints for visualization with cv.drawKeypoints.
    """
    if gray.dtype != np.float32:
        f32 = np.float32(gray)
    else:
        f32 = gray

    # Compute Harris response map
    R = cv.cornerHarris(src=f32, blockSize=block_size, ksize=ksize, k=k)

    # Optional: dilate for better marking (matches common tutorials)
    R_dilated = cv.dilate(R, None)

    # Normalize threshold relative to strongest response
    R_max = R_dilated.max() if R_dilated.size > 0 else 0
    if R_max <= 0:
        return []
    mask = R_dilated > (thresh * R_max)

    # Extract coordinates and convert to KeyPoints
    ys, xs = np.where(mask)
    keypoints = [cv.KeyPoint(float(x), float(y), _size=3) for (x, y) in zip(xs, ys)]

    return keypoints


# ------------------------------ FAST Corner Utilities ------------------------------
def fast_keypoints(
    gray: np.ndarray,
    threshold: int = 25,
    nonmax: bool = True,
    type_: int = cv.FAST_FEATURE_DETECTOR_TYPE_9_16,
) -> List[cv.KeyPoint]:
    """Run FAST corner detector and return keypoints.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image in uint8.
    threshold : int
        FAST intensity threshold.
    nonmax : bool
        Non-maximum suppression toggle.
    type_ : int
        FAST detector type (7_12, 9_16).
    """
    fast = cv.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax, type=type_)
    kps = fast.detect(gray, None)
    return kps


# ------------------------------- Visualization Utils -------------------------------
def draw_keypoints_bgr(img_bgr: np.ndarray, kps: List[cv.KeyPoint]) -> np.ndarray:
    """Return an image with keypoints rendered as small circles.

    Using cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS would draw orientation/size
    if available; for corners we stick to simple circles for clarity.
    """
    out = cv.drawKeypoints(
        image=img_bgr,
        keypoints=kps,
        outImage=None,
        color=(0, 255, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DEFAULT,
    )
    return out


def ensure_outdir(path: str | Path = "out") -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------- Runner --------------------------------------
def process_image(
    img_path: str,
    harris_params: Tuple[int, int, float, float],
    fast_params: Tuple[int, bool, int],
    save_dir: Path,
    show: bool = True,
) -> None:
    name = Path(img_path).stem

    # Read and prep
    bgr = cv.imread(img_path)
    if bgr is None:
        print(f"[WARN] Could not read image: {img_path}")
        return
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    # Unpack params
    hbsize, hksize, hk, hth = harris_params
    fth, fnms, ftype = fast_params

    # Harris
    harris_kps = harris_keypoints(gray, block_size=hbsize, ksize=hksize, k=hk, thresh=hth)
    img_harris = draw_keypoints_bgr(bgr, harris_kps)

    # FAST
    fast_kps = fast_keypoints(gray, threshold=fth, nonmax=fnms, type_=ftype)
    img_fast = draw_keypoints_bgr(bgr, fast_kps)

    # Save results
    out_harris = save_dir / f"{name}_HARRIS.jpg"
    out_fast = save_dir / f"{name}_FAST.jpg"
    cv.imwrite(str(out_harris), img_harris)
    cv.imwrite(str(out_fast), img_fast)

    print(f"[OK] {name}: Harris={len(harris_kps)} corners | FAST={len(fast_kps)} corners")

    if show:
        # Convert BGR->RGB for Matplotlib display
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        rgb_harris = cv.cvtColor(img_harris, cv.COLOR_BGR2RGB)
        rgb_fast = cv.cvtColor(img_fast, cv.COLOR_BGR2RGB)

        plt.figure(figsize=(13, 4))
        plt.suptitle(f"Corners on {name}")
        plt.subplot(1, 3, 1); plt.imshow(rgb); plt.title("Original"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(rgb_harris); plt.title(f"Harris ({len(harris_kps)})"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(rgb_fast); plt.title(f"FAST ({len(fast_kps)})"); plt.axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 5 – Harris & FAST Corner Detection")
    p.add_argument("--images", type=str, required=True, help="Glob pattern for images, e.g., 'data/*.jpg'")

    # Harris params
    p.add_argument("--harris_block_size", type=int, default=2, help="Neighborhood size for Harris (default: 2)")
    p.add_argument("--harris_ksize", type=int, default=3, help="Sobel aperture size (odd). Default: 3")
    p.add_argument("--harris_k", type=float, default=0.04, help="Harris k parameter (0.04–0.06). Default: 0.04")
    p.add_argument("--harris_thresh", type=float, default=0.01, help="Relative threshold fraction of max R. Default: 0.01")

    # FAST params
    p.add_argument("--fast_thresh", type=int, default=25, help="FAST threshold. Default: 25")
    p.add_argument("--fast_no_nms", action="store_true", help="Disable non-maximum suppression for FAST")
    p.add_argument(
        "--fast_type",
        type=str,
        default="9_16",
        choices=["7_12", "9_16"],
        help="FAST type (circle test). Default: 9_16",
    )

    # I/O and display
    p.add_argument("--out", type=str, default="out", help="Output directory (default: out)")
    p.add_argument("--no_show", action="store_true", help="Do not display figures (just save results)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve glob
    img_paths = sorted(glob.glob(args.images))
    if not img_paths:
        print(f"[ERROR] No images matched pattern: {args.images}")
        return

    save_dir = ensure_outdir(args.out)

    # Map FAST type string to OpenCV constant
    fast_type_map = {
        "7_12": cv.FAST_FEATURE_DETECTOR_TYPE_7_12,
        "9_16": cv.FAST_FEATURE_DETECTOR_TYPE_9_16,
    }

    harris_params = (
        args.harris_block_size,
        args.harris_ksize,
        args.harris_k,
        args.harris_thresh,
    )
    fast_params = (
        args.fast_thresh,
        (not args.fast_no_nms),
        fast_type_map[args.fast_type],
    )

    for p in img_paths:
        process_image(
            img_path=p,
            harris_params=harris_params,
            fast_params=fast_params,
            save_dir=save_dir,
            show=(not args.no_show),
        )


if __name__ == "__main__":
    main()

# Implement basic feature extraction algorithm given below 
# 1. Harris corner detection 
# 2. FAST corner detection 
# Apply it to different images and visualize detected keypoints