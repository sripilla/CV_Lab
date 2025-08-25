"""
Lab 5 – Custom Feature Descriptors: SIFT & SURF (and comparison to OpenCV)

What this script does
---------------------
1) Implements **custom SIFT descriptor** (128-D):
   - gradient magnitude/orientation, Gaussian weighting, 4x4 cells × 8 bins
   - orientation normalization using dominant orientation
   - scale-normalized sampling using keypoint size
2) Implements **custom SURF descriptor** (64-D):
   - integral image, Haar-wavelet responses
   - 4x4 subregions, each yields [sum(dx), sum(dy), sum(|dx|), sum(|dy|)]
3) Uses OpenCV SIFT/SURF (if available) for reference comparison
4) Evaluates **robustness** to: scale, rotation, affine transforms
   - Generates transformed versions of input
   - Matches descriptors (ratio test) vs original
   - Reports #keypoints, #matches, match rate; saves visualizations

Requirements & Notes
--------------------
- Python 3.8+
- OpenCV (cv2). For OpenCV SIFT/SURF reference:
  * SIFT on OpenCV 3.4.3 (xfeatures2d) – as requested. If you only have newer OpenCV,
    standard cv2.SIFT_create() also works for comparison but differs slightly.
  * SURF is patented and typically only in opencv_contrib (xfeatures2d). If missing,
    the script will skip the OpenCV SURF comparison but still run custom SURF.
- NumPy, Matplotlib

Usage
-----
python Lab5_SIFT_SURF_Custom.py --image data/chessboard.jpg --no_show
python Lab5_SIFT_SURF_Custom.py --image data/graffiti.png --sift_ref --surf_ref

Outputs
-------
- out/<name>_*_matches.jpg: match visualizations for each transform & method
- A printed table comparing custom vs OpenCV across transforms

"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# ------------------------------ Utility Helpers ------------------------------

def ensure_outdir(p: str | Path = "out") -> Path:
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# ------------------------------ Keypoint Detection ------------------------------

def detect_keypoints(gray: np.ndarray, max_kp: int = 800) -> List[cv.KeyPoint]:
    """Use OpenCV SIFT detector for keypoint *locations & scales & orientations*.
    If not available, fallback to goodFeaturesToTrack + gradient orientations.
    """
    kps: List[cv.KeyPoint] = []
    sift = None
    # Preferred: SIFT detector (robust scale/orientation)
    if hasattr(cv, "SIFT_create"):
        sift = cv.SIFT_create(nfeatures=max_kp)
    else:
        try:
            sift = cv.xfeatures2d.SIFT_create(nfeatures=max_kp)  # type: ignore[attr-defined]
        except Exception:
            sift = None
    if sift is not None:
        kps = sift.detect(gray, None)
        # prune to max_kp strongest
        kps = sorted(kps, key=lambda k: -k.response)[:max_kp]
        return kps

    # Fallback: Harris corners + simple orientation from gradient
    corners = cv.goodFeaturesToTrack(gray, maxCorners=max_kp, qualityLevel=0.01, minDistance=6)
    if corners is None:
        return []
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 360.0) % 360.0
    for c in corners.squeeze(1):
        x, y = float(c[0]), float(c[1])
        theta = float(ang[int(round(y)), int(round(x))])
        kps.append(cv.KeyPoint(x, y, _size=8, _angle=theta))
    return kps


# ------------------------------ Custom SIFT Descriptor ------------------------------
@dataclass
class SIFTParams:
    patch_radius: float = 8.0   # in *normalized* coordinates (~ half the window after scale)
    grid_n: int = 4             # 4x4 cells
    bins: int = 8               # 8 orientation bins per cell
    sigma_factor: float = 0.5   # Gaussian window relative to patch radius
    clip: float = 0.2           # descriptor vector clipping before renorm


def compute_gradients(gray_f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = cv.Sobel(gray_f, cv.CV_32F, 1, 0, ksize=1)
    gy = cv.Sobel(gray_f, cv.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx * gx + gy * gy)
    ori = (np.rad2deg(np.arctan2(gy, gx)) + 360.0) % 360.0
    return mag, ori


def dominant_orientation(mag: np.ndarray, ori: np.ndarray, cx: float, cy: float, scale: float) -> float:
    # Form a weighted orientation histogram around (cx, cy) within ~3*scale
    radius = int(round(3 * scale))
    if radius < 3:
        radius = 3
    H = np.zeros(36, dtype=np.float32)
    h, w = mag.shape
    y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius + 1))
    x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius + 1))
    sigma = 1.5 * scale if scale > 0 else 1.5
    if sigma <= 0:
        sigma = 1.5
    W = np.exp(-((np.arange(-radius, radius + 1)[:, None]) ** 2 + (np.arange(-radius, radius + 1)[None, :]) ** 2) / (2 * sigma * sigma))
    W = W[(y0 - int(cy) + radius):(y1 - int(cy) + radius), (x0 - int(cx) + radius):(x1 - int(cx) + radius)]

    m_patch = mag[y0:y1, x0:x1] * W
    o_patch = ori[y0:y1, x0:x1]
    bins = (o_patch / 10.0).astype(np.int32) % 36
    for b in range(36):
        H[b] = m_patch[bins == b].sum()
    # smooth histogram (simple circular smoothing)
    H = (np.roll(H, 1) + H + np.roll(H, -1)) / 3.0
    ang = (np.argmax(H) * 10.0) % 360.0
    return float(ang)


def sample_rotated_patch(gray_f: np.ndarray, cx: float, cy: float, s: float, theta_deg: float, half_width: int) -> np.ndarray:
    """Sample a square patch centered at (cx, cy) with size 2*half_width, rotated by theta, scaled by s.
    Use affine warp for proper interpolation.
    """
    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Build affine to map output patch coords to input image coords
    # Output coords u in [-half, half)
    S = s
    A = np.array([[ S * cos_t,  S * sin_t,  cx],
                  [-S * sin_t,  S * cos_t,  cy]], dtype=np.float32)
    # We want to map from patch coordinates to image: so create a grid and warpAffine
    patch_size = 2 * half_width
    patch = cv.warpAffine(gray_f, A, (patch_size, patch_size), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT101)
    return patch


def sift_descriptor_at(gray_f: np.ndarray, kp: cv.KeyPoint, prm: SIFTParams) -> np.ndarray:
    # Determine scale from keypoint size (OpenCV SIFT uses size ~ diameter of meaningful region)
    scale = max(1.0, kp.size / 2.0)
    # First, estimate dominant orientation (if angle < 0)
    mag, ori = compute_gradients(gray_f)
    theta = kp.angle if kp.angle >= 0 else dominant_orientation(mag, ori, kp.pt[0], kp.pt[1], scale)

    # Sample normalized patch aligned to theta and scaled
    half = int(prm.grid_n * prm.patch_radius)
    patch = sample_rotated_patch(gray_f, kp.pt[0], kp.pt[1], s=1.0 / scale, theta_deg=theta, half_width=half)
    pmag, pori = compute_gradients(patch)

    # Gaussian weighting across patch
    sigma = prm.sigma_factor * prm.grid_n * prm.patch_radius
    yy, xx = np.mgrid[-half:half, -half:half]
    W = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma)).astype(np.float32)
    pmag *= W

    # Pool into 4x4 cells, 8 bins each
    cell = (2 * half) // prm.grid_n
    desc = []
    for gy in range(prm.grid_n):
        for gx in range(prm.grid_n):
            y0, y1 = gy * cell, (gy + 1) * cell
            x0, x1 = gx * cell, (gx + 1) * cell
            m = pmag[y0:y1, x0:x1]
            o = pori[y0:y1, x0:x1]
            # relative orientation to kp theta
            o_rel = (o - theta + 360.0) % 360.0
            hist = np.zeros(prm.bins, dtype=np.float32)
            bin_idx = (o_rel / (360.0 / prm.bins)).astype(np.int32) % prm.bins
            for b in range(prm.bins):
                hist[b] = m[bin_idx == b].sum()
            desc.append(hist)
    v = np.concatenate(desc, axis=0)
    # Normalize → clip → renormalize (SIFT trick)
    v = v / (np.linalg.norm(v) + 1e-8)
    v = np.clip(v, 0, prm.clip)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v.astype(np.float32)


def compute_sift_custom(gray: np.ndarray, kps: List[cv.KeyPoint], prm: SIFTParams = SIFTParams()) -> Tuple[List[cv.KeyPoint], np.ndarray]:
    gray_f = gray.astype(np.float32)
    descs = []
    valid_kps: List[cv.KeyPoint] = []
    h, w = gray.shape
    border = int(prm.grid_n * prm.patch_radius + 2)
    for kp in kps:
        x, y = kp.pt
        if x < border or y < border or x >= w - border or y >= h - border:
            continue
        d = sift_descriptor_at(gray_f, kp, prm)
        if np.any(np.isnan(d)):
            continue
        valid_kps.append(kp)
        descs.append(d)
    if not descs:
        return [], np.zeros((0, prm.grid_n * prm.grid_n * prm.bins), dtype=np.float32)
    return valid_kps, np.vstack(descs)


# ------------------------------ Custom SURF Descriptor ------------------------------
@dataclass
class SURFParams:
    grid_n: int = 4          # 4x4 subregions
    sample_step: int = 5     # spacing for Haar sampling within each subregion
    haar_size: int = 9       # kernel size for Haar (odd)


def integral_image(gray: np.ndarray) -> np.ndarray:
    ii = cv.integral(gray)
    return ii.astype(np.float32)


def box_sum(ii: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    # integral image is (h+1, w+1)
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def haar_response(ii: np.ndarray, x: int, y: int, size: int, orientation: float) -> Tuple[float, float]:
    """Compute SURF-like Haar wavelet responses dx, dy around (x,y) aligned to orientation.
    We approximate alignment by rotating sample offsets instead of the box.
    """
    theta = np.deg2rad(orientation)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Define Haar rectangles in local coords: horizontal and vertical split
    half = size // 2
    # Sample points across a small region: use two adjacent boxes for dx and dy
    # We'll use four boxes to emulate Haar: left-right (dx) and top-bottom (dy)
    # Create helper to sum a small axis-aligned box via integral image
    def sum_box(cx: float, cy: float, hw: int, hh: int) -> float:
        x0 = int(round(cx - hw)); x1 = int(round(cx + hw))
        y0 = int(round(cy - hh)); y1 = int(round(cy + hh))
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(ii.shape[1] - 1, x1); y1 = min(ii.shape[0] - 1, y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return box_sum(ii, x0, y0, x1, y1)

    # Define offsets for left/right and top/bottom boxes in local frame
    # Rotate offsets by theta to map to image frame
    def rot(dx: float, dy: float) -> Tuple[float, float]:
        return (dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)

    # Horizontal pair for dx
    lx, ly = rot(-half/2, 0.0)
    rx, ry = rot(+half/2, 0.0)
    # Vertical pair for dy
    tx, ty = rot(0.0, -half/2)
    bx, by = rot(0.0, +half/2)

    # Box half-sizes (axis-aligned in image, approximate)
    hw = max(1, half // 2)
    hh = max(1, half // 2)

    sx, sy = x + lx, y + ly
    dx_pos = sum_box(sx, sy, hw, hh)
    sx, sy = x + rx, y + ry
    dx_neg = sum_box(sx, sy, hw, hh)

    sx, sy = x + tx, y + ty
    dy_pos = sum_box(sx, sy, hw, hh)
    sx, sy = x + bx, y + by
    dy_neg = sum_box(sx, sy, hw, hh)

    dx = (dx_pos - dx_neg) / (size * size)
    dy = (dy_pos - dy_neg) / (size * size)
    return dx, dy


def compute_surf_custom(gray: np.ndarray, kps: List[cv.KeyPoint], prm: SURFParams = SURFParams()) -> Tuple[List[cv.KeyPoint], np.ndarray]:
    ii = integral_image(gray)
    descs = []
    valid_kps: List[cv.KeyPoint] = []
    # SURF orientation: use gradient-based or keypoint-provided angle
    gray_f = gray.astype(np.float32)
    gx = cv.Sobel(gray_f, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray_f, cv.CV_32F, 0, 1, ksize=3)
    ori_map = (np.rad2deg(np.arctan2(gy, gx)) + 360.0) % 360.0

    for kp in kps:
        x, y = kp.pt
        x_i, y_i = int(round(x)), int(round(y))
        if x_i < 8 or y_i < 8 or x_i >= gray.shape[1]-8 or y_i >= gray.shape[0]-8:
            continue
        theta = kp.angle if kp.angle >= 0 else float(ori_map[y_i, x_i])

        # Build 4x4 subregions; per subregion, sample multiple Haar responses
        vec = []
        step = prm.sample_step
        sub = prm.haar_size
        span = prm.grid_n * step
        # Center the grid around the keypoint
        x0 = x_i - span // 2
        y0 = y_i - span // 2
        for gy_ in range(prm.grid_n):
            for gx_ in range(prm.grid_n):
                sum_dx = 0.0
                sum_dy = 0.0
                sum_adx = 0.0
                sum_ady = 0.0
                for sy in range(step):
                    for sx in range(step):
                        px = x0 + gx_ * step + sx
                        py = y0 + gy_ * step + sy
                        dx, dy = haar_response(ii, px, py, size=sub, orientation=theta)
                        sum_dx += dx; sum_dy += dy
                        sum_adx += abs(dx); sum_ady += abs(dy)
                vec.extend([sum_dx, sum_dy, sum_adx, sum_ady])
        v = np.array(vec, dtype=np.float32)
        nrm = np.linalg.norm(v) + 1e-8
        v = v / nrm
        valid_kps.append(kp)
        descs.append(v)
    if not descs:
        return [], np.zeros((0, prm.grid_n * prm.grid_n * 4), dtype=np.float32)
    return valid_kps, np.vstack(descs)


# ------------------------------ OpenCV Reference Descriptors ------------------------------

def opencv_sift(gray: np.ndarray, kps: List[cv.KeyPoint]) -> Tuple[List[cv.KeyPoint], np.ndarray]:
    sift = None
    if hasattr(cv, "SIFT_create"):
        sift = cv.SIFT_create()
    else:
        sift = cv.xfeatures2d.SIFT_create()  # type: ignore[attr-defined]
    kps2, desc = sift.compute(gray, kps)
    return kps2, desc


def opencv_surf(gray: np.ndarray, kps: List[cv.KeyPoint]) -> Optional[Tuple[List[cv.KeyPoint], np.ndarray]]:
    try:
        surf = cv.xfeatures2d.SURF_create(hessianThreshold=400)  # type: ignore[attr-defined]
    except Exception:
        return None
    kps2, desc = surf.compute(gray, kps)
    return kps2, desc


# ------------------------------ Matching & Evaluation ------------------------------

def ratio_matches(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.75) -> List[cv.DMatch]:
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    # Use L2 BF matcher
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def draw_and_save_matches(img1: np.ndarray, kps1: List[cv.KeyPoint], img2: np.ndarray, kps2: List[cv.KeyPoint], matches: List[cv.DMatch], path: Path, max_draw: int = 80) -> None:
    matches_sorted = sorted(matches, key=lambda m: m.distance)
    vis = cv.drawMatches(img1, kps1, img2, kps2, matches_sorted[:max_draw], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(str(path), vis)


# ------------------------------ Transforms for Robustness ------------------------------

def transform_scale(img: np.ndarray, s: float) -> np.ndarray:
    h, w = img.shape[:2]
    return cv.resize(img, (int(w * s), int(h * s)), interpolation=cv.INTER_LINEAR)


def transform_rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT101)


def transform_affine(img: np.ndarray, shear: float = 0.15) -> np.ndarray:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst = np.float32([[0, 0], [w - 1, 0], [int(shear * w), h - 1]])
    M = cv.getAffineTransform(src, dst)
    return cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT101)


# ------------------------------ Main Pipeline ------------------------------

def evaluate_on_pair(img_ref: np.ndarray, img_tgt: np.ndarray, method_name: str, desc_func) -> Tuple[int, int, int, float, List[cv.KeyPoint], List[cv.KeyPoint], List[cv.DMatch]]:
    gray1 = to_gray(img_ref)
    gray2 = to_gray(img_tgt)

    kps1 = detect_keypoints(gray1)
    kps2 = detect_keypoints(gray2)

    kps1, d1 = desc_func(gray1, kps1)
    kps2, d2 = desc_func(gray2, kps2)

    matches = ratio_matches(d1, d2)
    match_rate = len(matches) / max(1, min(len(kps1), len(kps2)))
    return len(kps1), len(kps2), len(matches), float(match_rate), kps1, kps2, matches


def method_selector(name: str):
    if name == "sift_custom":
        return lambda g, k: compute_sift_custom(g, k)
    if name == "surf_custom":
        return lambda g, k: compute_surf_custom(g, k)
    if name == "sift_ref":
        return lambda g, k: opencv_sift(g, k)
    if name == "surf_ref":
        return lambda g, k: opencv_surf(g, k) or ([], np.zeros((0,64), np.float32))
    raise ValueError(name)


def run_experiment(img: np.ndarray, out_dir: Path, include_ref_sift: bool, include_ref_surf: bool, show: bool) -> None:
    tests = [
        ("scale_0.75", lambda im: transform_scale(im, 0.75)),
        ("scale_1.5", lambda im: transform_scale(im, 1.5)),
        ("rot_15", lambda im: transform_rotate(im, 15)),
        ("rot_45", lambda im: transform_rotate(im, 45)),
        ("rot_90", lambda im: transform_rotate(im, 90)),
        ("affine_shear", lambda im: transform_affine(im, 0.18)),
    ]

    methods = ["sift_custom", "surf_custom"]
    if include_ref_sift:
        methods.append("sift_ref")
    if include_ref_surf:
        methods.append("surf_ref")

    img_name = "image"

    gray_ref = to_gray(img)
    base_kps = detect_keypoints(gray_ref)

    print("\n=== Reference image keypoints:", len(base_kps))

    rows = []
    for tname, tf in tests:
        img_t = tf(img)
        for m in methods:
            desc_func = method_selector(m)
            n1, n2, nm, rate, k1, k2, matches = evaluate_on_pair(img, img_t, m, desc_func)
            rows.append((tname, m, n1, n2, nm, rate))
            save_path = out_dir / f"{Path(args.image).stem}_{tname}_{m}_matches.jpg"
            draw_and_save_matches(img, k1, img_t, k2, matches, save_path)
            print(f"{tname:12s} | {m:10s} | k1={n1:4d} k2={n2:4d} matches={nm:4d} rate={rate:.3f}")

    # Pretty summary
    try:
        import pandas as pd
        import caas_jupyter_tools  # available in ChatGPT notebooks; ignore if missing in local
        df = pd.DataFrame(rows, columns=["transform", "method", "k_ref", "k_tgt", "good_matches", "match_rate"])
        csv_path = out_dir / f"{Path(args.image).stem}_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary: {csv_path}")
    except Exception:
        pass

    if show:
        # quick bar chart for match counts
        import matplotlib.pyplot as plt
        by_t = {}
        for t, m, _, _, nm, _ in rows:
            by_t.setdefault(t, []).append((m, nm))
        plt.figure(figsize=(10, 4))
        for i, (t, ms) in enumerate(by_t.items()):
            xs = np.arange(len(ms))
            plt.bar(xs + i * 0.0, [v for _, v in ms], label=t)
        plt.legend(); plt.title("Good matches by method (per transform)")
        plt.tight_layout(); plt.show()


# ------------------------------ CLI ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom SIFT & SURF descriptors + robustness evaluation")
    parser.add_argument("--image", required=True, help="Path to input reference image")
    parser.add_argument("--out", default="out", help="Output directory")
    parser.add_argument("--sift_ref", action="store_true", help="Include OpenCV SIFT for comparison")
    parser.add_argument("--surf_ref", action="store_true", help="Include OpenCV SURF for comparison (needs xfeatures2d)")
    parser.add_argument("--no_show", action="store_true", help="Do not show charts; only save images")
    args = parser.parse_args()

    img = cv.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")
    out_dir = ensure_outdir(args.out)

    run_experiment(img, out_dir, include_ref_sift=args.sift_ref, include_ref_surf=args.surf_ref, show=(not args.no_show))


# Implement your own version of the feature descriptors given below and compare them:
#  1. SIFT 
# 2. SURF 
# Assess the robustness of descriptors to change in scale, rotation and 
# affine transformations. 
# Also compare your implementation with the descriptors available in opencv library.
#  Use the earlier version of opencv(3.4.3) for the SIFT