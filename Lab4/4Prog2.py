# Q2: Detect lines using Hough Transform (manual accumulator, no cv2.HoughLines)

import cv2
import numpy as np
import math

def main():
    image_path = 'data/dog.jpg'
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Couldn't read: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Simple threshold for edges
    _, edges = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    H, W = edges.shape
    diag_len = int(round(math.hypot(H, W)))
    rhos = np.arange(-diag_len, diag_len + 1, 1)
    thetas = np.deg2rad(np.arange(-90, 90, 1))

    # Step 2: Accumulator
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(edges)

    for i in range(len(x_idxs)):
        x, y = x_idxs[i], y_idxs[i]
        for t_idx in range(len(thetas)):
            rho = int(x * np.cos(thetas[t_idx]) + y * np.sin(thetas[t_idx]))
            accumulator[rho + diag_len, t_idx] += 1

    # Step 3: Detect peaks (lines) by threshold
    out = img.copy()
    threshold = 100  # tune depending on image
    for r_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            if accumulator[r_idx, t_idx] > threshold:
                rho = rhos[r_idx]
                theta = thetas[t_idx]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * ( a))
                x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * ( a))
                cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Step 4: Show results
    cv2.imshow('Original Image', img)
    cv2.imshow('Binary Edge Map', edges)
    cv2.imshow('Detected Lines (manual Hough)', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# write a program to detect lines using hough transform