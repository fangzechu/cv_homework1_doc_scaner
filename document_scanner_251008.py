''' 
Document scanner - CV homework: Canny edges + contour approx + perspective transform
end to function `scan_document(image_path, out_dir)`
input: image_path
output: img in out_dir

'''

# It also runs once on the provided sample in /mnt/data and saves results.

import os
import cv2
import numpy as np
from typing import Tuple

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [tl, tr, br, bl]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perspective-warp the image given 4 corner points."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def find_document_quad(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return quadrilateral points of the doc and an edge visualization."""
    # Blur slightly, detect edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = auto_canny(blur)
    # Strengthen edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Find largest 4-point contour by area
    cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    quad = None
    for c in cnts[:10]:  # check top 10 contours
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype("float32")
            break

    # Fallback: use minAreaRect if no 4-point polygon found
    if quad is None and len(cnts) > 0:
        c = cnts[0]
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        quad = box.astype("float32")

    # Optional: refine corner locations to subpixel accuracy if possible
    if quad is not None:
        # Harris/GoodFeatures map to refine
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            corners = corners.reshape(-1, 2)
            refined = []
            for q in order_points(quad):
                # pick the nearest detected corner to each ordered vertex
                dists = np.linalg.norm(corners - q, axis=1)
                nearest = corners[np.argmin(dists)]
                refined.append(nearest)
            quad = np.array(refined, dtype="float32")

    return quad, edges

def enhance_scanned(warped: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return color and binarized (scanned-like) versions."""
    # Color correction (optional white balance-like)
    color = warped.copy()
    # Convert to grayscale and adaptive threshold to mimic a scanner
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # Use adaptive threshold and slight median blur
    gray_w = cv2.medianBlur(gray_w, 3)
    bw = cv2.adaptiveThreshold(gray_w, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    return color, bw

def scan_document(image_path: str, out_dir: str, debug: bool = False) -> Tuple[str, str]:
    """Main entry: path in, saves results (color & bw) to out_dir, returns paths."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # Keep ratio for later warping to original resolution
    orig = image.copy()
    ratio = image.shape[0] / 800.0  # scale to 800px height for detection
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 800))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    quad, edges = find_document_quad(gray)
    if quad is None:
        raise RuntimeError("Could not detect document boundary (quadrilateral not found).")

    # Scale quad back to original image coordinates
    quad_orig = quad * ratio
    warped = four_point_transform(orig, quad_orig)

    color, bw = enhance_scanned(warped)

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_color = os.path.join(out_dir, f"{base}_scanned_color.jpg")
    out_bw = os.path.join(out_dir, f"{base}_scanned_bw.jpg")
    cv2.imwrite(out_color, color)
    cv2.imwrite(out_bw, bw)

    # Debug composite with corners
    if debug:
        dbg = resized.copy()
        for p in quad.astype(int):
            cv2.circle(dbg, tuple(p), 8, (0, 255, 0), -1)
        out_dbg1 = os.path.join(out_dir, f"{base}_edges.jpg")
        out_dbg2 = os.path.join(out_dir, f"{base}_corners.jpg")
        cv2.imwrite(out_dbg1, edges)
        cv2.imwrite(out_dbg2, dbg)

    return out_color, out_bw

# ---- Run once on the provided sample and show output paths ----
sample_path = "./DS_scan_test.png"
out_dir = "./output"
try:
    color_p, bw_p = scan_document(sample_path, out_dir, debug=True)
    print("Saved:", color_p, "and", bw_p)
    print("Debug images also saved to:", out_dir)
except Exception as e:
    print("Error:", e)


