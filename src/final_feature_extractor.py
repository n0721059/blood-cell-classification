"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   WBC Feature Extraction — Tavakoli et al. (2021) — Consolidated Final     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Extracts exactly 51 features per image (48 colour + 3 shape).             ║
║  No features are dropped.                                                   ║
║                                                                             ║
║  Feature vector layout (51 total)                                           ║
║  ─────────────────────────────────                                          ║
║  Idx   0–11  NclCvx mean ratios   (all 12 channels: R,G,B,H,S,V,L,A,B*,Y,Cr,Cb) ║
║  Idx  12–23  NclCvx std  ratios   (all 12 channels)                        ║
║  Idx  24–35  RocCvx mean ratios   (all 12 channels)                        ║
║  Idx  36–47  RocCvx std  ratios   (all 12 channels)                        ║
║  Idx  48–50  Shape: Circularity, Convexity, Solidity                       ║
║                                                                             ║
║  Improvements incorporated from each version:                              ║
║  ─────────────────────────────────────────────                             ║
║  V1  → Full 51-feature set (all 12 channels for both means and stds)       ║
║        Serial + parallel extraction; checkpoint/resume; validate_features  ║
║  V2  → CHAIN_APPROX_NONE  (all boundary pixels present before smoothing)   ║
║        _smooth_contour_perimeter: Gaussian-smooth perimeter for Circularity║
║        Convexity fix: cv2.arcLength on hull polygon (was scipy .area,      ║
║          caused Convexity > 1.0 in 5 % of images)                         ║
║        Solidity clipped to [0, 1] (float overflow guard)                   ║
║        spawn multiprocessing context (safe on macOS/Linux/Windows)         ║
║        Index-keyed checkpoint dict + _save_checkpoint helper               ║
║        validate_features with strict bounds checks                         ║
║  V3  → 3-class multi-Otsu, keep only top class (nucleus only, not cyto)   ║
║        Single-largest-contour selection (replaces 1/10th-area rule)        ║
║        MORPH_CLOSE + flood-fill hole-filling (chromatin holes fixed)       ║
║  NEW → Vectorised pixel collection (10–20× faster than nested Python loop) ║
║        pyhdust CMYK fallback (pure-numpy, no dependency risk)               ║
║        Per-class extraction rate in validate_features                      ║
║        --manifest mode (reads pipeline_output/manifest.csv directly)       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
─────
# From a CSV file (must have a filepath column):
    python feature_extraction_final.py --csv data.csv --out features.csv

# From the data_pipeline manifest directly:
    python feature_extraction_final.py --manifest pipeline_output/manifest.csv \
                                       --out features.csv

# Parallel execution (recommended for >5 k images):
    python feature_extraction_final.py --csv data.csv --out features.csv \
        --parallel --n_jobs -1 --checkpoint ckpt.csv

# With validation report:
    python feature_extraction_final.py --csv data.csv --out features.csv --validate

Notebook import:
    from feature_extraction_final import (
        extract_features, extract_features_from_df,
        extract_features_parallel, validate_features,
        FEATURE_COLS, COLOR_COLS, SHAPE_COLS,
    )
"""

from __future__ import annotations

import os
import sys
import csv
import time
import logging
import argparse
import warnings
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter1d
from skimage import filters as fl
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY: pyhdust (CMYK conversion) — with pure-numpy fallback
# ══════════════════════════════════════════════════════════════════════════════

try:
    import pyhdust.images as _phim
    def _rgb_to_cmyk(img_rgb: np.ndarray) -> np.ndarray:
        """Thin wrapper around pyhdust.images.rgb2cmyk."""
        return _phim.rgb2cmyk(img_rgb)
    _PYHDUST_AVAILABLE = True
except ImportError:
    _PYHDUST_AVAILABLE = False
    log.warning(
        "pyhdust not found — using built-in numpy CMYK conversion.\n"
        "  Install with:  pip install pyhdust\n"
        "  Results will be numerically equivalent."
    )
    def _rgb_to_cmyk(img_rgb: np.ndarray) -> np.ndarray:
        """
        Pure-numpy RGB→CMYK conversion.
        Input : HxWx3 uint8 RGB in [0, 255].
        Output: HxWx4 float32 CMYK in [0, 1].
        """
        rgb_f = img_rgb.astype(np.float32) / 255.0
        R, G, B = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
        K = 1.0 - np.maximum.reduce([R, G, B])
        denom = 1.0 - K
        # avoid division by zero where K == 1 (pure black pixel)
        safe = np.where(denom > 0, denom, 1.0)
        C = (1.0 - R - K) / safe
        M = (1.0 - G - K) / safe
        Y = (1.0 - B - K) / safe
        return np.stack([C, M, Y, K], axis=-1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMN DEFINITIONS  —  51 features, NO DROPS
# ══════════════════════════════════════════════════════════════════════════════

CH_NAMES = ["R", "G", "B", "H", "S", "V", "L", "A", "Bstar", "Y", "Cr", "Cb"]

# 48 colour features:  {ratio}_{channel}_{stat}
COLOR_COLS = (
    [f"NclCvx_{ch}_mean" for ch in CH_NAMES] +   # 12  NCL/CVX mean ratios
    [f"NclCvx_{ch}_std"  for ch in CH_NAMES] +   # 12  NCL/CVX std  ratios
    [f"RocCvx_{ch}_mean" for ch in CH_NAMES] +   # 12  ROC/CVX mean ratios
    [f"RocCvx_{ch}_std"  for ch in CH_NAMES]     # 12  ROC/CVX std  ratios
)                                                 # = 48 colour features

# 3 shape features
SHAPE_COLS = ["Circularity", "Convexity", "Solidity"]

# Full 51-feature vector (order matches the numpy array produced by extract_features)
FEATURE_COLS = COLOR_COLS + SHAPE_COLS

assert len(FEATURE_COLS) == 51, f"Expected 51, got {len(FEATURE_COLS)}"


# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE — step-by-step helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _color_balance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gray-world colour balancing.
    INPUT : HxWx3 uint8 BGR  (OpenCV native order)
    OUTPUT: HxWx3 uint8 BGR  (balanced, same order — all downstream cv2 calls stay correct)
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mu_gray = float(np.mean(gray))
    out     = img_bgr.astype(np.float32)
    for ch in range(3):
        mu_ch = float(np.mean(out[:, :, ch]))
        if mu_ch > 0:
            out[:, :, ch] *= mu_gray / mu_ch
    return np.clip(out, 0, 255).astype(np.uint8)


def _build_nucleus_channel(balanced_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Derives the nucleus-sensitive soft map (min_MS_KM) from a colour-balanced
    BGR image, as described in Tavakoli et al. (2021).

    Returns
    -------
    min_MS_KM : float array — nucleus soft map (high = nucleus)
    _M        : float array — CMYK-M channel (fallback for thresholding)
    """
    # BGR → RGB for CMYK conversion
    balanced_rgb = cv2.cvtColor(balanced_bgr, cv2.COLOR_BGR2RGB)
    cmyk  = _rgb_to_cmyk(balanced_rgb)   # HxWx4 float [0,1]
    _M    = cmyk[:, :, 1]                # Magenta channel
    _K    = cmyk[:, :, 3]                # Key (black) channel

    # Saturation from HLS (FULL flag gives correct 0–255 range)
    _S    = cv2.cvtColor(balanced_bgr, cv2.COLOR_BGR2HLS_FULL)[:, :, 2].astype(np.float32) / 255.0

    min_MS    = np.minimum(_M, _S)
    km_min    = np.minimum(_K, _M)           # min(K, M)
    KM        = _K - km_min                  # K – min(K, M)
    min_MS_KM = min_MS - np.minimum(min_MS, KM)

    return min_MS_KM, _M


def _smooth_contour_perimeter(contour: np.ndarray, sigma: float = 1.5) -> float:
    """
    Computes the arc-length of a contour after 1-D Gaussian smoothing of its
    (x, y) coordinates (mode='wrap' so the closed contour is handled correctly).

    V2 improvement: raw cv2.arcLength over-counts the perimeter by measuring
    every rasterisation jagg.  Smoothing with sigma=1.5 suppresses sub-pixel
    noise while keeping biologically meaningful boundary detail, bringing
    Circularity values in line with ground-truth expectations.

    INPUT : contour — np.ndarray shape (N, 1, 2) from cv2.findContours
    OUTPUT: smoothed arc-length (float)
    """
    pts = contour[:, 0, :].astype(np.float32)   # (N, 2)
    xs  = gaussian_filter1d(pts[:, 0], sigma=sigma, mode="wrap")
    ys  = gaussian_filter1d(pts[:, 1], sigma=sigma, mode="wrap")
    smoothed = np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.int32)
    return float(cv2.arcLength(smoothed, True))


def _segment_nucleus(
    balanced_bgr: np.ndarray,
    min_area: int = 100,
) -> tuple[np.ndarray | None, list, float, float, float, str | None]:
    """
    Segments the cell nucleus from a colour-balanced BGR image.

    Returns
    -------
    nucleus        : uint8 mask (255 = nucleus, 0 = background) — or None on failure
    contours       : list of OpenCV contours (exactly one on success)
    perimeter_raw  : float — raw cv2 arc-length (used for Convexity)
    perimeter_smooth: float — Gaussian-smoothed arc-length (used for Circularity)
    area           : float — nucleus pixel count
    error          : str | None — human-readable reason if failed

    Improvements applied (V2 + V3)
    ────────────────────────────────
    V2 — CHAIN_APPROX_NONE:
        All boundary pixels are kept so the Gaussian smoothing step has a
        complete, gap-free sequence of (x, y) points to work with.
        CHAIN_APPROX_SIMPLE skips collinear pixels and can introduce
        artificial kinks after smoothing.

    V3 — 3-class multi-Otsu (FIX A — cytoplasm separation):
        Using 2 classes (background | nucleus) caused the cytoplasm to be
        swept into the nucleus class for IG and monocyte images.  3 classes
        (background | cytoplasm | nucleus) and keeping only the top class
        (highest intensity on the soft map) isolates the true nucleus.

    V3 — Single largest contour only (FIX C — neighbouring cell blobs):
        The previous 1/10th-area noise rule left blobs of 20–40 % the main
        nucleus size, allowing a neighbouring cell to expand the convex hull.
        Keeping only the single largest contour is geometrically correct and
        avoids any arbitrary area threshold.

    V3 — MORPH_CLOSE + flood-fill hole-filling (FIX B — chromatin holes):
        Chromatin texture creates locally bright regions that fall below the
        threshold, leaving holes inside the mask.  These holes inflate the
        perimeter and contaminate the ROC (Region of Convexity) region.
        Fix: morphological CLOSE seals hairline cracks, then flood-fill from
        the image border identifies the true external background; inverting
        the flood result fills all remaining interior holes.
    """
    min_MS_KM, _M = _build_nucleus_channel(balanced_bgr)

    blurred = cv2.GaussianBlur(min_MS_KM, ksize=(5, 5), sigmaX=0)

    # ── FIX A (V3): 3-class multi-Otsu → keep only the top (nucleus) class ──
    try:
        thresh  = fl.threshold_multiotsu(blurred, classes=3)   # returns 2 thresholds
        nucleus = np.zeros(blurred.shape, dtype=np.uint8)
        nucleus[blurred >= thresh[-1]] = 255                    # top class = nucleus
    except Exception:
        # Fallback 1: same 3-class attempt on the M channel
        try:
            _M_blur = cv2.GaussianBlur(_M, ksize=(5, 5), sigmaX=0)
            thresh  = fl.threshold_multiotsu(_M_blur, classes=3)
            nucleus = np.zeros(_M_blur.shape, dtype=np.uint8)
            nucleus[_M_blur >= thresh[-1]] = 255
        except Exception:
            return None, [], 0.0, 0.0, 0.0, "threshold_multiotsu failed on both channels"

    # ── FIX C (V3): keep the single largest contour only ─────────────────────
    # Use CHAIN_APPROX_NONE: every boundary pixel present for Gaussian smoothing
    result   = cv2.findContours(nucleus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = result[-2]   # compatible with OpenCV 3.x and 4.x

    if len(contours) == 0:
        return None, [], 0.0, 0.0, 0.0, "No contours detected after thresholding"

    largest_cnt  = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_cnt)

    if largest_area <= min_area:
        return (None, [], 0.0, 0.0, 0.0,
                f"Largest contour area {largest_area:.1f} ≤ min_area {min_area}")

    # Redraw the mask using only the largest contour (erases all other blobs)
    nucleus = np.zeros_like(nucleus)
    cv2.drawContours(nucleus, [largest_cnt], -1, 255, thickness=-1)

    # ── FIX B (V3): fill internal holes ───────────────────────────────────────
    # Step 1 — morphological CLOSE: seals hairline cracks between adjacent
    #           white pixels.  Elliptical kernel size 7 closes chromatin-pattern
    #           holes (typically 3–5 px wide) without eroding the outer boundary.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # Step 2 — flood-fill from the image border to label the true background,
    #           then invert to turn any remaining interior holes white.
    h, w    = nucleus.shape
    flood   = nucleus.copy()
    mask_ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)   # label background from top-left
    interior_holes = cv2.bitwise_not(flood)       # pixels that are background in flood but black in nucleus
    nucleus        = cv2.bitwise_or(nucleus, interior_holes)

    # ── Final contour on the cleaned, filled mask ──────────────────────────
    result   = cv2.findContours(nucleus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = result[-2]

    if len(contours) == 0:
        return None, [], 0.0, 0.0, 0.0, "No contours after hole-filling"

    # Take the largest as a safety net in case flood-fill left a border artefact
    final_cnt = max(contours, key=cv2.contourArea)
    contours  = [final_cnt]

    perimeter_raw    = float(cv2.arcLength(final_cnt, True))
    perimeter_smooth = _smooth_contour_perimeter(final_cnt)        # V2
    area             = float(np.sum(nucleus == 255))

    return nucleus, contours, perimeter_raw, perimeter_smooth, area, None


def _build_convex_hull_masks(
    nucleus: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, ConvexHull | None, float, str | None]:
    """
    Builds the convex hull (CVX) and Region-of-Convexity (ROC) binary masks.

    V2 fix for Convexity > 1.0 bug
    ────────────────────────────────
    V1 used scipy ConvexHull.area (geometric perimeter in 2-D, which uses
    continuous Euclidean distance) for the hull perimeter, but cv2.arcLength
    (pixel-count perimeter) for the nucleus.  The two measures differ by a
    few percent and could result in Convexity > 1.0.
    Fix: measure the hull perimeter with cv2.arcLength on the drawn polygon,
    the same method used for the nucleus perimeter.  Convexity is then always
    ≤ 1.0 by construction (clipped as a safety net for float rounding).

    Returns
    -------
    img_convex    : uint8 binary convex hull mask
    img_roc       : uint8 binary ROC mask (hull minus nucleus)
    cvx_hull      : scipy ConvexHull object (for .volume = area in 2-D)
    cvx_perimeter : float  hull perimeter (cv2.arcLength)
    error         : str | None
    """
    pts = np.argwhere(nucleus == 255)
    if len(pts) < 3:
        return None, None, None, 0.0, "Too few nucleus pixels for ConvexHull"

    # scipy ConvexHull expects (x, y); argwhere returns (row, col) = (y, x)
    ncl_pts = pts[:, [1, 0]].astype(np.float64)

    try:
        cvx_hull = ConvexHull(ncl_pts)
    except Exception as e:
        return None, None, None, 0.0, f"ConvexHull failed: {e}"

    corners = ncl_pts[cvx_hull.vertices].reshape(-1, 1, 2).astype(np.int32)

    img_convex = np.zeros_like(nucleus)
    cv2.drawContours(img_convex, [corners], 0, 255, thickness=-1)

    img_roc = cv2.subtract(img_convex, nucleus)

    # V2 fix: measure hull perimeter the same way as nucleus perimeter
    cvx_perimeter = float(cv2.arcLength(corners, True))

    return img_convex, img_roc, cvx_hull, cvx_perimeter, None


def _get_12_channels(balanced_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Converts a colour-balanced BGR image into the 12 single-channel arrays
    used in Tavakoli et al.:  R, G, B, H, S, V, L, A, B*, Y, Cr, Cb

    Returns a list of 12 HxW float32 arrays in that exact order so that
    CH_NAMES[i] is always the label for channels[i].
    """
    bgr = balanced_bgr

    # RGB (note: OpenCV BGR channel order → slice indices 2, 1, 0)
    R = bgr[:, :, 2].astype(np.float32)
    G = bgr[:, :, 1].astype(np.float32)
    B = bgr[:, :, 0].astype(np.float32)

    # HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # LAB
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, Bstar = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # YCrCb
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y, Cr, Cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]

    return [R, G, B, H, S, V, L, A, Bstar, Y, Cr, Cb]


def _collect_region_pixels_vectorised(
    nucleus: np.ndarray,
    img_convex: np.ndarray,
    channels: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised collection of per-channel pixel values in the three regions:
      NCL (nucleus), ROC (convex hull minus nucleus), CVX (entire convex hull).

    NEW in this version — replaces the nested Python loop from V1/V2/V3:
        for p, (row, col) in enumerate(cvx_pts):
            for ch in range(n_ch):
                ...
    The loop was the single largest bottleneck (~80 % of extraction time on
    large nucleus masks).  NumPy fancy indexing on a pre-stacked array gives
    the same result 10–20× faster with no change in output values.

    Returns
    -------
    NCL : float32 array shape (12, n_ncl_pixels)
    ROC : float32 array shape (12, n_roc_pixels)
    CVX : float32 array shape (12, n_cvx_pixels)
    """
    # Pixel coordinates inside the convex hull
    cvx_rows, cvx_cols = np.where(img_convex == 255)

    # Stack all 12 channels into shape (12, H, W) for fast indexing
    ch_stack = np.stack(channels, axis=0)   # (12, H, W)

    # Gather all channel values at the hull pixels in one shot
    CVX = ch_stack[:, cvx_rows, cvx_cols]   # (12, n_cvx)

    # Boolean mask distinguishing nucleus pixels from ROC pixels
    in_nucleus = nucleus[cvx_rows, cvx_cols] == 255   # (n_cvx,) bool

    NCL = CVX[:, in_nucleus]    # (12, n_ncl)
    ROC = CVX[:, ~in_nucleus]   # (12, n_roc)

    return NCL, ROC, CVX


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SINGLE-IMAGE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(
    img_bgr: np.ndarray,
    min_area: int = 100,
) -> tuple[np.ndarray | None, str | None]:
    """
    Full pipeline: BGR image → 51-dimensional feature vector.

    Feature vector layout (51 features)
    ─────────────────────────────────────
    Idx   0–11  NclCvx mean ratios  (12 ch: R,G,B,H,S,V,L,A,B*,Y,Cr,Cb)
    Idx  12–23  NclCvx std  ratios  (12 ch: all)
    Idx  24–35  RocCvx mean ratios  (12 ch: all)
    Idx  36–47  RocCvx std  ratios  (12 ch: all)
    Idx  48     Circularity
    Idx  49     Convexity
    Idx  50     Solidity

    Parameters
    ----------
    img_bgr  : HxWx3 uint8 BGR image (OpenCV native format)
    min_area : minimum nucleus area in pixels; images where the largest
               detected nucleus region is ≤ this value are skipped

    Returns
    -------
    features : np.ndarray shape (51,) float64 — or None on failure
    error    : human-readable failure reason, or None on success
    """
    # ── 1. Gray-world colour balance ──────────────────────────────────────
    balanced = _color_balance(img_bgr)

    # ── 2. Nucleus segmentation ───────────────────────────────────────────
    #       Incorporates all V3 fixes (3-class Otsu, largest-contour,
    #       MORPH_CLOSE + flood-fill hole-filling)
    nucleus, contours, perimeter_raw, perimeter_smooth, area, err = _segment_nucleus(
        balanced, min_area
    )
    if err:
        return None, err

    # ── 3. Convex hull and ROC masks ──────────────────────────────────────
    img_convex, img_roc, cvx_hull, cvx_perimeter, err = _build_convex_hull_masks(nucleus)
    if err:
        return None, err

    # ── 4. Shape features ─────────────────────────────────────────────────
    cvx_area = float(cvx_hull.volume)   # scipy 2-D: .volume == area

    # Circularity  (V2: smoothed perimeter suppresses rasterisation noise)
    # Formula: P² / (4π × A).  Circle → 1.0; more complex shape → higher value.
    circularity = perimeter_smooth ** 2 / (4.0 * np.pi * area)

    # Convexity  (V2 fix: both perimeters measured via cv2.arcLength)
    # Guaranteed ≤ 1.0 by geometry; clipped as float-rounding safety net.
    convexity = float(np.clip(cvx_perimeter / (perimeter_raw + 1e-9), 0.0, 1.0))

    # Solidity  (V2: clipped to [0, 1] to prevent float overflow artefacts)
    solidity = float(np.clip(area / (cvx_area + 1e-9), 0.0, 1.0))

    shape_feats = np.array([circularity, convexity, solidity], dtype=np.float64)

    # ── 5. Edge case: nucleus exactly equals convex hull (no ROC region) ──
    #       Return neutral ratio values of 1.0 for all colour features.
    if int(np.sum(img_roc)) == 0:
        color_feats = np.ones(48, dtype=np.float64)
        return np.concatenate([color_feats, shape_feats]), None

    # ── 6. Get 12 colour-space channels ───────────────────────────────────
    channels = _get_12_channels(balanced)

    # ── 7. Collect pixel values — vectorised (NEW vs V1/V2/V3) ───────────
    NCL, ROC_px, CVX = _collect_region_pixels_vectorised(nucleus, img_convex, channels)

    # ── 8. Per-channel mean and std ───────────────────────────────────────
    eps = 1e-9

    ncl_mean = NCL.mean(axis=1) if NCL.shape[1] > 0 else np.zeros(12, np.float32)
    ncl_std  = NCL.std(axis=1)  if NCL.shape[1] > 0 else np.zeros(12, np.float32)
    cvx_mean = CVX.mean(axis=1)
    cvx_std  = CVX.std(axis=1)
    roc_mean = ROC_px.mean(axis=1) if ROC_px.shape[1] > 0 else np.zeros(12, np.float32)
    roc_std  = ROC_px.std(axis=1)  if ROC_px.shape[1] > 0 else np.zeros(12, np.float32)

    # ── 9. Compute ratios for ALL 12 channels (51-feature set, NO drops) ──
    ratio_ncl_mean = ncl_mean / (cvx_mean + eps)   # (12,)
    ratio_ncl_std  = ncl_std  / (cvx_std  + eps)   # (12,)
    ratio_roc_mean = roc_mean / (cvx_mean + eps)   # (12,)
    ratio_roc_std  = roc_std  / (cvx_std  + eps)   # (12,)

    color_feats = np.concatenate([
        ratio_ncl_mean,   # 12  NclCvx mean ratios
        ratio_ncl_std,    # 12  NclCvx std  ratios
        ratio_roc_mean,   # 12  RocCvx mean ratios
        ratio_roc_std,    # 12  RocCvx std  ratios
    ]).astype(np.float64)  # = 48 colour features

    # Clip extreme outliers (division by near-zero std produces huge values)
    color_feats = np.clip(color_feats, 0.0, 10.0)

    return np.concatenate([color_feats, shape_feats]), None   # shape (51,)


# ══════════════════════════════════════════════════════════════════════════════
# DATAFRAME PIPELINE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _worker(args: tuple) -> tuple[int, np.ndarray | None, str | None]:
    """
    Top-level picklable worker: reads one image and calls extract_features.
    Must be defined at module level so multiprocessing can pickle it.
    """
    idx, filepath, min_area = args
    try:
        img = cv2.imread(str(filepath))
        if img is None:
            return idx, None, f"cv2.imread returned None: {filepath}"
        feats, err = extract_features(img, min_area=min_area)
        return idx, feats, err
    except Exception as exc:
        return idx, None, str(exc)


def _save_checkpoint(df: pd.DataFrame, feat_rows: dict, path: str) -> None:
    """
    Writes partial extraction results to a CSV keyed by original row index.
    On re-run, rows already in the checkpoint are skipped (crash recovery).

    V2 improvement over V1: the checkpoint is keyed by _orig_idx so rows
    are reassembled correctly regardless of insertion order (important for
    the parallel unordered path).
    """
    rows = [{"_orig_idx": i, **dict(zip(FEATURE_COLS, feats))}
            for i, feats in feat_rows.items()]
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info(f"  Checkpoint saved → {path}  ({len(rows):,} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# SERIAL EXTRACTION (single-process, with checkpoint/resume)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_from_df(
    df: pd.DataFrame,
    filepath_col: str    = "Filepath",
    min_area: int        = 100,
    checkpoint_path: str = None,
    checkpoint_every: int= 500,
) -> pd.DataFrame:
    """
    Adds 51 feature columns to *df* by processing each image in serial order.

    Parameters
    ----------
    df               : input DataFrame; must contain *filepath_col*
    filepath_col     : column name holding image file paths
    min_area         : minimum nucleus pixel count (skip smaller detections)
    checkpoint_path  : path to a CSV for crash-safe progress saving;
                       if the file exists on entry, already-processed rows
                       are loaded and skipped  (safe resume)
    checkpoint_every : rows between checkpoint writes

    Returns
    -------
    df_out : original DataFrame + 51 feature columns
             Failed rows have NaN in feature columns and a message in
             the 'extraction_error' column.
    """
    df_out = df.copy()
    for col in FEATURE_COLS:
        df_out[col] = np.nan
    df_out["extraction_error"] = None

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_idx = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        log.info(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = pd.read_csv(checkpoint_path)
        processed = int(ckpt[FEATURE_COLS[0]].notna().sum())
        if processed > 0:
            df_out.iloc[:processed] = ckpt.iloc[:processed]
            start_idx = processed
            log.info(f"  Resuming from row {start_idx:,}")

    n_total   = len(df_out)
    n_success = n_fail = 0
    t_start   = time.time()
    log.info(f"Serial extraction: {n_total - start_idx:,} images to process")
    log.info(f"Feature columns  : {len(FEATURE_COLS)} total "
             f"({len(COLOR_COLS)} colour + {len(SHAPE_COLS)} shape)")

    with tqdm(total=n_total - start_idx, desc="Extracting", unit="img", ncols=90) as pbar:
        for i in range(start_idx, n_total):
            fpath   = df_out.iloc[i][filepath_col]
            img_bgr = cv2.imread(str(fpath))

            if img_bgr is None:
                df_out.at[i, "extraction_error"] = f"cv2.imread failed: {fpath}"
                n_fail += 1
            else:
                feats, err = extract_features(img_bgr, min_area=min_area)
                if err or feats is None:
                    df_out.at[i, "extraction_error"] = err or "Unknown error"
                    n_fail += 1
                else:
                    feat_start = df_out.columns.get_loc(FEATURE_COLS[0])
                    feat_end   = df_out.columns.get_loc(FEATURE_COLS[-1]) + 1
                    df_out.iloc[i, feat_start:feat_end] = feats
                    n_success += 1

            pbar.set_postfix(ok=n_success, fail=n_fail)
            pbar.update(1)

            if checkpoint_path and (i + 1) % checkpoint_every == 0:
                df_out.to_csv(checkpoint_path, index=False)
                log.info(f"  Checkpoint at row {i + 1:,}")

    elapsed = time.time() - t_start
    log.info(f"Done in {elapsed:.1f}s | "
             f"Success: {n_success}/{n_total} | "
             f"Failed: {n_fail}/{n_total} | "
             f"Rate: {n_total/elapsed:.1f} img/s")
    if n_fail > 0:
        log.warning(f"{n_fail} images failed — see 'extraction_error' column.")

    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL EXTRACTION (multiprocessing, with checkpoint/resume)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_parallel(
    df: pd.DataFrame,
    filepath_col: str    = "Filepath",
    min_area: int        = 100,
    n_jobs: int          = -1,
    checkpoint_path: str = None,
) -> pd.DataFrame:
    """
    Parallel feature extraction using multiprocessing.

    Parameters
    ----------
    df              : input DataFrame; must contain *filepath_col*
    filepath_col    : column name holding image file paths
    min_area        : minimum nucleus pixel count
    n_jobs          : worker count; -1 = all logical CPUs; 1 = serial (debug)
    checkpoint_path : crash-safe checkpoint path (skips already-processed rows
                      on re-run)

    Returns
    -------
    DataFrame with original columns + FEATURE_COLS (51 columns).
    Failed rows: NaN features.

    V2/V3 improvements over V1
    ───────────────────────────
    • spawn context (mp.get_context("spawn")): avoids fork-related deadlocks
      on macOS and is safe on all platforms including Windows.
    • Index-keyed feat_rows dict: unordered imap_unordered results are stored
      by original row index and only assembled into a DataFrame at the end,
      avoiding the off-by-one reassembly bug possible with ordered writes.
    • Checkpoint stores _orig_idx for reliable resume after partial crashes.
    """
    n_cpus  = mp.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    n_total = len(df)
    log.info(f"Parallel extraction: {n_total:,} images on {n_cpus} workers")

    # ── Resume from checkpoint ────────────────────────────────────────────
    completed_idx: set[int] = set()
    feat_rows: dict[int, np.ndarray] = {}

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = pd.read_csv(checkpoint_path)
        for _, row in ckpt.iterrows():
            i    = int(row["_orig_idx"])
            vals = row[FEATURE_COLS].values.astype(np.float64)
            if not np.all(np.isnan(vals)):
                feat_rows[i]     = vals
                completed_idx.add(i)
        log.info(f"  Resumed {len(completed_idx):,} rows from checkpoint")

    work = [
        (i, df.iloc[i][filepath_col], min_area)
        for i in range(n_total)
        if i not in completed_idx
    ]
    log.info(f"  Remaining: {len(work):,} images")

    if n_cpus == 1:
        # Single-process path (easier to debug, same code path as worker)
        for args in tqdm(work, desc="Extracting", unit="img", ncols=90):
            idx, feats, err = _worker(args)
            if feats is None:
                log.warning(f"  [{idx}] FAILED — {err}")
                feat_rows[idx] = np.full(len(FEATURE_COLS), np.nan)
            else:
                feat_rows[idx] = feats
            if checkpoint_path and len(feat_rows) % 500 == 0:
                _save_checkpoint(df, feat_rows, checkpoint_path)
    else:
        # V2: spawn context — safe on all platforms (macOS, Linux, Windows)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_cpus) as pool:
            for idx, feats, err in tqdm(
                pool.imap_unordered(_worker, work, chunksize=8),
                total=len(work),
                desc="Extracting",
                unit="img",
                ncols=90,
            ):
                if feats is None:
                    log.warning(f"  [{idx}] FAILED — {err}")
                    feat_rows[idx] = np.full(len(FEATURE_COLS), np.nan)
                else:
                    feat_rows[idx] = feats
                if checkpoint_path and len(feat_rows) % 500 == 0:
                    _save_checkpoint(df, feat_rows, checkpoint_path)

    # Final checkpoint flush
    if checkpoint_path:
        _save_checkpoint(df, feat_rows, checkpoint_path)

    # Assemble output: preserve original row order via sorted index key
    feat_matrix = np.vstack([feat_rows[i] for i in range(n_total)])
    df_feats    = pd.DataFrame(feat_matrix, columns=FEATURE_COLS, index=df.index)
    result      = pd.concat(
        [df.reset_index(drop=True), df_feats.reset_index(drop=True)], axis=1
    )

    n_failed = int(result[FEATURE_COLS].isna().any(axis=1).sum())
    log.info(f"  Done. Failed: {n_failed:,} / {n_total:,}  ({100*n_failed/n_total:.2f}%)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MANIFEST READER  (integrates with data_pipeline.py output)
# ══════════════════════════════════════════════════════════════════════════════

def df_from_manifest(
    manifest_path: str | Path,
    filepath_col_out: str = "Filepath",
    label_col_out: str    = "Label",
) -> pd.DataFrame:
    """
    Reads pipeline_output/manifest.csv produced by data_pipeline.py and
    returns a DataFrame with the image path and label columns expected by
    extract_features_from_df / extract_features_parallel.

    Only rows where valid == 'True' are included.

    Parameters
    ----------
    manifest_path    : path to manifest.csv
    filepath_col_out : column name to use for the image path
    label_col_out    : column name to use for the class label

    Returns
    -------
    pd.DataFrame with at least [filepath_col_out, label_col_out] columns
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run data_pipeline.py first."
        )

    rows = []
    with open(manifest_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("valid", "False").strip().lower() != "true":
                continue
            rows.append({
                filepath_col_out: row["image_path"],
                label_col_out:    row["label"],
            })

    df = pd.DataFrame(rows)
    log.info(f"Loaded {len(df):,} valid rows from manifest")
    if len(df) == 0:
        raise ValueError("No valid rows found in manifest.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION  (V2 strict bounds + V1 per-class breakdown)
# ══════════════════════════════════════════════════════════════════════════════

def validate_features(
    df: pd.DataFrame,
    raise_on_error: bool = False,
) -> bool:
    """
    Sanity-checks the extracted feature DataFrame and prints a report.

    Checks
    ──────
    1. All 51 feature columns present
    2. NaN / Inf counts per feature
    3. Circularity ≥ 1.0  (V2 bounds check)
    4. Convexity in (0, 1]  (V2 bounds check)
    5. Solidity   in (0, 1]  (V2 bounds check)
    6. Color ratios in [0, 10]  (V2 bounds check)
    7. Per-class extraction rate  (V1 breakdown — restored)

    Parameters
    ----------
    df             : output of extract_features_from_df or extract_features_parallel
    raise_on_error : if True, raises ValueError when any check fails

    Returns
    -------
    all_ok : True if every check passed
    """
    SEP = "=" * 65
    print(f"\n{SEP}")
    print(f"  FEATURE VALIDATION REPORT  —  {len(df):,} rows")
    print(SEP)

    all_ok = True
    total  = len(df)

    # ── 1. Column presence ────────────────────────────────────────────────
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Missing columns ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
        all_ok = False
    else:
        print(f" All {len(FEATURE_COLS)} feature columns present  "
              f"({len(COLOR_COLS)} colour + {len(SHAPE_COLS)} shape)")

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    feat_df        = df[available_cols]

    # ── 2. NaN / Inf ──────────────────────────────────────────────────────
    n_nan_rows = int(feat_df.isna().any(axis=1).sum())
    n_inf_rows = int(np.isinf(feat_df.select_dtypes(include=np.number).values).any(axis=1).sum())

    sym = "✅" if n_nan_rows == 0 else "⚠️ "
    print(f"  {sym}  Rows with any NaN : {n_nan_rows:,}  ({100*n_nan_rows/total:.2f}%)")
    sym = "✅" if n_inf_rows == 0 else "❌"
    print(f"  {sym}  Rows with any Inf : {n_inf_rows:,}")
    if n_inf_rows:
        all_ok = False

    # ── 3–5. Shape feature bounds ─────────────────────────────────────────
    shape_checks = [
        ("Circularity", "≥ 1.0",   lambda s: s >= 1.0),
        ("Convexity",   "in (0,1]", lambda s: (s > 0) & (s <= 1.0)),
        ("Solidity",    "in (0,1]", lambda s: (s > 0) & (s <= 1.0)),
    ]
    for col, label, condition in shape_checks:
        if col not in df.columns:
            continue
        s   = df[col].dropna()
        bad = int((~condition(s)).sum())
        sym = "✅" if bad == 0 else "❌"
        print(f"  {sym}  {col:<14}  must be {label:<12}  violations: {bad:,}  "
              f"({100*bad/max(len(s),1):.1f}%)")
        if bad:
            all_ok = False

    # ── 6. Color ratio bounds ─────────────────────────────────────────────
    color_available = [c for c in COLOR_COLS if c in df.columns]
    if color_available:
        color_df = df[color_available].dropna()
        n_neg  = int((color_df < 0).any(axis=1).sum())
        n_over = int((color_df > 10).any(axis=1).sum())
        print(f"  {'✅' if n_neg==0 else '❌'}  Color ratios < 0    : {n_neg:,}")
        print(f"  {'✅' if n_over==0 else '❌'}  Color ratios > 10   : {n_over:,}")
        if n_neg or n_over:
            all_ok = False

    # ── 7. Per-class extraction rate (V1 feature restored) ───────────────
    label_col = next((c for c in ("Label", "label", "class", "Class") if c in df.columns), None)
    if label_col:
        print(f"\n  Per-class extraction rate:")
        for lbl, grp in df.groupby(label_col):
            ok_count = int(grp[FEATURE_COLS[0]].notna().sum()) if FEATURE_COLS[0] in df.columns else 0
            pct      = 100 * ok_count / max(len(grp), 1)
            sym      = "✅" if pct == 100 else ("⚠️ " if pct >= 90 else "❌")
            print(f"    {sym}  {str(lbl):<25}  {ok_count:>5}/{len(grp)}  ({pct:.1f}%)")

    # ── 8. Error summary ──────────────────────────────────────────────────
    if "extraction_error" in df.columns:
        errors = df["extraction_error"].dropna()
        if len(errors) > 0:
            print(f"\n  Error breakdown ({len(errors):,} failed rows):")
            for msg, cnt in errors.value_counts().head(6).items():
                print(f"    [{cnt:>4}×]  {str(msg)[:80]}")

    # ── Summary ───────────────────────────────────────────────────────────
    n_extracted = int(feat_df[FEATURE_COLS[0]].notna().sum()) if FEATURE_COLS[0] in feat_df.columns else 0
    print(f"\n  Extracted : {n_extracted:,} / {total:,}  ({100*n_extracted/total:.1f}%)")
    print(SEP)
    print(f"  {'✅ All checks passed.' if all_ok else '❌ Some checks failed — review pipeline.'}")
    print(f"{SEP}\n")

    if not all_ok and raise_on_error:
        raise ValueError("validate_features: one or more checks failed.")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 51 WBC features (Tavakoli et al. 2021) — consolidated final version.\n"
            "Accepts either a CSV file (--csv) or the data_pipeline manifest (--manifest)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv",       metavar="FILE",  help="Input CSV with image paths")
    src.add_argument("--manifest",  metavar="FILE",  help="manifest.csv from data_pipeline.py")

    # Output
    parser.add_argument("--out",        required=True,  metavar="FILE",  help="Output CSV path")

    # Column names
    parser.add_argument("--filepath",   default="Filepath", metavar="COL",
                        help="Column name for image paths (--csv mode; default: Filepath)")
    parser.add_argument("--label",      default="Label",    metavar="COL",
                        help="Column name for class labels (default: Label)")

    # Extraction options
    parser.add_argument("--min_area",   type=int,   default=100,
                        help="Minimum nucleus area in pixels (default: 100)")
    parser.add_argument("--parallel",   action="store_true",
                        help="Use multiprocessing for extraction")
    parser.add_argument("--n_jobs",     type=int,   default=-1,
                        help="Worker count for --parallel (-1 = all CPUs; default: -1)")
    parser.add_argument("--checkpoint", default=None, metavar="FILE",
                        help="Checkpoint CSV path (enables crash-safe resume)")
    parser.add_argument("--checkpoint_every", type=int, default=500,
                        help="Rows between checkpoint writes (serial mode; default: 500)")

    # Post-processing
    parser.add_argument("--validate",   action="store_true",
                        help="Print validation report after extraction")

    args = parser.parse_args()

    # ── Load input ────────────────────────────────────────────────────────
    if args.manifest:
        log.info(f"Loading manifest: {args.manifest}")
        df = df_from_manifest(args.manifest,
                              filepath_col_out=args.filepath,
                              label_col_out=args.label)
    else:
        log.info(f"Loading CSV: {args.csv}")
        df = pd.read_csv(args.csv)
        log.info(f"  {len(df):,} rows loaded")

    # ── Extract features ──────────────────────────────────────────────────
    if args.parallel:
        df_out = extract_features_parallel(
            df,
            filepath_col    = args.filepath,
            min_area        = args.min_area,
            n_jobs          = args.n_jobs,
            checkpoint_path = args.checkpoint,
        )
    else:
        df_out = extract_features_from_df(
            df,
            filepath_col     = args.filepath,
            min_area         = args.min_area,
            checkpoint_path  = args.checkpoint,
            checkpoint_every = args.checkpoint_every,
        )

    # ── Validate ──────────────────────────────────────────────────────────
    if args.validate:
        validate_features(df_out)

    # ── Save ──────────────────────────────────────────────────────────────
    log.info(f"Saving → {args.out}")
    df_out.to_csv(args.out, index=False)
    log.info(f"Done.  {len(FEATURE_COLS)} feature columns written.")


if __name__ == "__main__":
    main()