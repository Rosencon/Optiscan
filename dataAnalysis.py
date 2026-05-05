"""
bacteria_surface_analysis.py
─────────────────────────────────────────────────────────────────────────────
Surface profilometry analysis for bacterial detection on stainless steel.

Pipeline
  1. Load & parse  – read X / Y / Z sections from the semicolon-CSV
  2. Outlier mask  – flag raw Z residuals above MAX_HEIGHT_UM (default 20 µm)
                     to suppress scanner artefacts / scratches before fitting
  3. Plane fit     – least-squares baseline plane on UNMASKED points
  4. Detrend       – subtract the fitted plane from the full grid
  5. Threshold     – 3-sigma detection on residual (ignoring masked pixels)
  6. Label blobs   – connected-component analysis on the detection map
  7. Visualise     – four-panel figure + per-blob stats table
  8. Export        – save figure and CSV of detected regions
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import label, find_objects
import csv
import os

# ── User-tunable parameters ──────────────────────────────────────────────────
FILENAME         = "Testcsv(in).csv"   # input file (must be in same folder)
MAX_HEIGHT_UM    = 20.0                # residual values above this are masked
SIGMA_THRESHOLD  = 3.0                 # detection threshold (× noise sigma)
MIN_BLOB_PIXELS  = 5                   # ignore detections smaller than this
OUTPUT_FIGURE    = "surface_analysis.png"
OUTPUT_CSV       = "detected_regions.csv"
# ─────────────────────────────────────────────────────────────────────────────


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

def parse_section(lines):
    """Convert semicolon-separated rows → 2-D numpy array."""
    data = []
    for line in lines:
        vals = [v for v in line.strip().split(";") if v != ""]
        if vals:
            data.append([float(v) for v in vals])
    return np.array(data)


print("Loading data …")
with open(FILENAME, "r", encoding="utf-8-sig") as f:
    lines = f.readlines()

total = len(lines)
print(f"  Total lines in file: {total}")

# Section boundaries (0-indexed)
x_lines = lines[0:1250]
y_lines = lines[1251:2501]
z_lines = lines[2502:3752]

X = parse_section(x_lines)
Y = parse_section(y_lines)
Z = parse_section(z_lines)

print(f"  X shape: {X.shape}  Y shape: {Y.shape}  Z shape: {Z.shape}")

if not (X.shape == Y.shape == Z.shape):
    raise ValueError("X, Y, Z arrays must have the same shape.")

nrows, ncols = Z.shape


# ============================================================
# STEP 2: PRELIMINARY OUTLIER MASK (raw Z range)
# ============================================================
# We cannot apply the 20 µm residual cap until after the plane fit,
# so we do a first pass with a generous absolute-deviation mask
# to keep obviously bad scanner points from biasing the plane.

z_flat    = Z.flatten()
z_median  = np.median(z_flat)
# Generous initial mask: exclude points > 50 µm above the median
prelim_mask = np.abs(Z - z_median) < 50.0   # True = keep for plane fit


# ============================================================
# STEP 3: FIT BASELINE PLANE (on clean points only)
# ============================================================

x_flat  = X.flatten()
y_flat  = Y.flatten()
z_fit   = Z.flatten()
mask_f  = prelim_mask.flatten()

x_mean = np.mean(x_flat[mask_f])
y_mean = np.mean(y_flat[mask_f])

x_c = x_flat[mask_f] - x_mean
y_c = y_flat[mask_f] - y_mean

A      = np.c_[x_c, y_c, np.ones(mask_f.sum())]
coeffs, _, _, _ = np.linalg.lstsq(A, z_fit[mask_f], rcond=None)
a, b, c = coeffs

print(f"\nPlane fit (centered):  z = {a:.6f}·(x-x̄) + {b:.6f}·(y-ȳ) + {c:.4f}")
print(f"  x̄ = {x_mean:.2f} µm,  ȳ = {y_mean:.2f} µm")

Z_plane   = a * (X - x_mean) + b * (Y - y_mean) + c
Z_residual = Z - Z_plane


# ============================================================
# STEP 4: APPLY 20-MICRON CAP (residual outlier removal)
# ============================================================
# Points with residual > MAX_HEIGHT_UM are likely scratch artefacts,
# large contamination, or scanner errors – exclude from statistics
# and detection but keep visible in the map as a distinct colour.

outlier_mask = Z_residual > MAX_HEIGHT_UM          # True = outlier
clean_mask   = ~outlier_mask                        # True = clean pixel

print(f"\nOutlier pixels (>{MAX_HEIGHT_UM} µm residual): "
      f"{outlier_mask.sum()} / {Z_residual.size} "
      f"({100*outlier_mask.mean():.2f} %)")


# ============================================================
# STEP 5: THRESHOLD DETECTION ON CLEAN PIXELS
# ============================================================

clean_residuals = Z_residual[clean_mask]
mu    = np.mean(clean_residuals)
sigma = np.std(clean_residuals)
threshold = mu + SIGMA_THRESHOLD * sigma

print(f"\nClean-pixel statistics:")
print(f"  Mean   = {mu:.4f} µm")
print(f"  Sigma  = {sigma:.4f} µm")
print(f"  Threshold ({SIGMA_THRESHOLD}σ) = {threshold:.4f} µm")

# Detection: above threshold AND not an outlier
detections = (Z_residual > threshold) & clean_mask


# ============================================================
# STEP 6: CONNECTED-COMPONENT LABELLING
# ============================================================

labeled_array, n_blobs = label(detections)
print(f"\nConnected components (blobs) detected: {n_blobs}")

# Per-blob statistics
blob_stats = []
for blob_id in range(1, n_blobs + 1):
    blob_pixels = Z_residual[labeled_array == blob_id]
    if len(blob_pixels) < MIN_BLOB_PIXELS:
        continue
    rows_idx, cols_idx = np.where(labeled_array == blob_id)
    blob_stats.append({
        "blob_id"      : blob_id,
        "pixel_count"  : len(blob_pixels),
        "mean_height_um": float(np.mean(blob_pixels)),
        "max_height_um" : float(np.max(blob_pixels)),
        "x_center_um"  : float(np.mean(X[rows_idx, cols_idx])),
        "y_center_um"  : float(np.mean(Y[rows_idx, cols_idx])),
        "row_min"      : int(rows_idx.min()),
        "row_max"      : int(rows_idx.max()),
        "col_min"      : int(cols_idx.min()),
        "col_max"      : int(cols_idx.max()),
    })

valid_blobs = len(blob_stats)
print(f"  Blobs ≥ {MIN_BLOB_PIXELS} pixels: {valid_blobs}")
if valid_blobs > 0:
    print(f"\n  {'ID':>4}  {'Pixels':>7}  {'Mean(µm)':>10}  "
          f"{'Max(µm)':>9}  {'X-ctr(µm)':>11}  {'Y-ctr(µm)':>11}")
    for s in blob_stats[:20]:   # print first 20
        print(f"  {s['blob_id']:>4}  {s['pixel_count']:>7}  "
              f"{s['mean_height_um']:>10.3f}  {s['max_height_um']:>9.3f}  "
              f"{s['x_center_um']:>11.1f}  {s['y_center_um']:>11.1f}")
    if valid_blobs > 20:
        print(f"  … {valid_blobs - 20} more blobs not shown")


# ============================================================
# STEP 7: VISUALISATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Surface Profilometry – Bacterial Detection on Stainless Steel",
             fontsize=14, fontweight='bold')

# ── Panel 1: Raw Z surface ───────────────────────────────────
ax = axes[0, 0]
im = ax.imshow(Z, aspect='auto', cmap='terrain',
               extent=[X.min(), X.max(), Y.max(), Y.min()])
plt.colorbar(im, ax=ax, label="Absolute Height (µm)")
ax.set_title("Raw Surface Height")
ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")

# ── Panel 2: Baseline-corrected residual ─────────────────────
ax = axes[0, 1]
# Clip display to ±MAX_HEIGHT_UM so colour scale is meaningful
vmax = MAX_HEIGHT_UM
vmin = max(-MAX_HEIGHT_UM, Z_residual.min())
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
Z_display = np.clip(Z_residual, vmin, vmax)
im = ax.imshow(Z_display, aspect='auto', cmap='RdYlGn_r',
               norm=norm,
               extent=[X.min(), X.max(), Y.max(), Y.min()])
cb = plt.colorbar(im, ax=ax, label="Residual Height (µm)")
ax.set_title(f"Baseline-Corrected Surface (capped ±{MAX_HEIGHT_UM} µm)")
ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")

# ── Panel 3: Detection map ───────────────────────────────────
ax = axes[1, 0]
# Build an RGB overlay: grey=clean, yellow=outlier, red=detection
display_map = np.zeros((*Z.shape, 3))
display_map[clean_mask & ~detections] = [0.85, 0.85, 0.85]   # clean
display_map[outlier_mask]             = [0.95, 0.85, 0.20]   # outlier
display_map[detections]               = [0.90, 0.10, 0.10]   # detection

ax.imshow(display_map, aspect='auto',
          extent=[X.min(), X.max(), Y.max(), Y.min()])
ax.set_title(f"Detection Map  (threshold = {SIGMA_THRESHOLD}σ = {threshold:.3f} µm)")
ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")

patch_clean  = mpatches.Patch(color=[0.85, 0.85, 0.85], label="Clean surface")
patch_detect = mpatches.Patch(color=[0.90, 0.10, 0.10], label=f"Detection (>{SIGMA_THRESHOLD}σ)")
patch_out    = mpatches.Patch(color=[0.95, 0.85, 0.20], label=f"Outlier (>{MAX_HEIGHT_UM} µm)")
ax.legend(handles=[patch_clean, patch_detect, patch_out],
          loc='lower right', fontsize=8)

# ── Panel 4: Blob overlay on residual ───────────────────────
ax = axes[1, 1]
im = ax.imshow(Z_display, aspect='auto', cmap='RdYlGn_r', norm=norm,
               extent=[X.min(), X.max(), Y.max(), Y.min()])
plt.colorbar(im, ax=ax, label="Residual Height (µm)")

# Overlay bounding boxes of each valid blob
x_range = X.max() - X.min()
y_range = Y.max() - Y.min()
x_scale = x_range / ncols
y_scale = y_range / nrows

for s in blob_stats:
    cx = X.min() + s['col_min'] * x_scale
    cy = Y.min() + s['row_min'] * y_scale
    w  = (s['col_max'] - s['col_min'] + 1) * x_scale
    h  = (s['row_max'] - s['row_min'] + 1) * y_scale
    rect = mpatches.Rectangle((cx, cy), w, h,
                               linewidth=1.0, edgecolor='blue',
                               facecolor='none', alpha=0.7)
    ax.add_patch(rect)

ax.set_title(f"Detected Blobs ({valid_blobs} regions ≥ {MIN_BLOB_PIXELS} px)")
ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")

plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {OUTPUT_FIGURE}")
plt.show()


# ============================================================
# STEP 8: EXPORT BLOB STATISTICS TO CSV
# ============================================================

if blob_stats:
    keys = list(blob_stats[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(blob_stats)
    print(f"Blob stats saved → {OUTPUT_CSV}")
else:
    print("No blobs to export.")

print("\nAnalysis complete.")