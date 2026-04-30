import numpy as np
import matplotlib.pyplot as plt

filename = "Testcsv(in).csv"

def parse_section(lines):
    """Convert semicolon-separated rows into numpy array"""
    data = []
    for line in lines:
        vals = [v for v in line.strip().split(";") if v != ""]
        data.append([float(v) for v in vals])
    return np.array(data)

with open(filename, "r", encoding="utf-8-sig") as f:
    lines = f.readlines()

# Split into three sections
x_lines = lines[0:1250]
y_lines = lines[1251:2501]
z_lines = lines[2502:3752]

# Convert to arrays
X = parse_section(x_lines)
Y = parse_section(y_lines)
Z = parse_section(z_lines)

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
print("Shape of Z:", Z.shape)

# ============================================================
# STEP 2: FIT A BASELINE PLANE (NUMERICALLY STABLE)
# ============================================================

x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = Z.flatten()

# Center coordinates
x_mean = np.mean(x_flat)
y_mean = np.mean(y_flat)

x_c = x_flat - x_mean
y_c = y_flat - y_mean

# Plane model using centered coordinates
A = np.c_[x_c, y_c, np.ones_like(x_c)]

coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)

a, b, c = coeffs

print(f"Plane fit (centered): z = {a:.8f}(x-x̄) + {b:.8f}(y-ȳ) + {c:.4f}")

# Reconstruct plane
Z_plane = a * (X - x_mean) + b * (Y - y_mean) + c
# Reconstruct plane
Z_plane = a * (X - x_mean) + b * (Y - y_mean) + c

# ============================================================
# STEP 3: REMOVE BASELINE (DETRENDING)
# ============================================================

Z_residual = Z - Z_plane

# ============================================================
# STEP 4: THRESHOLD DETECTION
# ============================================================

# Estimate background noise
sigma = np.std(Z_residual)

# Detection threshold (adjust multiplier as needed)
threshold = 3 * sigma

print(f"Noise sigma = {sigma:.4f}")
print(f"Detection threshold = {threshold:.4f}")

# Binary detection map
detections = Z_residual > threshold

# ============================================================
# STEP 5: VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))
plt.imshow(Z_residual, aspect='auto', cmap='viridis')
plt.colorbar(label="Residual Height")
plt.title("Baseline-Corrected Surface")
plt.xlabel("X Index")
plt.ylabel("Y Index")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(detections, aspect='auto', cmap='gray')
plt.title("Threshold Detection Map")
plt.xlabel("X Index")
plt.ylabel("Y Index")
plt.show()