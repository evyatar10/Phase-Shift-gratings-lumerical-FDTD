import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# -----------------------------
# User settings
# -----------------------------
# Filter range in microns for the combined loss plot only
LAMBDA_MIN_UM = 1.57
LAMBDA_MAX_UM = 1.575

# Convert to nm because wl_nm is in nm
LAMBDA_MIN_NM = 1000.0 * LAMBDA_MIN_UM   # 1570 nm
LAMBDA_MAX_NM = 1000.0 * LAMBDA_MAX_UM   # 1575 nm

# Optional: highlight peak loss points with (x, y) text on the plots
HIGHLIGHT_PEAKS = True

# -----------------------------
# 1. Select .npz files
# -----------------------------

root = tk.Tk()
root.withdraw()

filepaths = filedialog.askopenfilenames(
    title="Select .npz files",
    filetypes=[("NumPy archive", "*.npz")]
)

if not filepaths:
    print("No files selected.")
    exit()

# -----------------------------
# 2. Select folder to save plots
# -----------------------------

save_folder = filedialog.askdirectory(
    title="Select folder to save plots"
)

if not save_folder:
    print("No save folder selected.")
    exit()

# -----------------------------
# Load data; compute filtered loss only for combined plot
# -----------------------------

datasets = []

for fp in filepaths:
    data = np.load(fp)

    wl_nm = data["wl_nm"]      # full wavelength axis [nm]
    T = data["T"]              # full T
    R = data["R"]              # full R
    loss = data["loss"]        # full loss

    # Mask for loss filtering in combined plot only
    mask = (wl_nm >= LAMBDA_MIN_NM) & (wl_nm <= LAMBDA_MAX_NM)
    wl_loss = wl_nm[mask]
    loss_filtered = loss[mask]

    datasets.append(
        {
            "name": os.path.splitext(os.path.basename(fp))[0],
            "wl_nm": wl_nm,       # unfiltered
            "T": T,               # unfiltered
            "R": R,               # unfiltered
            "loss": loss,         # unfiltered
            "wl_loss": wl_loss,   # filtered for combined plot
            "loss_f": loss_filtered,
        }
    )

# -----------------------------
# Plot 1: combined filtered loss only
# -----------------------------

plt.figure()
for ds in datasets:
    plt.plot(ds["wl_loss"], ds["loss_f"], label=ds["name"])

    # Optional peak highlight on combined filtered loss
    if HIGHLIGHT_PEAKS and ds["wl_loss"].size > 0:
        idx_peak = np.argmax(ds["loss_f"])
        x_peak = ds["wl_loss"][idx_peak]
        y_peak = ds["loss_f"][idx_peak]
        plt.plot(x_peak, y_peak, 'o')
        plt.text(
            x_peak,
            y_peak,
            f"({x_peak:.2f}, {y_peak:.3f})",
            fontsize=8,
            ha="left",
            va="bottom"
        )

plt.xlabel("Wavelength [nm]")
plt.ylabel("Loss (1 - R - T)")
plt.title("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()

combined_path = os.path.join(save_folder, "combined_loss.png")
plt.savefig(combined_path, dpi=300)
print(f"Saved: '{combined_path}'")

# -----------------------------
# Plots 2..N: per-file plots with full data (no filtering at all)
# -----------------------------

for ds in datasets:
    plt.figure()
    plt.plot(ds["wl_nm"], ds["T"], label="T")
    plt.plot(ds["wl_nm"], ds["R"], label="R")
    plt.plot(ds["wl_nm"], ds["loss"], label="loss (1 - R - T)")

    # Optional peak highlight on full loss curve
    if HIGHLIGHT_PEAKS and ds["wl_nm"].size > 0:
        idx_peak_full = np.argmax(ds["loss"])
        x_peak_full = ds["wl_nm"][idx_peak_full]
        y_peak_full = ds["loss"][idx_peak_full]
        plt.plot(x_peak_full, y_peak_full, 'o')
        plt.text(
            x_peak_full,
            y_peak_full,
            f"({x_peak_full:.2f}, {y_peak_full:.3f})",
            fontsize=8,
            ha="left",
            va="bottom"
        )

    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized power")
    plt.title(f"{ds['name']}  T, R, loss (1 - R - T)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_folder, f"{ds['name']}_TRloss.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved: '{out_path}'")

# -----------------------------
# Show all figures
# -----------------------------
plt.show()
