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
