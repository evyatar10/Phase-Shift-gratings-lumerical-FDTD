import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
import scipy.integrate  # Use scipy.integrate.trapezoid
import matplotlib.pyplot as plt
import config  # Import config to get the correct paths


def calculate_overlap_profile(file_long, file_short):
    """
    Calculates the field overlap integral between two devices as a function of X.
    Uses all 3 field components (Ex, Ey, Ez).
    Forces alignment to the Short Device's grid.
    """
    # Check if files exist
    if not os.path.exists(file_long):
        print(f"ERROR: Long file not found: {file_long}")
        return
    if not os.path.exists(file_short):
        print(f"ERROR: Short file not found: {file_short}")
        return

    print(f"Loading Long Device: {os.path.basename(file_long)}")
    # FIX: simplify_cells=True forces data into standard Python dicts
    data_L = sio.loadmat(file_long, simplify_cells=True)

    print(f"Loading Short Device: {os.path.basename(file_short)}")
    data_S = sio.loadmat(file_short, simplify_cells=True)

    # --- HELPER: UNPACK 3D DATA SAFELY ---
    def unpack_3d(data, name):
        # With simplify_cells=True, 'field_3d' is just a dictionary key
        if 'field_3d' not in data:
            raise ValueError(f"File {name} is missing 'field_3d'. Did you run with record_3d_fields=True?")

        f3d = data['field_3d']

        # Access fields directly from the dictionary
        # We assume standard saving keys: 'x', 'y', 'z', 'E_res'
        try:
            return f3d['x'], f3d['y'], f3d['z'], f3d['E_res']
        except KeyError as e:
            # Fallback debug in case keys are named differently
            print(f"Debug - Available keys in {name} field_3d: {f3d.keys()}")
            raise e

    try:
        xL, yL, zL, EL = unpack_3d(data_L, "Long")
        xS, yS, zS, ES = unpack_3d(data_S, "Short")
    except Exception as e:
        print(f"Data Unpacking Error: {e}")
        return

    # --- 1. SETUP INTERPOLATORS ---
    print("Building interpolators for Long Field (Ex, Ey, Ez)...")
    # EL shape is (Nx, Ny, Nz, 3)

    # Safety Check for Dimensions
    if EL.ndim != 4:
        print(f"Warning: Unexpected EL dimensions {EL.shape}. Expected (Nx, Ny, Nz, 3).")
        # Handle cases where singleton dimensions might exist
        if EL.ndim == 5: EL = np.squeeze(EL)

    interp_Ex = RegularGridInterpolator((xL, yL, zL), EL[..., 0], bounds_error=False, fill_value=0j)
    interp_Ey = RegularGridInterpolator((xL, yL, zL), EL[..., 1], bounds_error=False, fill_value=0j)
    interp_Ez = RegularGridInterpolator((xL, yL, zL), EL[..., 2], bounds_error=False, fill_value=0j)

    # --- 2. DEFINE THE COMMON GRID ---
    x_min = max(xL.min(), xS.min())
    x_max = min(xL.max(), xS.max())

    valid_mask = (xS >= x_min) & (xS <= x_max)
    x_common = xS[valid_mask]

    overlap_vals = []
    print(f"Calculating Overlap for {len(x_common)} slices...")

    YY, ZZ = np.meshgrid(yS, zS, indexing='ij')

    for i, x_val in enumerate(x_common):
        # A. Short Field Slice
        idx_S = np.where(xS == x_val)[0][0]
        E_S_slice = ES[idx_S, :, :, :]  # Shape (Ny, Nz, 3)

        # B. Long Field Slice (Interpolated)
        pts_y = YY.flatten()
        pts_z = ZZ.flatten()
        pts_x = np.full_like(pts_y, x_val)
        query_pts = np.column_stack((pts_x, pts_y, pts_z))

        Ex_L = interp_Ex(query_pts).reshape(len(yS), len(zS))
        Ey_L = interp_Ey(query_pts).reshape(len(yS), len(zS))
        Ez_L = interp_Ez(query_pts).reshape(len(yS), len(zS))

        E_L_slice = np.stack((Ex_L, Ey_L, Ez_L), axis=-1)

        # C. Overlap Integral
        # Dot product E_long . E_short*
        dot_prod = np.sum(E_L_slice * np.conj(E_S_slice), axis=-1)

        # Integration (dy * dz)
        # Using scipy.integrate.trapezoid
        integ_overlap = scipy.integrate.trapezoid(scipy.integrate.trapezoid(dot_prod, zS, axis=1), yS, axis=0)

        norm_L = scipy.integrate.trapezoid(
            scipy.integrate.trapezoid(np.sum(np.abs(E_L_slice) ** 2, axis=-1), zS, axis=1), yS, axis=0)
        norm_S = scipy.integrate.trapezoid(
            scipy.integrate.trapezoid(np.sum(np.abs(E_S_slice) ** 2, axis=-1), zS, axis=1), yS, axis=0)

        if norm_L > 0 and norm_S > 0:
            O_x = (np.abs(integ_overlap) ** 2) / (norm_L * norm_S)
        else:
            O_x = 0.0
        overlap_vals.append(O_x)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_common * 1e6, overlap_vals, linewidth=2)
    plt.xlabel('Position X [um]')
    plt.ylabel('Mode Overlap Factor (0-1)')
    plt.title(f'Overlap Profile\n{os.path.basename(file_long)} vs {os.path.basename(file_short)}')
    plt.grid(True)
    plt.ylim(bottom=0, top=1.05)
    plt.show()


if __name__ == "__main__":
    # --- USER CONFIGURATION ---
    # Update these numbers to match your actual file names
    N_short = 80
    N_long = 100

    # Paths constructed dynamically from config
    filename_short = r"C:\Users\evyat\Lumerical\long_bragg_grating_interconnect\bragg_fdtd_elements_v5_3d_profiles\results\result_40_periods_CONST_3D_crop.mat"  # doesnt have to be this formating
    filename_long = r"C:\Users\evyat\Lumerical\long_bragg_grating_interconnect\bragg_fdtd_elements_v5_3d_profiles\results\result_100_periods_CONST_3D_crop.mat"

    path_short = os.path.join(config.RESULTS_DIR, filename_short)
    path_long = os.path.join(config.RESULTS_DIR, filename_long)

    # Run Analysis
    calculate_overlap_profile(path_long, path_short)