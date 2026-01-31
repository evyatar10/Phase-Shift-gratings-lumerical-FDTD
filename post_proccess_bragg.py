import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import importlib.util
import config
import analysis  # Added for T-matrix calculation
from run_simulation import extract_and_process_field_profile

# --- LUMAPI INITIALIZATION ---
try:
    import lumapi
except ImportError:
    LUMAPI_PATH = r"C:\\Program Files\\Lumerical\\v252\\api\\python\\lumapi.py"
    spec = importlib.util.spec_from_file_location("lumapi", LUMAPI_PATH)
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)


def run_manual_reprocessing(layout_path, manual_resonance_nm=None, save_images=True, save_full_mat=True):
    """
    1. Loads existing simulation (no re-run).
    2. Recalculates Resonance, Q-factor (abs), and Side-Relative FWHM.
    3. Saves data to .mat file.
    4. Plots 3 separate figures (Linear, dB, Profile).
    5. Optionally saves the dB and Profile images to disk.
    6. (New) Optionally saves a full result .mat file (like run_simulation output)
       with the updated field profile and resonance info.
    """
    print(f"Opening Lumerical Layout: {layout_path}")
    fdtd = lumapi.FDTD(filename=layout_path)

    # --- 1. Get Spectral Data (Port 1 & 2) ---
    # We need Port 1 now for Reflection/Loss if we are saving the full mat
    res1 = fdtd.getresult("FDTD::ports::Port_1", "expansion for port monitor")
    res2 = fdtd.getresult("FDTD::ports::Port_2", "expansion for port monitor")

    wl = np.squeeze(res1["lambda"])
    wl_nm = wl * 1e9

    # S-Parameters (Complex)
    S11 = np.squeeze(res1["S"])
    S21 = np.squeeze(res2["S"])

    # Power Data
    T_linear = np.abs(S21) ** 2
    R_linear = np.abs(S11) ** 2
    Loss = 1.0 - R_linear - T_linear
    T_db = 10 * np.log10(T_linear)

    # --- 2. Resonance Selection ---
    if manual_resonance_nm:
        idx_res = np.argmin(np.abs(wl_nm - manual_resonance_nm))
    else:
        idx_res = np.argmax(T_linear)

    target_wl_m = wl[idx_res]
    res_wl_nm = wl_nm[idx_res]
    print(f"Targeting Resonance at: {res_wl_nm:.3f} nm")

    # --- 3. Q-Factor Calculation (Absolute) ---
    peak_val = T_linear[idx_res]
    half_power = peak_val / 2.0

    idx_below = np.where(T_linear < half_power)[0]
    left_side = idx_below[idx_below < idx_res]
    right_side = idx_below[idx_below > idx_res]

    if len(left_side) > 0 and len(right_side) > 0:
        fwhm_spec_nm = wl_nm[right_side[0]] - wl_nm[left_side[-1]]
        q_factor = abs(res_wl_nm / fwhm_spec_nm)
    else:
        q_factor = 0.0
        print("Warning: Could not find 3dB points for Q-factor.")

    # --- 4. Field Profile Extraction ---
    class SimProxy:
        def __init__(self, fdtd_obj):
            self.fdtd = fdtd_obj
            # This value matches your typical simulation size
            # WARNING: Ideally this should be read from the file to be safe
            self.x_grating_end = 75.4925e-6

    proxy = SimProxy(fdtd)
    f_x, I_x_1D, I_x_env, _, _ = extract_and_process_field_profile(proxy, target_wl_m)

    # --- 5. FWHM Calculation (Relative to Side-Minima) ---
    y_max = np.max(I_x_env)
    y_min_sides = (I_x_env[0] + I_x_env[-1]) / 2.0
    target_level = y_min_sides + 0.5 * (y_max - y_min_sides)

    signs = np.sign(I_x_env - target_level)
    crossings = np.where(np.diff(signs))[0]

    if len(crossings) >= 2:
        x_left = f_x[crossings[0]]
        x_right = f_x[crossings[-1]]
        fwhm_spatial_um = (x_right - x_left) * 1e6
    else:
        fwhm_spatial_um = 0.0

    results_tag = os.path.basename(layout_path).replace(".fsp", "")

    # --- 6a. Save Summary Data (Lite Version) ---
    save_path_mat = os.path.join(config.RESULTS_DIR, f"reprocessed_{results_tag}.mat")
    sio.savemat(save_path_mat, {
        'wl_nm': wl_nm,
        'T_linear': T_linear,
        'T_db': T_db,
        'Q_factor': q_factor,
        'res_wl_nm': res_wl_nm,
        'fwhm_um': fwhm_spatial_um,
        'field_x': f_x,
        'field_envelope': I_x_env,
        'field_energy_density_1D': I_x_1D
    })
    print(f"Summary data saved to: {save_path_mat}")

    # --- 6b. Save Full Simulation Format (Optional) ---
    if save_full_mat:
        # Calculate T-matrix for completeness
        _, _, _, T_matrix = analysis.calculate_physics_matrices(S11, S21)

        full_save_path = os.path.join(config.RESULTS_DIR, f"result_reprocessed_FULL_{results_tag}.mat")

        full_data = {
            'wl_m': wl,
            'wl_nm': wl_nm,
            'T': T_linear,
            'R': R_linear,
            'loss': Loss,
            'T_matrix': T_matrix,
            'S11_complex': S11,  # Note: Raw S-params (phase corrections from run_sim might be missing if not baked in)
            'S21_complex': S21,
            'L_device': 2.0 * proxy.x_grating_end,
            'field_x': f_x,
            'field_energy_density_1D': I_x_1D,
            'field_envelope_1D': I_x_env,
            'fwhm_m': fwhm_spatial_um * 1e-6
        }

        sio.savemat(full_save_path, full_data)
        print(f"Full original-format data saved to: {full_save_path}")

    # --- 7. Plotting & Image Saving ---

    # Plot 1: Transmission Linear (Visual only)
    fig1 = plt.figure(figsize=(8, 5))
    plt.plot(wl_nm, T_linear, 'b-', label='T (Linear)')
    plt.plot(res_wl_nm, T_linear[idx_res], 'ro', label=f'Peak: {res_wl_nm:.2f}nm')
    plt.title(f"Transmission (Linear)\nQ-Factor: {q_factor:.2f}")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission")
    plt.grid(True)
    plt.legend()

    # Plot 2: Transmission dB (Save optional)
    fig2 = plt.figure(figsize=(8, 5))
    plt.plot(wl_nm, T_db, 'k-', label='T (dB)')
    plt.axvline(res_wl_nm, color='r', linestyle='--', alpha=0.5)
    plt.title("Transmission (dB)")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission [dB]")
    plt.grid(True)
    plt.legend()

    if save_images:
        path_db = os.path.join(config.RESULTS_DIR, f"plot_dB_{results_tag}.png")
        fig2.savefig(path_db, dpi=300)
        print(f"Saved dB plot: {path_db}")

    # Plot 3: Field Profile (Save optional)
    fig3 = plt.figure(figsize=(10, 6))

    # Rapid changes (Raw) - Thin Blue
    plt.plot(f_x * 1e6, I_x_1D, 'b-', alpha=0.3, linewidth=0.5, label="Raw |E|^2")

    # Envelope - Thick Red
    plt.plot(f_x * 1e6, I_x_env, 'r-', linewidth=2.5, label="Field Envelope")

    # FWHM Line & Text (Clean, no box)
    if fwhm_spatial_um > 0:
        half_width = fwhm_spatial_um / 2.0
        plt.hlines(target_level, -half_width, half_width, colors='k', linestyles='dashed', linewidth=2)
        plt.text(0, target_level * 1.05, f"FWHM = {fwhm_spatial_um:.2f} um",
                 ha='center', color='black', fontweight='bold')

    plt.title(f"Mode Profile @ {res_wl_nm:.2f} nm")
    plt.xlabel("Position x [um]")
    plt.ylabel("Integrated Energy Density (a.u.)")
    plt.legend()
    plt.grid(True)

    if save_images:
        path_prof = os.path.join(config.RESULTS_DIR, f"plot_profile_{results_tag}.png")
        fig3.savefig(path_prof, dpi=300)
        print(f"Saved profile plot: {path_prof}")

    plt.show()

    fdtd.close()


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    MY_LAYOUT = os.path.join(config.LAYOUTS_DIR, "layout_150_periods_CONST.fsp")

    # Set save_full_mat=True to generate the new file
    run_manual_reprocessing(MY_LAYOUT, manual_resonance_nm=1562.37, save_images=True, save_full_mat=True)