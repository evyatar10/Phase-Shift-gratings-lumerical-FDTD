import shutil
import time
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.integrate  # For trapezoid
import scipy.signal  # For find_peaks
from scipy.interpolate import interp1d
import numpy as np
from bragg_device import PiShiftBraggFDTD
import analysis  # Ensure this module has the export function!
import config


def find_bragg_resonance(wl, T):
    """
    Finds the resonance peak by identifying the 'Stopband Bounds'.
    """
    stopband_indices = np.where(T < 0.6)[0]

    if len(stopband_indices) == 0:
        stopband_indices = np.where(T < 0.85)[0]

    if len(stopband_indices) == 0:
        print("Warning: No stopband detected. Using global maximum.")
        return np.argmax(T)

    idx_start = stopband_indices[0]
    idx_end = stopband_indices[-1]

    T_roi = T[idx_start: idx_end + 1]
    local_peak_idx = np.argmax(T_roi)

    return idx_start + local_peak_idx


def calculate_fwhm_relative(x, y):
    """
    Calculates FWHM relative to the minimum (floor) of the signal.
    """
    y_max = np.max(y)
    y_min = np.min(y)
    target_level = y_min + 0.5 * (y_max - y_min)

    print(f"FWHM Calc -> Max: {y_max:.2e}, Min: {y_min:.2e}, Target: {target_level:.2e}")

    signs = np.sign(y - target_level)
    zero_crossings = np.where(np.diff(signs))[0]

    if len(zero_crossings) >= 2:
        x_left_idx = zero_crossings[0]
        x_right_idx = zero_crossings[-1]

        def get_x_interp(idx):
            y1 = y[idx]
            y2 = y[idx + 1]
            x1 = x[idx]
            x2 = x[idx + 1]
            slope = (y2 - y1) / (x2 - x1)
            if slope == 0: return x1
            return x1 + (target_level - y1) / slope

        x_left = get_x_interp(x_left_idx)
        x_right = get_x_interp(x_right_idx)
        return x_right - x_left
    else:
        return 0.0


def extract_envelope_peaks(x, y):
    """
    Extracts the envelope by connecting the peaks of the standing wave.
    Extrapolates the nearest peak value to the edges (Nearest Neighbor).
    """
    peaks, _ = scipy.signal.find_peaks(y)

    if len(peaks) < 2:
        return y

    x_peaks = x[peaks]
    y_peaks = y[peaks]

    # Fill value holds the first and last peak values to the edges
    f_interp = interp1d(
        x_peaks,
        y_peaks,
        kind='cubic',
        bounds_error=False,
        fill_value=(y_peaks[0], y_peaks[-1])
    )

    return f_interp(x)


def extract_and_process_field_profile(sim, target_wl):
    """
    Retrieves the field profile, integrates it, crops it to the grating area,
    extracts the envelope, and calculates FWHM.
    """
    # 1. Retrieve Data from Monitor
    field_data = sim.fdtd.getresult("field_profile", "E")
    f_lam = np.squeeze(field_data['lambda'])
    f_x_raw = np.squeeze(field_data['x'])
    f_y = np.squeeze(field_data['y'])
    f_E = field_data['E']

    # 2. Find closest wavelength index
    idx_mon = np.argmin(np.abs(f_lam - target_wl))
    print(f"Extracting field at monitor wavelength: {f_lam[idx_mon] * 1e9:.3f} nm")

    # 3. Extract E-field at resonance
    if f_E.ndim == 5:
        E_res = f_E[:, :, 0, idx_mon, :]
    elif f_E.ndim == 4:
        E_res = f_E[:, :, idx_mon, :]
    else:
        E_res = f_E[..., idx_mon, :]

    # 4. Integrate |E|^2 over Y-axis
    I_xy = np.abs(E_res[..., 0]) ** 2 + np.abs(E_res[..., 1]) ** 2 + np.abs(E_res[..., 2]) ** 2
    I_x_1D_raw = scipy.integrate.trapezoid(I_xy, f_y, axis=1)

    # 5. Crop Data to Grating Length
    x_limit = sim.x_grating_end
    valid_indices = np.abs(f_x_raw) <= x_limit

    f_x = f_x_raw[valid_indices]
    I_x_1D = I_x_1D_raw[valid_indices]

    # 6. Extract Envelope
    I_x_envelope = extract_envelope_peaks(f_x, I_x_1D)

    # 7. Calculate FWHM
    fwhm_val = calculate_fwhm_relative(f_x, I_x_envelope)

    return f_x, I_x_1D, I_x_envelope, fwhm_val, f_lam[idx_mon]


def run_single_sim():
    # 1. Parameters
    lambda_res_est = 1.560e-6 #1.610
    scan_width_nm = 16.0
    n_points = 3001
    avg_corr = 800e-9
    corr_depth = 200e-9
    w_wide = avg_corr + corr_depth / 2
    w_narrow = avg_corr - corr_depth / 2
    core_h = 350e-9

    calc_y_span = w_wide + 1.8 * lambda_res_est
    calc_z_span = core_h + 1.8 * lambda_res_est

    # 2. Initialize Simulation
    sim = PiShiftBraggFDTD(
        pitch=500e-9,
        n_periods_each_side=60,
        n_apod_periods_each_side=20,
        width_narrow=w_narrow,
        width_wide=w_wide,
        width_port=1000e-9,
        core_height=core_h,
        substrate_thickness=4e-6,
        override_cavity_length_nm=False, # False = pitch/2
        y_span=calc_y_span,
        z_span=calc_z_span,
        material_db_path=config.MATERIAL_DB_PATH,
        n_periods_dist_to_port=20,
        n_wls_dist_port_to_pml=5.0,
        n_eff_guess=1.55,
        n_wl_points=n_points,
        use_apodization=False,
        center_mod_depth_nm=4.0,
        use_cavity_mesh_override=True,
        use_symmetry=True, # y symmetry
        use_z_symmetry=True,
        use_constant_materials=True,
        n_core_const=1.977
    )

    # 3. Generate Filenames
    N = sim.n_periods_each_side
    Napod = sim.n_apod_periods_each_side
    use_apod = bool(sim.use_apodization) and (Napod is not None) and (Napod > 0)
    cav_tag = f"_L_cav_{int(sim.cavity_length * 1e9)}" if sim.cavity_length != sim.pitch / 2.0 else ""
    mat_tag = "_CONST" if sim.use_constant_materials else ""
    tag = f"{N}_periods_{Napod}_apodizations{cav_tag}{mat_tag}" if use_apod else f"{N}_periods{cav_tag}{mat_tag}"

    layout_path = os.path.join(config.LAYOUTS_DIR, f"layout_{tag}.fsp")
    results_path = os.path.join(config.RESULTS_DIR, f"result_{tag}.mat")

    # 4. Run
    sim.build()
    sim.update_scan(center_lambda_m=lambda_res_est, width_nm=scan_width_nm, n_points=n_points)
    sim.fdtd.save(layout_path)
    print(f"Saved layout to: {layout_path}")

    start = time.perf_counter()
    sim.fdtd.run()
    print(f"Simulation time: {time.perf_counter() - start:.3f} seconds")

    # 5. Process S-parameters
    # UPDATED to use the new flags instead of 'correct_phase'
    wl, R, T, Loss, T_mat, S11, S21 = sim.get_s_and_t_matrix(
        neff_mat_file=config.NEFF_DATA_PATH,
        correct_length=True,
        correct_envelope_and_t_phase=True
    )

    # 6. Find Peak & Process Field Profile
    print("Finding Resonance Peak...")
    idx_peak = find_bragg_resonance(wl, T)
    target_wl = wl[idx_peak]
    print(f"Peak detected at: {target_wl * 1e9:.3f} nm (T = {T[idx_peak]:.3f})")

    # --- CALL NEW FUNCTION ---
    f_x, I_x_1D, I_x_envelope, fwhm_val, actual_mon_wl = extract_and_process_field_profile(sim, target_wl)

    print(f"Calculated FWHM (Relative): {fwhm_val * 1e6:.4f} um")

    # 7. Save
    mat_data = {
        'wl_m': wl, 'wl_nm': wl * 1e9,
        'T': T, 'R': R, 'loss': Loss,
        'T_matrix': T_mat, 'S11_complex': S11, 'S21_complex': S21,
        'L_device': 2.0 * sim.x_grating_end,
        'field_x': f_x,
        'field_energy_density_1D': I_x_1D,
        'field_envelope_1D': I_x_envelope,
        'fwhm_m': fwhm_val
    }
    sio.savemat(results_path, mat_data)
    print(f"Data saved to: {results_path}")

    # 8. Export for Interconnect
    print("Exporting for Interconnect...")
    # Fetch matrices AGAIN with specific flags for Interconnect:
    # correct_length=True (Removes pigtails/arms)
    # correct_envelope_and_t_phase=False (KEEPS physical group delay/phase slope)
    _, _, _, _, _, S11_interconnect, S21_interconnect = sim.get_s_and_t_matrix(
        neff_mat_file=config.NEFF_DATA_PATH,
        correct_length=True,
        correct_envelope_and_t_phase=True
    )

    interconnect_file = os.path.join(config.RESULTS_DIR, f"interconnect_symmetric_{tag}.txt")
    # Calling the symmetric export function from analysis.py
    # NOTE: Ensure your analysis.py actually has this function!
    analysis.export_for_interconnect_symmetric(interconnect_file, wl, S11_interconnect, S21_interconnect)
    print(f"Interconnect data saved to: {interconnect_file}")

    # 9. Plot
    fig1 = plt.figure(num="Spectrum", figsize=(10, 6))
    plt.plot(wl * 1e9, T, label="T (Modal)")
    plt.plot(wl * 1e9, R, label="R (Modal)")
    plt.plot(target_wl * 1e9, T[idx_peak], 'ro', label="Detected Resonance")
    plt.title(f"Spectral Response\nScan: {tag}")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized power")
    plt.legend()
    plt.grid(True)

    fig2 = plt.figure(num="Field Profile", figsize=(10, 6))
    plt.plot(f_x * 1e6, I_x_envelope, 'r-', linewidth=2.5, label="Field Envelope (Peak Tracing)")
    plt.plot(f_x * 1e6, I_x_1D, 'b-', alpha=0.3, linewidth=0.5, label="Raw |E|^2")

    y_max =  np.max(I_x_envelope)
    #y_min = np.min(I_x_envelope)
    y_min = 0 # make it relative to zero, as edge is not when profile fully decays
    target_y = y_min + 0.5 * (y_max - y_min)
    plt.hlines(target_y, -fwhm_val * 1e6 / 2, fwhm_val * 1e6 / 2, colors='k', linestyles='dashed')
    plt.text(0, target_y * 1.05, f"FWHM = {fwhm_val * 1e6:.2f} um", ha='center', color='black', fontweight='bold')

    plt.title(f"Mode Profile (Cropped)\nResonance at {actual_mon_wl * 1e9:.2f} nm")
    plt.xlabel("Position x [um]")
    plt.ylabel("Integrated Energy Density (a.u.)")
    plt.legend()
    plt.grid(True)

    print("Displaying plots...")
    plt.show()

    sim.close()


if __name__ == "__main__":
    run_single_sim()