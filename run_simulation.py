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

# ── Far-field extraction config ──────────────────────────────────────────
FF_RES  = 201   # ux/uy grid resolution  (must match plot_farfield.m)
IDX_F   = 1     # 1-based frequency index (monitors record 1 point)


def extract_farfield(fdtd, monitor_name):
    """
    Pull far-field data from one monitor via Lumerical eval().
    Returns dict with E2, ux, uy, lam  or  None on failure.
    """
    print(f"  Extracting far-field: {monitor_name}")
    try:
        res = fdtd.getresult(monitor_name, "E")
        lam = float(np.squeeze(res["lambda"]))
        print(f"    lam = {lam*1e9:.3f} nm")
    except Exception as e:
        print(f"    ERROR: no data [{e}]")
        return None

    script = f"""
    mname = '{monitor_name}';
    idx   = {IDX_F};
    res   = {FF_RES};
    E2 = farfield3d(mname, idx, res, res);
    ux = farfieldux(mname, idx, res, res);
    uy = farfielduy(mname, idx, res, res);
    """
    try:
        fdtd.eval(script)
    except Exception as e:
        print(f"    ERROR in eval [{e}]")
        return None

    E2 = np.squeeze(fdtd.getv("E2"))
    ux = np.squeeze(fdtd.getv("ux"))
    uy = np.squeeze(fdtd.getv("uy"))
    return {"E2": E2, "ux": ux, "uy": uy, "lam": lam}


def extract_monitor_nearfield(fdtd, monitor_name):
    """
    Extract the 2D E-field recorded on a planar profile monitor surface.
    Returns a dict with x, y, z, E_res (complex, all freq pts), lambda_arr,
    or None on failure.
    """
    print(f"  Extracting near-field from monitor surface: {monitor_name}")
    try:
        res = fdtd.getresult(monitor_name, "E")
    except Exception as e:
        print(f"    ERROR: could not read E from {monitor_name} [{e}]")
        return None

    return {
        "x":          np.squeeze(res["x"]),
        "y":          np.squeeze(res["y"]),
        "z":          np.squeeze(res["z"]),
        "E_res":      res["E"],          # complex array, all freq points
        "lambda_arr": np.squeeze(res["lambda"]),
    }


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
    print(f"Extracting 1D core field tracking at monitor wavelength: {f_lam[idx_mon] * 1e9:.3f} nm")

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
    cleanup_lumerical_data = False  # Set True to delete .h5 files after run (saves disk, but hides data in Lumerical GUI)
    lambda_res_est = 1.5625e-6  # Center of Scan
    scan_width_nm = 16.0

    n_points_global = 3001  # For high-res S-parameters
    n_2d_points = 51  # Odd number ensures center wl is captured

    avg_corr = 800e-9
    corr_depth = 200e-9
    w_wide = avg_corr + corr_depth / 2
    w_narrow = avg_corr - corr_depth / 2
    core_h = 350e-9

    span_multiplier = 4
    calc_y_span = w_wide + span_multiplier * lambda_res_est
    calc_z_span = core_h + span_multiplier * lambda_res_est

    # --- DEVICE & RECORDING CONFIG ---
    pitch = 500e-9
    N_periods = 100

    N_periods_target_overlap = N_periods
    cav_len = pitch / 2.0
    overlap_len_m = 2.0 * (N_periods_target_overlap * pitch) + cav_len + 1.0e-6

    # far-field distance for Y
    farfield_y_wls = 0.5
    calc_farfield_y = (calc_y_span / 2.0) - (farfield_y_wls * lambda_res_est)

    # far-field distance for Z
    farfield_z_wls = 0.5
    calc_farfield_z = (calc_z_span / 2.0) - (farfield_z_wls * lambda_res_est)

    # far-field monitor X window (centered on defect)
    farfield_x_span_m = 20e-6  # 20 µm window around the phase-shift defect

    # 2. Initialize Simulation
    sim = PiShiftBraggFDTD(
        pitch=pitch,
        n_periods_each_side=N_periods,
        n_apod_periods_each_side=10,
        width_narrow=w_narrow,
        width_wide=w_wide,
        width_port=1000e-9, #1562.684
        core_height=core_h,
        substrate_thickness=10e-6,
        override_cavity_length_nm=False,  # False = pitch/2
        y_span=calc_y_span,
        z_span=calc_z_span,
        material_db_path=config.MATERIAL_DB_PATH,
        n_periods_dist_to_port=20,
        n_wls_dist_port_to_pml=5.0,
        n_eff_guess=1.55,
        n_wl_points=n_points_global,
        use_apodization=True,
        center_mod_depth_nm=10.0,
        use_cavity_mesh_override=True,
        use_symmetry=True,  # y symmetry
        use_z_symmetry=True,
        use_constant_materials=True,
        n_core_const=1.977,

        # --- 2D TOP & CROSS FIELD SETTINGS ---
        record_2d_fields_top_and_cross=True,
        field_2d_x_span_m=None,  # None means record the full X span
        monitor_y_span_m=calc_y_span - 0.5 * lambda_res_est,
        monitor_z_span_m=calc_z_span - 0.5 * lambda_res_est,
        downsample_yz=1,  # Keep default high resolution
        
        # --- 3D MONITORS ---
        record_3d_fields=False,
        field_3d_span_m=None,
        
        # --- FAR FIELD ---
        record_farfield=True,
        farfield_x_span_m=farfield_x_span_m,
        farfield_y_dist_m=calc_farfield_y,
        farfield_z_dist_m=calc_farfield_z
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
    sim.update_scan(center_lambda_m=lambda_res_est, width_nm=scan_width_nm, n_points=n_points_global)

    # --- Override Large Monitor Points & Spatial Crop ---
    if sim.record_2d_fields_top_and_cross:
        sim.fdtd.setnamed("field_profile_2D_XY", "frequency points", n_2d_points)
        sim.fdtd.setnamed("field_profile_2D_YZ_cross", "frequency points", n_2d_points)
        print(f"Override: Set 2D monitors to {n_2d_points} points.")
    if getattr(sim, 'record_3d_fields', False):
        sim.fdtd.setnamed("field_profile_3D", "frequency points", n_2d_points)
        print(f"Override: Set 3D monitor to {n_2d_points} points.")
    if sim.record_farfield:
        # Force 1 point so both monitors record only at the resonance wavelength
        sim.fdtd.setnamed("side_monitor", "frequency points", 1)
        sim.fdtd.setnamed("top_monitor",  "frequency points", 1)
        print("Override: Far-field monitors set to 1 frequency point (resonance wavelength only).")

    sim.fdtd.save(layout_path)
    print(f"Saved layout to: {layout_path}")

    start = time.perf_counter()
    sim.fdtd.run()
    print(f"Simulation time: {time.perf_counter() - start:.3f} seconds")

    # 5. Process S-parameters
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

    f_x, I_x_1D, I_x_envelope, fwhm_val, actual_mon_wl = extract_and_process_field_profile(sim, target_wl)

    print(f"Calculated FWHM (Relative): {fwhm_val * 1e6:.4f} um")

    # --- 7. EXTRACT 2D TOP & CROSS SECTION DATA ---
    field_xy_data = {}
    field_yz_cross_data = {}
    
    if sim.record_2d_fields_top_and_cross:
        print("Extracting 2D XY (Top View) data...")
        res_xy = sim.fdtd.getresult("field_profile_2D_XY", "E")
        lam_xy = np.squeeze(res_xy['lambda'])

        field_xy_data = {
            'x': np.squeeze(res_xy['x']),
            'y': np.squeeze(res_xy['y']),
            'z': np.squeeze(res_xy['z']),
            'E_res': res_xy['E'],  # Contains all freq points
            'lambda_3d': lam_xy
        }
        del res_xy
        
        print("Extracting 2D YZ (Cross Section View) data...")
        res_yz = sim.fdtd.getresult("field_profile_2D_YZ_cross", "E")
        
        field_yz_cross_data = {
            'x': np.squeeze(res_yz['x']),
            'y': np.squeeze(res_yz['y']),
            'z': np.squeeze(res_yz['z']),
            'E_res': res_yz['E'],  # Contains all freq points
            'lambda_3d': lam_xy
        }
        del res_yz

    field_3d_data = {}
    if getattr(sim, 'record_3d_fields', False):
        print("Extracting full 3D Field data...")
        res_3d = sim.fdtd.getresult("field_profile_3D", "E")
        lam_3d = np.squeeze(res_3d['lambda'])
        
        field_3d_data = {
            'x': np.squeeze(res_3d['x']),
            'y': np.squeeze(res_3d['y']),
            'z': np.squeeze(res_3d['z']),
            'E_res': res_3d['E'],
            'lambda_3d': lam_3d
        }
        del res_3d

    # --- 7.5  FAR-FIELD DATA (E2, ux, uy) + NEAR-FIELD (E on monitor surface) ---
    farfield_side  = {}
    farfield_top   = {}
    nearfield_side = {}
    nearfield_top  = {}
    if sim.record_farfield:
        print("\nExtracting far-field data...")
        d = extract_farfield(sim.fdtd, "side_monitor")
        if d is not None:
            farfield_side = d
        d = extract_farfield(sim.fdtd, "top_monitor")
        if d is not None:
            farfield_top = d

        print("\nExtracting near-field (E on monitor surfaces)...")
        nf = extract_monitor_nearfield(sim.fdtd, "side_monitor")
        if nf is not None:
            nearfield_side = nf
        nf = extract_monitor_nearfield(sim.fdtd, "top_monitor")
        if nf is not None:
            nearfield_top = nf

    # 8. Save
    mat_data = {
        'wl_m': wl, 'wl_nm': wl * 1e9,
        'T': T, 'R': R, 'loss': Loss,
        'T_matrix': T_mat, 'S11_complex': S11, 'S21_complex': S21,
        'L_device': 2.0 * sim.x_grating_end,
        'field_x': f_x,
        'field_energy_density_1D': I_x_1D,
        'field_envelope_1D': I_x_envelope,
        'fwhm_m': fwhm_val,
        'field_xy': field_xy_data,
        'field_yz_cross': field_yz_cross_data,
        'field_3d': field_3d_data,
        # --- far-field ---
        'farfield_side':  farfield_side,
        'farfield_top':   farfield_top,
        'farfield_config': {'ff_res': FF_RES},
        # --- near-field (E on monitor surfaces) ---
        'nearfield_side': nearfield_side,
        'nearfield_top':  nearfield_top,
    }
    sio.savemat(results_path, mat_data)
    print(f"Data saved to: {results_path}")

    # 9. Export for Interconnect
    print("Exporting for Interconnect...")
    _, _, _, _, _, S11_interconnect, S21_interconnect = sim.get_s_and_t_matrix(
        neff_mat_file=config.NEFF_DATA_PATH,
        correct_length=True,
        correct_envelope_and_t_phase=True
    )
    interconnect_file = os.path.join(config.RESULTS_DIR, f"interconnect_symmetric_{tag}.txt")
    analysis.export_for_interconnect_symmetric(interconnect_file, wl, S11_interconnect, S21_interconnect)
    print(f"Interconnect data saved to: {interconnect_file}")

    # 10. Plot
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

    y_max = np.max(I_x_envelope)
    y_min = 0
    target_y = y_min + 0.5 * (y_max - y_min)
    plt.hlines(target_y, -fwhm_val * 1e6 / 2, fwhm_val * 1e6 / 2, colors='k', linestyles='dashed')
    plt.text(0, target_y * 1.05, f"FWHM = {fwhm_val * 1e6:.2f} um", ha='center', color='black', fontweight='bold')

    plt.title(f"Mode Profile (Cropped)\nResonance at {actual_mon_wl * 1e9:.2f} nm")
    plt.xlabel("Position x [um]")
    plt.ylabel("Integrated Energy Density (a.u.)")
    plt.legend()
    plt.grid(True)

    print("Displaying plots...")

    # 11. Optionally clean up Lumerical .h5 data files
    if cleanup_lumerical_data:
        try:
            import glob
            data_dir = layout_path.replace(".fsp", "")
            if os.path.exists(data_dir):
                h5_files = glob.glob(os.path.join(data_dir, "*.h5"))
                for h5_file in h5_files:
                    os.remove(h5_file)
                    print(f"Cleaned up {h5_file}")
        except Exception as e:
            print(f"Warning: Could not clean up Lumerical temp files: {e}")

    plt.show()

    sim.close()


if __name__ == "__main__":
    run_single_sim()