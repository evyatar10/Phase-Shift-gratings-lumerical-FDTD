import shutil
import time
import os
import gc
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.integrate
import scipy.signal
from scipy.interpolate import interp1d
import numpy as np
from bragg_device import PiShiftBraggFDTD
import analysis
import config


# --- HELPER FUNCTIONS ---

def find_bragg_resonance(wl, T):
    stopband_indices = np.where(T < 0.6)[0]
    if len(stopband_indices) == 0:
        stopband_indices = np.where(T < 0.85)[0]
    if len(stopband_indices) == 0:
        print("Warning: No stopband detected. Using global maximum.")
        return np.argmax(T)
    idx_start = stopband_indices[0]
    idx_end = stopband_indices[-1]
    T_roi = T[idx_start: idx_end + 1]
    return idx_start + np.argmax(T_roi)


def calculate_fwhm_relative(x, y):
    y_max, y_min = np.max(y), np.min(y)
    target_level = y_min + 0.5 * (y_max - y_min)
    signs = np.sign(y - target_level)
    zero_crossings = np.where(np.diff(signs))[0]
    if len(zero_crossings) >= 2:
        def get_x_interp(idx):
            y1, y2 = y[idx], y[idx + 1]
            x1, x2 = x[idx], x[idx + 1]
            slope = (y2 - y1) / (x2 - x1)
            return x1 + (target_level - y1) / slope if slope != 0 else x1

        return get_x_interp(zero_crossings[-1]) - get_x_interp(zero_crossings[0])
    return 0.0


def extract_envelope_peaks(x, y):
    y = np.squeeze(y)
    x = np.squeeze(x)
    if y.ndim == 0: y = np.atleast_1d(y)
    if x.ndim == 0: x = np.atleast_1d(x)
    peaks, _ = scipy.signal.find_peaks(y)
    if len(peaks) < 2: return y
    f_interp = interp1d(x[peaks], y[peaks], kind='cubic', bounds_error=False, fill_value=(y[peaks[0]], y[peaks[-1]]))
    return f_interp(x)


def extract_and_process_field_profile(sim, target_wl):
    field_data = sim.fdtd.getresult("field_profile", "E")
    f_lam = np.squeeze(field_data['lambda'])
    f_x = np.squeeze(field_data['x'])
    f_y = np.squeeze(field_data['y'])
    idx_mon = np.argmin(np.abs(f_lam - target_wl))

    f_E = field_data['E']
    if f_E.ndim == 5:
        E_res = f_E[:, :, 0, idx_mon, :]
    elif f_E.ndim == 4:
        E_res = f_E[:, :, idx_mon, :]
    else:
        E_res = f_E[..., idx_mon, :]

    I_xy = np.sum(np.abs(E_res) ** 2, axis=-1)
    I_x_1D = scipy.integrate.trapezoid(I_xy, f_y, axis=1)
    I_x_1D = np.squeeze(I_x_1D)

    valid_indices = np.abs(f_x) <= sim.x_grating_end
    f_x = f_x[valid_indices]
    I_x_1D = I_x_1D[valid_indices]
    I_x_envelope = extract_envelope_peaks(f_x, I_x_1D)
    fwhm_val = calculate_fwhm_relative(f_x, I_x_envelope)

    return f_x, I_x_1D, I_x_envelope, fwhm_val, f_lam[idx_mon]


# --- SWEEP RUNNER ---

def run_parameter_sweep():
    # Sweep Parameters
    N_periods_list = [80,150]

    # Common Config
    lambda_res_est = 1.562673e-6    #1.56232e-6
    n_3d_points = 51
    record_3d_fields = True
    export_interconnect_data = True

    pitch = 500e-9
    cav_len = pitch / 2.0
    N_reference_for_crop = 80 #N_periods_list[0]
    crop_len_m = 2.0 * (N_reference_for_crop * pitch) + cav_len + 1.0e-6

    scan_width_nm = 30.0
    n_points_global = 3001
    core_h = 350e-9
    w_wide = 900e-9
    w_narrow = 700e-9

    print(f"--- SWEEP CONFIGURATION ---")
    print(f"Sweeping N_periods: {N_periods_list}")
    print(f"Center Wavelength: {lambda_res_est * 1e6:.3f} um")
    print(f"3D Monitor: {n_3d_points} points")
    print(f"---------------------------\n")

    for i, N_periods in enumerate(N_periods_list):
        print(f"\n>>> STARTING RUN {i + 1}/{len(N_periods_list)}: N_periods = {N_periods} <<<")

        sim = PiShiftBraggFDTD(
            pitch=pitch,
            n_periods_each_side=N_periods,
            n_apod_periods_each_side=5,
            width_narrow=w_narrow,
            width_wide=w_wide,
            width_port=1000e-9,
            core_height=core_h,
            substrate_thickness=4e-6,
            override_cavity_length_nm=False,
            y_span=w_wide + 1.8 * lambda_res_est,
            z_span=core_h + 1.8 * lambda_res_est,
            material_db_path=config.MATERIAL_DB_PATH,
            n_periods_dist_to_port=20,
            n_wls_dist_port_to_pml=5.0,
            n_eff_guess=1.55,
            n_wl_points=n_points_global,
            use_apodization=True,
            center_mod_depth_nm=10.0,
            use_cavity_mesh_override=True,
            use_symmetry=True,
            use_z_symmetry=True,
            use_constant_materials=True,
            n_core_const=1.977,
            record_3d_fields=record_3d_fields,
            field_3d_span_m=crop_len_m,
            downsample_yz=1
        )

        N = sim.n_periods_each_side
        Napod = sim.n_apod_periods_each_side
        use_apod = bool(sim.use_apodization) and (Napod is not None) and (Napod > 0)
        cav_tag = f"_L_cav_{int(sim.cavity_length * 1e9)}" if sim.cavity_length != sim.pitch / 2.0 else ""
        mat_tag = "_CONST" if sim.use_constant_materials else ""
        base_tag = f"{N}_periods_{Napod}_apodizations{cav_tag}{mat_tag}" if use_apod else f"{N}_periods{cav_tag}{mat_tag}"
        file_tag = f"{base_tag}_3D_crop" if record_3d_fields else base_tag

        layout_path = os.path.join(config.LAYOUTS_DIR, f"layout_{file_tag}.fsp")
        results_path = os.path.join(config.RESULTS_DIR, f"result_{file_tag}.mat")

        try:
            sim.build()
            sim.update_scan(center_lambda_m=lambda_res_est, width_nm=scan_width_nm, n_points=n_points_global)

            if record_3d_fields:
                sim.fdtd.setnamed("field_profile_3D", "frequency points", n_3d_points)
                print(f"Override: Set 3D monitor to {n_3d_points} points.")

            sim.fdtd.save(layout_path)

            start_time = time.perf_counter()
            sim.fdtd.run()
            print(f"Run finished in {time.perf_counter() - start_time:.2f} sec")

            # --- PROCESS S-PARAMS (Standard) ---
            wl, R, T, Loss, T_mat, S11, S21 = sim.get_s_and_t_matrix(
                neff_mat_file=config.NEFF_DATA_PATH, correct_length=True, correct_envelope_and_t_phase=True
            )
            idx_peak = find_bragg_resonance(wl, T)
            target_wl = wl[idx_peak]
            print(f"Resonance at: {target_wl * 1e9:.2f} nm")

            # --- PROCESS 2D PROFILE ---
            f_x, I_x_1D, I_x_envelope, fwhm_val, actual_mon_wl = extract_and_process_field_profile(sim, target_wl)

            # --- EXTRACT 3D DATA (Modified to keep all wavelengths) ---
            field_3d_data = {}
            if sim.record_3d_fields:
                print("Extracting 3D Field...")
                res_3d = sim.fdtd.getresult("field_profile_3D", "E")
                lam_3d = np.squeeze(res_3d['lambda'])

                print(f"Saving 3D field for all {len(lam_3d)} frequency points.")

                # IMPORTANT: Save full data (no slicing on freq axis)
                E_full = res_3d['E']

                field_3d_data = {
                    'x': np.squeeze(res_3d['x']),
                    'y': np.squeeze(res_3d['y']),
                    'z': np.squeeze(res_3d['z']),
                    'E_res': E_full,
                    'lambda_3d': lam_3d
                }
                del res_3d, E_full

                # --- SAVE RESULTS (Updated to match run_simulation.py keys) ---
            mat_data = {
                'wl_m': wl,
                'wl_nm': wl * 1e9,  # ADDED: Standard wavelength vector in nm
                'T': T,
                'R': R,  # ADDED: Reflection
                'loss': Loss,  # ADDED: Loss
                'T_matrix': T_mat,  # ADDED: T-Matrix
                'S11_complex': S11,  # ADDED: Complex S11
                'S21_complex': S21,  # ADDED: Complex S21 (same as previous 'S21_complex')
                'L_device': 2.0 * sim.x_grating_end,
                'field_x': f_x,
                'field_energy_density_1D': I_x_1D,
                'field_envelope_1D': I_x_envelope,  # ADDED: Envelope 1D
                'fwhm_m': fwhm_val,  # ADDED: FWHM
                'field_3d': field_3d_data
            }
            sio.savemat(results_path, mat_data)
            print(f"Saved: {results_path}")

            if export_interconnect_data:
                _, _, _, _, _, S11_int, S21_int = sim.get_s_and_t_matrix(
                    neff_mat_file=config.NEFF_DATA_PATH, correct_length=True, correct_envelope_and_t_phase=True
                )
                int_file = os.path.join(config.RESULTS_DIR, f"interconnect_symmetric_{file_tag}.txt")
                analysis.export_for_interconnect_symmetric(int_file, wl, S11_int, S21_int)

        finally:
            sim.close()
            del sim
            gc.collect()
            print("Memory cleared.\n")


if __name__ == "__main__":
    run_parameter_sweep()