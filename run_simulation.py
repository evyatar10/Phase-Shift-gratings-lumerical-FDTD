import shutil
import time
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from bragg_device import PiShiftBraggFDTD
import config


def run_single_sim():
    # 1. Parameters
    lambda_res_est = 1.616e-6
    scan_width_nm = 42.0
    n_points = 3001
    avg_corr = 800e-9
    corr_depth = 350e-9
    w_wide = avg_corr + corr_depth / 2
    w_narrow = avg_corr - corr_depth / 2
    core_h = 350e-9

    calc_y_span = w_wide + 1.8 * lambda_res_est
    calc_z_span = core_h + 1.8 * lambda_res_est

    # 2. Initialize Simulation
    sim = PiShiftBraggFDTD(
        pitch=520e-9,
        n_periods_each_side=145,
        n_apod_periods_each_side=20,
        width_narrow=w_narrow,
        width_wide=w_wide,
        width_port=1000e-9,
        core_height=core_h,
        substrate_thickness=4e-6,

        override_cavity_length_nm=185.0,

        y_span=calc_y_span,
        z_span=calc_z_span,
        material_db_path=config.MATERIAL_DB_PATH,
        n_periods_dist_to_port=30,
        n_wls_dist_port_to_pml=5.0,
        n_eff_guess=1.55,
        n_wl_points=n_points,
        use_apodization=True,
        center_mod_depth_nm=4.0,

        use_cavity_mesh_override=True,

        # --- SYMMETRY SETTINGS ---
        use_symmetry=True,    # Y-axis (Anti-Symmetric for TE)
        use_z_symmetry=True,  # Z-axis (Symmetric for TE)

        # --- MATERIAL SETTINGS ---
        use_constant_materials=False,
        n_core_const=1.98  # Accurate SiN index at 1550nm
    )

    # 3. Generate Filenames
    N = sim.n_periods_each_side
    Napod = sim.n_apod_periods_each_side
    use_apod = bool(sim.use_apodization) and (Napod is not None) and (Napod > 0)

    cav_tag = ""
    if sim.cavity_length != sim.pitch / 2.0:
        cav_tag = f"_L_cav_{int(sim.cavity_length * 1e9)}"

    # Update tag to reflect material choice
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

    # 5. Process
    wl, R, T, Loss, T_mat, S11, S21 = sim.get_s_and_t_matrix(
        correct_phase=True,
        neff_mat_file=config.NEFF_DATA_PATH
    )

    # 6. Save
    mat_data = {
        'wl_m': wl,
        'wl_nm': wl * 1e9,
        'T': T, 'R': R, 'loss': Loss,
        'T_matrix': T_mat,
        'S11_complex': S11, 'S21_complex': S21,
        'L_device': 2.0 * sim.x_grating_end
    }
    sio.savemat(results_path, mat_data)
    print(f"Data saved to: {results_path}")

    # The data folder has the same name as the layout file
    folder_to_delete = os.path.splitext(layout_path)[0]

    if os.path.exists(folder_to_delete):
        try:
            shutil.rmtree(folder_to_delete)
            print(f"CLEANUP: Deleted data folder: {os.path.basename(folder_to_delete)}")
        except Exception as e:
            print(f"CLEANUP ERROR: Could not delete {folder_to_delete}: {e}")

    # 7. Plot
    plt.figure()
    plt.plot(wl * 1e9, T, label="T (Modal)")
    plt.plot(wl * 1e9, R, label="R (Modal)")
    plt.plot(wl * 1e9, Loss, label="Radiation Loss")
    plt.title(f"Scan: {tag}")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    sim.close()


if __name__ == "__main__":
    run_single_sim()