# run_simulation.py
import time
import os
import matplotlib.pyplot as plt
import scipy.io as sio

# --- IMPORTS FROM YOUR NEW FILES ---
from bragg_device import PiShiftBraggFDTD
import config


def run_single_sim():
    # 1. Parameters
    lambda_res_est = 1.5725e-6
    scan_width_nm = 42.0
    n_points = 3001
    w_wide = 900e-9
    core_h = 350e-9

    # Dynamic span calculation
    calc_y_span = w_wide + 1.8 * lambda_res_est
    calc_z_span = core_h + 1.8 * lambda_res_est

    # 2. Initialize Simulation
    sim = PiShiftBraggFDTD(
        pitch=500e-9,
        n_periods_each_side=40,
        n_apod_periods_each_side=10,
        width_narrow=700e-9,
        width_wide=w_wide,
        width_port=1000e-9,
        core_height=core_h,
        substrate_thickness=4e-6,

        # Exact spans from original
        y_span=calc_y_span,
        z_span=calc_z_span,

        material_db_path=config.MATERIAL_DB_PATH,

        # Specific geometry params from original
        n_periods_dist_to_port=20,
        n_wls_dist_port_to_pml=2.0,

        n_eff_guess=1.55,
        n_wl_points=n_points,
        use_apodization=False,  # Matches original (False)
        center_mod_depth_nm=10.0,

        use_cavity_mesh_override = False,
        dx_cavity_nm = 10.0,

    )

    # 3. Generate Filenames (Exact naming logic from original)
    N = sim.n_periods_each_side
    Napod = sim.n_apod_periods_each_side
    use_apod = bool(sim.use_apodization) and (Napod is not None) and (Napod > 0)

    tag = f"{N}_periods_{Napod}_apodizations" if use_apod else f"{N}_periods"

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

    # 5. Process Results
    wl, R, T, Loss, T_mat, S11, S21 = sim.get_s_and_t_matrix(
        correct_phase=True,
        neff_mat_file=config.NEFF_DATA_PATH
    )

    # 6. Save Data
    mat_data = {
        'wl_m': wl,
        'wl_nm': wl * 1e9,  # Added wl_nm to match original save
        'T': T,
        'R': R,
        'loss': Loss,
        'T_matrix': T_mat,
        'S11_complex': S11,
        'S21_complex': S21,
        'L_device': 2.0 * sim.x_grating_end
    }
    sio.savemat(results_path, mat_data)
    print(f"Data saved to: {results_path}")

    # 7. Quick Plot
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