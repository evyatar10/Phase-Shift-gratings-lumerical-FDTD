import numpy as np
import os
import time
import importlib.util
import matplotlib.pyplot as plt
import scipy.io as sio

# Import your existing modules
import config
import analysis

# Try to import lumapi
try:
    import lumapi
except ImportError:
    LUMAPI_PATH = r"C:\\Program Files\\Lumerical\\v252\\api\\python\\lumapi.py"
    spec = importlib.util.spec_from_file_location("lumapi", LUMAPI_PATH)
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)


class SimpleBraggFDTD:
    def __init__(self,
                 pitch=500e-9,
                 n_periods=40,  # Total number of periods
                 width_narrow=700e-9,
                 width_wide=900e-9,
                 width_port=1000e-9,
                 core_height=350e-9,
                 substrate_thickness=4e-6,
                 y_span=4e-6,
                 z_span=8e-6,
                 n_periods_dist_to_port=5,
                 n_wls_dist_port_to_pml=2.0,
                 material_db_path=None,
                 core_material="Si3N4 (Silicon Nitride) - Luke",
                 clad_material="SiO2 (Glass) - Palik",
                 n_eff_guess=1.55,
                 coarse_width_nm=150,
                 n_wl_points=401,
                 use_symmetry=True,
                 use_z_symmetry=True,
                 use_constant_materials=False,
                 n_core_const=1.977,
                 n_clad_const=1.44):

        self.pitch = pitch
        self.n_periods = n_periods
        self.width_narrow = width_narrow
        self.width_wide = width_wide
        self.width_port = width_port
        self.core_height = core_height
        self.substrate_thickness = substrate_thickness
        self.y_span = y_span
        self.z_span = z_span

        # Materials
        self.material_db_path = material_db_path
        self.core_material = core_material
        self.clad_material = clad_material
        self.use_constant_materials = use_constant_materials
        self.n_core_const = n_core_const
        self.n_clad_const = n_clad_const

        # Simulation settings
        self.use_symmetry = use_symmetry
        self.use_z_symmetry = use_z_symmetry
        self.n_eff_guess = n_eff_guess
        self.n_wl_points = n_wl_points

        # Calculate Geometry
        self.lambda_B = 2 * self.n_eff_guess * self.pitch

        # Total length of the grating
        self.L_grating = self.n_periods * self.pitch

        # We center the grating at x=0, so the end coordinate is half the length
        self.x_grating_end = self.L_grating / 2.0

        self.dist_grating_to_port = n_periods_dist_to_port * self.pitch
        self.x_port = self.x_grating_end + self.dist_grating_to_port
        self.dist_port_to_pml = n_wls_dist_port_to_pml * self.lambda_B
        self.x_sim_boundary = self.x_port + self.dist_port_to_pml
        self.sim_x_span = 2.0 * self.x_sim_boundary

        # Scan range
        self.coarse_width_nm = coarse_width_nm
        half_w = 0.5 * self.coarse_width_nm * 1e-9
        self.lam_min = self.lambda_B - half_w
        self.lam_max = self.lambda_B + half_w

        self.fdtd = lumapi.FDTD()
        self._setup_materials()

    def _setup_materials(self):
        # COPY OF ORIGINAL MATERIAL SETUP
        if self.use_constant_materials:
            print(f"Using CONSTANT Materials: SiN={self.n_core_const}, SiO2={self.n_clad_const}")
            const_sin = "SiN_Const_Custom"
            const_sio2 = "SiO2_Const_Custom"

            eps_sin = self.n_core_const ** 2
            eps_sio2 = self.n_clad_const ** 2

            script = f'''
            f_vector = [0; 1000e12];
            eps_sin_vector = [{eps_sin}; {eps_sin}];
            eps_sio2_vector = [{eps_sio2}; {eps_sio2}];

            if (materialexists("{const_sin}")) {{ deletematerial("{const_sin}"); }}
            if (materialexists("{const_sio2}")) {{ deletematerial("{const_sio2}"); }}

            new_mat = addmaterial("Sampled data");
            setmaterial(new_mat, "name", "{const_sin}");
            data_sin = matrix(2, 2);
            data_sin(1:2, 1) = f_vector;
            data_sin(1:2, 2) = eps_sin_vector; 
            setmaterial("{const_sin}", "sampled data", data_sin);

            new_mat2 = addmaterial("Sampled data");
            setmaterial(new_mat2, "name", "{const_sio2}");
            data_sio2 = matrix(2, 2);
            data_sio2(1:2, 1) = f_vector;
            data_sio2(1:2, 2) = eps_sio2_vector;
            setmaterial("{const_sio2}", "sampled data", data_sio2);
            '''
            self.fdtd.eval(script)
            self.core_material = const_sin
            self.clad_material = const_sio2
            return

        if self.material_db_path and os.path.exists(self.material_db_path):
            print(f"Importing material DB from: {self.material_db_path}")
            self.fdtd.importmaterialdb(self.material_db_path)
            src_core = 'LGT Si3N4 Sellmeier'
            src_clad = 'LGT SiO2 Sellmeier'
        else:
            print("Using Standard Library materials (Luke/Palik).")
            src_core = self.core_material
            src_clad = self.clad_material

        custom_sin = "SiN_custom"
        custom_sio2 = "SiO2_custom"
        safe_lam_min = 1.35e-6
        safe_lam_max = 1.85e-6

        script = f'''
        if (materialexists("{custom_sin}")) {{ deletematerial("{custom_sin}"); }}
        if (materialexists("{custom_sio2}")) {{ deletematerial("{custom_sio2}"); }}

        if (materialexists("{src_core}")) {{
            m1 = copymaterial("{src_core}");
            setmaterial(m1, "name", "{custom_sin}");
        }} else {{
            addmaterial("Dielectric");
            set("name", "{custom_sin}");
            set("Refractive Index", 2.0); 
        }}

        if (materialexists("{src_clad}")) {{
            m2 = copymaterial("{src_clad}");
            setmaterial(m2, "name", "{custom_sio2}");
        }} else {{
            addmaterial("Dielectric");
            set("name", "{custom_sio2}");
            set("Refractive Index", 1.44);
        }}

        if (materialexists("{custom_sin}")) {{
            setmaterial("{custom_sin}", "specify fit range", 1);
            setmaterial("{custom_sin}", "wavelength min", {safe_lam_min});
            setmaterial("{custom_sin}", "wavelength max", {safe_lam_max});
            setmaterial("{custom_sin}", "imaginary weight", 2); 
            setmaterial("{custom_sin}", "max coefficients", 8);
            setmaterial("{custom_sin}", "tolerance", 0.001);
            setmaterial("{custom_sin}", "make fit passive", 1);
            setmaterial("{custom_sin}", "improve numerical stability", 1);
        }}

        if (materialexists("{custom_sio2}")) {{
            setmaterial("{custom_sio2}", "specify fit range", 1);
            setmaterial("{custom_sio2}", "wavelength min", {safe_lam_min});
            setmaterial("{custom_sio2}", "wavelength max", {safe_lam_max});
            setmaterial("{custom_sio2}", "imaginary weight", 2); 
            setmaterial("{custom_sio2}", "max coefficients", 8);
            setmaterial("{custom_sio2}", "tolerance", 0.001);
            setmaterial("{custom_sio2}", "make fit passive", 1);
            setmaterial("{custom_sio2}", "improve numerical stability", 1);
        }}
        '''
        self.fdtd.eval(script)
        self.core_material = custom_sin
        self.clad_material = custom_sio2

    def _reset_layout(self):
        self.fdtd.switchtolayout()
        self.fdtd.selectall()
        self.fdtd.delete()

    def build(self):
        self._reset_layout()
        self._add_fdtd_region()
        self._add_aligned_mesh_override()
        self._add_bragg_core()
        self._add_source_and_monitors()

    def _add_fdtd_region(self):
        fdtd = self.fdtd
        fdtd.addfdtd()
        fdtd.set("x", 0)
        fdtd.set("x span", self.sim_x_span)
        fdtd.set("y", 0)
        fdtd.set("y span", self.y_span)
        fdtd.set("z", 0)
        fdtd.set("z span", self.z_span)

        for bc in ["x min bc", "x max bc", "y min bc", "y max bc", "z min bc", "z max bc"]:
            fdtd.set(bc, "PML")

        if self.use_symmetry:
            fdtd.set("y min bc", "Anti-Symmetric")
            fdtd.set("force symmetric y mesh", 1)

        if self.use_z_symmetry:
            fdtd.set("z min bc", "Symmetric")
            fdtd.set("force symmetric z mesh", 1)

        fdtd.set("dimension", "3D")
        fdtd.setdevice("GPU")
        fdtd.set("background material", self.clad_material)
        fdtd.set("simulation time", 100e-12)
        fdtd.set("auto shutoff min", 1e-6)
        fdtd.set("mesh accuracy", 3)
        fdtd.set("dt stability factor", 0.5)

    def _add_aligned_mesh_override(self, cells_per_half_period=5):
        # EXACTLY matching the mesh logic from previous code,
        # but simplified to a single box since there is no cavity.
        fdtd = self.fdtd
        half_pitch = 0.5 * self.pitch
        n_cells_half = max(1, int(cells_per_half_period))
        dx_grating = half_pitch / float(n_cells_half)
        dy_global = self.width_narrow / 13.0
        dz_global = self.core_height / 7.0

        max_device_width = max(self.width_port, self.width_wide, self.width_narrow)
        y_span_override = max_device_width * 1.2
        z_span_override = self.core_height

        # Create one single mesh override covering the entire grating area
        # from -x_grating_end to +x_grating_end
        fdtd.addmesh()
        fdtd.set("name", "mesh_grating")
        fdtd.set("x", 0)
        fdtd.set("x span", 2.0 * self.x_grating_end)  # Full length
        fdtd.set("y", 0.0)
        fdtd.set("y span", y_span_override)
        fdtd.set("z", 0.0)
        fdtd.set("z span", z_span_override)
        fdtd.set("override x mesh", 1)
        fdtd.set("override y mesh", 1)
        fdtd.set("override z mesh", 1)
        fdtd.set("dx", dx_grating)
        fdtd.set("dy", dy_global)
        fdtd.set("dz", dz_global)

    def _add_bragg_core(self):
        fdtd = self.fdtd
        z_core_center = 0.0
        half_pitch = self.pitch / 2.0
        seg_id = 0

        def add_core_segment(x1, x2, width, name_prefix="core_seg"):
            nonlocal seg_id
            seg_id += 1
            fdtd.addrect()
            fdtd.set("name", f"{name_prefix}_{seg_id:d}")
            fdtd.set("material", self.core_material)
            fdtd.set("y", 0)
            fdtd.set("y span", width)
            fdtd.set("z", z_core_center)
            fdtd.set("z span", self.core_height)
            fdtd.set("x min", x1)
            fdtd.set("x max", x2)

        # 1. Start from the left edge of the grating
        x_start = -self.x_grating_end
        x = x_start

        # 2. Add Left Infinite Waveguide
        add_core_segment(-self.x_sim_boundary - 1e-6, x_start, self.width_port, name_prefix="wg_left_inf")

        # 3. Add Grating Periods
        # No apodization, no cavity. Simple loop.
        for i in range(self.n_periods):
            # Narrow segment
            add_core_segment(x, x + half_pitch, self.width_narrow, name_prefix=f"narrow_{i}")
            x += half_pitch
            # Wide segment
            add_core_segment(x, x + half_pitch, self.width_wide, name_prefix=f"wide_{i}")
            x += half_pitch

        # 4. Add Right Infinite Waveguide
        add_core_segment(x, self.x_sim_boundary + 1e-6, self.width_port, name_prefix="wg_right_inf")

    def _add_source_and_monitors(self):
        fdtd = self.fdtd
        cells_per_half_period = 5
        half_pitch = 0.5 * self.pitch
        n_cells_half = max(1, int(cells_per_half_period))
        dx_mesh = half_pitch / float(n_cells_half)

        # Snap port distance to mesh
        dist_snapped = round(self.dist_grating_to_port / dx_mesh) * dx_mesh
        self.dist_grating_to_port = dist_snapped
        self.x_port = self.x_grating_end + dist_snapped

        # Port 1 (Forward)
        fdtd.addport()
        fdtd.set("name", "Port_1")
        fdtd.set("injection axis", "x")
        fdtd.set("x", -(self.x_grating_end + dist_snapped))
        fdtd.set("y", 0)
        fdtd.set("y span", 1.2 * self.y_span)
        fdtd.set("z", 0)
        fdtd.set("z span", 1.2 * self.z_span)
        fdtd.set("direction", "forward")
        fdtd.set("mode selection", "fundamental TE mode")
        fdtd.set("frequency dependent profile", 1)

        # Port 2 (Backward)
        fdtd.addport()
        fdtd.set("name", "Port_2")
        fdtd.set("injection axis", "x")
        fdtd.set("x", (self.x_grating_end + dist_snapped))
        fdtd.set("y", 0)
        fdtd.set("y span", 1.2 * self.y_span)
        fdtd.set("z", 0)
        fdtd.set("z span", 1.2 * self.z_span)
        fdtd.set("direction", "backward")
        fdtd.set("mode selection", "fundamental TE mode")
        fdtd.set("frequency dependent profile", 1)

    def update_scan(self, center_lambda_m, width_nm, n_points):
        self.n_wl_points = n_points
        half_w = 0.5 * width_nm * 1e-9
        self.lam_min = center_lambda_m - half_w
        self.lam_max = center_lambda_m + half_w
        self.fdtd.switchtolayout()
        self.fdtd.setglobalsource("wavelength start", self.lam_min)
        self.fdtd.setglobalsource("wavelength stop", self.lam_max)
        self.fdtd.setglobalmonitor("frequency points", self.n_wl_points)
        self.fdtd.setnamed("FDTD::ports", "monitor frequency points", self.n_wl_points)

    def close(self):
        try:
            self.fdtd.close()
        except Exception:
            pass

    def get_s_and_t_matrix(self, neff_mat_file=None, correct_length=True, correct_envelope_and_t_phase=True):
        """
        UPDATED: Now accepts separate flags for length correction and envelope/phase correction.
        """
        res1 = self.fdtd.getresult("FDTD::ports::Port_1", "expansion for port monitor")
        res2 = self.fdtd.getresult("FDTD::ports::Port_2", "expansion for port monitor")
        wl = np.squeeze(res1["lambda"])
        S11_raw = np.squeeze(res1["S"])
        S21_raw = np.squeeze(res2["S"])

        neff1_data, neff2_data = None, None
        use_single_neff = False
        single_neff_val = None

        # Only gather neff if length correction is actually requested
        if correct_length:
            if self.use_constant_materials:
                use_single_neff = True
                n1_res = self.fdtd.getresult("FDTD::ports::Port_1", "neff")
                neff_vec = np.atleast_1d(np.squeeze(n1_res["neff"]))
                mid_idx = len(neff_vec) // 2
                single_neff_val = neff_vec[mid_idx]
                print(f"Using Single Neff for Correction: {np.real(single_neff_val):.4f}")

            elif not (neff_mat_file and os.path.exists(neff_mat_file)):
                n1_res = self.fdtd.getresult("FDTD::ports::Port_1", "neff")
                n2_res = self.fdtd.getresult("FDTD::ports::Port_2", "neff")
                neff1_data = np.squeeze(n1_res["neff"])
                neff2_data = np.squeeze(n2_res["neff"])

        S11_sim, S21_sim = analysis.apply_phase_correction(
            wl, S11_raw, S21_raw,
            self.pitch, self.dist_grating_to_port, self.x_grating_end,
            neff_mat_file, neff1_data, neff2_data,
            use_single_neff=use_single_neff,
            single_neff_val=single_neff_val,
            do_length_correction=correct_length,
            do_envelope_correction=correct_envelope_and_t_phase
        )

        R_modal, T_modal, Loss_radiation, T_matrix = analysis.calculate_physics_matrices(S11_sim, S21_sim)
        return wl, R_modal, T_modal, Loss_radiation, T_matrix, S11_sim, S21_sim


def run_simple_sim():
    # 1. Parameters
    lambda_res_est = 1.560e-6
    scan_width_nm = 42.0
    n_points = 3001
    avg_corr = 800e-9
    corr_depth = 200e-9
    w_wide = avg_corr + corr_depth / 2
    w_narrow = avg_corr - corr_depth / 2
    core_h = 350e-9

    calc_y_span = w_wide + 1.8 * lambda_res_est
    calc_z_span = core_h + 1.8 * lambda_res_est

    # User requested 40 periods total
    N_total = 40

    # 2. Initialize Simulation
    sim = SimpleBraggFDTD(
        pitch=500e-9,
        n_periods=N_total,  # Total periods
        width_narrow=w_narrow,
        width_wide=w_wide,
        width_port=1000e-9,
        core_height=core_h,
        substrate_thickness=4e-6,
        y_span=calc_y_span,
        z_span=calc_z_span,
        material_db_path=config.MATERIAL_DB_PATH,
        n_periods_dist_to_port=30,
        n_wls_dist_port_to_pml=5.0,
        n_eff_guess=1.55,
        n_wl_points=n_points,
        use_symmetry=True,  # y symmetry
        use_z_symmetry=True,
        use_constant_materials=True,
        n_core_const=1.977
    )

    # 3. Generate Filenames
    tag = f"SimpleBragg_{N_total}_periods"
    if sim.use_constant_materials:
        tag += "_CONST"

    # --- SAVE LOCATION ---
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
    # UPDATED: Length correction ON, Envelope/Phase correction ON (User request)
    wl, R, T, Loss, T_mat, S11, S21 = sim.get_s_and_t_matrix(
        neff_mat_file=config.NEFF_DATA_PATH,
        correct_length=True,
        correct_envelope_and_t_phase=False
    )

    # 6. Save Data
    mat_data = {
        'wl_m': wl, 'wl_nm': wl * 1e9,
        'T': T, 'R': R, 'loss': Loss,
        'T_matrix': T_mat, 'S11_complex': S11, 'S21_complex': S21,
        'L_device': 2.0 * sim.x_grating_end,
    }
    sio.savemat(results_path, mat_data)
    print(f"Data saved to: {results_path}")

    # 7. Export for INTERCONNECT (New addition)
    print("Exporting for Interconnect...")
    interconnect_file = os.path.join(config.RESULTS_DIR, f"interconnect_symmetric_{tag}.txt")
    analysis.export_for_interconnect_symmetric(interconnect_file, wl, S11, S21)
    print(f"Interconnect data saved to: {interconnect_file}")

    # 8. Plot R and T
    plt.figure(figsize=(10, 6))
    plt.plot(wl * 1e9, T, label="T (Modal) = |S21|^2", color='blue')
    plt.plot(wl * 1e9, R, label="R (Modal) = |S11|^2", color='red')
    plt.title(f"Simple Bragg Grating Response\nPeriods: {N_total}")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized power")
    plt.legend()
    plt.grid(True)

    print("Displaying plots...")
    plt.show()

    sim.close()


if __name__ == "__main__":
    run_simple_sim()