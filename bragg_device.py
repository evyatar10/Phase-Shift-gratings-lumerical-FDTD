import numpy as np
import os
import importlib.util
import analysis  # Importing the file above

# Try to import lumapi normally
try:
    import lumapi
except ImportError:
    # Adjust this path if needed
    LUMAPI_PATH = r"C:\\Program Files\\Lumerical\\v252\\api\\python\\lumapi.py"
    spec = importlib.util.spec_from_file_location("lumapi", LUMAPI_PATH)
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)


class PiShiftBraggFDTD:
    def __init__(self,
                 pitch=500e-9,
                 n_periods_each_side=10,
                 n_apod_periods_each_side=None,
                 width_narrow=700e-9,
                 width_wide=900e-9,
                 width_port=1000e-9,
                 core_height=350e-9,
                 substrate_thickness=4e-6,
                 y_span=4e-6,
                 z_span=8e-6,
                 # NEW GEOMETRY PARAMETERS
                 n_periods_dist_to_port=5,
                 n_wls_dist_port_to_pml=2.0,

                 # MATERIALS
                 material_db_path=None,
                 core_material="Si3N4 (Silicon Nitride) - Luke",
                 clad_material="SiO2 (Glass) - Palik",

                 n_eff_guess=1.55,
                 coarse_width_nm=150,
                 n_wl_points=401,
                 use_apodization=False,
                 center_mod_depth_nm=40.0,
                 # MESH CONTROLS
                 use_cavity_mesh_override=False,
                 dx_cavity_nm=10.0):

        self.pitch = pitch
        self.n_periods_each_side = n_periods_each_side

        # Apodization logic
        if n_apod_periods_each_side is None:
            self.n_apod_periods_each_side = n_periods_each_side
        else:
            self.n_apod_periods_each_side = max(
                1, min(n_apod_periods_each_side, n_periods_each_side)
            )

        self.width_narrow = width_narrow
        self.width_wide = width_wide
        self.width_port = width_port
        self.core_height = core_height
        self.substrate_thickness = substrate_thickness
        self.y_span = y_span
        self.z_span = z_span

        self.material_db_path = material_db_path
        self.core_material = core_material
        self.clad_material = clad_material

        self.n_eff_guess = n_eff_guess
        self.n_wl_points = n_wl_points
        self.use_apodization = use_apodization
        self.center_mod_depth = center_mod_depth_nm * 1e-9

        # Mesh settings
        self.use_cavity_mesh_override = use_cavity_mesh_override
        self.dx_cavity = dx_cavity_nm * 1e-9

        # --- GEOMETRY CALCULATION ---
        self.lambda_B = 2 * self.n_eff_guess * self.pitch
        # Modified: Removed the offset as requested
        self.cavity_length = pitch / 2.0

        # 1. Grating Extent
        self.x_grating_end = (self.n_periods_each_side * self.pitch) + (self.cavity_length / 2.0)

        # 2. Port Location
        self.dist_grating_to_port = n_periods_dist_to_port * self.pitch
        self.x_port = self.x_grating_end + self.dist_grating_to_port

        # 3. PML Boundary
        self.dist_port_to_pml = n_wls_dist_port_to_pml * self.lambda_B
        self.x_sim_boundary = self.x_port + self.dist_port_to_pml

        # Total Simulation Span
        self.sim_x_span = 2.0 * self.x_sim_boundary

        # Frequency range
        self.coarse_width_nm = coarse_width_nm
        half_w = 0.5 * self.coarse_width_nm * 1e-9
        self.lam_min = self.lambda_B - half_w
        self.lam_max = self.lambda_B + half_w

        self.fdtd = lumapi.FDTD()
        self._setup_materials()

    def _setup_materials(self):
        """Create editable copies of materials and apply fit settings."""
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

        script = f'''
        if (haveresult("{custom_sin}", "material")) {{ deletematerial("{custom_sin}"); }}
        if (haveresult("{custom_sio2}", "material")) {{ deletematerial("{custom_sio2}"); }}

        m1 = copymaterial("{src_core}");
        setmaterial(m1, "name", "{custom_sin}");
        m2 = copymaterial("{src_clad}");
        setmaterial(m2, "name", "{custom_sio2}");

        setmaterial("{custom_sin}",  "specify fit range", 1);
        setmaterial("{custom_sio2}", "specify fit range", 1);

        setmaterial("{custom_sin}",  "wavelength min", {self.lam_min});
        setmaterial("{custom_sin}",  "wavelength max", {self.lam_max});
        setmaterial("{custom_sio2}", "wavelength min", {self.lam_min});
        setmaterial("{custom_sio2}", "wavelength max", {self.lam_max});

        setmaterial("{custom_sin}",  "tolerance", 0.001);
        setmaterial("{custom_sio2}", "tolerance", 0.001);

        setmaterial("{custom_sin}",  "make fit passive", 1);
        setmaterial("{custom_sio2}", "make fit passive", 1);

        setmaterial("{custom_sin}",  "improve numerical stability", 1);
        setmaterial("{custom_sio2}", "improve numerical stability", 1);
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
        fdtd.set("y", 0)
        fdtd.set("z", 0)
        fdtd.set("x span", self.sim_x_span)
        fdtd.set("y span", self.y_span)
        fdtd.set("z span", self.z_span)
        fdtd.set("dimension", "3D")
        fdtd.setdevice("GPU")
        fdtd.set("background material", self.clad_material)

        for bc in ["x min bc", "x max bc", "y min bc", "y max bc", "z min bc", "z max bc"]:
            fdtd.set(bc, "PML")

        fdtd.set("simulation time", 25e-12)
        fdtd.set("auto shutoff min", 1e-6)
        fdtd.set("mesh accuracy", 3)
        fdtd.set("dt stability factor", 0.95)

    def _add_aligned_mesh_override(self, cells_per_half_period=5):
        fdtd = self.fdtd

        # --- 1. X Step (Propagation - Unchanged) ---
        half_pitch = 0.5 * self.pitch
        n_cells_half = max(1, int(cells_per_half_period))
        dx_global = half_pitch / float(n_cells_half)

        # --- 2. Y Step (Resolution Control) ---
        # We stick to your request: 13 cells inside the narrow width.
        # This gives dy ~ 54nm.
        # Note: Since the teeth are ~100nm deep, they will get ~2 cells.
        # Lumerical's "Conformal Mesh" will handle the exact edge position
        # sub-pixel, so 2 cells is acceptable for general trends.
        dy_global = self.width_narrow / 13.0

        # --- 3. Z Step (Exact Height) ---
        dz_global = self.core_height / 7.0

        # --- 4. Y Span (Robustness Control) ---
        # We find the WIDEST part of the geometry to ensure the mesh box
        # covers everything. This prevents grading artifacts on the teeth.
        max_device_width = max(self.width_port, self.width_wide, self.width_narrow)
        y_span_override = max_device_width * 1.2  # 20% buffer to be safe

        # --- Add Mesh Override ---
        fdtd.addmesh()
        fdtd.set("name", "mesh_waveguide_core")

        # Position: Centered
        fdtd.set("x", 0.0)
        fdtd.set("y", 0.0)
        fdtd.set("z", 0.0)

        # --- SPANS ---
        fdtd.set("x span", self.sim_x_span)

        # Y: Covers the entire device + buffer
        fdtd.set("y span", y_span_override)

        # Z: Exact match to core height (Perfect alignment for top/bottom)
        fdtd.set("z span", self.core_height)

        # Enable Overrides
        fdtd.set("override x mesh", 1)
        fdtd.set("override y mesh", 1)
        fdtd.set("override z mesh", 1)

        fdtd.set("set maximum mesh step", 1)
        fdtd.set("dx", dx_global)
        fdtd.set("dy", dy_global)
        fdtd.set("dz", dz_global)

        # --- Cavity Specific Mesh ---
        if self.use_cavity_mesh_override:
            fdtd.addmesh()
            fdtd.set("name", "mesh_cavity_override")
            fdtd.set("x", 0.0)
            fdtd.set("x span", self.cavity_length)

            # Match the Y/Z spans of the main override
            fdtd.set("y", 0.0)
            fdtd.set("y span", y_span_override)
            fdtd.set("z", 0.0)
            fdtd.set("z span", self.core_height)

            fdtd.set("override x mesh", 1)
            fdtd.set("override y mesh", 0)
            fdtd.set("override z mesh", 0)
            fdtd.set("set maximum mesh step", 1)
            fdtd.set("dx", self.dx_cavity)

    def _add_bragg_core(self):
        fdtd = self.fdtd
        z_core_center = 0.0
        pitch = self.pitch
        half_pitch = pitch / 2.0

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

        avg_width = 0.5 * (self.width_narrow + self.width_wide)
        full_depth_edge = self.width_wide - self.width_narrow

        if self.use_apodization:
            full_depth_center = self.center_mod_depth
        else:
            full_depth_center = full_depth_edge

        n_total = self.n_periods_each_side
        n_apod = self.n_apod_periods_each_side

        def get_mod_depth(d):
            if d <= n_apod and n_total > 1:
                denom = (n_apod - 1) if (n_apod > 1 and n_apod == n_total) else n_apod
                if denom == 0: return full_depth_center
                frac = (d - 1) / float(denom)
                return full_depth_center + (full_depth_edge - full_depth_center) * frac
            else:
                return full_depth_edge

        W_narrow = {}
        W_wide = {}
        for d in range(1, n_total + 1):
            mod_depth = get_mod_depth(d)
            delta = mod_depth / 2.0
            W_narrow[d] = avg_width - delta
            W_wide[d] = avg_width + delta

        x_grating_start = -self.x_grating_end
        x = x_grating_start

        x_pml_left = -self.x_sim_boundary - 1e-6
        add_core_segment(x_pml_left, x_grating_start, self.width_port, name_prefix="wg_left_inf")

        for d in range(n_total, 0, -1):
            w_n, w_w = W_narrow[d], W_wide[d]
            x1 = x
            x2 = x1 + half_pitch
            add_core_segment(x1, x2, w_n, name_prefix=f"L_narrow_{d}")
            x = x2
            x1 = x
            x2 = x1 + half_pitch
            add_core_segment(x1, x2, w_w, name_prefix=f"L_wide_{d}")
            x = x2

        w_cavity = W_narrow[1]
        x1 = x
        x2 = x1 + self.cavity_length
        add_core_segment(x1, x2, w_cavity, name_prefix="cavity")
        x = x2

        for d in range(1, n_total + 1):
            w_n, w_w = W_narrow[d], W_wide[d]
            x1 = x
            x2 = x1 + half_pitch
            add_core_segment(x1, x2, w_n, name_prefix=f"R_narrow_{d}")
            x = x2
            x1 = x
            x2 = x1 + half_pitch
            add_core_segment(x1, x2, w_w, name_prefix=f"R_wide_{d}")
            x = x2

        x_pml_right = self.x_sim_boundary + 1e-6
        add_core_segment(x, x_pml_right, self.width_port, name_prefix="wg_right_inf")

    def _add_source_and_monitors(self):
        fdtd = self.fdtd

        cells_per_half_period = 5
        half_pitch = 0.5 * self.pitch
        n_cells_half = max(1, int(cells_per_half_period))
        dx_mesh = half_pitch / float(n_cells_half)

        dist_theoretical = self.dist_grating_to_port
        dist_snapped = round(dist_theoretical / dx_mesh) * dx_mesh
        self.dist_grating_to_port = dist_snapped

        x_Port_1 = -(self.x_grating_end + dist_snapped)
        x_Port_2 = (self.x_grating_end + dist_snapped)

        self.x_port = self.x_grating_end + dist_snapped

        z_center = 0.0
        y_center = 0.0
        monitor_ratio = 1.2

        fdtd.addport()
        fdtd.set("name", "Port_1")
        fdtd.set("injection axis", "x")
        fdtd.set("x", x_Port_1)
        fdtd.set("y", y_center)
        fdtd.set("y span", monitor_ratio * self.y_span)
        fdtd.set("z", z_center)
        fdtd.set("z span", monitor_ratio * self.z_span)
        fdtd.set("direction", "forward")
        fdtd.set("mode selection", "fundamental TE mode")

        fdtd.set("frequency dependent profile", 1)

        fdtd.addport()
        fdtd.set("name", "Port_2")
        fdtd.set("injection axis", "x")
        fdtd.set("x", x_Port_2)
        fdtd.set("y", y_center)
        fdtd.set("y span", monitor_ratio * self.y_span)
        fdtd.set("z", z_center)
        fdtd.set("z span", monitor_ratio * self.z_span)
        fdtd.set("direction", "backward")
        fdtd.set("mode selection", "fundamental TE mode")

        fdtd.set("frequency dependent profile", 1)

        fdtd.addmovie()
        fdtd.set("name", "movie_xy")
        fdtd.set("monitor type", "2D Z-normal")
        fdtd.set("x", 0)
        fdtd.set("x span", self.sim_x_span)
        fdtd.set("y", y_center)
        fdtd.set("y span", self.y_span)
        fdtd.set("z", 0.0)
        fdtd.set("lock aspect ratio", 1)
        fdtd.set("horizontal resolution", 600)

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

    def get_s_and_t_matrix(self, neff_mat_file=None, correct_phase=True):
        # 1. Get raw expansion results
        res1 = self.fdtd.getresult("FDTD::ports::Port_1", "expansion for port monitor")
        res2 = self.fdtd.getresult("FDTD::ports::Port_2", "expansion for port monitor")

        wl = np.squeeze(res1["lambda"])

        S11_raw = np.squeeze(res1["S"])
        S21_raw = np.squeeze(res2["S"])

        S11_sim = S11_raw
        S21_sim = S21_raw

        # 2. Phase Correction (Delegated to analysis.py)
        if correct_phase:
            # We need to fetch internal neff data if external file is missing
            neff1_data = None
            neff2_data = None

            if not (neff_mat_file and os.path.exists(neff_mat_file)):
                # Fetch internal neff from simulation if we aren't loading external file
                n1_res = self.fdtd.getresult("FDTD::ports::Port_1", "neff")
                n2_res = self.fdtd.getresult("FDTD::ports::Port_2", "neff")
                neff1_data = np.squeeze(n1_res["neff"])
                neff2_data = np.squeeze(n2_res["neff"])

            S11_sim, S21_sim = analysis.apply_phase_correction(
                wl, S11_raw, S21_raw,
                self.pitch,
                self.dist_grating_to_port,
                self.x_grating_end,
                neff_mat_file,
                neff1_data,
                neff2_data
            )
        else:
            print("Skipping phase correction (Returning raw S-parameters at Port).")

        # 3. Physics Conversion (Delegated to analysis.py)
        R_modal, T_modal, Loss_radiation, T_matrix = analysis.calculate_physics_matrices(S11_sim, S21_sim)

        return wl, R_modal, T_modal, Loss_radiation, T_matrix, S11_sim, S21_sim