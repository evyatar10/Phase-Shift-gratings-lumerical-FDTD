import numpy as np
import os
import importlib.util
import math  # <--- Added import
import analysis

# Try to import lumapi normally
try:
    import lumapi
except ImportError:
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
                 n_periods_dist_to_port=5,
                 n_wls_dist_port_to_pml=2.0,
                 override_cavity_length_nm=None,
                 material_db_path=None,
                 core_material="Si3N4 (Silicon Nitride) - Luke",
                 clad_material="SiO2 (Glass) - Palik",
                 n_eff_guess=1.55,
                 coarse_width_nm=150,
                 n_wl_points=401,
                 use_apodization=False,
                 center_mod_depth_nm=40.0,
                 use_cavity_mesh_override=False):

        self.pitch = pitch
        self.n_periods_each_side = n_periods_each_side

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

        self.use_cavity_mesh_override = use_cavity_mesh_override

        self.lambda_B = 2 * self.n_eff_guess * self.pitch

        if override_cavity_length_nm:
            self.cavity_length = override_cavity_length_nm * 1e-9
        else:
            self.cavity_length = pitch / 2.0

        self.x_grating_end = (self.n_periods_each_side * self.pitch) + (self.cavity_length / 2.0)
        self.dist_grating_to_port = n_periods_dist_to_port * self.pitch
        self.x_port = self.x_grating_end + self.dist_grating_to_port
        self.dist_port_to_pml = n_wls_dist_port_to_pml * self.lambda_B
        self.x_sim_boundary = self.x_port + self.dist_port_to_pml
        self.sim_x_span = 2.0 * self.x_sim_boundary

        self.coarse_width_nm = coarse_width_nm
        half_w = 0.5 * self.coarse_width_nm * 1e-9
        self.lam_min = self.lambda_B - half_w
        self.lam_max = self.lambda_B + half_w

        self.fdtd = lumapi.FDTD()
        self._setup_materials()

    def _setup_materials(self):
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

        # Updated range based on your successful test (Wide enough for stability, narrow enough for accuracy)
        safe_lam_min = 1.35e-6
        safe_lam_max = 1.85e-6

        script = f'''
        if (haveresult("{custom_sin}", "material")) {{ deletematerial("{custom_sin}"); }}
        if (haveresult("{custom_sio2}", "material")) {{ deletematerial("{custom_sio2}"); }}

        m1 = copymaterial("{src_core}");
        setmaterial(m1, "name", "{custom_sin}");
        m2 = copymaterial("{src_clad}");
        setmaterial(m2, "name", "{custom_sio2}");

        # --- 1. FIT RANGE ---
        setmaterial("{custom_sin}",  "specify fit range", 1);
        setmaterial("{custom_sio2}", "specify fit range", 1);
        setmaterial("{custom_sin}",  "wavelength min", {safe_lam_min});
        setmaterial("{custom_sin}",  "wavelength max", {safe_lam_max});
        setmaterial("{custom_sio2}", "wavelength min", {safe_lam_min});
        setmaterial("{custom_sio2}", "wavelength max", {safe_lam_max});

        # --- 2. HIGH-Q OPTIMIZATION (CRITICAL) ---
        # "imaginary weight" prioritizes zero loss (10^-10) over refractive index accuracy
        setmaterial("{custom_sin}",  "imaginary weight", 2); 
        setmaterial("{custom_sio2}", "imaginary weight", 2);

        setmaterial("{custom_sin}",  "max coefficients", 8);
        setmaterial("{custom_sio2}", "max coefficients", 8);
        setmaterial("{custom_sin}",  "tolerance", 0.001);
        setmaterial("{custom_sio2}", "tolerance", 0.001);

        # --- 3. STABILITY CHECKS ---
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
        fdtd.setdevice("CPU")
        fdtd.set("background material", self.clad_material)
        for bc in ["x min bc", "x max bc", "y min bc", "y max bc", "z min bc", "z max bc"]:
            fdtd.set(bc, "PML")
        fdtd.set("simulation time", 60e-12)
        fdtd.set("auto shutoff min", 1e-6)
        fdtd.set("mesh accuracy", 3)
        fdtd.set("dt stability factor", 0.9)

    def _add_aligned_mesh_override(self, cells_per_half_period=5, max_cavity_dx=40e-9):
        """
        Adds mesh overrides.
        UPDATED: Now enforces a high-resolution mesh (max_cavity_dx) inside the cavity
        to ensure we have more than just 4 cells.
        """
        fdtd = self.fdtd
        import math

        # --- 1. Calculate General Steps ---
        half_pitch = 0.5 * self.pitch
        n_cells_half = max(1, int(cells_per_half_period))
        dx_grating = half_pitch / float(n_cells_half)  # Target: ~50nm

        dy_global = self.width_narrow / 13.0
        dz_global = self.core_height / 7.0

        max_device_width = max(self.width_port, self.width_wide, self.width_narrow)
        y_span_override = max_device_width * 1.2
        z_span_override = self.core_height

        # --- 2. Anchor Points ---
        x_cav_right = self.cavity_length / 2.0
        x_cav_left = -self.cavity_length / 2.0
        x_sim_left = -self.sim_x_span / 2.0 - 1e-6
        x_sim_right = self.sim_x_span / 2.0 + 1e-6

        print(f"DEBUG: Grating Step (dx) = {dx_grating * 1e9:.2f} nm")

        # --- Region A: LEFT GRATING ARM ---
        fdtd.addmesh()
        fdtd.set("name", "mesh_left_arm")
        len_left = x_cav_left - x_sim_left
        fdtd.set("x", x_sim_left + len_left / 2.0)
        fdtd.set("x span", len_left)
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

        # --- Region B: RIGHT GRATING ARM ---
        fdtd.addmesh()
        fdtd.set("name", "mesh_right_arm")
        len_right = x_sim_right - x_cav_right
        fdtd.set("x", x_cav_right + len_right / 2.0)
        fdtd.set("x span", len_right)
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

        # --- Region C: CAVITY (Central) ---
        if self.use_cavity_mesh_override:
            # 1. Determine Target Step
            # We want the step to be AT LEAST as small as the grating,
            # but preferably smaller (e.g. 10nm) to resolve the phase shift.
            target_dx = min(dx_grating, max_cavity_dx)

            # 2. Calculate integer number of cells to fit cavity
            n_cells_cav = max(1, math.ceil(self.cavity_length / target_dx))

            # 3. Recalculate exact dx to fill the length perfectly
            dx_cav_snapped = self.cavity_length / float(n_cells_cav)

            print(f"DEBUG: Cavity Length = {self.cavity_length * 1e9:.2f} nm")
            print(f"DEBUG: Cavity Target Resolution = {max_cavity_dx * 1e9:.2f} nm")
            print(f"DEBUG: Cavity Actual Cells = {n_cells_cav}")
            print(f"DEBUG: Cavity Step (dx) = {dx_cav_snapped * 1e9:.2f} nm")

            fdtd.addmesh()
            fdtd.set("name", "mesh_cavity")
            fdtd.set("x", 0.0)
            fdtd.set("x span", self.cavity_length)
            fdtd.set("y", 0.0)
            fdtd.set("y span", y_span_override)
            fdtd.set("z", 0.0)
            fdtd.set("z span", z_span_override)
            fdtd.set("override x mesh", 1)
            fdtd.set("override y mesh", 1)
            fdtd.set("override z mesh", 1)

            # Apply the strictly smaller, aligned step
            fdtd.set("dx", dx_cav_snapped)
            fdtd.set("dy", dy_global)
            fdtd.set("dz", dz_global)

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
        full_depth_center = self.center_mod_depth if self.use_apodization else full_depth_edge
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

        W_narrow, W_wide = {}, {}
        for d in range(1, n_total + 1):
            mod_depth = get_mod_depth(d)
            delta = mod_depth / 2.0
            W_narrow[d] = avg_width - delta
            W_wide[d] = avg_width + delta

        x_grating_start = -self.x_grating_end
        x = x_grating_start
        add_core_segment(-self.x_sim_boundary - 1e-6, x_grating_start, self.width_port, name_prefix="wg_left_inf")

        for d in range(n_total, 0, -1):
            add_core_segment(x, x + half_pitch, W_narrow[d], name_prefix=f"L_narrow_{d}")
            x += half_pitch
            add_core_segment(x, x + half_pitch, W_wide[d], name_prefix=f"L_wide_{d}")
            x += half_pitch

        add_core_segment(x, x + self.cavity_length, W_narrow[1], name_prefix="cavity")
        x += self.cavity_length

        for d in range(1, n_total + 1):
            add_core_segment(x, x + half_pitch, W_narrow[d], name_prefix=f"R_narrow_{d}")
            x += half_pitch
            add_core_segment(x, x + half_pitch, W_wide[d], name_prefix=f"R_wide_{d}")
            x += half_pitch

        add_core_segment(x, self.x_sim_boundary + 1e-6, self.width_port, name_prefix="wg_right_inf")

    def _add_source_and_monitors(self):
        fdtd = self.fdtd
        cells_per_half_period = 5
        half_pitch = 0.5 * self.pitch
        n_cells_half = max(1, int(cells_per_half_period))
        dx_mesh = half_pitch / float(n_cells_half)
        dist_snapped = round(self.dist_grating_to_port / dx_mesh) * dx_mesh
        self.dist_grating_to_port = dist_snapped
        self.x_port = self.x_grating_end + dist_snapped

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

    def get_s_and_t_matrix(self, neff_mat_file=None, correct_phase=True):
        res1 = self.fdtd.getresult("FDTD::ports::Port_1", "expansion for port monitor")
        res2 = self.fdtd.getresult("FDTD::ports::Port_2", "expansion for port monitor")
        wl = np.squeeze(res1["lambda"])
        S11_raw = np.squeeze(res1["S"])
        S21_raw = np.squeeze(res2["S"])

        if correct_phase:
            neff1_data, neff2_data = None, None
            if not (neff_mat_file and os.path.exists(neff_mat_file)):
                n1_res = self.fdtd.getresult("FDTD::ports::Port_1", "neff")
                n2_res = self.fdtd.getresult("FDTD::ports::Port_2", "neff")
                neff1_data = np.squeeze(n1_res["neff"])
                neff2_data = np.squeeze(n2_res["neff"])

            S11_sim, S21_sim = analysis.apply_phase_correction(
                wl, S11_raw, S21_raw,
                self.pitch, self.dist_grating_to_port, self.x_grating_end,
                neff_mat_file, neff1_data, neff2_data
            )
        else:
            S11_sim, S21_sim = S11_raw, S21_raw

        R_modal, T_modal, Loss_radiation, T_matrix = analysis.calculate_physics_matrices(S11_sim, S21_sim)
        return wl, R_modal, T_modal, Loss_radiation, T_matrix, S11_sim, S21_sim