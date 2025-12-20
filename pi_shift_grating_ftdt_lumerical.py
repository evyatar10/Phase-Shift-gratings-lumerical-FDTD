import numpy as np
import matplotlib.pyplot as plt
import time
import os
import scipy.io as sio

# Try to import lumapi normally
try:
    import lumapi
except ImportError:
    import importlib.util
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
                 core_height=350e-9,
                 substrate_thickness=4e-6,
                 y_span=4e-6,
                 z_span=8e-6,
                 # NEW GEOMETRY PARAMETERS
                 n_periods_dist_to_port=5,      # Distance from grating edge to port (in pitches)
                 n_wls_dist_port_to_pml=2.0,    # Distance from port to PML (in wavelengths)

                 core_material="Si3N4 (Silicon Nitride) - Luke",
                 clad_material="SiO2 (Glass) - Palik",
                 n_eff_guess=1.55,
                 coarse_width_nm=150,
                 n_wl_points=401,
                 use_apodization=False,
                 center_mod_depth_nm=40.0):

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
        self.core_height = core_height
        self.substrate_thickness = substrate_thickness
        self.y_span = y_span
        self.z_span = z_span

        self.core_material = core_material
        self.clad_material = clad_material
        self.n_eff_guess = n_eff_guess
        self.n_wl_points = n_wl_points
        self.use_apodization = use_apodization
        self.center_mod_depth = center_mod_depth_nm * 1e-9

        # --- GEOMETRY CALCULATION ---
        self.lambda_B = 2 * self.n_eff_guess * self.pitch
        self.cavity_length = pitch / 2

        # 1. Grating Extent (Half length from x=0)
        # N periods + half the cavity
        self.x_grating_end = (self.n_periods_each_side * self.pitch) + (self.cavity_length / 2.0)

        # 2. Port Location
        # User defined: Grating Edge + N pitches
        self.dist_grating_to_port = n_periods_dist_to_port * self.pitch
        self.x_port = self.x_grating_end + self.dist_grating_to_port

        # 3. PML Boundary
        # User defined: Port + N wavelengths (using lambda_B as reference)
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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _setup_materials(self):
        """Create editable copies of materials and apply fit settings."""
        custom_sin = "SiN_custom"
        custom_sio2 = "SiO2_custom"

        # (Same material setup script as before)
        script = f'''
        if (haveresult("{custom_sin}", "material")) {{ deletematerial("{custom_sin}"); }}
        if (haveresult("{custom_sio2}", "material")) {{ deletematerial("{custom_sio2}"); }}

        m1 = copymaterial("{self.core_material}");
        setmaterial(m1, "name", "{custom_sin}");
        m2 = copymaterial("{self.clad_material}");
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

    # ------------------------------------------------------------------
    # Build simulation
    # ------------------------------------------------------------------
    def build(self):
        self._reset_layout()
        self._add_fdtd_region()
        self._add_x_aligned_mesh_override()
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

        fdtd.set("simulation time", 10e-12)
        fdtd.set("auto shutoff min", 1e-6)
        fdtd.set("mesh accuracy", 3)

    def _add_x_aligned_mesh_override(self, cells_per_half_period=5):
        """
        Mesh override covers the entire simulation x-span now to ensure
        consistency from port to port.
        """
        fdtd = self.fdtd
        half_pitch = 0.5 * self.pitch
        n_cells_half = max(1, int(cells_per_half_period))
        dx = half_pitch / float(n_cells_half)

        fdtd.addmesh()
        fdtd.set("name", "mesh_x_aligned")
        fdtd.set("x", 0.0)
        fdtd.set("x span", self.sim_x_span) # Cover full span
        fdtd.set("y", 0.0)
        fdtd.set("y span", self.y_span)
        fdtd.set("z", 0.0)
        fdtd.set("z span", self.z_span)

        fdtd.set("override x mesh", 1)
        fdtd.set("override y mesh", 0)
        fdtd.set("override z mesh", 0)
        fdtd.set("set maximum mesh step", 1)
        fdtd.set("dx", dx)

    def _add_bragg_core(self):
        fdtd = self.fdtd
        z_core_center = 0.0
        pitch = self.pitch
        half_pitch = pitch / 2.0

        # Helper to draw rectangles
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

        # ---------------------------------------------------------
        # Apodization Calc
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Build Geometry
        # ---------------------------------------------------------

        # Start X drawing position
        # We draw from Left Grating Edge moving Right.
        # The Infinite Waveguides are drawn separately.

        x_grating_start = -self.x_grating_end
        x = x_grating_start

        # A. LEFT INFINITE WAVEGUIDE
        # Extends from PML (with margin) to the start of the grating
        x_pml_left = -self.x_sim_boundary - 1e-6 # Extra 1um into PML
        add_core_segment(x_pml_left, x_grating_start, self.width_narrow, name_prefix="wg_left_inf")

        # B. Left Grating
        for d in range(n_total, 0, -1):
            w_n, w_w = W_narrow[d], W_wide[d]
            x1 = x; x2 = x1 + half_pitch
            add_core_segment(x1, x2, w_n, name_prefix=f"L_narrow_{d}")
            x = x2
            x1 = x; x2 = x1 + half_pitch
            add_core_segment(x1, x2, w_w, name_prefix=f"L_wide_{d}")
            x = x2

        # C. Cavity
        w_cavity = W_narrow[1]
        x1 = x; x2 = x1 + self.cavity_length
        add_core_segment(x1, x2, w_cavity, name_prefix="cavity")
        x = x2

        # D. Right Grating
        for d in range(1, n_total + 1):
            w_n, w_w = W_narrow[d], W_wide[d]
            x1 = x; x2 = x1 + half_pitch
            add_core_segment(x1, x2, w_n, name_prefix=f"R_narrow_{d}")
            x = x2
            x1 = x; x2 = x1 + half_pitch
            add_core_segment(x1, x2, w_w, name_prefix=f"R_wide_{d}")
            x = x2

        # E. RIGHT INFINITE WAVEGUIDE
        # Extends from current x (end of grating) to PML (with margin)
        x_pml_right = self.x_sim_boundary + 1e-6
        add_core_segment(x, x_pml_right, self.width_narrow, name_prefix="wg_right_inf")

    def _add_source_and_monitors(self):
        fdtd = self.fdtd

        # Use calculated port positions
        x_Port_1 = -self.x_port
        x_Port_2 = self.x_port

        z_center = 0.0
        y_center = 0.0
        monitor_ratio = 1.05

        # Port 1 (Source)
        fdtd.addport()
        fdtd.set("name", "Port_1")
        fdtd.set("injection axis", "x")
        fdtd.set("x", x_Port_1)
        fdtd.set("y", y_center)
        fdtd.set("y span", monitor_ratio*self.y_span)
        fdtd.set("z", z_center)
        fdtd.set("z span", monitor_ratio*self.z_span)
        fdtd.set("direction", "forward")
        fdtd.set("mode selection", "fundamental mode")

        # Port 2 (Monitor)
        fdtd.addport()
        fdtd.set("name", "Port_2")
        fdtd.set("injection axis", "x")
        fdtd.set("x", x_Port_2)
        fdtd.set("y", y_center)
        fdtd.set("y span", monitor_ratio*self.y_span)
        fdtd.set("z", z_center)
        fdtd.set("z span", monitor_ratio*self.z_span)
        fdtd.set("direction", "backward")
        fdtd.set("mode selection", "fundamental mode")

        fdtd.addmovie()
        fdtd.set("name", "movie_xy")
        fdtd.set("monitor type", "2D Z-normal")
        fdtd.set("x", 0)
        fdtd.set("x span", self.sim_x_span)
        fdtd.set("y", y_center)
        fdtd.set("y span", self.y_span)
        fdtd.set("z", 0.0)
        fdtd.set("lock aspect ratio", 1)
        fdtd.set("horizontal resolution", 400)

    def get_s_and_t_matrix(self):
        # Result extraction logic remains identical
        res1 = self.fdtd.getresult("FDTD::ports::Port_1", "expansion for port monitor")
        res2 = self.fdtd.getresult("FDTD::ports::Port_2", "expansion for port monitor")

        wl = np.squeeze(res1["lambda"])
        S11 = np.squeeze(res1["S"])
        S21 = np.squeeze(res2["S"])

        S11 = np.conj(S11)
        S21 = np.conj(S21)
        S12 = S21
        S22 = S11

        R_modal = np.abs(S11)**2
        T_modal = np.abs(S21)**2
        Loss_radiation = 1.0 - R_modal - T_modal

        S21_c = S21.astype(complex)
        T11 = np.zeros_like(S11, dtype=complex)
        T12 = np.zeros_like(S11, dtype=complex)
        T21 = np.zeros_like(S11, dtype=complex)
        T22 = np.zeros_like(S11, dtype=complex)

        # Left-to-Right T-matrix conversion
        # Allows chaining: T_total = T3 @ T2 @ T1
        mask = np.abs(S21_c) > 1e-15

        # Formula: T22 = 1 / S21
        T22[mask] = 1.0 / S21_c[mask]

        # Formula: T21 = -S11 / S21
        T21[mask] = -S11[mask] / S21_c[mask]

        # Formula: T12 = S22 / S21
        T12[mask] = S22[mask] / S21_c[mask]

        # Formula: T11 = S12 - (S11 * S22) / S21
        T11[mask] = S12[mask] - (S11[mask] * S22[mask]) / S21_c[mask]

        # Stack results
        T_matrix = np.stack([
            np.stack([T11, T12], axis=1),
            np.stack([T21, T22], axis=1)
        ], axis=1)

        return wl, R_modal, T_modal, Loss_radiation, T_matrix, S11, S21

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

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Saving Location
    base_save_dir = r"C:\Users\evyat\Lumerical\pi_shifts_FDTD_results\18_12"

    # Simulation Parameters
    lambda_res_est = 1.573e-6
    scan_width_nm = 40.0
    n_points = 1001
    w_wide = 900e-9
    core_h = 350e-9

    # ------------------------------------------------------------------
    # 2. SIMULATION SETUP
    # ------------------------------------------------------------------
    sim = PiShiftBraggFDTD(
        pitch=500e-9,
        n_periods_each_side=70,
        n_apod_periods_each_side=10,
        width_narrow=700e-9,
        width_wide=w_wide,
        core_height=core_h,
        substrate_thickness=4e-6,
        y_span=w_wide + 1.8 * lambda_res_est,
        z_span=core_h + 1.8 * lambda_res_est,
        n_periods_dist_to_port=20,   # Distance from grating edge to Port
        n_wls_dist_port_to_pml=2.0, # Distance from Port to PML (wavelengths)
        core_material="Si3N4 (Silicon Nitride) - Luke",
        clad_material="SiO2 (Glass) - Palik",
        n_eff_guess=1.55,
        coarse_width_nm=150,
        n_wl_points=n_points,
        use_apodization=True,
        center_mod_depth_nm=40.0
    )

    # ------------------------------------------------------------------
    # 3. FOLDER & NAMING LOGIC
    # ------------------------------------------------------------------
    layouts_dir = os.path.join(base_save_dir, "layouts")
    results_dir = os.path.join(base_save_dir, "results")
    os.makedirs(layouts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    N = sim.n_periods_each_side
    Napod = sim.n_apod_periods_each_side
    use_apod = bool(sim.use_apodization) and (Napod is not None) and (Napod > 0)
    tag = f"{N}_periods_{Napod}_apodizations" if use_apod else f"{N}_periods"

    layout_filename = f"layout_{tag}.fsp"
    results_filename = f"result_{tag}.mat"
    layout_path = os.path.join(layouts_dir, layout_filename)
    results_path = os.path.join(results_dir, results_filename)

    # ------------------------------------------------------------------
    # 4. RUN & SAVE
    # ------------------------------------------------------------------
    sim.build()
    sim.update_scan(center_lambda_m=lambda_res_est, width_nm=scan_width_nm, n_points=n_points)

    sim.fdtd.save(layout_path)
    print(f"Saved layout to: {layout_path}")

    start = time.perf_counter()
    sim.fdtd.run()
    end = time.perf_counter()
    print(f"Simulation time: {end - start:.3f} seconds")

    wl, R_modal, T_modal, Loss_radiation, T_matrix, S11, S21 = sim.get_s_and_t_matrix()
    wl_nm = wl * 1e9

    mat_data = {
        'wl_m': wl,
        'wl_nm': wl_nm,
        'T': T_modal,
        'R': R_modal,
        'loss': Loss_radiation,
        'T_matrix': T_matrix,
        'S11_complex': S11,
        'S21_complex': S21
    }

    sio.savemat(results_path, mat_data)

    plt.figure()
    plt.plot(wl_nm, T_modal, label="T (Modal)")
    plt.plot(wl_nm, R_modal, label="R (Modal)")
    plt.plot(wl_nm, Loss_radiation, label="Radiation Loss")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized power")
    plt.legend()
    plt.grid(True)
    plt.title(f"Scan: {tag}")
    plt.tight_layout()
    plt.show()

    sim.close()