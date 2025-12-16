"""
Pi-shift Bragg grating in Lumerical FDTD driven from Python.

Geometry:
    - Propagation along x  (horizontal in XY view)
    - SiN core height:        350 nm  (along z)
    - SiO2 cladding:          everywhere (background material)
    - Width modulation along y:
          narrow  = 700 nm
          wide    = 900 nm
    - Grating pitch along x:  500 nm
    - N periods left plus one cavity period plus N periods right
    - Plus 2 straight periods (700 nm wide) at each end

Spectral setup:
    - Bragg wavelength guess: "lambda_B approx 2 n_eff Lambda"
    - Scan widths are defined explicitly in nanometers

Apodization option:
    - Keep average width at 800 nm
    - At edges: full modulation depth = 200 nm  (700 / 900)
    - At cavity: full modulation depth = center_mod_depth_nm
      for example 40 nm gives 780 / 820 around the cavity
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tkinter as tk
from tkinter import filedialog

# Try to import lumapi normally
try:
    import lumapi
except ImportError:
    # Fallback; adjust path if needed
    import importlib.util
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
                 buffer_x=4e-6,
                 core_material="Si3N4 (Silicon Nitride) - Luke",
                 clad_material="SiO2 (Glass) - Palik",
                 n_eff_guess=1.55,
                 coarse_width_nm=150,
                 n_wl_points=401,
                 use_apodization=False,
                 center_mod_depth_nm=40.0):

        self.pitch = pitch
        self.n_periods_each_side = n_periods_each_side

        # New parameter; how many periods near the cavity are apodized on each side
        # Default; all periods are apodized (old behavior)
        if n_apod_periods_each_side is None:
            self.n_apod_periods_each_side = n_periods_each_side
        else:
            # Clamp to valid range
            self.n_apod_periods_each_side = max(
                1,
                min(n_apod_periods_each_side, n_periods_each_side)
            )

        self.width_narrow = width_narrow
        self.width_wide = width_wide
        self.core_height = core_height
        self.substrate_thickness = substrate_thickness
        self.y_span = y_span
        self.z_span = z_span
        self.buffer_x = buffer_x
        self.core_material = core_material
        self.clad_material = clad_material

        self.n_eff_guess = n_eff_guess
        self.n_wl_points = n_wl_points

        self.use_apodization = use_apodization
        self.center_mod_depth = center_mod_depth_nm * 1e-9

        self.cavity_length = pitch / 2
        self.grating_length = 2 * n_periods_each_side * pitch
        self.n_straight_periods_each_side = 3
        self.straight_length_each_side = self.n_straight_periods_each_side * pitch

        self.device_length = (
            2 * self.straight_length_each_side
            + self.grating_length
            + self.cavity_length
        )
        self.sim_x_span = self.device_length + 2 * buffer_x

        self.lambda_B = 2 * self.n_eff_guess * self.pitch

        self.coarse_width_nm = coarse_width_nm
        half_w = 0.5 * self.coarse_width_nm * 1e-9
        self.lam_min = self.lambda_B - half_w
        self.lam_max = self.lambda_B + half_w

        self.fdtd = lumapi.FDTD()

        # Correct placement of material setup
        self._setup_materials()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _setup_materials(self):
        """Create editable copies of materials and apply fit settings."""

        custom_sin = "SiN_custom"
        custom_sio2 = "SiO2_custom"

        script = f'''
        if (haveresult("{custom_sin}", "material")) {{
            deletematerial("{custom_sin}");
        }}
        if (haveresult("{custom_sio2}", "material")) {{
            deletematerial("{custom_sio2}");
        }}

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
        #self._add_mesh_override_y()
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
        #fdtd.set("dimension", 2)
        fdtd.set("dimension", "3D")
        fdtd.setdevice("GPU")
        fdtd.set("background material", self.clad_material)

        for bc in ["x min bc", "x max bc",
                   "y min bc", "y max bc",
                   "z min bc", "z max bc"]:
            fdtd.set(bc, "PML")

        fdtd.set("simulation time", 5e-12) #5e-12
        fdtd.set("auto shutoff min", 1e-6)
        fdtd.set("mesh accuracy", 3)

    def _add_x_aligned_mesh_override(self, cells_per_half_period=5):
        """
        Create a mesh override that aligns the x mesh with the grating periods.
        """

        fdtd = self.fdtd

        # Half period in x
        half_pitch = 0.5 * self.pitch

        # Ensure at least one cell
        n_cells_half = max(1, int(cells_per_half_period))

        # Mesh step in x so that half_pitch is an integer multiple of dx
        dx = half_pitch / float(n_cells_half)

        # Override region; cover the whole device in x, full y and z spans
        fdtd.addmesh()
        fdtd.set("name", "mesh_x_aligned")

        fdtd.set("x", 0.0)
        fdtd.set("x span", self.device_length)   # from −L/2 to +L/2, same as grating

        fdtd.set("y", 0.0)
        fdtd.set("y span", self.y_span)

        fdtd.set("z", 0.0)
        fdtd.set("z span", self.z_span)

        # Only override x; leave y and z to the automatic mesh
        fdtd.set("override x mesh", 1)
        fdtd.set("override y mesh", 0)
        fdtd.set("override z mesh", 0)

        fdtd.set("set maximum mesh step", 1)
        fdtd.set("dx", dx)

    def _add_mesh_override_y(self):
        """Local mesh refinement over the grating region, Y only.
        """
        fdtd = self.fdtd
        # (Preserved method body)
        lambda_core = 2.0 * self.pitch
        ppw_y_center = float(getattr(self, "ppw_y_center", 8.0))
        ppw_y_edge   = float(getattr(self, "ppw_y_edge", 8.0))
        dy_ppw_center = lambda_core / ppw_y_center
        dy_ppw_edge   = lambda_core / ppw_y_edge
        full_depth_edge = self.width_wide - self.width_narrow
        if self.use_apodization:
            full_depth_min = min(self.center_mod_depth, full_depth_edge)
        else:
            full_depth_min = full_depth_edge
        full_depth_min = max(full_depth_min, 10e-9)
        dy_feature = full_depth_min / 2.0
        dy_edge = min(dy_ppw_edge, dy_feature)
        dy_center = dy_ppw_center
        y_narrow_half = 0.5 * self.width_narrow
        y_wide_half   = 0.5 * self.width_wide
        margin_y = 2.0 * dy_edge
        y_edge_half_span = (y_wide_half - y_narrow_half) / 2.0 + margin_y
        y_edge_center_pos = 0.5 * (y_narrow_half + y_wide_half)
        y_edge_center_neg = -y_edge_center_pos
        y_center_span = 2.0 * y_narrow_half
        x_center = 0.0
        x_span   = self.device_length
        z_center = self.core_height / 2.0
        z_span   = 2.0 * self.core_height
        fdtd.addmesh()
        fdtd.set("name", "mesh_grating_center")
        fdtd.set("x", x_center)
        fdtd.set("x span", x_span)
        fdtd.set("y", 0.0)
        fdtd.set("y span", y_center_span)
        fdtd.set("z", z_center)
        fdtd.set("z span", z_span)
        fdtd.set("override x mesh", 0)
        fdtd.set("override y mesh", 1)
        fdtd.set("override z mesh", 0)
        fdtd.set("set maximum mesh step", 1)
        fdtd.set("dy", dy_center)
        for name, y_c in [("mesh_grating_side_pos", y_edge_center_pos),
                          ("mesh_grating_side_neg", y_edge_center_neg)]:
            fdtd.addmesh()
            fdtd.set("name", name)
            fdtd.set("x", x_center)
            fdtd.set("x span", x_span)
            fdtd.set("y", y_c)
            fdtd.set("y span", 2.0 * y_edge_half_span)
            fdtd.set("z", z_center)
            fdtd.set("z span", z_span)
            fdtd.set("override x mesh", 0)
            fdtd.set("override y mesh", 1)
            fdtd.set("override z mesh", 0)
            fdtd.set("set maximum mesh step", 1)
            fdtd.set("dy", dy_edge)

    def _add_bragg_core(self):
        fdtd = self.fdtd
        z_core_center = self.core_height / 2.0
        pitch = self.pitch
        half_pitch = pitch / 2.0

        # Calculate starting position
        x_start = -self.device_length / 2.0
        x = x_start
        seg_id = 0

        # Helper to draw rectangles
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
        # 1. Precompute Widths for Apodization
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
            # d=1 is closest to cavity, d=N is closest to edge
            if d <= n_apod and n_total > 1:
                if n_apod == n_total:
                    denom = n_apod - 1 if n_apod > 1 else 1
                else:
                    denom = n_apod

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
        # 2. Build Geometry
        # ---------------------------------------------------------
        # A. Left Straight
        x1 = x
        x2 = x1 + self.straight_length_each_side
        add_core_segment(x1, x2, self.width_narrow, name_prefix="wg_left")
        x = x2
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
        # E. Right Straight
        x1 = x
        x2 = x1 + self.straight_length_each_side
        add_core_segment(x1, x2, self.width_narrow, name_prefix="wg_right")
        x = x2

    # ------------------------------------------------------------------
    # Monitors & Sources (UPDATED TO PORTS - FIXED)
    # ------------------------------------------------------------------
    def _add_source_and_monitors(self):
        fdtd = self.fdtd

        lam_min = self.lam_min
        lam_max = self.lam_max
        n_freq = self.n_wl_points

        x_device_start = -self.device_length / 2.0
        x_device_end = +self.device_length / 2.0
        x_wg_left_end = x_device_start + self.straight_length_each_side
        x_wg_right_start = x_device_end - self.straight_length_each_side

        x_Port_1 = x_wg_left_end - 0.4 * self.straight_length_each_side
        x_Port_2 = x_wg_right_start + 0.4 * self.straight_length_each_side
        z_center = 0.0
        y_center = 0.0

        # Port 1 (Source)
        fdtd.addport()
        fdtd.set("name", "Port_1")
        fdtd.set("injection axis", "x")
        fdtd.set("x", x_Port_1)
        fdtd.set("y", y_center)
        fdtd.set("y span", self.y_span)
        fdtd.set("z", z_center)
        fdtd.set("z span", self.z_span)
        fdtd.set("direction", "forward")
        fdtd.set("mode selection", "fundamental mode") #fundamental TE mode?


        # Port 2 (Monitor)
        fdtd.addport()
        fdtd.set("name", "Port_2")
        fdtd.set("injection axis", "x")
        fdtd.set("x", x_Port_2)
        fdtd.set("y", y_center)
        fdtd.set("y span", self.y_span)
        fdtd.set("z", z_center)
        fdtd.set("z span", self.z_span)
        fdtd.set("direction", "forward")
        fdtd.set("mode selection", "fundamental mode")

        # GLOBAL Source Settings
        #fdtd.setglobalsource("wavelength start", lam_min)
        #fdtd.setglobalsource("wavelength stop", lam_max)
        #fdtd.setglobalmonitor("frequency points", n_freq)



        fdtd.addmovie()
        fdtd.set("name", "movie_xy")
        fdtd.set("monitor type", "2D Z-normal")
        fdtd.set("x", 0)
        fdtd.set("x span", self.device_length + self.pitch)
        fdtd.set("y", y_center)
        fdtd.set("y span", self.y_span)
        fdtd.set("z", self.core_height / 2.0)
        fdtd.set("lock aspect ratio", 1)
        fdtd.set("horizontal resolution", 400)

    # ------------------------------------------------------------------
    # Run and spectra (FIXED: Get "expansion" result, not "S")
    # ------------------------------------------------------------------

    def get_s_and_t_matrix(self):
        #fdtd = self.fdtd

        # ------------------------------------------------------------------
        # FIX: The internal name is just "expansion"
        # ------------------------------------------------------------------
        # "expansion for port monitor" is the GUI display name.
        # "expansion" is the ID the code needs.
        #try:
        #    res1 = fdtd.getresult("Port_1", "expansion")
        #    res2 = fdtd.getresult("Port_2", "expansion")
        #except Exception as e:
            # Diagnostic: If this fails, this prints what names exist so you can see them
        #    print("\n--- DEBUG INFO ---")
        #    print(f"Could not find 'expansion'. Exact error: {e}")
         #   print("Available results in Port_1:")
         #   print(fdtd.eval("?getresult('Port_1');"))
         #   print("------------------\n")
         #   raise

        res1 = self.fdtd.getresult("FDTD::ports::Port_1", "expansion for port monitor")
        res2 = self.fdtd.getresult("FDTD::ports::Port_2", "expansion for port monitor")

        # Extract wavelength
        wl = np.squeeze(res1["lambda"])

        # Port 1 (left); direction = forward
        # a1: wave traveling +x (launched / forward)
        # b1: wave traveling -x (reflected / backward)
        a1 = np.squeeze(res1["a"])
        b1 = np.squeeze(res1["b"])

        # Port 2 (right); you set direction = forward
        # a2: wave traveling +x (transmitted forward out of the device)
        # b2: wave traveling -x (would correspond to a wave coming from the right into the device)
        a2 = np.squeeze(res2["a"])

        epsilon = 1e-20

        # S11: reflection at port 1
        S11 = b1 / (a1 + epsilon)

        #S11 = np.squeeze(res1["S"])
        #S21 = np.squeeze(res2["S"])

        # S21: transmission from port 1 to port 2
        S21 = a2 / (a1 + epsilon)

        # (Keep the rest of your T-Matrix code exactly the same as before...)
        S12 = S21
        S22 = np.zeros_like(S11)

        R_modal = np.abs(S11)**2
        T_modal = np.abs(S21)**2
        Loss_radiation = 1.0 - R_modal - T_modal

        # ... (Include the rest of your T-matrix logic here) ...

        # Copy-paste the T-matrix calculation block from your previous code:
        S21_c = S21.astype(complex)
        T11 = np.zeros_like(S11, dtype=complex)
        T12 = np.zeros_like(S11, dtype=complex)
        T21 = np.zeros_like(S11, dtype=complex)
        T22 = np.zeros_like(S11, dtype=complex)

        mask = np.abs(S21_c) > 1e-12
        T11[mask] = 1.0 / S21_c[mask]
        T12[mask] = -S22[mask] / S21_c[mask]
        T21[mask] = S11[mask] / S21_c[mask]
        T22[mask] = S12[mask] - (S11[mask] * S22[mask]) / S21_c[mask]

        T_matrix = np.stack([
            np.stack([T11, T12], axis=1),
            np.stack([T21, T22], axis=1)
        ], axis=1)

        return wl, R_modal, T_modal, Loss_radiation, T_matrix
    
    #erase this function
    def get_spectra(self):
        wl, R, T, loss, _ = self.get_s_and_t_matrix()
        return wl, T, R, loss

    def update_scan(self, center_lambda_m, width_nm, n_points):
        self.n_wl_points = n_points
        half_w = 0.5 * width_nm * 1e-9
        self.lam_min = center_lambda_m - half_w
        self.lam_max = center_lambda_m + half_w

        fdtd = self.fdtd
        fdtd.switchtolayout()

        fdtd.setglobalsource("wavelength start", self.lam_min)
        fdtd.setglobalsource("wavelength stop", self.lam_max)
        fdtd.setglobalmonitor("frequency points", self.n_wl_points)

        fdtd.setnamed("FDTD::ports", "monitor frequency points", self.n_wl_points)


    #erase this function
    def close(self):
        try:
            self.fdtd.close()
        except Exception:
            pass


if __name__ == "__main__":
    lambda_res_est = 1.573e-6
    scan_width_nm = 40.0
    n_points = 1001

    sim = PiShiftBraggFDTD(
        pitch=500e-9,
        n_periods_each_side=20,
        n_apod_periods_each_side=10,
        width_narrow=700e-9,
        width_wide=900e-9,
        core_height=350e-9,
        substrate_thickness=4e-6,
        y_span=4e-6,
        z_span=4e-6,
        buffer_x=5e-6,
        core_material="Si3N4 (Silicon Nitride) - Luke",
        clad_material="SiO2 (Glass) - Palik",
        n_eff_guess=1.55,
        coarse_width_nm=150,
        n_wl_points=n_points,
        use_apodization=True,
        center_mod_depth_nm=40.0
    )

    root = tk.Tk()
    root.withdraw()
    save_dir = filedialog.askdirectory(title="Select folder to save NPZ results")
    if not save_dir:
        save_dir = os.getcwd()

    N = sim.n_periods_each_side
    Napod = sim.n_apod_periods_each_side

    if sim.use_apodization and Napod > 0:
        filename = f"{N}_periods_{Napod}_apodizations_SMatrix.npz"
    else:
        filename = f"{N}_periods_SMatrix.npz"

    save_path = os.path.join(save_dir, filename)

    sim.build()
    sim.update_scan(center_lambda_m=lambda_res_est,
                    width_nm=scan_width_nm,
                    n_points=n_points)

    start = time.perf_counter()
    sim.fdtd.run()
    end = time.perf_counter()
    print(f"Simulation time: {end - start:.3f} seconds")

    wl, R_modal, T_modal, Loss_radiation, T_matrix = sim.get_s_and_t_matrix()
    wl_nm = wl * 1e9

    np.savez(save_path,
             wl_m=wl,
             wl_nm=wl_nm,
             T=T_modal,
             R=R_modal,
             loss=Loss_radiation,
             T_matrix=T_matrix)

    print(f"Saved spectrum and T-matrix to: {save_path}")

    plt.figure()
    plt.plot(wl_nm, T_modal, label="T (Modal)")
    plt.plot(wl_nm, R_modal, label="R (Modal)")
    plt.plot(wl_nm, Loss_radiation, label="Radiation Loss")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized power")
    plt.legend()
    plt.grid(True)
    plt.title("Single scan (Modal Orthogonality)")
    plt.tight_layout()
    plt.show()

    sim.close()