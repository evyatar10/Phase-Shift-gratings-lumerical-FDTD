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
        self.n_straight_periods_each_side = 2
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

        fdtd.set("simulation time", 5e-12)
        fdtd.set("auto shutoff min", 1e-6)
        fdtd.set("mesh accuracy", 3)

    def _add_x_aligned_mesh_override(self, cells_per_half_period=5):
        """
        Create a mesh override that aligns the x mesh with the grating periods.

        Idea:
          * The grating boundaries in x occur every half_pitch = pitch / 2
          * We choose dx so that half_pitch = cells_per_half_period * dx
          * Then every interface at k * (half_pitch) falls exactly on a mesh line

        Parameters
        ----------
        cells_per_half_period : int
            Number of mesh cells per half period in x.
            For example:
              - 4  -> dx = half_pitch / 4   (coarser)
              - 8  -> dx = half_pitch / 8   (finer)
              - 12 -> dx ≈ 20 nm for pitch = 500 nm
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

        Strategy:
          * X,Z: keep automatic mesh (use global mesh accuracy)
          * Y: piecewise
                - coarser dy in the central waveguide region
                - finer dy only in thin strips around the sidewalls
        """

        fdtd = self.fdtd

        # 1. Effective wavelength and PPW constraints
        lambda_core = 2.0 * self.pitch

        ppw_y_center = float(getattr(self, "ppw_y_center", 8.0))
        ppw_y_edge   = float(getattr(self, "ppw_y_edge", 8.0))

        dy_ppw_center = lambda_core / ppw_y_center
        dy_ppw_edge   = lambda_core / ppw_y_edge

        # 2. Y direction: feature-based dy near sidewalls
        full_depth_edge = self.width_wide - self.width_narrow

        if self.use_apodization:
            full_depth_min = min(self.center_mod_depth, full_depth_edge)
        else:
            full_depth_min = full_depth_edge

        full_depth_min = max(full_depth_min, 10e-9)

        # Feature criterion; aim for about 2 cells across the smallest depth
        dy_feature = full_depth_min / 2.0
        dy_edge = min(dy_ppw_edge, dy_feature)

        # Central region dy
        dy_center = dy_ppw_center

        # Sidewall positions
        y_narrow_half = 0.5 * self.width_narrow
        y_wide_half   = 0.5 * self.width_wide

        margin_y = 2.0 * dy_edge
        y_edge_half_span = (y_wide_half - y_narrow_half) / 2.0 + margin_y

        y_edge_center_pos = 0.5 * (y_narrow_half + y_wide_half)
        y_edge_center_neg = -y_edge_center_pos

        y_center_span = 2.0 * y_narrow_half

        # Common X/Z region used for all overrides
        x_center = 0.0
        x_span   = self.device_length

        z_center = self.core_height / 2.0
        z_span   = 2.0 * self.core_height

        # 3. Central coarse Y mesh override
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

        # 4. Fine sidewall Y mesh overrides
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
        x_start = -self.device_length / 2.0
        x = x_start
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
        delta_edge = 0.5 * full_depth_edge
        delta_center = 0.5 * full_depth_center

        # Number of periods near the cavity that are apodized on each side
        n_apod = self.n_apod_periods_each_side
        n_total = self.n_periods_each_side

        # Left straight input
        x1 = x
        x2 = x1 + self.straight_length_each_side
        add_core_segment(x1, x2, self.width_narrow, name_prefix="wg_left")
        x = x2

        # Left grating periods; from left facet to cavity
        for i in range(self.n_periods_each_side):
            if self.use_apodization:
                if n_apod >= n_total:
                    # Old behavior; apodization over full grating
                    if n_total > 1:
                        j = n_total - 1 - i
                        frac = j / (n_total - 1)
                    else:
                        frac = 0.0
                    delta = delta_center + (delta_edge - delta_center) * frac
                else:
                    # New behavior; outer periods uniform; inner n_apod periods apodized toward the cavity
                    n_outer_uniform = n_total - n_apod
                    if i < n_outer_uniform:
                        delta = delta_edge
                    else:
                        if n_apod > 1:
                            j = i - n_outer_uniform   # 0 .. n_apod-1
                            frac = j / (n_apod - 1)
                        else:
                            frac = 0.0
                        # At boundary; delta = delta_edge; at cavity; delta = delta_center
                        delta = delta_edge + (delta_center - delta_edge) * frac

                width_wide_i = avg_width + delta
                width_narrow_i = avg_width - delta
            else:
                width_wide_i = self.width_wide
                width_narrow_i = self.width_narrow

            x1 = x
            x2 = x1 + half_pitch
            add_core_segment(x1, x2, width_wide_i)
            x = x2

            x1 = x
            x2 = x1 + half_pitch
            add_core_segment(x1, x2, width_narrow_i)
            x = x2

        # Cavity
        if self.use_apodization:
            width_cavity = avg_width - delta_center
        else:
            width_cavity = self.width_narrow

        x1 = x
        x2 = x1 + self.cavity_length
        add_core_segment(x1, x2, width_cavity, "cavity")
        x = x2

        # Right grating periods; from cavity to right facet
        for i in range(self.n_periods_each_side):
            if self.use_apodization:
                if n_apod >= n_total:
                    # Old behavior; apodization over full grating
                    if n_total > 1:
                        frac = i / (n_total - 1)
                    else:
                        frac = 0.0
                    delta = delta_center + (delta_edge - delta_center) * frac
                else:
                    # New behavior; inner n_apod periods near cavity apodized; outer periods uniform
                    if i < n_apod:
                        if n_apod > 1:
                            frac = i / (n_apod - 1)
                        else:
                            frac = 0.0
                        # At cavity; delta = delta_center; at boundary; delta = delta_edge
                        delta = delta_center + (delta_edge - delta_center) * frac
                    else:
                        delta = delta_edge

                width_wide_i = avg_width + delta
                width_narrow_i = avg_width - delta
            else:
                width_wide_i = self.width_wide
                width_narrow_i = self.width_narrow

            x1 = x
            x2 = x1 + half_pitch
            add_core_segment(x1, x2, width_wide_i)
            x = x2

            x1 = x
            x2 = x1 + half_pitch
            add_core_segment(x1, x2, width_narrow_i)
            x = x2

        # Right straight output
        x1 = x
        x2 = x1 + self.straight_length_each_side
        add_core_segment(x1, x2, self.width_narrow, "wg_right")
        x = x2

    def _add_source_and_monitors(self):
        fdtd = self.fdtd

        lam_min = self.lam_min
        lam_max = self.lam_max
        n_freq = self.n_wl_points

        x_device_start = -self.device_length / 2.0
        x_device_end   = +self.device_length / 2.0

        x_wg_left_start  = x_device_start
        x_wg_left_end    = x_wg_left_start + self.straight_length_each_side
        x_wg_right_end   = x_device_end
        x_wg_right_start = x_wg_right_end - self.straight_length_each_side

        x_source = 0.5 * (x_wg_left_start + x_wg_left_end)
        x_R_mon = x_wg_left_end - 0.25 * self.straight_length_each_side
        x_T_mon = x_wg_right_start + 0.25 * self.straight_length_each_side

        # If you want "full height" in z, center at 0 with span = z_span
        # Same idea in y if you want full width of the FDTD window
        z_center = 0.0
        y_center = 0.0

        # ------------------- Source -------------------
        fdtd.addmode()
        fdtd.set("name", "input_mode")
        fdtd.set("injection axis", "x")
        fdtd.set("direction", "Forward")
        fdtd.set("x", x_source)

        fdtd.set("y", y_center)
        fdtd.set("y span", self.y_span)          # "full" y span

        fdtd.set("z", z_center)
        fdtd.set("z span", self.z_span)          # "full" z span

        fdtd.set("wavelength start", lam_min)
        fdtd.set("wavelength stop", lam_max)
        fdtd.set("mode selection", "fundamental TE mode")

        # ------------------- Transmission monitor -------------------
        fdtd.adddftmonitor()
        fdtd.set("name", "T_monitor")
        fdtd.set("monitor type", "2D X-normal")
        fdtd.set("x", x_T_mon)

        fdtd.set("y", y_center)
        fdtd.set("y span", self.y_span)          # full y

        fdtd.set("z", z_center)
        fdtd.set("z span", self.z_span)          # full z

        fdtd.set("override global monitor settings", 1)
        fdtd.set("use source limits", 1)
        fdtd.set("frequency points", n_freq)

        # ------------------- Reflection monitor -------------------
        fdtd.adddftmonitor()
        fdtd.set("name", "R_monitor")
        fdtd.set("monitor type", "2D X-normal")
        fdtd.set("x", x_R_mon)

        fdtd.set("y", y_center)
        fdtd.set("y span", self.y_span)          # full y

        fdtd.set("z", z_center)
        fdtd.set("z span", self.z_span)          # full z

        fdtd.set("override global monitor settings", 1)
        fdtd.set("use source limits", 1)
        fdtd.set("frequency points", n_freq)

        # ------------------- Movie monitor -------------------
        fdtd.addmovie()
        fdtd.set("name", "movie_xy")
        fdtd.set("monitor type", "2D Z-normal")
        fdtd.set("x", 0)
        fdtd.set("x span", self.device_length + self.pitch)

        fdtd.set("y", y_center)
        fdtd.set("y span", self.y_span)          # full y

        # For Z-normal movie you usually want it at the core plane, not full z
        # so here I keep it at core height; you can change if you want:
        fdtd.set("z", self.core_height / 2.0)

        fdtd.set("lock aspect ratio", 1)
        fdtd.set("horizontal resolution", 400)

    # ------------------------------------------------------------------
    # Run and spectra
    # ------------------------------------------------------------------

    def get_spectra(self):
        T_res = self.fdtd.getresult("T_monitor", "T")
        wl = np.squeeze(T_res["lambda"])

        T_val = np.squeeze(np.real(self.fdtd.transmission("T_monitor")))
        T_plane = np.squeeze(np.real(self.fdtd.transmission("R_monitor")))
        R_val = 1.0 - T_plane

        T = np.clip(T_val, 0.0, None)
        R = np.clip(R_val, 0.0, None)
        loss = 1.0 - T - R
        #loss = np.clip(loss, 0.0, None)

        return wl, T, R, loss

    def update_scan(self, center_lambda_m, width_nm, n_points):
        self.n_wl_points = n_points
        half_w = 0.5 * width_nm * 1e-9
        self.lam_min = center_lambda_m - half_w
        self.lam_max = center_lambda_m + half_w

        fdtd = self.fdtd
        fdtd.switchtolayout()
        fdtd.setnamed("input_mode", "wavelength start", self.lam_min)
        fdtd.setnamed("input_mode", "wavelength stop", self.lam_max)
        fdtd.setnamed("T_monitor", "frequency points", self.n_wl_points)
        fdtd.setnamed("R_monitor", "frequency points", self.n_wl_points)

    def close(self):
        try:
            self.fdtd.close()
        except Exception:
            pass


if __name__ == "__main__":

    # Estimated resonance center and scan width around it
    lambda_res_est = 1.573e-6      # [m]
    scan_width_nm = 40.0           # full width in nm (±scan_width_nm/2)
    n_points = 1001                # number of wavelength points in the scan

    sim = PiShiftBraggFDTD(
        pitch=500e-9,
        n_periods_each_side=50,
        n_apod_periods_each_side=5,
        width_narrow=700e-9,
        width_wide=900e-9,
        core_height=350e-9,
        substrate_thickness=4e-6,
        y_span=4.5e-6,  #4e-6
        z_span=8e-6,  #8e-6
        buffer_x=4e-6,  #4e-6
        core_material="Si3N4 (Silicon Nitride) - Luke",
        clad_material="SiO2 (Glass) - Palik",
        n_eff_guess=1.55,
        coarse_width_nm=150,
        n_wl_points=n_points,
        use_apodization=True,
        center_mod_depth_nm=40.0
    )

    sim.build()
    sim.update_scan(center_lambda_m=lambda_res_est,
                    width_nm=scan_width_nm,
                    n_points=n_points)

    start = time.perf_counter()

    sim.fdtd.run()
    end = time.perf_counter()
    print(f"Simulation time: {end - start:.3f} seconds")

    # Build file name: "pi_shift_<N>periods[_apodization].fsp"
    #base_name = f"pi_shift_{sim.n_periods_each_side}periods"
    #if sim.use_apodization:
        #base_name += "_apodization"
    #sim.fdtd.save(base_name + ".fsp")

    wl, T, R, loss = sim.get_spectra()
    wl_nm = wl * 1e9

    # --- Build dynamic filename ---
    N = sim.n_periods_each_side
    Napod = sim.n_apod_periods_each_side

    if sim.use_apodization and Napod > 0:
        filename = f"{N}_periods_{Napod}_apodizations.npz"
    else:
        filename = f"{N}_periods.npz"

    # --- Save the spectra ---
    np.savez(filename,
             wl_m=wl,
             wl_nm=wl_nm,
             T=T,
             R=R,
             loss=loss)

    print(f"Saved spectrum to: {filename}")

    plt.figure()
    plt.plot(wl_nm, T, label="T")
    plt.plot(wl_nm, R, label="R")
    plt.plot(wl_nm, loss, label="loss")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized power")
    plt.legend()
    plt.grid(True)
    plt.title("Single scan around estimated resonance")
    plt.tight_layout()

    plt.show()

    sim.close()
