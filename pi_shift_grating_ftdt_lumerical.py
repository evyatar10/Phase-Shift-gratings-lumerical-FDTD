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
        fdtd.set("dimension", 2)
        fdtd.setdevice("GPU")
        fdtd.set("background material", self.clad_material)

        for bc in ["x min bc", "x max bc",
                   "y min bc", "y max bc",
                   "z min bc", "z max bc"]:
            fdtd.set(bc, "PML")

        fdtd.set("simulation time", 5e-12)
        fdtd.set("auto shutoff min", 1e-6)
        fdtd.set("mesh accuracy", 3)

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

        x1 = x
        x2 = x1 + self.straight_length_each_side
        add_core_segment(x1, x2, self.width_narrow, name_prefix="wg_left")
        x = x2

        for i in range(self.n_periods_each_side):
            if self.use_apodization:
                if self.n_periods_each_side > 1:
                    j = self.n_periods_each_side - 1 - i
                    frac = j / (self.n_periods_each_side - 1)
                else:
                    frac = 0.0
                delta = delta_center + (delta_edge - delta_center) * frac
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

        if self.use_apodization:
            width_cavity = avg_width - delta_center
        else:
            width_cavity = self.width_narrow

        x1 = x
        x2 = x1 + self.cavity_length
        add_core_segment(x1, x2, width_cavity, "cavity")
        x = x2

        for i in range(self.n_periods_each_side):
            if self.use_apodization:
                if self.n_periods_each_side > 1:
                    frac = i / (self.n_periods_each_side - 1)
                else:
                    frac = 0.0
                delta = delta_center + (delta_edge - delta_center) * frac
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
        z_core_center = self.core_height / 2.0

        fdtd.addmode()
        fdtd.set("name", "input_mode")
        fdtd.set("injection axis", "x")
        fdtd.set("direction", "Forward")
        fdtd.set("x", x_source)
        fdtd.set("y", 0)
        fdtd.set("y span", self.y_span * 0.8)
        fdtd.set("z", z_core_center)
        fdtd.set("z span", self.core_height * 4.0)
        fdtd.set("wavelength start", lam_min)
        fdtd.set("wavelength stop", lam_max)
        fdtd.set("mode selection", "fundamental TE mode")

        fdtd.adddftmonitor()
        fdtd.set("name", "T_monitor")
        fdtd.set("monitor type", "2D X-normal")
        fdtd.set("x", x_T_mon)
        fdtd.set("y", 0)
        fdtd.set("y span", self.y_span * 0.8)
        fdtd.set("z", self.core_height)
        fdtd.set("z span", self.core_height * 3.0)
        fdtd.set("override global monitor settings", 1)
        fdtd.set("use source limits", 1)
        fdtd.set("frequency points", n_freq)

        fdtd.adddftmonitor()
        fdtd.set("name", "R_monitor")
        fdtd.set("monitor type", "2D X-normal")
        fdtd.set("x", x_R_mon)
        fdtd.set("y", 0)
        fdtd.set("y span", self.y_span * 0.8)
        fdtd.set("z", self.core_height)
        fdtd.set("z span", self.core_height * 3.0)
        fdtd.set("override global monitor settings", 1)
        fdtd.set("use source limits", 1)
        fdtd.set("frequency points", n_freq)

        fdtd.addmovie()
        fdtd.set("name", "movie_xy")
        fdtd.set("monitor type", "2D Z-normal")
        fdtd.set("x", 0)
        fdtd.set("x span", self.device_length + self.pitch)
        fdtd.set("y", 0)
        fdtd.set("y span", self.y_span * 0.8)
        fdtd.set("z", z_core_center)
        fdtd.set("lock aspect ratio", 1)
        fdtd.set("horizontal resolution", 400)

    # ------------------------------------------------------------------
    # Run and spectra
    # ------------------------------------------------------------------
    def run(self):
        self.fdtd.run()

    def get_spectra(self):
        T_res = self.fdtd.getresult("T_monitor", "T")
        wl = np.squeeze(T_res["lambda"])

        T_val = np.squeeze(np.real(self.fdtd.transmission("T_monitor")))
        T_plane = np.squeeze(np.real(self.fdtd.transmission("R_monitor")))
        R_val = 1.0 - T_plane

        T = np.clip(T_val, 0.0, None)
        R = np.clip(R_val, 0.0, None)
        loss = 1.0 - T - R
        loss = np.clip(loss, 0.0, None)

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

    sim = PiShiftBraggFDTD(
        pitch=500e-9,
        n_periods_each_side=40,
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
        center_mod_depth_nm=40.0
    )

    sim.build()
    sim.run()
    wl_c, T_c, R_c, loss_c = sim.get_spectra()
    wl_c_nm = wl_c * 1e9

    lambda_res_est = 1.565e-6
    print(f"Estimated resonance from coarse scan: {lambda_res_est*1e9:.2f} nm")

    fine_width_nm = 20.0
    sim.update_scan(center_lambda_m=lambda_res_est,
                    width_nm=fine_width_nm,
                    n_points=801)

    sim.run()
    wl_f, T_f, R_f, loss_f = sim.get_spectra()
    wl_f_nm = wl_f * 1e9

    plt.figure()
    plt.plot(wl_c_nm, T_c, label="T coarse")
    plt.plot(wl_c_nm, R_c, label="R coarse")
    plt.plot(wl_c_nm, loss_c, label="loss coarse")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized power")
    plt.legend()
    plt.grid(True)
    plt.title("Coarse scan")
    plt.tight_layout()

    plt.figure()
    plt.plot(wl_f_nm, T_f, label="T fine")
    plt.plot(wl_f_nm, R_f, label="R fine")
    plt.plot(wl_f_nm, loss_f, label="loss fine")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized power")
    plt.legend()
    plt.grid(True)
    plt.title("Fine scan around resonance")
    plt.tight_layout()

    plt.show()

    sim.close()
