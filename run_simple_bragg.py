"""
Simple (uniform) Bragg grating simulation — no pi-shift cavity, no apodization.

Uses SimpleBraggFDTD device class with shared config from simulation_config.py.

Usage:
    python run_simple_bragg.py
"""

import os
import time
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import analysis
import config
from post_processing import SParameters, export_interconnect
from simulation_config import SimulationConfig

# Import lumapi
try:
    import lumapi
except ImportError:
    spec = importlib.util.spec_from_file_location("lumapi", config.LUMAPI_PATH)
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)


# ═══════════════════════════════════════════════════════════════════════════════
# SimpleBraggFDTD Device Class
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleBraggFDTD:
    """Uniform Bragg grating (no cavity, no apodization)."""

    def __init__(self,
                 pitch=500e-9,
                 n_periods=40,
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
                 cells_per_half_period=5,
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
        self.cells_per_half_period = int(cells_per_half_period)

        # Derived geometry
        self.lambda_B = 2 * self.n_eff_guess * self.pitch
        self.L_grating = self.n_periods * self.pitch
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
        self._add_aligned_mesh_override(self.cells_per_half_period)
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
        fdtd.setdevice("GPU" if config.USE_GPU else "CPU")
        fdtd.set("background material", self.clad_material)
        fdtd.set("simulation time", 100e-12)
        fdtd.set("auto shutoff min", 1e-6)
        fdtd.set("mesh accuracy", 3)
        fdtd.set("dt stability factor", 0.5)

    def _add_aligned_mesh_override(self, cells_per_half_period=5):
        fdtd = self.fdtd
        half_pitch = 0.5 * self.pitch
        n_cells_half = max(1, int(cells_per_half_period))
        dx_grating = half_pitch / float(n_cells_half)
        dy_global = self.width_narrow / 13.0
        dz_global = self.core_height / 7.0

        max_device_width = max(self.width_port, self.width_wide, self.width_narrow)
        y_span_override = max_device_width * 1.2
        z_span_override = self.core_height

        fdtd.addmesh()
        fdtd.set("name", "mesh_grating")
        fdtd.set("x", 0)
        fdtd.set("x span", 2.0 * self.x_grating_end)
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

        x_start = -self.x_grating_end
        x = x_start

        # Left infinite waveguide
        add_core_segment(-self.x_sim_boundary - 1e-6, x_start, self.width_port, name_prefix="wg_left_inf")

        # Grating periods (no apodization, no cavity)
        for i in range(self.n_periods):
            add_core_segment(x, x + half_pitch, self.width_narrow, name_prefix=f"narrow_{i}")
            x += half_pitch
            add_core_segment(x, x + half_pitch, self.width_wide, name_prefix=f"wide_{i}")
            x += half_pitch

        # Right infinite waveguide
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
        res1 = self.fdtd.getresult("FDTD::ports::Port_1", "expansion for port monitor")
        res2 = self.fdtd.getresult("FDTD::ports::Port_2", "expansion for port monitor")
        wl = np.squeeze(res1["lambda"])
        S11_raw = np.squeeze(res1["S"])
        S21_raw = np.squeeze(res2["S"])

        neff1_data, neff2_data = None, None
        use_single_neff = False
        single_neff_val = None

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


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_simple_sim(cfg: SimulationConfig = None):
    """Run a simple (uniform) Bragg grating simulation."""
    if cfg is None:
        cfg = SimulationConfig()
        # Defaults for simple grating that differ from pi-shift
        cfg.spectral.center_wavelength_m = 1.560e-6
        cfg.spectral.scan_width_nm = 42.0
        cfg.farfield.enabled = False
        cfg.phase_correction.do_envelope_correction = False

    # Create device from shared config
    sim = SimpleBraggFDTD(**cfg.to_simple_device_kwargs())

    # Generate filename
    tag = f"SimpleBragg_{sim.n_periods}_periods"
    if sim.use_constant_materials:
        tag += "_CONST"

    layout_path = os.path.join(config.LAYOUTS_DIR, f"layout_{tag}.fsp")
    results_path = os.path.join(config.RESULTS_DIR, f"result_{tag}.mat")

    # Build and run
    sim.build()
    sim.update_scan(
        center_lambda_m=cfg.spectral.center_wavelength_m,
        width_nm=cfg.spectral.scan_width_nm,
        n_points=cfg.spectral.n_wl_points,
    )

    sim.fdtd.save(layout_path)
    print(f"Saved layout to: {layout_path}")

    start = time.perf_counter()
    sim.fdtd.run()
    print(f"Simulation time: {time.perf_counter() - start:.3f} seconds")

    # Process S-parameters
    wl, R, T, Loss, T_mat, S11, S21 = sim.get_s_and_t_matrix(
        neff_mat_file=config.NEFF_DATA_PATH,
        correct_length=cfg.phase_correction.do_length_correction,
        correct_envelope_and_t_phase=cfg.phase_correction.do_envelope_correction,
    )
    s_params = SParameters(wl=wl, R=R, T=T, Loss=Loss, T_mat=T_mat, S11=S11, S21=S21)

    # Save
    mat_data = {
        'wl_m': wl, 'wl_nm': wl * 1e9,
        'T': T, 'R': R, 'loss': Loss,
        'T_matrix': T_mat, 'S11_complex': S11, 'S21_complex': S21,
        'L_device': 2.0 * sim.x_grating_end,
    }
    sio.savemat(results_path, mat_data)
    print(f"Data saved to: {results_path}")

    # Export for INTERCONNECT (respects cfg flag)
    if cfg.run.export_interconnect:
        print("Exporting for INTERCONNECT...")
        export_interconnect(s_params, config.RESULTS_DIR, tag)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(wl * 1e9, T, label="T (Modal) = |S21|^2", color='blue')
    plt.plot(wl * 1e9, R, label="R (Modal) = |S11|^2", color='red')
    plt.title(f"Simple Bragg Grating Response\nPeriods: {sim.n_periods}")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized power")
    plt.legend()
    plt.grid(True)

    print("Displaying plots...")
    plt.show()

    sim.close()


if __name__ == "__main__":
    # Create config — override simple grating defaults here:
    cfg = SimulationConfig()
    cfg.spectral.center_wavelength_m = 1.560e-6
    cfg.spectral.scan_width_nm = 42.0
    cfg.mesh.span_multiplier = 1.8
    cfg.phase_correction.do_envelope_correction = False

    # Example overrides:
    # cfg.simple_grating.n_periods_total = 80
    # cfg.geometry.core_height_m = 400e-9

    run_simple_sim(cfg)
