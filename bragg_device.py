import numpy as np
import os
import importlib.util
import math
import analysis

# Try to import lumapi normally
try:
    import lumapi
except ImportError:
    import config as _cfg
    spec = importlib.util.spec_from_file_location("lumapi", _cfg.LUMAPI_PATH)
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
                 apod_method='linear',
                 tanh_steepness=2.0,
                 use_cavity_mesh_override=False,
                 cells_per_half_period=5,
                 use_symmetry=True,
                 use_z_symmetry=True,
                 use_constant_materials=False,
                 n_core_const=1.977,
                 n_clad_const=1.44,
                 # --- NEW OPTIONAL 2D PARAMS ---
                 record_2d_fields_top_and_cross=False,
                 field_2d_x_span_m=None,  # If None, records full device. If set, crops X span for XY.
                 monitor_y_span_m=None,
                 monitor_z_span_m=None,
                 downsample_yz=1,       # Default 1 to preserve resolution at interfaces
                 # --- NEW OPTIONAL 3D PARAMS ---
                 record_3d_fields=False,
                 field_3d_span_m=None,
                 # --- NEW OPTIONAL FAR-FIELD MONITOR ---
                 record_farfield=False,
                 farfield_x_span_m=20e-6,
                 farfield_y_dist_m=None,
                 farfield_z_dist_m=None):

        self.pitch = pitch
        self.n_periods_each_side = n_periods_each_side

        if n_apod_periods_each_side is None:
            self.n_apod_periods_each_side = 0
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

        self.use_symmetry = use_symmetry
        self.use_z_symmetry = use_z_symmetry
        self.use_constant_materials = use_constant_materials
        self.n_core_const = n_core_const
        self.n_clad_const = n_clad_const

        self.n_eff_guess = n_eff_guess
        self.n_wl_points = n_wl_points
        self.use_apodization = use_apodization
        self.center_mod_depth = center_mod_depth_nm * 1e-9
        self.apod_method = apod_method
        self.tanh_steepness = tanh_steepness

        self.use_cavity_mesh_override = use_cavity_mesh_override
        self.cells_per_half_period = int(cells_per_half_period)

        # --- NEW STATE VARS ---
        self.record_2d_fields_top_and_cross = record_2d_fields_top_and_cross
        self.field_2d_x_span_m = field_2d_x_span_m
        
        self.record_3d_fields = record_3d_fields
        self.field_3d_span_m = field_3d_span_m
        
        self.monitor_y_span_m = monitor_y_span_m
        self.monitor_z_span_m = monitor_z_span_m
        self.downsample_yz = downsample_yz

        self.record_farfield = record_farfield
        self.farfield_x_span_m = farfield_x_span_m
        self.farfield_y_dist_m = farfield_y_dist_m
        self.farfield_z_dist_m = farfield_z_dist_m

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
        fdtd.setdevice("GPU")
        fdtd.set("background material", self.clad_material)
        fdtd.set("simulation time", 1000e-12)
        fdtd.set("auto shutoff min", 1e-6)
        fdtd.set("mesh accuracy", 3)
        fdtd.set("dt stability factor", 0.7)

    def _add_aligned_mesh_override(self, cells_per_half_period=5):
        fdtd = self.fdtd
        half_pitch = 0.5 * self.pitch
        n_cells_half = max(1, int(cells_per_half_period))
        dx_grating = half_pitch / float(n_cells_half)
        dy = self.width_narrow / 13.0
        dz = self.core_height / 7.0

        # Grating region: x_grating_start to x_grating_end (both tooth boundaries)
        x_grating_start = -self.x_grating_end
        x_grating_end = self.x_grating_end
        grating_span = x_grating_end - x_grating_start

        # Verify span / dx is integer (guaranteed when cavity_length = half_pitch)
        n_cells_total = round(grating_span / dx_grating)
        dx_actual = grating_span / float(n_cells_total)
        if abs(dx_actual - dx_grating) / dx_grating > 0.05:
            print(f"  WARNING: dx adjusted from {dx_grating*1e9:.1f}nm to "
                  f"{dx_actual*1e9:.1f}nm to fit grating span")
        dx_grating = dx_actual

        # Y/Z extent: waveguide + 1 evanescent decay length on each side
        _dn_sq = max(self.n_eff_guess**2 - self.n_clad_const**2, 0.01)
        _decay_len = self.lambda_B / (2.0 * math.pi * math.sqrt(_dn_sq))
        y_span_override = self.width_wide + 2.0 * _decay_len
        z_span_override = self.core_height + 2.0 * _decay_len

        # Single mesh override covering entire grating
        x_center = (x_grating_start + x_grating_end) / 2.0
        fdtd.addmesh()
        fdtd.set("name", "mesh_grating")
        fdtd.set("x", x_center)
        fdtd.set("x span", grating_span)
        fdtd.set("y", 0.0)
        fdtd.set("y span", y_span_override)
        fdtd.set("z", 0.0)
        fdtd.set("z span", z_span_override)
        fdtd.set("override x mesh", 1)
        fdtd.set("override y mesh", 1)
        fdtd.set("override z mesh", 1)
        fdtd.set("dx", dx_grating)
        fdtd.set("dy", dy)
        fdtd.set("dz", dz)

        print(f"Mesh: 1 override box, dx={dx_grating*1e9:.1f}nm, "
              f"dy={dy*1e9:.1f}nm, dz={dz*1e9:.1f}nm, "
              f"cells/half_period={n_cells_half}")
        print(f"  Grating: [{x_grating_start*1e6:.4f}, {x_grating_end*1e6:.4f}] um "
              f"({n_cells_total} cells)")
        print(f"  Y/Z override: {y_span_override*1e6:.2f} x {z_span_override*1e6:.2f} um")

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

        apod_method = self.apod_method
        tanh_steepness = self.tanh_steepness

        def get_mod_depth(d):
            if d <= n_apod and n_total > 1:
                denom = (n_apod - 1) if (n_apod > 1 and n_apod == n_total) else n_apod
                if denom == 0: return full_depth_center
                frac = (d - 1) / float(denom)
                if apod_method == 'tanh':
                    frac = np.tanh(tanh_steepness * 2.0 * frac) / np.tanh(2.0 * tanh_steepness)
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
        half_pitch = 0.5 * self.pitch
        n_cells_half = max(1, int(self.cells_per_half_period))
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

        # --- Monitor: Central Mode Tracking (1D X-axis) ---
        fdtd.addprofile()
        fdtd.set("name", "field_profile")
        fdtd.set("monitor type", "2D Z-normal")  # This is usually kept thin to trace peak field
        fdtd.set("x", 0)
        fdtd.set("x span", 2.0 * self.x_grating_end + 2.0e-6)
        fdtd.set("y", 0)
        fdtd.set("y span", 1.5 * self.width_wide)
        fdtd.set("z", 0)
        fdtd.set("override global monitor settings", 1)
        fdtd.set("use source limits", 1)
        fdtd.set("frequency points", 501)

        # --- Monitors: Time Domain (3 Points) ---
        def add_time_mon(name, x_pos):
            fdtd.addtime()
            fdtd.set("name", name)
            fdtd.set("monitor type", "Point")
            fdtd.set("x", x_pos)
            fdtd.set("y", 0)
            fdtd.set("z", 0)

        add_time_mon("time_input", -self.x_grating_end - 0.5e-6)
        add_time_mon("time_cavity", 0.0)
        add_time_mon("time_output", self.x_grating_end + 0.5e-6)

        y_span_val = self.monitor_y_span_m if self.monitor_y_span_m else 1.5 * self.width_wide
        z_span_val = self.monitor_z_span_m if self.monitor_z_span_m else 1.5 * self.core_height

        # --- OPTIONAL TOP & CROSS 2D MONITORS ---
        if self.record_2d_fields_top_and_cross:
            # 1. Top View (XY Plane)
            if self.field_2d_x_span_m:
                x_span_xy = self.field_2d_x_span_m
                print(f"2D XY Monitor: CROP Mode active. Span limited to {x_span_xy * 1e6:.2f} um")
            else:
                x_span_xy = 2.0 * self.x_grating_end + 1.0e-6

            fdtd.addprofile()
            fdtd.set("name", "field_profile_2D_XY")
            fdtd.set("monitor type", "2D Z-normal")
            fdtd.set("x", 0)
            fdtd.set("x span", x_span_xy)
            fdtd.set("y", 0)
            fdtd.set("y span", y_span_val)
            fdtd.set("z", 0)  # Locked to core center

            # Defaults (overridden by apply_monitor_overrides after build)
            fdtd.set("override global monitor settings", 1)
            fdtd.set("use source limits", 1)
            fdtd.set("frequency points", 5)
            # Downsampling
            fdtd.set("down sample x", 1)
            fdtd.set("down sample y", self.downsample_yz)

            # 2. Cross Section View (YZ Plane)
            fdtd.addprofile()
            fdtd.set("name", "field_profile_2D_YZ_cross")
            fdtd.set("monitor type", "2D X-normal")
            fdtd.set("x", self.cavity_length / 2.0)  # Locked to Phase Shift Defect Center
            fdtd.set("y", 0)
            fdtd.set("y span", y_span_val)
            fdtd.set("z", 0)
            fdtd.set("z span", z_span_val)

            # Safe settings to save space
            fdtd.set("override global monitor settings", 1)
            fdtd.set("use source limits", 1)
            fdtd.set("frequency points", 5)
            # Downsampling
            fdtd.set("down sample y", self.downsample_yz)
            fdtd.set("down sample z", self.downsample_yz)

            # 3. Side View (XZ Plane) — Y-normal at y=0
            fdtd.addprofile()
            fdtd.set("name", "field_profile_2D_XZ_side")
            fdtd.set("monitor type", "2D Y-normal")
            fdtd.set("x", 0)
            fdtd.set("x span", x_span_xy)
            fdtd.set("y", 0)
            fdtd.set("z", 0)
            fdtd.set("z span", z_span_val)

            fdtd.set("override global monitor settings", 1)
            fdtd.set("use source limits", 1)
            fdtd.set("frequency points", 5)
            # Downsampling
            fdtd.set("down sample x", 1)
            fdtd.set("down sample z", self.downsample_yz)

        # --- OPTIONAL FULL 3D MONITOR ---
        if self.record_3d_fields:
            if self.field_3d_span_m:
                x_span_3d = self.field_3d_span_m
                print(f"3D Monitor: CROP Mode active. Span limited to {x_span_3d * 1e6:.2f} um")
            else:
                x_span_3d = 2.0 * self.x_grating_end + 1.0e-6

            fdtd.addprofile()
            fdtd.set("name", "field_profile_3D")
            fdtd.set("monitor type", "3D")
            fdtd.set("x", 0)
            fdtd.set("x span", x_span_3d)
            fdtd.set("y", 0)
            fdtd.set("y span", y_span_val)
            fdtd.set("z", 0)
            fdtd.set("z span", z_span_val)
            
            # Safe settings to save space
            fdtd.set("override global monitor settings", 1)
            fdtd.set("use source limits", 1)
            fdtd.set("frequency points", 5)
            # Downsampling
            fdtd.set("down sample x", 1)
            fdtd.set("down sample y", self.downsample_yz)
            fdtd.set("down sample z", self.downsample_yz)

        # --- OPTIONAL FARFIELD MONITORS ---
        if self.record_farfield:
            # ---- Side monitor (2D Y-normal): captures radiation out the waveguide sides ----
            fdtd.addprofile()
            fdtd.set("name", "side_monitor")
            fdtd.set("monitor type", "2D Y-normal")

            y_pos = self.farfield_y_dist_m if self.farfield_y_dist_m is not None else 1.5 * self.width_wide
            fdtd.set("x", self.cavity_length / 2.0)  # centered on phase-shift defect
            fdtd.set("x span", self.farfield_x_span_m)
            fdtd.set("y", y_pos)

            z_span_ff = self.monitor_z_span_m if self.monitor_z_span_m else 1.5 * self.core_height
            fdtd.set("z", 0)
            fdtd.set("z span", z_span_ff)

            fdtd.set("override global monitor settings", 1)
            fdtd.set("use source limits", 1)
            fdtd.set("frequency points", 1)  # record at resonance wavelength only

            # ---- Top monitor (2D Z-normal): captures radiation out the top (vertical) ----
            fdtd.addprofile()
            fdtd.set("name", "top_monitor")
            fdtd.set("monitor type", "2D Z-normal")

            z_pos = self.farfield_z_dist_m if self.farfield_z_dist_m is not None else 1.5 * self.core_height
            fdtd.set("x", self.cavity_length / 2.0)  # centered on phase-shift defect
            fdtd.set("x span", self.farfield_x_span_m)
            fdtd.set("y", 0)
            y_span_ff = self.monitor_y_span_m if self.monitor_y_span_m else 1.5 * self.width_wide
            fdtd.set("y span", y_span_ff)
            fdtd.set("z", z_pos)

            fdtd.set("override global monitor settings", 1)
            fdtd.set("use source limits", 1)
            fdtd.set("frequency points", 1)  # record at resonance wavelength only

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
        # Note: frequency points for 2D/3D/far-field monitors are set by
        # apply_monitor_overrides() in sim_helpers.py after this call.

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