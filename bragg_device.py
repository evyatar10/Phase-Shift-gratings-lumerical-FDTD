import numpy as np
import os
import importlib.util
import math
import analysis
import config as _cfg

# Try to import lumapi normally
try:
    import lumapi
except ImportError:
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
                 cells_per_half_period=5,
                 simulation_mode="accurate",
                 use_symmetry=True,
                 use_z_symmetry=True,
                 polarization="TE",
                 use_constant_materials=False,
                 n_core_const=1.97,
                 n_clad_const=1.444,
                 const_material_mode="sampled",
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
                 farfield_z_dist_m=None,
                 cavity_width_option="narrow",
                 cavity_width_m=None,
                 innermost_tooth_shift_m=0.0,
                 lengthen_cavity=True,
                 n_free_inner_teeth=1,
                 inner_dw_nm=None,
                 inner_shift_nm=None,
                 width_narrow_per_tooth_m=None,
                 width_wide_per_tooth_m=None,
                 # --- NEW: two side-by-side coupled devices ---
                 n_devices=1,
                 device_gap_m=1.0e-6,
                 device_stagger_m=0.0,
                 width_narrow_2=None,
                 width_wide_2=None,
                 device2_closed=False,
                 # --- NEW: small dielectric scatterer(s) (radiation-recycling study) ---
                 scatterer_enabled=False,
                 scatterer_shape="cylinder",
                 scatterer_radius_m=150e-9,
                 scatterer_x_m=0.0,
                 scatterer_y_m=1.0e-6,
                 scatterer_index=None,
                 scatterer_mirrored_y=True,
                 scatterer_height_m=None):

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
        if polarization not in ("TE", "TM"):
            raise ValueError(f"polarization must be 'TE' or 'TM', got {polarization!r}")
        self.polarization = polarization
        self.use_constant_materials = use_constant_materials
        self.n_core_const = n_core_const
        self.n_clad_const = n_clad_const
        self.const_material_mode = const_material_mode
        # True only for the "object" constant-index backend; gates the per-object
        # set("index", n) and set("background index", n) calls below.
        self._object_index_mode = False

        self.n_eff_guess = n_eff_guess
        self.n_wl_points = n_wl_points
        self.use_apodization = use_apodization
        self.center_mod_depth = center_mod_depth_nm * 1e-9
        self.apod_method = apod_method
        self.tanh_steepness = tanh_steepness

        self.cells_per_half_period = max(1, int(cells_per_half_period))
        self.dx_override = (pitch / 2.0) / float(self.cells_per_half_period)
        self.simulation_mode = simulation_mode

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

        self.cavity_width_option = cavity_width_option
        self.cavity_width_m = cavity_width_m

        if override_cavity_length_nm:
            self.cavity_length = override_cavity_length_nm * 1e-9
        else:
            self.cavity_length = pitch / 2.0

        self.innermost_tooth_shift_m = float(innermost_tooth_shift_m)
        self.lengthen_cavity = bool(lengthen_cavity)
        half_pitch_val = pitch / 2.0

        # Generalized per-tooth DW and shift for the inverse-design path.
        # If `inner_shift_nm` is supplied it takes precedence over the legacy
        # scalar `innermost_tooth_shift_m`. With n_free_inner_teeth=1 and
        # inner_shift_nm=[X], the geometry is bit-identical to the legacy path
        # with innermost_tooth_shift_m=X*1e-9.
        self.n_free_inner_teeth = int(n_free_inner_teeth)
        if self.n_free_inner_teeth < 1:
            raise ValueError(f"n_free_inner_teeth must be >= 1, got {self.n_free_inner_teeth}.")
        if self.n_free_inner_teeth > n_periods_each_side - 1:
            raise ValueError(
                f"n_free_inner_teeth ({self.n_free_inner_teeth}) must be "
                f"<= n_periods_each_side - 1 ({n_periods_each_side - 1}); "
                f"a tooth at d+1 on the right is required to absorb the shift."
            )

        self.inner_dw_nm = list(inner_dw_nm) if inner_dw_nm is not None else None
        if self.inner_dw_nm is not None and len(self.inner_dw_nm) != self.n_free_inner_teeth:
            raise ValueError(
                f"inner_dw_nm length ({len(self.inner_dw_nm)}) "
                f"must equal n_free_inner_teeth ({self.n_free_inner_teeth})."
            )

        if inner_shift_nm is not None:
            self.inner_shift_nm = [float(s) for s in inner_shift_nm]
        else:
            # Legacy scalar path → length-1 list
            self.inner_shift_nm = [self.innermost_tooth_shift_m * 1e9]
            if self.n_free_inner_teeth > 1:
                # Pad with zeros so subsequent freed teeth have no shift unless explicitly provided
                self.inner_shift_nm += [0.0] * (self.n_free_inner_teeth - 1)
        if len(self.inner_shift_nm) != self.n_free_inner_teeth:
            raise ValueError(
                f"inner_shift_nm length ({len(self.inner_shift_nm)}) "
                f"must equal n_free_inner_teeth ({self.n_free_inner_teeth})."
            )
        for d, s_nm in enumerate(self.inner_shift_nm, start=1):
            s_m = s_nm * 1e-9
            if not (0.0 <= s_m < half_pitch_val):
                raise ValueError(
                    f"inner_shift_nm[{d-1}] must be in [0, half_pitch) nm. "
                    f"Got {s_nm:.1f} nm, half_pitch={half_pitch_val * 1e9:.1f} nm."
                )

        total_shift_m = sum(s * 1e-9 for s in self.inner_shift_nm)
        cavity_extra = 2.0 * total_shift_m if self.lengthen_cavity else 0.0
        self.cavity_length_effective = self.cavity_length + cavity_extra

        # Explicit per-tooth width arrays (m), indexed d=1..len = innermost → outermost.
        # When supplied, W_narrow[d]/W_wide[d] are taken verbatim for d <= len; teeth
        # beyond the array length fall back to the apodization-envelope / uniform path.
        # Unlike inner_dw_nm (per-tooth depth around a fixed avg_width), these allow the
        # average width to vary per tooth (needed for designs whose cavity-side average
        # tapers toward a narrower cavity).
        self.width_narrow_per_tooth_m = (
            list(width_narrow_per_tooth_m) if width_narrow_per_tooth_m is not None else None
        )
        self.width_wide_per_tooth_m = (
            list(width_wide_per_tooth_m) if width_wide_per_tooth_m is not None else None
        )
        if (self.width_narrow_per_tooth_m is None) != (self.width_wide_per_tooth_m is None):
            raise ValueError(
                "width_narrow_per_tooth_m and width_wide_per_tooth_m must both be set or "
                "both be None."
            )
        if self.width_narrow_per_tooth_m is not None:
            if len(self.width_narrow_per_tooth_m) != len(self.width_wide_per_tooth_m):
                raise ValueError(
                    f"width_narrow_per_tooth_m ({len(self.width_narrow_per_tooth_m)}) and "
                    f"width_wide_per_tooth_m ({len(self.width_wide_per_tooth_m)}) must have "
                    f"equal length."
                )
            if len(self.width_narrow_per_tooth_m) > n_periods_each_side:
                raise ValueError(
                    f"per-tooth width arrays length ({len(self.width_narrow_per_tooth_m)}) "
                    f"must be <= n_periods_each_side ({n_periods_each_side})."
                )

        self.x_grating_end = (self.n_periods_each_side * self.pitch) + (self.cavity_length / 2.0)
        self.dist_grating_to_port = n_periods_dist_to_port * self.pitch
        self.x_port = self.x_grating_end + self.dist_grating_to_port
        self.dist_port_to_pml = n_wls_dist_port_to_pml * self.lambda_B
        self.x_sim_boundary = self.x_port + self.dist_port_to_pml
        self.sim_x_span = 2.0 * self.x_sim_boundary

        # ── Two side-by-side coupled devices ──────────────────────────────────
        # Device 1 (driven) is centered at (x=0, y=+s/2); device 2 (passive) at
        # (x=Δx, y=-s/2). The y-symmetry plane is broken (single-guide drive), so
        # it is forced OFF here regardless of the incoming flag; z-symmetry stays.
        self.n_devices = int(n_devices)
        if self.n_devices not in (1, 2):
            raise ValueError(f"n_devices must be 1 or 2, got {self.n_devices}.")
        self.device_gap_m = float(device_gap_m)
        self.device2_closed = bool(device2_closed)  # closed recycler (no feed waveguides / drain ports)
        self.width_narrow_2 = float(width_narrow_2) if width_narrow_2 is not None else self.width_narrow
        self.width_wide_2 = float(width_wide_2) if width_wide_2 is not None else self.width_wide
        # Snap the longitudinal stagger to the x-mesh so device 2's teeth stay
        # aligned to the same cell grid as device 1 (preserves mesh accuracy).
        self.device_stagger_m = round(float(device_stagger_m) / self.dx_override) * self.dx_override

        if self.n_devices == 2:
            self.use_symmetry = False  # broken by single-guide excitation
            # center-to-center lateral separation (edge-to-edge gap on the wide teeth = device_gap_m)
            s = self.device_gap_m + 0.5 * self.width_wide + 0.5 * self.width_wide_2
            self.y_dev1 = +0.5 * s
            self.y_dev2 = -0.5 * s
            self.x_dev1 = 0.0
            self.x_dev2 = self.device_stagger_m
            # Guard: the port feed lines (width_port) and wide teeth must not overlap.
            half1 = 0.5 * max(self.width_wide, self.width_port)
            half2 = 0.5 * max(self.width_wide_2, self.width_port)
            if (s - half1 - half2) <= 0.0:
                raise ValueError(
                    f"device_gap_m={self.device_gap_m*1e9:.0f} nm too small: the two guides' "
                    f"widest features (incl. {self.width_port*1e9:.0f} nm port feeds) overlap. "
                    f"Increase device_gap_m."
                )
        else:
            self.y_dev1 = 0.0
            self.y_dev2 = 0.0
            self.x_dev1 = 0.0
            self.x_dev2 = 0.0

        # Unified x-domain extent: symmetric about x=0 (keeps the existing mesh-edge
        # parity logic keyed to x=0), but widened to enclose the staggered device 2.
        self.fdtd_x_center = 0.0
        self.fdtd_x_span = 2.0 * (self.x_sim_boundary + abs(self.x_dev2))

        # Coupling outputs (populated by get_s_and_t_matrix when n_devices==2)
        self.coupling_left = None
        self.coupling_right = None
        self.coupling_total = None
        self.loss_4port = None
        self.S31_complex = None
        self.S41_complex = None

        # ── Small dielectric scatterer(s) — radiation-recycling study ──────────
        # Active only when enabled AND radius > 0 (radius 0 = clean in-study
        # control with identical numerics/domain). Drawn by _add_scatterers().
        self.scatterer_shape = str(scatterer_shape or "cylinder").lower()
        if self.scatterer_shape != "cylinder":
            raise ValueError(f"scatterer_shape must be 'cylinder', got {scatterer_shape!r}")
        self.scatterer_radius_m = float(scatterer_radius_m)
        self.scatterer_x_m = float(scatterer_x_m)
        self.scatterer_y_m = float(scatterer_y_m)
        self.scatterer_index = scatterer_index  # None → n_core_const at placement
        self.scatterer_mirrored_y = bool(scatterer_mirrored_y)
        self.scatterer_height_m = (float(scatterer_height_m) if scatterer_height_m
                                   else self.core_height)
        self._has_scatterer = bool(scatterer_enabled) and self.scatterer_radius_m > 0.0
        if (self._has_scatterer and not self.scatterer_mirrored_y
                and abs(self.scatterer_y_m) > 0.0 and self.use_symmetry):
            # A single off-axis scatterer breaks the y=0 mirror plane; with the TM
            # 'Symmetric' BC left on, Lumerical would silently simulate the
            # MIRRORED PAIR. Refuse rather than mislead.
            raise ValueError(
                "Single (non-mirrored) off-axis scatterer with the y-symmetry plane ON "
                "would silently simulate a mirrored pair. Set symmetry.use_y_symmetry="
                "False, or use scatterer_mirrored_y=True."
            )
        if self._has_scatterer:
            y_edge = abs(self.scatterer_y_m) + self.scatterer_radius_m
            if y_edge > 0.5 * self.y_span:
                raise ValueError(
                    f"Scatterer extends to |y|={y_edge*1e6:.2f} µm, outside the domain "
                    f"(half y-span {0.5*self.y_span*1e6:.2f} µm). Widen the domain "
                    f"(SimulationConfig.y_span_override_m)."
                )
            clearance = 0.5 * self.y_span - y_edge
            lam_clad = self.lambda_B / self.n_clad_const
            if clearance < lam_clad:
                print(f"  WARNING: scatterer-to-PML clearance {clearance*1e6:.2f} µm < "
                      f"λ/n_clad ({lam_clad*1e6:.2f} µm) — boundary artifacts possible.")
            if abs(self.scatterer_y_m) - self.scatterer_radius_m < 0.5 * self.width_wide:
                print(f"  WARNING: scatterer inner edge |y|={ (abs(self.scatterer_y_m)-self.scatterer_radius_m)*1e9:.0f} nm "
                      f"overlaps/touches the wide teeth (half-width {0.5*self.width_wide*1e9:.0f} nm).")

        self.coarse_width_nm = coarse_width_nm
        half_w = 0.5 * self.coarse_width_nm * 1e-9
        self.lam_min = self.lambda_B - half_w
        self.lam_max = self.lambda_B + half_w

        self.fdtd = lumapi.FDTD()
        self._setup_materials()

    def _setup_materials(self):
        if self.use_constant_materials and self.const_material_mode == "object":
            # Direct index: assign "<Object defined dielectric>" to each object and
            # set its index explicitly (done in _add_bragg_core / _add_fdtd_region).
            # No DB material is created — this bypasses Lumerical's material fitting.
            print(f"Using CONSTANT Materials (object-defined dielectric): "
                  f"SiN={self.n_core_const}, SiO2={self.n_clad_const}")
            self._object_index_mode = True
            self.core_material = "<Object defined dielectric>"
            self.clad_material = "<Object defined dielectric>"
            return

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
            set("Refractive Index", 1.444);
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
        self._add_scatterers()
        self._add_source_and_monitors()

    def _add_fdtd_region(self):
        fdtd = self.fdtd
        fdtd.addfdtd()
        fdtd.set("x", self.fdtd_x_center)
        fdtd.set("x span", self.fdtd_x_span)
        fdtd.set("y", 0)
        fdtd.set("y span", self.y_span)
        fdtd.set("z", 0)
        fdtd.set("z span", self.z_span)

        for bc in ["x min bc", "x max bc", "y min bc", "y max bc", "z min bc", "z max bc"]:
            fdtd.set(bc, "PML")

        # Symmetry-plane parity follows polarization: the BC must match the
        # parity of the injected mode's E field or the mode is filtered out.
        #   TE (E along y): y=0 Anti-Symmetric, z=0 Symmetric
        #   TM (E along z): y=0 Symmetric,      z=0 Anti-Symmetric
        if self.use_symmetry:
            fdtd.set("y min bc", "Anti-Symmetric" if self.polarization == "TE" else "Symmetric")
            fdtd.set("force symmetric y mesh", 1)

        if self.use_z_symmetry:
            fdtd.set("z min bc", "Symmetric" if self.polarization == "TE" else "Anti-Symmetric")
            fdtd.set("force symmetric z mesh", 1)

        if self.n_devices == 2:
            # The side-by-side pair has no y-symmetry plane (only device 1 is driven);
            # y-min must stay PML or device 2 becomes a phantom mirror image.
            assert fdtd.get("y min bc") == "PML", (
                "Two-device run requires y-symmetry OFF (y min bc must be PML)."
            )
            print(f"  [2-device] y-symmetry OFF; guides at y=±{self.y_dev1*1e6:.3f} µm, "
                  f"device-2 stagger Δx={self.x_dev2*1e6:.3f} µm")

        fdtd.set("dimension", "3D")
        if _cfg.USE_GPU:
            fdtd.setdevice("GPU")
        fdtd.set("background material", self.clad_material)
        if self._object_index_mode:
            fdtd.set("background index", self.n_clad_const)
        # Sim time stays 2000 ps project-wide; OPT-IN override via TM_SIM_TIME_PS
        # for one-off convergence checks only (empty/unset -> unchanged default).
        fdtd.set("simulation time", (float(os.environ.get("TM_SIM_TIME_PS") or "2000")) * 1e-12)
        fdtd.set("auto shutoff min", 1e-7)

        fdtd.set("mesh type", "custom non-uniform")
        fdtd.set("define x mesh by", "maximum mesh step")
        fdtd.set("dx", self.dx_override)
        # Pin dy/dz absolutely so transverse sampling is constant across runs
        # at different λ (we scan 1300–1700 nm). 50 nm matches the 14-cells-per-λ
        # rule at ~1384 nm with n_SiN=1.977 — slightly conservative below that,
        # mildly relaxed above. Was previously "mesh cells per wavelength"=14.
        fdtd.set("define y mesh by", "maximum mesh step")
        fdtd.set("dy", 50e-9)
        fdtd.set("define z mesh by", "maximum mesh step")
        fdtd.set("dz", 50e-9)
        fdtd.set("allow grading in x", 0)
        fdtd.set("allow grading in y", 1)
        fdtd.set("allow grading in z", 1)
        fdtd.set("grading factor", 1.41421)
        fdtd.set("mesh refinement", "conformal variant 0")

        fdtd.set("dt stability factor", 0.7)

    def _add_aligned_mesh_override(self):
        # Single override box centered at x=0 with dx = pitch/(2*N) — period-based,
        # independent of cavity_length_effective. Span = M*dx with M parity-matched
        # to N so a cell-edge sits at x=0 for even N (cell-center for odd N), which
        # places mesh edges exactly on the wide/narrow transitions at ±(2k+1)·pitch/4
        # in the nominal (no shift, no detuning) case.
        fdtd = self.fdtd
        N = self.cells_per_half_period
        dx = self.dx_override

        if dx > 0.25 * self.pitch:
            print(
                f"  WARNING: dx={dx*1e9:.2f} nm > pitch/4 ({self.pitch*0.25e9:.2f} nm). "
                f"Consider bumping cells_per_half_period (e.g. simulation_mode='accurate')."
            )

        min_span = self.fdtd_x_span + 2e-6
        M = math.ceil(min_span / dx)
        if (M % 2) != (N % 2):
            M += 1
        box_span = M * dx

        if self.n_devices == 2:
            # One union box covering BOTH guides and the gap between them (where the
            # coupling lives). Centered at y=0 (the pair is symmetric about y=0).
            s = self.y_dev1 - self.y_dev2  # center-to-center separation
            margin = 0.5 * max(self.width_wide, self.width_wide_2)
            y_span_override = s + 0.5 * self.width_wide + 0.5 * self.width_wide_2 + 2.0 * margin
            dy_global = min(self.width_narrow, self.width_narrow_2) / 13.0
        else:
            max_device_width = max(self.width_port, self.width_wide, self.width_narrow)
            y_span_override = max_device_width * 1.2
            dy_global = self.width_narrow / 13.0
        z_span_override = self.core_height
        dz_global = self.core_height / 7.0

        fdtd.addmesh()
        fdtd.set("name", "mesh_override")
        fdtd.set("x", 0.0)
        fdtd.set("x span", box_span)
        fdtd.set("y", 0.0)
        fdtd.set("y span", y_span_override)
        fdtd.set("z", 0.0)
        fdtd.set("z span", z_span_override)
        fdtd.set("override x mesh", 1)
        fdtd.set("override y mesh", 1)
        fdtd.set("override z mesh", 1)
        fdtd.set("dx", dx)
        fdtd.set("dy", dy_global)
        fdtd.set("dz", dz_global)

    def _add_bragg_core(self):
        fdtd = self.fdtd
        z_core_center = 0.0
        pitch = self.pitch
        half_pitch = pitch / 2.0
        seg_id = 0
        x_domain_half = 0.5 * self.fdtd_x_span

        def add_core_segment(x1, x2, width, name_prefix="core_seg", y_center=0.0):
            nonlocal seg_id
            seg_id += 1
            fdtd.addrect()
            fdtd.set("name", f"{name_prefix}_{seg_id:d}")
            fdtd.set("material", self.core_material)
            if self._object_index_mode:
                fdtd.set("index", self.n_core_const)
            fdtd.set("y", y_center)
            fdtd.set("y span", width)
            fdtd.set("z", z_core_center)
            fdtd.set("z span", self.core_height)
            fdtd.set("x min", x1)
            fdtd.set("x max", x2)

        def build_grating_arms(y_center, x_offset, dev_width_narrow, dev_width_wide, featured, draw_feed=True):
            """Build one pi-shift grating (two arms + central cavity) at the given
            lateral (y_center) and longitudinal (x_offset) offset.

            featured=True  → device 1: honours apodization, inner-tooth shifts/DW,
                             explicit per-tooth width arrays, and cavity_width_m.
            featured=False → device 2: a plain uniform pi-shift grating with its own
                             corrugation depth (no apodization / shifts / per-tooth)."""
            avg_width = 0.5 * (dev_width_narrow + dev_width_wide)
            full_depth_edge = dev_width_wide - dev_width_narrow
            full_depth_center = self.center_mod_depth if (featured and self.use_apodization) else full_depth_edge
            n_total = self.n_periods_each_side
            n_apod = self.n_apod_periods_each_side if featured else 0
            n_free = self.n_free_inner_teeth
            apod_method = self.apod_method
            tanh_steepness = self.tanh_steepness

            def get_mod_depth(d):
                # Per-tooth DW override for the freed inner teeth (inverse-design path).
                if featured and d <= n_free and self.inner_dw_nm is not None:
                    return float(self.inner_dw_nm[d - 1]) * 1e-9
                if featured and d <= n_apod and n_total > 1:
                    denom = (n_apod - 1) if (n_apod > 1 and n_apod == n_total) else n_apod
                    if denom == 0: return full_depth_center
                    frac = (d - 1) / float(denom)
                    if apod_method == 'tanh':
                        frac = np.tanh(tanh_steepness * 2.0 * frac) / np.tanh(2.0 * tanh_steepness)
                    return full_depth_center + (full_depth_edge - full_depth_center) * frac
                return full_depth_edge

            W_narrow, W_wide = {}, {}
            per_tooth_n = self.width_narrow_per_tooth_m if featured else None
            n_explicit = len(per_tooth_n) if per_tooth_n is not None else 0
            for d in range(1, n_total + 1):
                if d <= n_explicit:
                    # Explicit per-tooth widths take precedence over the envelope.
                    W_narrow[d] = self.width_narrow_per_tooth_m[d - 1]
                    W_wide[d] = self.width_wide_per_tooth_m[d - 1]
                else:
                    mod_depth = get_mod_depth(d)
                    delta_w = mod_depth / 2.0
                    W_narrow[d] = avg_width - delta_w
                    W_wide[d] = avg_width + delta_w

            # Per-tooth shift map (m). Nonzero only for the featured device's freed teeth.
            shift_for_tooth = {d: 0.0 for d in range(0, n_total + 2)}
            if featured:
                for d in range(1, n_free + 1):
                    shift_for_tooth[d] = float(self.inner_shift_nm[d - 1]) * 1e-9
            cavity_extra = (2.0 * sum(shift_for_tooth[d] for d in range(1, n_free + 1))) \
                if (featured and self.lengthen_cavity) else 0.0

            x_grating_start = -self.x_grating_end + x_offset
            x = x_grating_start
            # Feed (access) waveguides run to the PML. Omitted for a CLOSED device 2
            # (device2_closed): with no guided escape path, the power it catches from
            # device 1's radiation re-radiates — a fraction back into device 1 — instead
            # of draining out a through-port. The 80-period arms are strong mirrors, so
            # device 2 acts as a self-contained recycling cavity.
            if draw_feed:
                add_core_segment(-x_domain_half - 1e-6, x_grating_start, self.width_port,
                                 name_prefix="wg_left_inf", y_center=y_center)

            # Left grating: walk d = n_total ... 1 (outside → cavity).
            for d in range(n_total, 0, -1):
                s_d = shift_for_tooth[d]
                if d == 1 and self.cavity_width_option == "avg_ext":
                    w_ln = avg_width
                else:
                    w_ln = W_narrow[d]
                add_core_segment(x, x + half_pitch - s_d, w_ln, name_prefix=f"L_narrow_{d}", y_center=y_center)
                x += half_pitch - s_d
                add_core_segment(x, x + half_pitch, W_wide[d], name_prefix=f"L_wide_{d}", y_center=y_center)
                x += half_pitch

            # Cavity (length expanded by 2*Σ shifts to keep x_grating_end constant)
            if featured and self.cavity_width_m is not None:
                W_cavity = self.cavity_width_m
            else:
                W_cavity = avg_width if self.cavity_width_option in ("avg", "avg_ext") else W_narrow[1]
            add_core_segment(x, x + self.cavity_length + cavity_extra, W_cavity,
                             name_prefix="cavity", y_center=y_center)
            x += self.cavity_length + cavity_extra

            # Right grating: walk d = 1 ... n_total (cavity → outside).
            for d in range(1, n_total + 1):
                s_prev = shift_for_tooth[d - 1] if d >= 2 else 0.0
                if d == 1 and self.cavity_width_option == "avg_ext":
                    w_rn = avg_width
                else:
                    w_rn = W_narrow[d]
                add_core_segment(x, x + half_pitch - s_prev, w_rn, name_prefix=f"R_narrow_{d}", y_center=y_center)
                x += half_pitch - s_prev
                add_core_segment(x, x + half_pitch, W_wide[d], name_prefix=f"R_wide_{d}", y_center=y_center)
                x += half_pitch

            if draw_feed:
                add_core_segment(x, x_domain_half + 1e-6, self.width_port,
                                 name_prefix="wg_right_inf", y_center=y_center)

        # Device 1 (driven, full-featured). y_dev1 = 0 for single-device runs.
        build_grating_arms(self.y_dev1, self.x_dev1, self.width_narrow, self.width_wide, featured=True)

        # Device 2 (passive, plain uniform grating with its own corrugation depth).
        # device2_closed → no feed waveguides (recycler, not a drain).
        if self.n_devices == 2:
            build_grating_arms(self.y_dev2, self.x_dev2, self.width_narrow_2, self.width_wide_2,
                               featured=False, draw_feed=not self.device2_closed)

    def _add_scatterers(self):
        """Small dielectric cylinder(s) beside the guide (radiation-recycling study).

        Vertical cylinder(s), z-centered on the core, full core height — the z=0
        mirror parity is preserved. With scatterer_mirrored_y both (x, ±y) copies
        are drawn, so the scene stays valid whether the y=0 symmetry BC is on
        (only the +y half is solved) or off. Index defaults to the core index
        (SiN post in the oxide background); pass n_clad for an oxide hole inside
        the core — the mesh-order override lets the scatterer win overlaps."""
        if not self._has_scatterer:
            return
        fdtd = self.fdtd
        n_scat = (float(self.scatterer_index) if self.scatterer_index is not None
                  else self.n_core_const)
        y_centers = ([self.scatterer_y_m, -self.scatterer_y_m]
                     if self.scatterer_mirrored_y else [self.scatterer_y_m])
        for k, y_c in enumerate(y_centers, start=1):
            fdtd.addcircle()                      # cylinder, axis = z
            fdtd.set("name", f"scatterer_{k}")
            fdtd.set("material", self.core_material)
            if self._object_index_mode:
                fdtd.set("index", n_scat)
            fdtd.set("radius", self.scatterer_radius_m)
            fdtd.set("x", self.scatterer_x_m)
            fdtd.set("y", y_c)
            fdtd.set("z", 0.0)
            fdtd.set("z span", self.scatterer_height_m)
            fdtd.set("override mesh order from material database", 1)
            fdtd.set("mesh order", 1)             # wins overlaps (needed for in-core holes)
        print(f"  [scatterer] {len(y_centers)}x cylinder r={self.scatterer_radius_m*1e9:.0f} nm, "
              f"n={n_scat:.4g}, at (x={self.scatterer_x_m*1e6:.3f} µm, "
              f"y=±{abs(self.scatterer_y_m)*1e6:.3f} µm)" if self.scatterer_mirrored_y else
              f"  [scatterer] cylinder r={self.scatterer_radius_m*1e9:.0f} nm, n={n_scat:.4g}, "
              f"at (x={self.scatterer_x_m*1e6:.3f} µm, y={self.scatterer_y_m*1e6:.3f} µm)")

    def _add_source_and_monitors(self):
        fdtd = self.fdtd
        dx_mesh = self.dx_override
        dist_snapped = round(self.dist_grating_to_port / dx_mesh) * dx_mesh
        self.dist_grating_to_port = dist_snapped
        self.x_port = self.x_grating_end + dist_snapped

        # Port y span: full-domain for a single device (legacy); for the two-device
        # pair, shrink to a local window around each guide so the port mode solve
        # locks onto its OWN waveguide instead of a two-guide supermode.
        if self.n_devices == 2:
            port_y_span = max(self.width_wide, self.width_wide_2, self.width_port) + 1.2e-6
        else:
            port_y_span = 1.2 * self.y_span

        x_p = self.x_grating_end + dist_snapped

        def add_port(name, x_pos, y_pos, direction):
            fdtd.addport()
            fdtd.set("name", name)
            fdtd.set("injection axis", "x")
            fdtd.set("x", x_pos)
            fdtd.set("y", y_pos)
            fdtd.set("y span", port_y_span)
            fdtd.set("z", 0)
            fdtd.set("z span", 1.2 * self.z_span)
            fdtd.set("direction", direction)
            fdtd.set("mode selection", f"fundamental {self.polarization} mode")
            fdtd.set("frequency dependent profile", 1)

        # Device 1 (driven). y_dev1 = 0 for single-device runs (legacy-identical).
        add_port("Port_1", -x_p, self.y_dev1, "forward")
        add_port("Port_2", +x_p, self.y_dev1, "backward")

        if self.n_devices == 2 and not self.device2_closed:
            # Device 2 (passive, OPEN): mirror of Port_1/Port_2 at device-2's y center,
            # longitudinally shifted by the stagger Δx. Port_1 stays the only source.
            add_port("Port_3", self.x_dev2 - x_p, self.y_dev2, "forward")
            add_port("Port_4", self.x_dev2 + x_p, self.y_dev2, "backward")
            fdtd.setnamed("FDTD::ports", "source port", "Port_1")
        # device2_closed → no device-2 ports (it's a recycler, not measured); FoM is
        # device-1 T from Port_1/Port_2 only.

        # --- Monitor: Central Mode Tracking (1D X-axis) ---
        fdtd.addprofile()
        fdtd.set("name", "field_profile")
        fdtd.set("monitor type", "2D Z-normal")  # This is usually kept thin to trace peak field
        fdtd.set("x", 0)
        fdtd.set("x span", 2.0 * self.x_grating_end + 2.0e-6)
        fdtd.set("y", self.y_dev1)  # track the driven (device-1) guide; 0 for single-device
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
            fdtd.set("y", self.y_dev1)  # track the driven (device-1) guide; 0 for single-device
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

        # Two-device side-by-side: read the passive guide's ports (3,4). Their raw
        # |S|² is the fraction of input power coupled into device-2's fundamental
        # mode (backward / forward). NO phase correction is applied to the cross
        # terms — coupling magnitude is the physical quantity of interest.
        if self.n_devices == 2 and not self.device2_closed:
            res3 = self.fdtd.getresult("FDTD::ports::Port_3", "expansion for port monitor")
            res4 = self.fdtd.getresult("FDTD::ports::Port_4", "expansion for port monitor")
            S31 = np.squeeze(res3["S"])
            S41 = np.squeeze(res4["S"])
            self.S31_complex = S31
            self.S41_complex = S41
            self.coupling_left = np.abs(S31) ** 2
            self.coupling_right = np.abs(S41) ** 2
            self.coupling_total = self.coupling_left + self.coupling_right
            # Device-1 pure radiation loss after subtracting what device 2 captured.
            self.loss_4port = Loss_radiation - self.coupling_total

        return wl, R_modal, T_modal, Loss_radiation, T_matrix, S11_sim, S21_sim