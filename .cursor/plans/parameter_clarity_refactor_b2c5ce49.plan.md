---
name: Parameter Clarity Refactor
overview: Refactor the project to centralize all simulation parameters into grouped, documented dataclasses, enforce consistent naming conventions, and add workflow documentation — making the relationship between inputs, physics, and outputs transparent.
todos:
  - id: params-dataclass
    content: Create simulation_params.py with GeometryParams, MeshParams, MaterialParams, ScanParams, MonitorParams, DerivedParams dataclasses
    status: pending
  - id: update-bragg-device
    content: Refactor bragg_device.py to accept SimulationParams and expose DerivedParams as a public attribute
    status: pending
  - id: update-run-scripts
    content: Update run_simulation.py and run_sweep.py to construct and pass SimulationParams
    status: pending
  - id: document-analysis
    content: Add phase-correction pipeline docblock to analysis.py and rename opaque variables
    status: pending
  - id: matlab-params
    content: Create params.m shared constants file and update MATLAB scripts to use it
    status: pending
  - id: readme
    content: Write README.md with workflow diagram, parameter guide, and output descriptions
    status: pending
isProject: false
---

# Parameter Clarity & Project Organization Refactor

## Problem Summary

Parameters are currently scattered across `bragg_device.py`, `run_simulation.py`, `run_sweep.py`, `config.py`, and MATLAB scripts. There is no single source of truth, no consistent unit-suffix convention, and the derivation chains (e.g. how `pitch` → `lambda_B` → `x_sim_boundary`) are buried inside methods with no documentation.

---

## 1. Create `simulation_params.py` — a single, grouped parameter file

Replace the flat keyword-argument style with Python dataclasses organized by physical role. Each field gets a docstring-style comment and an explicit unit suffix.

```python
# simulation_params.py
from dataclasses import dataclass, field

@dataclass
class GeometryParams:
    pitch_nm: float = 500.0          # Grating period Λ [nm]
    n_periods_each_side: int = 100   # Bragg periods per arm
    n_apod_periods: int = 8          # Periods over which modulation ramps up
    avg_corrugation_nm: float = 800.0  # Mean waveguide width [nm]
    corrugation_depth_nm: float = 200.0  # Full ΔW = W_wide − W_narrow [nm]
    center_mod_depth_nm: float = 10.0   # ΔW at cavity center (apodization) [nm]
    port_width_nm: float = 1000.0    # Straight waveguide at I/O ports [nm]
    core_height_nm: float = 350.0    # Waveguide core thickness [nm]
    substrate_thickness_um: float = 4.0  # SiO₂ below core [µm]
    override_cavity_length_nm: float = None  # Override λ/4 defect length [nm]

@dataclass
class MeshParams:
    cells_per_half_period: int = 5   # Mesh cells per Λ/2 → dx = Λ/(2*5)
    dy_divisions_of_narrow: int = 13 # dy = W_narrow / 13
    dz_divisions_of_core: int = 7    # dz = core_height / 7
    max_cavity_dx_nm: float = 40.0   # Fine mesh inside cavity [nm]
    use_cavity_mesh_override: bool = False

@dataclass
class MaterialParams:
    use_constant_index: bool = True
    n_core: float = 1.977            # SiN refractive index (constant mode)
    n_clad: float = 1.44             # SiO₂ refractive index
    n_eff_guess: float = 1.55        # Approximate modal neff for Bragg λ estimate
    core_material: str = "Si3N4 (Silicon Nitride) - Luke"
    clad_material: str = "SiO2 (Glass) - Palik"

@dataclass
class ScanParams:
    center_wavelength_nm: float = 1562.5  # Scan center ≈ Bragg resonance [nm]
    scan_width_nm: float = 30.0           # Total scan range [nm]
    n_spectral_points: int = 3001         # Points for S-parameter monitors
    n_field_points: int = 51              # Points for 2D/3D field monitors

@dataclass
class MonitorParams:
    record_farfield: bool = True
    farfield_x_span_um: float = 30.0     # X extent of far-field monitors [µm]
    farfield_resolution: int = 201        # Grid size for ux/uy far-field map
    farfield_monitor_y_offset_wls: float = 0.8  # Monitor distance from PML [λ]
    farfield_monitor_z_offset_wls: float = 0.8
    n_periods_to_port: int = 20          # Grating-to-port gap [periods]
    n_wls_port_to_pml: float = 5.0       # Port-to-PML buffer [λ_B]
    span_multiplier: float = 3.0         # FDTD cladding buffer = mult × λ_res

@dataclass
class SimulationParams:
    geometry: GeometryParams = field(default_factory=GeometryParams)
    mesh: MeshParams = field(default_factory=MeshParams)
    material: MaterialParams = field(default_factory=MaterialParams)
    scan: ScanParams = field(default_factory=ScanParams)
    monitor: MonitorParams = field(default_factory=MonitorParams)
```

Key improvements:

- Every parameter has a unit suffix (`_nm`, `_um`, `_m`, `_wls`) so there is never ambiguity at a call site
- Parameters are grouped by physical role, not by order of use
- `SimulationParams` is the single object passed to `PiShiftBraggFDTD` and `run_simulation`

---

## 2. Add a `DerivedParams` class with explicit derivation chains

All computed quantities (currently scattered across `__init__` and private methods) should be computed once, in one place, with the formula written as a comment:

```python
@dataclass
class DerivedParams:
    """All values computed from SimulationParams. Never set manually."""
    width_narrow_nm: float   # = avg_corrugation - corrugation_depth/2
    width_wide_nm: float     # = avg_corrugation + corrugation_depth/2
    cavity_length_nm: float  # = pitch/2  (λ/4 phase shift)
    lambda_bragg_nm: float   # = 2 * n_eff_guess * pitch
    dx_nm: float             # = pitch / (2 * cells_per_half_period)
    dy_nm: float             # = width_narrow / dy_divisions_of_narrow
    dz_nm: float             # = core_height / dz_divisions_of_core
    x_grating_end_um: float  # = n_periods * pitch + cavity/2
    x_sim_boundary_um: float # = x_grating_end + port_gap + pml_buffer
    y_span_um: float         # = width_wide + span_mult * lambda_res
    z_span_um: float         # = core_height + span_mult * lambda_res

    @classmethod
    def from_params(cls, p: SimulationParams) -> "DerivedParams": ...
```

This makes the derivation chain auditable: any reader can trace exactly how the simulation box size or mesh step arises from the fundamental inputs.

---

## 3. Rename for unit consistency across Python files

Current inconsistencies to fix:


| Current name             | Rename to                                     | Reason                       |
| ------------------------ | --------------------------------------------- | ---------------------------- |
| `pitch`                  | `pitch_nm` (or use `p.geometry.pitch_nm`)     | no unit, usually nm          |
| `core_h` / `core_height` | `core_height_nm`                              | inconsistent alias           |
| `avg_corr`               | `avg_corrugation_nm`                          | cryptic abbreviation         |
| `corr_depth`             | `corrugation_depth_nm`                        | cryptic                      |
| `n_2d_points`            | `n_field_points`                              | unclear what "2D" means here |
| `span_mult`              | `span_multiplier`                             | abbreviation                 |
| `FF_RES` / `IDX_F`       | `farfield_resolution` / `farfield_freq_index` | ALL_CAPS not a constant      |
| `farfield_y_wls`         | `farfield_monitor_y_offset_wls`               | unclear meaning              |


---

## 4. Document phase-correction parameters in `analysis.py`

The de-embedding logic is the hardest part to understand. Add a block comment before `apply_phase_correction()` explaining:

```python
# Phase correction pipeline:
#
# 1. Feed de-embedding:
#    S_corr = S_raw * exp(+i * beta(lambda) * L_feed * 2)
#    Removes phase accumulated in the port-to-grating waveguide.
#    L_feed = dist_grating_to_port [m], beta = 2π * neff / lambda
#
# 2. Bloch carrier removal:
#    S_corr *= exp(+i * beta_0 * L_device)
#    beta_0 = pi / pitch  (grating Bloch vector at Bragg condition)
#    Removes the linear phase slope across the full device.
#
# 3. Absolute phase alignment:
#    Rotate so that angle(S21) = pi/2 at the resonance peak.
#    This sets the phase origin to the physical convention where
#    a lossless resonator has S21 = +i at resonance.
```

---

## 5. Add a `README.md` with a workflow diagram

```
calc_neff_vs_wl.py ──► FDE_sweep_results.mat
                              │ neff(λ)
simulation_params.py ──► bragg_device.py ──► run_simulation.py
   (all parameters)      (FDTD geometry)        │
                                              analysis.py
                                                 │
                                           result_*.mat
                                                 │
                    ┌────────────────────────────┤
                    ▼                            ▼
              plot_fdtd.m             plot_farfield.m
           (T/R/Loss, Q, φ)      (near-field, far-field, FWHM angle)
                    │
        analyze_core_k_space.m
        compare_simulations.m
        overlap_analysis_bg.m
```

---

## 6. Add a shared parameter header block to MATLAB scripts

All MATLAB scripts currently repeat magic numbers. Extract these into a `params.m` script that each `.m` file runs first:

```matlab
% params.m — shared physical constants for all MATLAB post-processing
PITCH_NM        = 500;     % Grating period [nm]
N_EFF_GUESS     = 1.55;    % Approximate modal effective index
N_CLAD          = 1.44;    % SiO2 cladding index
CRITICAL_ANGLE_DEG = asind(N_CLAD / N_EFF_GUESS);  % TIR critical angle
LAMBDA_BRAGG_NM = 2 * N_EFF_GUESS * PITCH_NM;      % Bragg wavelength estimate
FF_RES          = 201;     % Far-field grid resolution
```

This removes duplicated literals in `plot_farfield.m`, `analyze_core_k_space.m`, and `compare_simulations.m`.

---

## Files Changed

- **New file** `[simulation_params.py](simulation_params.py)` — all grouped + documented params
- **Modified** `[bragg_device.py](bragg_device.py)` — accept `SimulationParams`, expose `DerivedParams`
- **Modified** `[run_simulation.py](run_simulation.py)` — use `SimulationParams`
- **Modified** `[run_sweep.py](run_sweep.py)` — use `SimulationParams`
- **Modified** `[analysis.py](analysis.py)` — add phase-correction documentation
- **New file** `[params.m](params.m)` — shared MATLAB constants
- **Modified** `[plot_farfield.m](plot_farfield.m)`, `[analyze_core_k_space.m](analyze_core_k_space.m)`, `[compare_simulations.m](compare_simulations.m)` — use `params.m`
- **New file** `[README.md](README.md)` — workflow diagram and parameter guide

