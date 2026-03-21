# Pi-Shift Bragg Grating FDTD Simulation

3D FDTD simulation of pi-phase-shifted Bragg gratings in silicon nitride (Si3N4) waveguides using Lumerical FDTD. Computes spectral response (S-parameters), intracavity field profiles, far-field radiation patterns, and exports data for Lumerical INTERCONNECT.

## Prerequisites

- **Lumerical FDTD** (v252 or later) with GPU acceleration
- **Python 3.10+** with packages: `numpy`, `scipy`, `matplotlib`
- **MATLAB** (for post-processing plots and analysis)

## Project Structure

```
.
├── simulation_config.py         # All simulation parameters (edit this)
├── config.py                    # Machine-specific paths (edit this)
├── run_simulation.py            # Core simulation pipeline
├── run_sweep.py                 # Parameter sweep (thin wrapper)
├── run_simple_bragg.py          # Simple uniform grating (no cavity)
├── bragg_device.py              # PiShiftBraggFDTD device builder
├── analysis.py                  # Phase correction & S-parameter processing
├── sim_helpers.py               # Shared helper functions
├── calc_neff_vs_wl.py           # FDE mode solver for neff data
├── plots.py                     # Python plotting utility
├── post_proccess_bragg.py       # Re-process saved simulations
├── convergence_testing/
│   └── run_convergence.py       # Far-field convergence test
├── matlab_plotting/
│   ├── plot_farfield.m          # Near-field + far-field visualization
│   ├── plot_fdtd.m              # T/R/Loss/Phase spectra
│   └── plot_convergence.m       # Far-field vs monitor distance
└── matlab_analysis/
    ├── analyze_farfield_radiation.m   # 3D spherical radiation patterns
    ├── analyze_core_k_space.m         # FFT k-space analysis
    ├── analyze_yz_circle_power.m      # YZ circular power extraction
    ├── compare_simulations.m          # Compare two simulation results
    ├── calculate_profile.m            # 3D field profile FWHM
    ├── overlap_analysis_many.m        # Batch overlap integrals
    └── overlap_analysis_bg.m          # Overlap utility
```

## Quick Start

1. **Set your local paths** in `config.py`:
   - `BASE_SAVE_DIR` — where layouts (.fsp) and results (.mat) are saved
   - `NEFF_DATA_PATH` — pre-computed neff vs wavelength data
   - `LUMAPI_PATH` — path to Lumerical Python API

2. **Adjust simulation parameters** in `simulation_config.py` (or override in the runner script).

3. **Run a single simulation:**
   ```bash
   python run_simulation.py
   ```

4. **Run a parameter sweep:**
   ```bash
   python run_sweep.py
   ```

## Configuration

### Two config files

| File | Purpose |
|------|---------|
| `config.py` | Machine-specific paths (save directories, Lumerical API path) |
| `simulation_config.py` | All physics and simulation parameters |

### Parameter groups in `simulation_config.py`

| Group | Key fields | Description |
|-------|-----------|-------------|
| `GeometryConfig` | `avg_corrugation_width_m`, `corrugation_depth_m`, `core_height_m` | Waveguide cross-section |
| `GratingConfig` | `pitch_m`, `n_periods_each_side` | Bragg grating periodicity |
| `ApodizationConfig` | `enabled`, `n_apod_periods_each_side`, `center_mod_depth_nm` | Width modulation tapering |
| `SpectralConfig` | `center_wavelength_m`, `scan_width_nm`, `n_wl_points` | Wavelength scan range |
| `MeshConfig` | `span_multiplier`, `n_periods_dist_to_port` | Domain sizing and mesh |
| `MaterialConfig` | `use_constant_materials`, `n_core_const`, `n_eff_guess` | Refractive index settings |
| `SymmetryConfig` | `use_y_symmetry`, `use_z_symmetry` | Boundary condition symmetry |
| `MonitorConfig` | `record_2d_fields`, `record_3d_fields` | Field monitor settings |
| `FarFieldConfig` | `enabled`, `farfield_x_span_m`, `farfield_dist_wls` | Far-field monitors |
| `PhaseCorrectionConfig` | `do_length_correction`, `do_envelope_correction` | S-parameter corrections |
| `SweepConfig` | `parameter`, `values` | Parameter sweep control |

### How to change key parameters

```python
from simulation_config import SimulationConfig

cfg = SimulationConfig()

# Number of periods
cfg.grating.n_periods_each_side = 120

# Enable/disable apodization
cfg.apodization.enabled = False

# Apodization depth at the cavity center
cfg.apodization.center_mod_depth_nm = 20.0

# Center wavelength
cfg.spectral.center_wavelength_m = 1.560e-6

# Use dispersive materials instead of constant index
cfg.material.use_constant_materials = False
```

## Apodization

The grating supports **linear gradient apodization** for sidelobe suppression.

When enabled, the corrugation depth (width modulation) varies linearly:
- At the **cavity center**: modulation = `center_mod_depth_nm` (small)
- At the **grating edges**: modulation = full `corrugation_depth_m` (large)
- The transition occurs over `n_apod_periods_each_side` periods on each side

This creates a smooth taper that reduces abrupt impedance discontinuities at the grating boundaries.

**Parameters:**

| Field | Default | Description |
|-------|---------|-------------|
| `apodization.enabled` | `True` | Enable/disable apodization |
| `apodization.n_apod_periods_each_side` | `10` | Number of tapered periods per side |
| `apodization.center_mod_depth_nm` | `10.0` | Modulation depth at cavity (nm) |

When `enabled = False`, all periods use the full corrugation depth (uniform grating with cavity).

## Phase Correction

S-parameter phase correction is applied in `analysis.py` to extract the intrinsic grating response. It has three stages, controlled by two flags:

### Stage A: Feed-Line De-embedding (`do_length_correction`)

Removes the propagation phase accumulated in the straight waveguide sections between the FDTD ports and the grating edges.

- Uses neff (effective index) from one of three sources:
  1. Constant scalar value (when `use_constant_materials = True`)
  2. External .mat file (neff vs wavelength from FDE solver)
  3. FDTD port monitor data
- Formula: `S11 *= exp(-j * beta * L_feed)^2`, `S21 *= exp(-j * beta1 * L) * exp(-j * beta2 * L)`
- `L_feed = dist_grating_to_port` (distance from grating edge to port)

### Stage B: Bragg Carrier Phase Removal (`do_envelope_correction`)

Removes the expected linear phase slope across the device:
- `beta_0 = pi / pitch` (the Bragg wave vector)
- `slope_correction = exp(-j * beta_0 * device_length)`
- Applied to S21 only

### Stage C: Phase Alignment (`do_envelope_correction`)

Rotates the S21 phase so the resonance peak sits at exactly -90 degrees (pi/2 radians):
1. Finds the resonance peak within the stopband (T < 0.5)
2. Measures the S21 phase at that peak
3. Applies a global phase rotation to align it to the target

### Control

```python
cfg.phase_correction.do_length_correction = True   # Stage A
cfg.phase_correction.do_envelope_correction = True  # Stages B + C
```

For simple Bragg gratings (no cavity), Stage B+C should typically be disabled since there is no resonance peak to align to.

## Parameter Sweep

`run_sweep.py` is a thin wrapper over `run_simulation.run_single_sim()`. It iterates over a list of values for any config parameter using dot notation:

```python
cfg = SimulationConfig()
cfg.sweep.parameter = "grating.n_periods_each_side"
cfg.sweep.values = [80, 100, 120]
```

Other sweep examples:
```python
# Sweep apodization depth
cfg.sweep.parameter = "apodization.center_mod_depth_nm"
cfg.sweep.values = [5.0, 10.0, 20.0, 40.0]

# Sweep number of apodized periods
cfg.sweep.parameter = "apodization.n_apod_periods_each_side"
cfg.sweep.values = [5, 10, 20, 50]
```

Each sweep iteration creates a deep copy of the config, overrides the target parameter, and runs the full simulation pipeline.

## MATLAB Post-Processing

### Plotting (`matlab_plotting/`)

- **plot_fdtd.m** — Transmission, reflection, loss, and phase from .mat result files
- **plot_farfield.m** — Near-field monitor surface and far-field radiation patterns
- **plot_convergence.m** — Far-field convergence with monitor distance

### Analysis (`matlab_analysis/`)

- **analyze_farfield_radiation.m** — 3D spherical radiation pattern (hemisphere surf plot + polar cuts)
- **analyze_core_k_space.m** — FFT analysis of intracavity field profile in k-space
- **analyze_yz_circle_power.m** — Power distribution along circular boundary in YZ cross-section
- **compare_simulations.m** — Side-by-side comparison of two simulation results
- **calculate_profile.m** — 3D field profile FWHM analysis
- **overlap_analysis_many.m** — Batch overlap integral computation between devices
- **overlap_analysis_bg.m** — Overlap integral utility function

## Output Format

Results are saved as `.mat` files with the following fields:

| Field | Description |
|-------|-------------|
| `wl_m`, `wl_nm` | Wavelength vectors |
| `T`, `R`, `loss` | Transmission, reflection, radiation loss |
| `S11_complex`, `S21_complex` | Complex S-parameters |
| `T_matrix` | Transfer matrix |
| `field_x`, `field_energy_density_1D` | 1D field profile along grating |
| `field_envelope_1D`, `fwhm_m` | Envelope and FWHM of intracavity mode |
| `field_xy`, `field_yz_cross` | 2D field slices (if enabled) |
| `field_3d` | Full 3D field volume (if enabled) |
| `farfield_side`, `farfield_top` | Far-field patterns (if enabled) |
| `nearfield_side`, `nearfield_top` | Near-field on monitor surfaces (if enabled) |

S-parameters are also exported in Lumerical INTERCONNECT format as `.txt` files.
