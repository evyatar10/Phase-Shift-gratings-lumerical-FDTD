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
├── run_experiment.py            # Experiment card runner (examples)
├── run_simple_bragg.py          # Simple uniform grating (no cavity)
├── experiment_card.py           # ExperimentCard dataclass + run_card/run_cards
├── bragg_device.py              # PiShiftBraggFDTD device builder (incl. innermost-tooth shift)
├── analysis.py                  # Phase correction & S-parameter processing
├── post_processing.py           # Full post-simulation analysis pipeline
├── sim_helpers.py               # Shared helper functions
├── python_tools/
│   ├── calc_neff_vs_wl.py       # FDE mode solver for neff data
│   ├── plot_loss_spectra.py     # Plot radiation loss spectra
│   ├── farfield_export.py       # Far-field data export utility
│   ├── analyze_farfield.py      # Far-field analysis tool
│   └── overlap_analysis.py      # Overlap integral tool
├── experiment_examples/
│   └── p8RC1_tanh.py            # Example experiment card (tanh apodization)
├── convergence_testing/
│   ├── run_convergence.py       # Far-field convergence test
│   └── run_mesh_convergence.py  # Coordinate-descent mesh convergence test
├── ToothShift/
│   ├── run_sweep_innermost_shift.py  # Innermost tooth shift sweep
│   ├── run_sweep_inner_tooth_size.py # 2D sweep: inner tooth size × shift
│   └── optimize_innermost_shift.py   # Brent's method optimizer for shift
├── zeus/
│   ├── deploy.sh                # Upload project to Zeus and submit PBS job
│   ├── scripts/
│   │   └── server_run.py        # Server-side pipeline wrapper (patches config paths)
│   └── jobs/
│       ├── run_python_job.sh    # PBS job script for Python pipeline
│       └── run_fsp_job.sh       # PBS job script for .fsp file
├── matlab_plotting/
│   ├── plot_fdtd.m              # T/R/Loss/Phase spectra
│   ├── plot_farfield.m          # Near-field + far-field visualization
│   ├── plot_convergence.m       # Far-field vs monitor distance
│   ├── plot_field_poynting.m    # Field + Poynting vector plots
│   ├── plot_field_poynting_overlay.m  # Overlaid field/Poynting visualization
│   ├── plot_field_poynting_zoom.m     # Zoomed field/Poynting view
│   ├── plot_mode_profile.m            # Mode profile (|E|² envelope + FWHM) comparison
│   ├── plot_mode_profile_xz.m         # Mode profile from XZ side-view monitor
│   ├── plot_transmission_compare.m    # Transmission spectra multi-file comparison
│   ├── plot_resonance_vs_param.m      # Resonance / peak T vs shift or inner tooth size
│   └── save_figures_interactive.m     # Interactive figure export helper
└── matlab_analysis/
    ├── analyze_farfield_radiation.m   # 3D spherical radiation patterns
    ├── analyze_core_k_space.m         # FFT k-space analysis
    ├── analyze_yz_circle_power.m      # YZ circular power extraction
    ├── analyze_radiation_recycling.m  # Radiation recycling analysis
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

5. **Run an innermost tooth shift sweep:**
   ```bash
   python ToothShift/run_sweep_innermost_shift.py
   ```

6. **Run a 2D inner tooth size × shift sweep:**
   ```bash
   python ToothShift/run_sweep_inner_tooth_size.py
   ```

7. **Optimize the innermost tooth shift automatically:**
   ```bash
   python ToothShift/optimize_innermost_shift.py
   ```

8. **Run with an experiment card:**
   ```bash
   python run_experiment.py
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
| `ApodizationConfig` | `enabled`, `method`, `n_apod_periods_each_side`, `center_mod_depth_nm`, `tanh_steepness` | Width modulation tapering |
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

# Apodization profile: 'none', 'linear', or 'tanh'
cfg.apodization.method = 'tanh'
cfg.apodization.tanh_steepness = 2.5

# Apodization depth at the cavity center
cfg.apodization.center_mod_depth_nm = 20.0

# Center wavelength
cfg.spectral.center_wavelength_m = 1.560e-6

# Use dispersive materials instead of constant index
cfg.material.use_constant_materials = False
```

## Apodization

The grating supports three apodization profiles for sidelobe suppression, controlled by `ApodizationConfig.method`:

- **`'none'`** — uniform grating, constant corrugation depth throughout
- **`'linear'`** — modulation depth varies linearly from `center_mod_depth_nm` at the cavity to the full `corrugation_depth_m` at the grating edges
- **`'tanh'`** — modulation depth follows a tanh profile, concentrating the transition near the grating edge for a flatter center region

The transition occurs over `n_apod_periods_each_side` periods on each side.

**Parameters:**

| Field | Default | Description |
|-------|---------|-------------|
| `apodization.enabled` | `False` | Enable/disable apodization |
| `apodization.method` | `'linear'` | Profile shape: `'none'`, `'linear'`, or `'tanh'` |
| `apodization.n_apod_periods_each_side` | `1` | Number of tapered periods per side |
| `apodization.center_mod_depth_nm` | `160.0` | Modulation depth at cavity (nm) |
| `apodization.tanh_steepness` | `2.0` | Steepness `a` for tanh profile |

## Innermost Tooth Shift

`PiShiftBraggFDTD` accepts an `innermost_tooth_shift_m` parameter that shifts the innermost grating tooth on each side of the cavity by a distance `delta`:

```
Before: ...[L_narrow_1: hp][L_wide_1: hp][cavity][R_narrow_1: hp][R_wide_1: hp]...
After:  ...[L_narrow_1: hp-δ][L_wide_1: hp][cavity+2δ][R_narrow_1: hp][R_wide_1: hp-δ]...
```

With `lengthen_cavity=True` (default), the cavity grows by `2*delta` so total device length is preserved; the innermost narrow half-periods shrink by `delta`. With `lengthen_cavity=False` the cavity is fixed and total length shrinks by `2*delta`.

**Usage:**
```python
from bragg_device import PiShiftBraggFDTD

sim = PiShiftBraggFDTD(**kwargs, innermost_tooth_shift_m=105e-9)
```

The `ToothShift/` directory contains three scripts for exploring this design parameter:

- **`run_sweep_innermost_shift.py`** — sweeps a list of shift values, saves a `.mat` result per shift, and plots all transmission spectra on one figure.
- **`run_sweep_inner_tooth_size.py`** — runs a 2D sweep: for each shift value in `TOOTH_SHIFT_VALUES_NM`, iterates over all corrugation depth values in `INNER_SIZE_VALUES_NM`.
- **`optimize_innermost_shift.py`** — finds the shift that maximizes resonance transmission using Brent's method (bounded golden-section + parabolic interpolation) within a configurable budget of FDTD evaluations.

The MATLAB script `matlab_plotting/plot_resonance_vs_param.m` can visualize both resonance wavelength and peak transmission vs. shift or inner tooth size from the saved `.mat` files.

## Innermost Tooth Size

Setting an independent corrugation depth for the innermost tooth is equivalent to 1-period apodization — use the existing apodization parameters:

```python
# Custom innermost-tooth depth (e.g. 120 nm) + optional shift
sim = PiShiftBraggFDTD(
    **kwargs,
    innermost_tooth_shift_m=105e-9,
    use_apodization=True,
    n_apod_periods_each_side=1,
    center_mod_depth_nm=120.0,
)
```

With `n_apod_periods_each_side=1`, `get_mod_depth(d)` returns `center_mod_depth` for `d=1` and the standard edge depth for all other periods — byte-identical to a direct override of the innermost widths.

## Experiment Cards

`experiment_card.py` provides a simplified parameter interface for comparing against fabricated devices. An `ExperimentCard` specifies only the parameters that differ between experiments; everything else uses `SimulationConfig` defaults.

```python
from experiment_card import ExperimentCard, run_card, run_cards

# Single device
card = ExperimentCard(
    n_periods_each_side=50,
    center_mod_depth_nm=15,
    apod_method='tanh',
    tanh_steepness=2.5,
    label="Sample A",
)
run_card(card)

# Compare multiple devices
run_cards([
    ExperimentCard(n_periods_each_side=40, center_mod_depth_nm=10, label="Dev1"),
    ExperimentCard(n_periods_each_side=60, center_mod_depth_nm=20, label="Dev2"),
])
```

**Supported card fields:**

| Field | Config path | Notes |
|-------|------------|-------|
| `n_periods_each_side` | `grating.n_periods_each_side` | |
| `center_mod_depth_nm` | `apodization.center_mod_depth_nm` | |
| `apod_method` | `apodization.method` | `'none'`, `'linear'`, `'tanh'` |
| `tanh_steepness` | `apodization.tanh_steepness` | |
| `corrugation_depth_nm` | `geometry.corrugation_depth_m` | auto-converted to m |
| `pitch_nm` | `grating.pitch_m` | auto-converted to m |
| `n_apod_periods_each_side` | `apodization.n_apod_periods_each_side` | |
| `cavity_length_nm` | `grating.override_cavity_length_nm` | `None` = default (pitch/2) |
| `center_wavelength_nm` | `spectral.center_wavelength_m` | auto-converted to m |
| `scan_width_nm` | `spectral.scan_width_nm` | |

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

## Post-Processing Pipeline

`post_processing.py` is the main analysis entry point after the FDTD solver completes. It runs the following pipeline stages in order:

1. S-parameter extraction and phase correction
2. Resonance detection
3. Field profile extraction and FWHM
4. 2D field monitor extraction
5. 3D field monitor extraction
6. Far-field and near-field extraction
7. Results assembly
8. Save to `.mat`
9. INTERCONNECT export
10. Plotting

Each stage is a standalone function and can be called independently (e.g., to re-run only far-field analysis on an already-loaded session).

## MATLAB Post-Processing

### Plotting (`matlab_plotting/`)

- **plot_fdtd.m** — Transmission, reflection, loss, and phase from .mat result files
- **plot_farfield.m** — Near-field monitor surface and far-field radiation patterns
- **plot_convergence.m** — Far-field convergence with monitor distance
- **plot_field_poynting.m** — Field intensity and Poynting vector visualization
- **plot_field_poynting_overlay.m** — Overlaid field and Poynting vector plots
- **plot_field_poynting_zoom.m** — Zoomed-in field/Poynting view
- **plot_mode_profile.m** — Side-by-side comparison of |E|² envelope + FWHM across simulations
- **plot_mode_profile_xz.m** — Mode profile from the XZ side-view 2D monitor (integrates |E|² over z)
- **plot_transmission_compare.m** — Overlay transmission spectra from multiple .mat result files
- **plot_resonance_vs_param.m** — Resonance wavelength and/or peak transmission vs. shift or inner tooth size (auto-detected from filename)
- **save_figures_interactive.m** — Interactive helper for exporting figures

### Analysis (`matlab_analysis/`)

- **analyze_farfield_radiation.m** — 3D spherical radiation pattern (hemisphere surf plot + polar cuts)
- **analyze_core_k_space.m** — FFT analysis of intracavity field profile in k-space
- **analyze_yz_circle_power.m** — Power distribution along circular boundary in YZ cross-section
- **analyze_radiation_recycling.m** — Radiation recycling analysis
- **compare_simulations.m** — Side-by-side comparison of two simulation results
- **calculate_profile.m** — 3D field profile FWHM analysis
- **overlap_analysis_many.m** — Batch overlap integral computation between devices
- **overlap_analysis_bg.m** — Overlap integral utility function

## Convergence Testing

`convergence_testing/` contains two convergence scripts:

- **`run_convergence.py`** — Far-field monitor distance convergence test.
- **`run_mesh_convergence.py`** — Coordinate-descent mesh convergence. Tests two mesh parameters sequentially:
  - **Phase A** — `cells_per_half_period` (controls dx)
  - **Phase B** — `dz_divisor` (controls dz = core_height / divisor)

  The convergence metric is configurable (`"Q"` for Q-factor or `"lambda"` for resonance wavelength). Early stopping activates when the metric changes by less than the threshold for two consecutive mesh values, and checkpoints are written to a JSON file after every run.

## Zeus Deployment (CPU / PBS)

The `zeus/` directory contains scripts for running simulations on the Zeus PBS cluster at Technion.

### Workflow

1. **Upload and submit:**
   ```bash
   bash zeus/deploy.sh
   ```
   This syncs all Python source files and neff data to Zeus via `scp`, then submits a PBS job with `qsub`.

2. **Upload only (no job submission):**
   ```bash
   bash zeus/deploy.sh --upload-only
   ```

3. **Download results after the job finishes:**
   ```bash
   bash zeus/deploy.sh --results
   ```
   Results are saved locally to `./results_from_server/`.

### Configuration

Edit `zeus/zeus.conf` to set:

| Variable | Description |
|----------|-------------|
| `ZEUS_USER` | Your Zeus username |
| `ZEUS_HOST` | Zeus hostname (default: `zeus.technion.ac.il`) |
| `REMOTE_BASE` | Remote working directory |

### Files

| File | Description |
|------|-------------|
| `zeus/deploy.sh` | Main deploy/submit/download script |
| `zeus/scripts/server_run.py` | Server-side pipeline wrapper; patches `config.py` paths for Zeus at runtime |
| `zeus/jobs/run_python_job.sh` | PBS job script that runs the Python pipeline |
| `zeus/jobs/run_fsp_job.sh` | PBS job script for running a standalone `.fsp` file |

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
