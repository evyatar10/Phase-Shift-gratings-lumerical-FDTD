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
├── experiment_card.py           # ExperimentCard dataclass + run_card/run_cards
├── bragg_device.py              # PiShiftBraggFDTD device builder (incl. innermost-tooth shift)
├── analysis.py                  # Phase correction & S-parameter processing
├── post_processing.py           # Full post-simulation analysis pipeline
├── sim_helpers.py               # Shared helper functions
├── runners/                     # All studies — see runners/README.md for the full guide
│   ├── README.md                # The two study patterns + deploy-menu contract
│   ├── optimization_common.py   # Shared base config for the optimization studies
│   ├── single/
│   │   ├── run_simulation.py            # Core simulation pipeline
│   │   ├── run_experiment.py            # Experiment card runner (examples)
│   │   └── run_simple_bragg.py          # [legacy] Simple uniform grating (no cavity)
│   ├── sweeps/
│   │   ├── sweep_spec.py                  # SweepSpec engine + run_sweep_spec dispatcher
│   │   ├── number_of_periods.py           # Sweep over n_periods_each_side
│   │   ├── innermost_shift.py             # Sweep over innermost tooth shift
│   │   ├── inner_tooth_size.py            # 2D sweep: inner tooth size × shift
│   │   ├── apod_and_shift.py              # 2D sweep: n_apod_periods × shift
│   │   └── optimize_innermost_shift.py    # [legacy] Brent's method optimizer for shift
│   ├── inverse_design/          # Optimization: lumopt adjoint (L-BFGS-B)
│   ├── gradient_free_design/    # Optimization: Python-driven Lumerical PSO
│   ├── fd_gradient_design/      # Optimization: scipy + finite-difference gradients
│   ├── lumerical_native_optimization/  # Optimization: native addsweep("Optimization")
│   ├── experiment_comparison/   # Simulation vs fabricated IT11 devices
│   └── visualization/           # Shared optimization-run plotting helpers
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
│   ├── run_mesh_convergence.py  # Coordinate-descent mesh convergence test
│   └── run_auto_shutoff_convergence.py  # Auto-shutoff threshold convergence
├── athena/                      # PRIMARY HPC target — SLURM GPU cluster
│   ├── deploy_athena.sh         # Upload / submit / download orchestrator (interactive + flags)
│   ├── athena.conf              # Host, remote paths, partitions, license
│   ├── jobs/                    # SLURM job scripts (single, array, aggregate)
│   └── scripts/                 # Server-side entry points (athena_run.py, ...)
├── dgx/                         # Legacy A100 cluster (R470 driver; needs NVML shim)
├── container/                   # Apptainer container build for Athena/DGX
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
   python -m runners.single.run_simulation
   ```

4. **Run a parameter sweep.** Each file in `runners/sweeps/` is one study,
   defined declaratively as a `SweepSpec`. Edit the lists inside the file, then:
   ```bash
   python -m runners.sweeps.number_of_periods    # sweep n_periods_each_side
   python -m runners.sweeps.innermost_shift      # sweep tooth shift
   python -m runners.sweeps.inner_tooth_size     # 2D: shift × inner-tooth depth
   python -m runners.sweeps.apod_and_shift        # 2D: n_apod periods × shift
   ```
   To create a new sweep: copy any of these files and change the `SweepSpec`
   field lists. The same file runs locally (sequential) or on Athena as a
   parallel SLURM array via `bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.<study>`.

5. **Run an optimization study.** Four methods live under `runners/`
   (inverse_design, gradient_free_design, fd_gradient_design,
   lumerical_native_optimization), each with the same layout: an engine file,
   a production `optimize_transmission.py`, and a fast `smoke_test.py`.
   Always smoke-test on Athena before a production run:
   ```bash
   bash athena/deploy_athena.sh --gradient-free-design=runners.gradient_free_design.smoke_test
   bash athena/deploy_athena.sh --gradient-free-design=runners.gradient_free_design.optimize_transmission
   # other families: --inverse-design= / --lumerical-native= / --fd-gradient-design=
   ```
   See [runners/README.md](runners/README.md) for the full guide, including
   the deploy-menu discovery contract (what makes a file appear in the
   Athena menus) and the shared-module map.

   (The older `python -m runners.sweeps.optimize_innermost_shift` Brent's-method
   shift optimizer still works but is superseded by these.)

6. **Run with an experiment card:**
   ```bash
   python -m runners.single.run_experiment
   ```

> **Where things actually run:** local Python is used for building scenes and
> quick checks; production FDTD runs are dispatched to the Athena GPU cluster
> via `athena/deploy_athena.sh` (interactive menu, or the flags shown above).
> `--status` shows the queue, `--results-no-fsp` downloads data, and
> `--license-probe` checks FlexLM seats before submitting.

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

The `runners/sweeps/` directory contains studies for exploring this design parameter:

- **`innermost_shift.py`** — `SweepSpec` over `innermost_tooth_shift_nm`, saves a `.mat` per shift.
- **`inner_tooth_size.py`** — 2D `SweepSpec`: `innermost_tooth_shift_nm × center_mod_depth_nm` (innermost-tooth depth via 1-period apodization).
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

Sweeps are declarative `SweepSpec` instances. Each file in `runners/sweeps/`
is one study. The sweep engine ([`runners/sweeps/sweep_spec.py`](runners/sweeps/sweep_spec.py))
takes a `SweepSpec`, expands the cartesian (or zipped) product of all populated
fields into a list of `SimulationConfig` objects, and runs them.

```python
from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig

SPEC = SweepSpec(
    n_periods_each_side      = [80, 100, 120],
    center_mod_depth_nm      = [5.0, 10.0, 20.0, 40.0],
    label = "periods_x_mod_depth",
)

if __name__ == "__main__":
    base = SimulationConfig()
    base.mesh.simulation_mode = "optimization"
    run_sweep_spec(SPEC, target="local", base=base)   # 12 sims, sequential
```

Sweepable fields are listed in `experiment_card._CARD_FIELD_MAP`. To add a new
one, add an entry there once and it becomes available to every `SweepSpec`.

The same study file dispatches to a SLURM array on Athena (one task per
combo) via `bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.<study>`.

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
