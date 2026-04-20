# File Naming Convention

Output files (`.fsp` layouts, `.mat` results) use compact parameter abbreviations
to keep paths short enough for Lumerical's internal path-length limit.

## Parameter abbreviations

| New token | Old token | Description |
|---|---|---|
| `N80` | `80_periods` | Number of grating periods each side of cavity |
| `A1` | `1_apod` | Number of apodization periods each side |
| `_th` | `_tanh` | Tanh apodization profile (omitted = linear) |
| `_D5p76` | `_neg_det_5.76nm` | Negative cavity detuning from pitch/2 (nm); `.` → `p` |
| `_avg` | `_avg_wgd` | Cavity width = average of narrow + wide waveguide |
| `_avgx` | `_avg_ext_wgd` | Cavity width = average including extended region |
| `_d` | `_disp` | Dispersive material model (omitted = constant index) |
| `_ff` | `_ff` | Far-field monitors enabled (unchanged) |
| `_S90` | `_shift_90.0nm` | Innermost tooth shift (nm) |
| `_I100` | `_innersize_100nm` | Innermost tooth corrugation depth (nm) |
| `_fc` | `_fixed_cav` | Fixed cavity length (no adaptive lengthening) |

## Example

```
Old: layout_80_periods_1_apod_neg_det_5.76nm_avg_wgd_shift_90.0nm_innersize_100nm.fsp
New: layout_N80_A1_D5p76_avg_S90_I100.fsp
```

## Directory structure

Sweep subfolders follow the same convention:

```
layouts/
  S90/              ← formerly shift_90nm/
    layout_N80_A1_D5p76_avg_S90_I100.fsp
results/
  S90/
    result_N80_A1_D5p76_avg_S90_I100.mat
```

## Where abbreviations are generated

- `generate_file_tag()` in `sim_helpers.py` — produces the base tag (`N{periods}[_A{apod}]…`)
- `run_sweep_inner_tooth_size.py` — appends `_S{shift}_I{innersize}`
- `run_sweep_innermost_shift.py` — appends `_S{shift}`
