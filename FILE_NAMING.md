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
| `_M125` | `_innersize_125nm` | Center modulation depth (nm); shown only when use_apod and value differs from default 100 |
| `_fc` | `_fixed_cav` | Fixed cavity length (no adaptive lengthening) |

## Example

```
Old: layout_80_periods_1_apod_neg_det_5.76nm_avg_wgd_shift_90.0nm_innersize_125nm.fsp
New: layout_N80_A1_M125_D5p76_avg_S90.fsp
```

## Where abbreviations are generated

- `generate_file_tag()` in `sim_helpers.py` — produces the full tag for all
  sweep dimensions (periods, apod count, mod depth, detuning, shift, etc.).
  Sweep runners no longer append per-study suffixes; everything is one place.
