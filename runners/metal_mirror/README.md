# Metal-mirror studies (lateral reflector, "option B")

Follow-up of the scatterer/Green's program (`runners/scatterers/`, stage-H
FINDINGS): can a near-ideal mirror parallel to the guide recycle/suppress the
grazing in-plane leak (ux = 0.980, theta ~ 11.5 deg) that every dielectric
reflector failed to touch? Mechanism + design rationale in the runner header
(`metal_mirror_dscan.py`) and `results_from_athena/scat_h_retrocomb/FINDINGS.md`.

## Studies

| runner | what it does |
|---|---|
| `metal_mirror_dscan.py` | PEC film (200 nm thick, core height, +/-41.3 um long, mirrored +/-y), standoff scan d = 3.0..5.7 um over one 2.68 um interference cycle + in-study control. 6 tasks. |

## Run it

```bash
# queue must be EMPTY of other --option3 arrays (shared sweep_list.txt!)
ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
    --option3 --spec=runners.metal_mirror.metal_mirror_dscan --max-concurrent=3
# results:
bash athena/deploy_athena.sh --results-no-fsp     # -> results_from_athena/metal_mirror_dscan/
```

Read the verdict from ports: `resonance_transmission` (T), loss, lambda, Q per d
vs the control row — NOT from the side far-field monitor (it sits behind the
mirror and is shadowed). T(d) oscillating at ~2.68 um period above the 0.0018
jitter floor = mirror couples to the leak; flat = this film geometry is closed.

## Knobs (top of the runner)

- `D_SCAN_NM` — standoff list (keep d >= 2.5 um near-field floor; PML assert guards the top).
- `MIRROR_MATERIAL` — `"PEC (Perfect Electrical Conductor)"` (ideal) or a real
  metal, e.g. `"Al (Aluminium) - Palik"` (then check the material FIT over the
  window in Material Explorer; fine mesh over the film is the accurate option).
- `FILM_HEIGHT_NM` — `None` = core height 350 nm; set e.g. `6000.0` for the
  tall-wall "can ANY mirror help" variant.
- Material machinery: `scatterer_material` field (SweepSpec/ScattererConfig) —
  added 2026-07-18; any named Lumerical DB material works for scatterer objects.
