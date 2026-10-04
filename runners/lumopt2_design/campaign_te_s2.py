"""TE lane, seed S2 — Itai's re-optimized Nt60 "overshoot" apodization (the
job-63722 geometry: pitch 491.06, bulk 752.9/1247.1, cavity 951.4, 20 µm mode),
run through the same d1 two-constraint engine as campaign_te_s1.

Study dir: runners/lumopt2_design/  |  Created 2026-10-04  |  Job(s): TBD
Purpose: the second TE seed (user 2026-10-04): a physics-informed start in a
DIFFERENT basin from the uniform grating. Its 60 non-bulk teeth (d=1..60, from
`runners/sweeps/itai_hh_nt60w20.APOD_*_NM`) are exactly the 60 free periods per
side; the frozen outer teeth are his bulk (corr 494.2 / avg 1000). No shift in
his design (seed shifts = 0); no comb (bare).

Seed → 296-vector: corr_d = wide_d − narrow_d, avg_d = (wide_d + narrow_d)/2,
shift 0, inert comb slots at the engine seed (sliver-frozen, absent from the
scene), cavity width 951.4 nm. Tooth d=1 is his zero-corrugation tooth
(951.4/951.4) ⇒ corr_min_nm = 0 and that one parameter sits ON its lower bound
(trust clamp keeps the plain box there — skill item 22).

N: bulk κ = 0.0572 /µm (DERIVED from 63722→63752, d ln Q_c/dN =
0.0561/period at corr 494.2 on W1000; the apodized core is weaker) — the
engine's two_kappa_L on the actual seed vector decides N (gate_te_local prints
it): N = 98/side = his length (user decision 2026-10-04) → 2κL 5.11. Stored
N=98 row: Q_L 7680 → linewidth ~0.20 nm ⇒ window 2 nm @ 4 pm (501 pts) = 51 pts
across it (margin if PVA raises Q); recenter 0.6 nm (< half-window). Ring-down τ ≈ 6 ps, fine.

Stored anchors for the FAMILY (conformal mesher — NOT the PVA numbers the
canary will read; never cross-quote): N=98 λ 1559.8597 / T 0.97498 / Q_L 7680 /
fwhm_m 19.633 (results_from_igum/itai_hh_nt60w20_summary.csv, job 63722);
far-field row 164893 reproduces it (1559.867 / 0.97308 / 19.630).

Same gate order as S1 (validate_te tasks 10-19: 10 λ-finder, 11 anchors,
12-13 noise, 14-15 C_port, 16-17 C_field, 18 smoke, 19 toy). C_port/C_field:
S1's values are VERIFIED here at ≤10 % per class (the C-recipe's second point)
before being adopted; if they fail, S2 gets its own fit. main() refuses to run
until the measured constants are filled in.
"""
import dataclasses
import os

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_te_s1 import SPEC as S1
from runners.sweeps.itai_hh_nt60w20 import (APOD_NARROW_NM, APOD_WIDE_NM,
                                            BULK_NARROW_NM, BULK_WIDE_NM,
                                            CAVITY_W_NM, PITCH_NM as S2_PITCH_NM)

# ── MEASURED by validate_te, pasted here (None = not yet measured) ──────────
SCAN_CENTER_NM = 1560.464    # MEASURED task 10 (job 168375_10, PVA): λ_pk 1560.4642; conformal 1559.867 (+0.60 nm) | T 0.9645 Q 7570 W 19.64
FWHM0_UM       = 19.636      # MEASURED task 11 (168530_11, 2 nm/501): fwhm_env 19.6360; T 0.97309 Q 7694 λ 1560.407 — task 10
SOFTW0_UM      = 19.558      # MEASURED task 11: softw_adj_um (twin sample; raw-line 19.609) — task 10
ADJ_FIX_PORT   = None        # S1's fit, VERIFIED on S2 by tasks 14+15 (or S2's own)
ADJ_FIX_FIELD  = None        # S1's fit, VERIFIED on S2 by tasks 16+17 (or S2's own)
FOM_SLACK      = 1.5e-3      # EXPECTED until task 12 measures S2's floor

N_SIDE     = 98              # Itai's length (user 2026-10-04); 2κL = 5.11 on the seed vector
N_FREE     = 60
CORR0_NM   = round(BULK_WIDE_NM - BULK_NARROW_NM, 1)      # 494.2
AVG_W_NM   = round((BULK_WIDE_NM + BULK_NARROW_NM) / 2, 1)  # 1000.0
KAPPA_BULK = 0.0572          # /µm (DERIVED, see docstring)


def seed_vector():
    """His 60 apodized teeth as (corr, avg) + zero shifts + inert comb + cavity."""
    nar = np.asarray(APOD_NARROW_NM[:N_FREE], float)
    wid = np.asarray(APOD_WIDE_NM[:N_FREE], float)
    base = dataclasses.replace(S1, n_free=N_FREE, corr0_nm=CORR0_NM,
                               avg_w_nm=AVG_W_NM, seed_override=None,
                               corr_seed_nm=None, avg_seed_nm=None)
    p = eng.seed_params(base)                 # uniform bulk + comb lattice
    L = eng.layout(N_FREE)
    p[L.SL_CORR] = wid - nar
    p[L.SL_AVG] = (wid + nar) / 2.0
    p[L.SL_SHIFT] = 0.0
    p[L.I_CAV] = CAVITY_W_NM
    return tuple(float(v) for v in p)


SEED = seed_vector()

SPEC = dataclasses.replace(
    S1,
    label="lumopt2_te_s2",
    pitch_nm=S2_PITCH_NM,                # 491.06
    corr0_nm=CORR0_NM, avg_w_nm=AVG_W_NM,
    kappa_per_um=KAPPA_BULK,
    n_free=N_FREE, n_periods_side=N_SIDE,
    seed_override=SEED,
    corr_min_nm=0.0, corr_max_nm=750.0,      # seed spans 0..634 nm
    avg_bounds_nm=(900.0, 1100.0),           # seed avg 951..1030
    wcav_bounds_nm=(850.0, 1250.0),          # seed 951.4
    region_dx_nm=S2_PITCH_NM / eng.CELLS_PER_PITCH,   # 49.106 pitch-locked
    region_y_half_nm=1250.0,             # (1100+750/2)/2 = 737 nm + 500 margin; 501 λ ≈ 24 GB
    scan_center_nm=SCAN_CENTER_NM or 1563.0,
    # MEASURED fwhm 0.206 nm → FOM window ±0.52 nm inside ±1.0; recenter 0.4 keeps the
    # window unclipped (audit 2026-10-04). 4 pm grid = 51 pts across the line.
    scan_width_nm=2.0, n_wl_points=501, recenter_nm=0.4,
    fwhm0_um=FWHM0_UM, wgp_target_um=FWHM0_UM, wg_anchor=None,
    # avg bounds are 4× wider than S1's ⇒ 16× the D-metric weight (item 21):
    # trust-clamp avg to ±25 so the avg block carries S1's weight.
    trust_nm={"shift": 15.0, "avg": 25.0},
    wgp_fom_slack=FOM_SLACK,
    two_kl_floor=None,                   # back to the 3.5 rule (S1's 3.3 is S1-specific); 5.11 here
    wg_src_tiles=4,                      # 2·(98·0.491+0.123)+2 = 98.5 µm / 49.1 nm = 2005 cells → 501/tile
)

N_TASKS = 1


def main(task_idx=0):
    for name, val in (("SCAN_CENTER_NM", SCAN_CENTER_NM), ("FWHM0_UM", FWHM0_UM),
                      ("SOFTW0_UM", SOFTW0_UM), ("ADJ_FIX_PORT", ADJ_FIX_PORT),
                      ("ADJ_FIX_FIELD", ADJ_FIX_FIELD)):
        assert val is not None, f"{name} not measured — run validate_te first"
    spec = dataclasses.replace(
        SPEC, scan_center_nm=SCAN_CENTER_NM, fwhm0_um=FWHM0_UM,
        wgp_target_um=FWHM0_UM, wg_anchor={"softw": SOFTW0_UM, "fwhm": FWHM0_UM},
        adj_fix_re=ADJ_FIX_PORT[0], adj_fix_im=ADJ_FIX_PORT[1])
    spec.adj_fix_field_re, spec.adj_fix_field_im = ADJ_FIX_FIELD
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    best = eng.run_campaign(spec, out_dir)
    print(f"[te-s2] done: best_fom {best['fom']:.5f} — read "
          f"{spec.label}_proj.jsonl for rho_T/cap_nm/rLam_nm")
