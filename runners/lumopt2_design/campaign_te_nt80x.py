"""Gradient-only lumopt2 lane on Itai's Nt80x 15.5-um overshoot (the lab device).

Study dir: runners/lumopt2_design/ | Created 2026-10-20 | Job(s): TBD
Purpose: ONE measured adjoint gradient (gT, gW, gLam) of the Nt80x at its own
numerics, for route A: project it (lambda, W held) and line-search with cards on
the conformal short-run path. NOT a campaign (full campaign NO-GO, plan 2026-10-20).

Device = runners/sweeps/farfield_sph_15um ROUND C_NT80: pitch 514, N 96/side,
bulk 750/1250 (corr 500, avg 1000), cavity 950.3, 81 apodized teeth (d=81 is
760.1/1227.7, not bulk -> 81 free; the outer 15 frozen teeth are drawn at bulk).
Three teeth sit on Itai's clamp 600.1/1800.1 (corr 1200, inside the 1250 bound).
Numerics = the TE lane standing defaults (campaign_te_n98 LANE_DEFAULTS +
SEED_DEFAULTS["s2"]), except: FIXED 340 ps (= C_NT80 PROD 265046, ~2 amplitude
decay times at Q_L 9.9e4), 1.0 nm / 401 pts (2.5 pm; 501 lambda segfaulted at 98
free teeth), lambda trust 4 pm (1/4 of the 16 pm line), box 6.8 x 6.81 um.

★The width twin is a single-lambda monitor pinned at the scan centre on the
first eval. On a 16 pm line an un-measured PVA offset (S2: +0.60 nm vs conformal)
would put iterate 0's gW tens of linewidths off resonance. So main() first runs
ONE forward-only LOCATE (3 nm window, same 340 ps) and takes from it lambda_pk
(-> scan centre + exact twin pin via wg_track_resonance) and FWHM0/softW0 (the
lane's own width band). Budget: locate + iterate-0 forward (fields needed for
the adjoints) + one trial step = MAX_FEVAL 3; gradients of every accepted point
are saved as <label>_grads_it*.npz (raw gT, gW, gW_eff, gLam).
The logged t_pk is the RAW port T of a truncated run (~T_corr / gamma^2,
gamma^2 = 1.34 at 340 ps): EXPECTED ~0.49 for corrected 0.658 (265046).

Dispatch (Athena):
  SBATCH_MEM=256G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=24:00:00 \
      bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_te_nt80x --gpu=h200
"""
import dataclasses
import os

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import persistence
from runners.lumopt2_design import campaign_te_s2 as s2
from runners.lumopt2_design.campaign_te_n98 import LANE_DEFAULTS, SEED_DEFAULTS
from runners.sweeps import farfield_sph_15um as ff

N_SIDE, N_FREE = ff.N_SIDE, 81            # 96/side, 15 bulk periods
SIM_TIME_PS = 340.0                       # C_NT80 PROD run length (265046)
os.environ["TM_SIM_TIME_PS"] = str(SIM_TIME_PS)   # ff sets 150 at import; keep builder + tags consistent
SCAN_CENTER_NM = 1620.07                  # conformal 1619.4705 MEASURED (265046) + ~0.6 PVA offset (S2, EXPECTED)
LOCATE_WINDOW_NM = 3.0                    # locate forward: covers offsets -0.9..+2.1 nm vs conformal
FWHM_PROV_UM = 16.15                      # conformal W (265046) - provisional, replaced by the locate row
MAX_FEVAL = 3
KAPPA = 0.0572 * 500.0 / 494.2           # DERIVED: S2 bulk kappa scaled by corr (as itai_nt80_seed)
assert len(ff.NT80X_NARROW_NM) == len(ff.NT80X_WIDE_NM) == N_FREE


def _spec_base():
    lane = dict(LANE_DEFAULTS, **SEED_DEFAULTS["s2"])
    spec = dataclasses.replace(
        s2.SPEC_V3, label="lumopt2_te_nt80x_grad" + os.environ.get("NT80X_LABEL_SUFFIX", ""),   # suffix: twin lanes on other partitions (own results dir)
        pitch_nm=ff.PITCH_ITAI_NM, n_periods_side=N_SIDE, n_free=N_FREE,
        corr0_nm=ff.BULK_WIDE_NM - ff.BULK_NARROW_NM,
        avg_w_nm=(ff.BULK_WIDE_NM + ff.BULK_NARROW_NM) / 2.0,
        kappa_per_um=KAPPA, seed_override=None, corr_seed_nm=None, avg_seed_nm=None,
        corr_min_nm=0.0, corr_max_nm=1250.0, avg_bounds_nm=(925.0, 1225.0),
        wcav_bounds_nm=(850.0, 1250.0),
        region_dx_nm=ff.PITCH_ITAI_NM / eng.CELLS_PER_PITCH,      # 51.4 nm pitch-locked
        region_y_half_nm=1150.0,          # widest tooth at bounds (1225+625)/2 = 925 nm
        box_y_um=6.8, box_z_mult=3.99,    # z = 0.35 + 3.99*1.620 = 6.81 um
        scan_center_nm=SCAN_CENTER_NM, scan_width_nm=1.0, n_wl_points=401, recenter_nm=0.2,
        fwhm0_um=FWHM_PROV_UM, wgp_target_um=FWHM_PROV_UM,
        wg_anchor={"softw": FWHM_PROV_UM, "fwhm": FWHM_PROV_UM},
        adj_fix_re=1.0, adj_fix_im=0.0,   # port C = (1, 0) as S2 (gain-ratio gate)
        max_feval=MAX_FEVAL, best_metric="fom",
        wgp_d_halfrange_nm=lane["metric"], trust_nm=dict(lane["trust"]),
        wgp_v3_perp_glam=lane["perp"], wgp_v3_dlam_nm=0.004)
    spec.adj_fix_field_re, spec.adj_fix_field_im = s2.ADJ_FIX_FIELD   # S2's own fit (Nt80x unfitted)
    spec.adj_fix_rad_re, spec.adj_fix_rad_im = lane["adj_fix_rad"]
    spec.wgp_gw_shift_um_per_nm = lane["gw_shift"]
    spec.sim_time_ps, spec.auto_shutoff_min = SIM_TIME_PS, lane["auto_shutoff_min"]
    assert spec.wg_track_resonance and spec.wgp_v3 and spec.bare
    return spec


def seed_vector(spec):
    """Itai's 81 apodized teeth as (corr, avg) + zero shifts + inert comb + cavity."""
    nar = np.asarray(ff.NT80X_NARROW_NM, float)
    wid = np.asarray(ff.NT80X_WIDE_NM, float)
    L = eng.layout(N_FREE)
    p = eng.seed_params(spec)
    p[L.SL_CORR] = wid - nar
    p[L.SL_AVG] = (wid + nar) / 2.0
    p[L.SL_SHIFT] = 0.0
    p[L.I_CAV] = ff.CAVITY_W_NM
    return tuple(float(v) for v in p)


_base = _spec_base()
SPEC = dataclasses.replace(_base, seed_override=seed_vector(_base))
LOCATE_SPEC = dataclasses.replace(SPEC, scan_width_nm=LOCATE_WINDOW_NM)
N_TASKS = 1


def main(task_idx=0):
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    log = os.path.join(out_dir, f"{SPEC.label}_evals.jsonl")
    rows = persistence.read_rows(log)
    loc = rows[0] if rows else eng.run_canary(LOCATE_SPEC, out_dir)   # row 0 = the locate forward
    lam, t_pk = loc.get("lam_pk_nm"), loc.get("t_pk")
    w, sw = loc.get("fwhm_env_um"), loc.get("softw_um")
    print(f"[nt80x locate] lam_pk {lam} t_pk {t_pk} W {w} softW {sw} Q_L {loc.get('q_loaded')}", flush=True)
    # CLAUDE.md §2 sanity: a dead or off-window device must stop here, not get a gradient
    half = LOCATE_WINDOW_NM / 2.0
    assert lam is not None and abs(lam - SCAN_CENTER_NM) < half - 0.05, "locate: no resonance in window"
    assert t_pk is not None and t_pk > 0.1, f"locate: dead device (t_pk {t_pk})"
    assert w and sw and 14.0 < w < 18.5, f"locate: width {w} off-family (conformal 16.15)"
    spec = dataclasses.replace(SPEC, scan_center_nm=round(float(lam), 2),
                               fwhm0_um=float(w), wgp_target_um=float(w),
                               wg_anchor={"softw": float(sw), "fwhm": float(w)})
    spec._wg_lam_track = float(lam)       # exact twin pin for iterate 0 (func reads it per call)
    best = eng.run_campaign(spec, out_dir)
    print(f"[te-nt80x {spec.label}] done: best_fom {best['fom']:.5f}; gradients in {spec.label}_grads_it*.npz")
