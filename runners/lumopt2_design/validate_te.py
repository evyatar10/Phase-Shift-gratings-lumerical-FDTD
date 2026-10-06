"""TE-lane gate ladder (validate_c325 pattern) for seeds S1 (plain) and S2
(overshoot): every TM-fitted constant is re-measured on the TE device before
either campaign may start.

Study dir: runners/lumopt2_design/  |  Created 2026-10-04  |  Job(s): TBD
Tasks 0-9 = S1 (campaign_te_s1), 10-19 = S2 (campaign_te_s2); k = task % 10:
  k=0  λ-FINDER canary: one forward, 8 nm window / 161 pts (50 pm) centred on
       the stored conformal λ + 4 nm (covers PVA shifts 0..+8 nm). ★168240 lesson:
       the region DFT monitor stores every cell × every λ — 801 pts blew the 40 GB
       A100 (88.8 GiB). Peak located to ~25 pm; k=1 refines at the production grid. Reads lam_pk at
       PVA numerics → paste into the campaign file's SCAN_CENTER_NM.
  k=1  PRODUCTION-WINDOW canary at the measured centre → the seed's own
       fwhm_env (= the width SPEC), softW anchor, T0, Q0. These anchors are
       taken at the campaign's window/points (a canary at another window is a
       different numerics identity — validate_c325 task 45 lesson).
  k=2,3 NOISE FLOOR: the k=1 forward repeated with ONE outer free tooth's corr
       moved +0.5 nm (tooth 60 / tooth 59 — sub-cell, physics-negligible). The
       |ΔT| spread vs k=1 is the TE jitter floor behind wgp_fom_slack
       (CLAUDE.md §2: measure the floor inside the study; TM's 0.0015 is VOID).
  k=4  C_port Re + FD (naive adjoint, adj_phase_fix OFF; 6 indices across
       corr / avg / shift / wcav; central FD = 12 forwards + fwd + adj).
       ★TE-specific readout: residual PER CLASS after the global C fit —
       corr/avg move E-NORMAL walls (Johnson/Kottke: the hard case), shifts
       move E-parallel end faces. corr/avg >10 % while shift passes ⇒ the
       `bc_patch` route, not a bigger C.
  k=5  C_port Im (adjoint-only at adj_fix=(0,1), same detune point).
  k=6  C_field FD + Re (wg_pure=True: J = −softW; 3 indices; production
       field-adjoint config: fieldregion tiles=4 on GPU).
  k=7  C_field Im (adjoint-only, adj_fix_field=(0,1)).
  k=8  PIPELINE SMOKE: the full campaign spec on an N=70 surrogate, 2 iterates,
       honesty guards disarmed (two_kl_floor=0, fwhm0=None). Same region size
       as production ⇒ an honest peak-RSS reading for the 256G lane. NOT
       physics. Semantic PASS = ns2 law executed + optstate sidecar written.
  k=9  TOY: 3 iterates of the full spec (falsification readout: rho_T,
       |Δλ|, |ΔW| per iterate; verdict to the user BEFORE the 96 h campaign).
Order per seed: 0 → 1 → (2,3) → (4,5) ∥ (6,7) → fit → 8 → 9 → campaign.
Gradient tasks run at `te_point` (seed + the probed shifts at 5 nm + cavity
width +10 nm), NOT at the TM detune (60 shifts × 20 nm = 2.4 µm of cavity and
outside the 15 nm shift trust bounds — review 2026-10-04). After the toy, copy
`validate_te_sN/<label>_toy_evals.jsonl` + `_optstate.json` into the campaign
out_dir under the campaign label (CLAUDE.md §6: never re-derive iterates).
S2 reuses S1's C values and VERIFIES them (k=4..7 on S2 are the C-recipe's
mandatory second operating point); S2 gets its own fit only if ≤10 % fails.
Fit helper: `python -m runners.lumopt2_design.validate_te fit <fd> <re> <im>`
prints the ENGINE tuple (a, b) = (s cosφ, −s sinφ) for adj_fix_re/im.

Lane sizing (skill items 29/30): k=4 → 14 concurrent sims ≈ 50 GB scratch,
≥10 h, LUMOPT2_QOS=12h_4g; k=6 → fwd + adj + 6 legs ≈ 8 h, 250G; k=1/2/3/5/7
≈ 1-2 h each, 160G; k=8 ≈ 2 h, 256G; k=9 ≈ 6-8 h, 256G.
Dispatch: SBATCH_MEM=<see above> bash athena/deploy_athena.sh \
    --lumopt2-design=runners.lumopt2_design.validate_te --array-tasks=<i>
"""
import dataclasses
import json
import os
import sys

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import campaign_te_s1 as S1M
from runners.lumopt2_design import campaign_te_s2 as S2M

SEEDS = {0: S1M, 1: S2M}
N_TASKS = 40      # 0-19 baseline engine (S1, S2) | 20-39 v3 engine (S1: 28 smoke, 29 toy; S2: 38, 39)

# Provisional anchors for canaries ONLY (make_project needs a wg_anchor when
# width_grad is on): the stored CONFORMAL widths of each family (job 164893).
# Never quoted as the spec — the k=1 row replaces them.
PROV_FWHM_UM = {0: 19.13, 1: 19.63}
EXPECTED_PVA_SHIFT_NM = 4.0     # TM measured +5.2 nm PVA vs conformal; 8 nm window covers 0..+8
STORED_CONFORMAL_LAM = {0: 1559.990, 1: 1559.867}   # 164893 rows, box 6.8/6.81


def _indices(spec, n=6):
    L = eng.layout(spec.n_free)
    six = {"corr_1": L.SL_CORR.start, "corr_30": L.SL_CORR.start + 29,
           "avg_1": L.SL_AVG.start, "shift_1": L.SL_SHIFT.start,
           "shift_30": L.SL_SHIFT.start + 29, "wcav": L.I_CAV}
    three = {"corr_1": L.SL_CORR.start, "shift_1": L.SL_SHIFT.start, "wcav": L.I_CAV}
    d = six if n == 6 else three
    return list(d.values()), list(d.keys())


def te_point(spec):
    """C-recipe operating point for the TE lane (replaces detune_params):
    the seed with ONLY the probed shifts moved to 5 nm (inside the 0..15 nm
    shift trust bounds with room for the ±2/±4 nm FD legs; cavity +20 nm, not
    the TM detune's +2.4 µm), cavity width +10 nm, and S2's zero-corrugation
    tooth 1 lifted to 5 nm so its FD leg stays ≥ corr_min. Shifts off their
    0-bound is the whole point (skill item 8)."""
    L = eng.layout(spec.n_free)
    p = eng.seed_params(spec).copy()
    p[L.SL_SHIFT.start] = p[L.SL_SHIFT.start + 29] = 5.0
    p[L.I_CAV] += 10.0
    p[L.SL_CORR.start] = max(p[L.SL_CORR.start], spec.corr_min_nm + 5.0)
    return p


def check_points():
    """Zero-GPU: the k=4..7 operating point and its FD legs sit inside bounds
    for both seeds (the review-found blocker class). Called by gate_te_local.
    Uses the seed window when SCAN_CENTER is not yet measured."""
    ok = True
    for i, mod in SEEDS.items():
        for n, pert in ((6, 2.0), (3, 4.0)):
            spec = dataclasses.replace(_forward_spec(mod, i, f"_chk{n}"),
                                       width_grad=(n == 3), wg_pure=(n == 3))
            idx, _ = _indices(spec, n)
            try:
                eng._assert_fd_legs_in_bounds(spec, te_point(spec), idx, pert)
                print(f"  {spec.label}: point + FD legs ±{pert} nm in bounds ({n} indices)")
            except AssertionError as e:
                print(f"  {spec.label}: FAIL {e}")
                ok = False
    return ok


def _centre(mod):
    assert mod.SCAN_CENTER_NM is not None, \
        f"{mod.SPEC.label}: run task k=0 first and paste SCAN_CENTER_NM"
    return mod.SCAN_CENTER_NM


def _forward_spec(mod, seed_i, suffix, **kw):
    """Forward-only spec: optimizer off, width_grad on (anchors need softW),
    provisional anchor, no trips."""
    prov = PROV_FWHM_UM[seed_i]
    return dataclasses.replace(
        mod.SPEC, label=mod.SPEC.label + suffix,
        wg_project=False, wgp_ns2=False, wg_lam_chain=False,
        wgp_target_um=prov, fwhm0_um=prov,       # make_project precondition;
        wg_anchor={"softw": prov, "fwhm": prov},  # a canary only logs (trip caught)
        adj_phase_fix=False, **kw)


def _port_spec(mod, seed_i, suffix, **kw):
    """Port-only gradient spec for the C_port recipe (no width entry)."""
    return dataclasses.replace(
        _forward_spec(mod, seed_i, suffix, scan_center_nm=_centre(mod)),
        width_grad=False, **kw)


def _field_spec(mod, seed_i, suffix, **kw):
    """wg_pure spec for the C_field recipe at the production field-adjoint
    config (fieldregion, tiles=4, GPU); coarse λ grid is the same physics for
    J = −softW (validate_c325 _w_spec note) and spares ~25 GB."""
    return dataclasses.replace(
        _forward_spec(mod, seed_i, suffix, scan_center_nm=_centre(mod)),
        width_grad=True, wg_pure=True, n_wl_points=151, **kw)


def _campaign_spec(mod, suffix, v3=False, **kw):
    for name in ("SCAN_CENTER_NM", "FWHM0_UM", "SOFTW0_UM", "ADJ_FIX_PORT",
                 "ADJ_FIX_FIELD"):
        assert getattr(mod, name) is not None, f"{mod.SPEC.label}: {name} unmeasured"
    base = mod.SPEC_V3 if v3 else mod.SPEC
    fields = dict(label=base.label + suffix, scan_center_nm=mod.SCAN_CENTER_NM,
                  fwhm0_um=mod.FWHM0_UM, wgp_target_um=mod.FWHM0_UM,
                  wg_anchor={"softw": mod.SOFTW0_UM, "fwhm": mod.FWHM0_UM},
                  adj_fix_re=mod.ADJ_FIX_PORT[0], adj_fix_im=mod.ADJ_FIX_PORT[1])
    fields.update(kw)                     # the smoke overrides fwhm0_um=None (review #4)
    spec = dataclasses.replace(base, **fields)
    spec.adj_fix_field_re, spec.adj_fix_field_im = mod.ADJ_FIX_FIELD
    return spec


def fit_port(fd, re, im, labels):
    """C-recipe fit by EXACT linear least squares FD ≈ a·Re + b·Im (the engine
    applies exactly a·RE + b·IM), full precision, with the design-matrix
    condition number and leave-one-out held-out errors.
    ★2026-10-05 (GPT review A7, verified): the old 0.05° phase grid + 4-decimal
    tuple cannot resolve cancellation-dominated entries (|Z| up to 1300× the
    gradient needs ~0.004° for 10 %) — it produced a spurious −12.3 % on
    shift_30 where the exact fit gives −0.1 %."""
    fd, re, im = (np.asarray(v, float) for v in (fd, re, im))
    A = np.stack([re, im], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, fd, rcond=None)
    model = a * re + b * im
    print(f"C fit (exact LSQ): a {a:.6f} b {b:+.6f} | s {np.hypot(a, b):.5f} "
          f"phi {np.degrees(np.arctan2(-b, a)):+.4f} deg | cond {np.linalg.cond(A):.0f} | "
          f"vector resid {np.linalg.norm(model - fd) / np.linalg.norm(fd):.3%}")
    print(f"ENGINE TUPLE (adj_fix_re, adj_fix_im) = ({a:.6f}, {b:+.6f})")
    worst = {}
    for i, (lab, m, f) in enumerate(zip(labels, model, fd)):
        res = m / f - 1.0 if f != 0 else float("nan")
        k = [j for j in range(len(fd)) if j != i]            # leave-one-out
        (a1, b1), *_ = np.linalg.lstsq(A[k], fd[k], rcond=None)
        loo = (a1 * re[i] + b1 * im[i]) / f - 1.0 if f != 0 else float("nan")
        cls = lab.split("_")[0]
        worst[cls] = max(worst.get(cls, 0.0), abs(res))
        print(f"  {lab:>9s}: FD {f:+.5e} model {m:+.5e} resid {res:+.1%} "
              f"held-out {loo:+.1%} |Z|/|FD| {np.hypot(re[i], im[i]) / abs(f):.0f} "
              f"sign {'OK' if np.sign(m) == np.sign(f) else 'FLIP'}")
    print("  per-class worst |resid|: " + ", ".join(f"{k} {v:.1%}" for k, v in worst.items()))
    bad = [k for k, v in worst.items() if v > 0.10]
    if bad:
        print(f"  ★FAIL >10 % in class(es) {bad}. If corr/avg fail while shift "
              f"passes → E-normal boundary error (TE hard case): try bc_patch, "
              f"not a bigger C.")
    return (float(a), float(b)), worst


def _upgrade_markers(out_dir, label, require_reuse):
    """Engagement audit of the 2026-10-04 optimizer upgrades on a hardware run:
    count each feature's own log marker in <label>_proj.jsonl (U3 mac and U4
    range_alpha fire every iterate; U2 broyden only on a REUSED row; U1 only on
    a noise-level reject). A toy must have reused at least once or U2 was never
    exercised on hardware."""
    rows = [json.loads(l) for l in open(os.path.join(out_dir, f"{label}_proj.jsonl"),
                                        encoding="utf-8")]
    n = dict(iterates=len(rows),
             mac=sum(1 for r in rows if r.get("mac") is not None),
             range_alpha=sum(1 for r in rows if r.get("range_alpha") is not None),
             reused=sum(1 for r in rows if r.get("gw_reused")),
             broyden=sum(1 for r in rows if r.get("broyden_rel") is not None
                         or r.get("broyden_skipped")),
             noise_reject=sum(1 for r in rows if r.get("noise_reject")),
             rejects=sum(1 for r in rows if "retry" in str(r.get("phase", ""))))
    n["v3_mode"] = sum(1 for r in rows if r.get("v3_mode"))
    n["v3_radius"] = sum(1 for r in rows if r.get("v3_radius"))
    ev = [json.loads(l) for l in open(os.path.join(out_dir, f"{label}_evals.jsonl"),
                                      encoding="utf-8")]
    n["cw_measured"] = sum(1 for r in ev if r.get("cw_um_per_nm") is not None)
    n["cw_errors"] = sum(1 for r in ev if r.get("cw_error"))
    print(f"[upgrade markers {label}] {n}")
    if "_v3" in label and (n["v3_mode"] == 0 or n["cw_measured"] == 0):
        raise RuntimeError("V3 run: no v3 step logged or dW/dlambda never measured "
                           "(see cw_error in the evals log) — v3 engine not exercised")
    if require_reuse and n["reused"] == 0:
        raise RuntimeError("TOY: width row never REUSED — U2 (Broyden) unexercised; "
                           "do not start the campaign on this evidence")


def main(task_idx):
    seed_i, k = divmod(int(task_idx), 10)
    v3 = seed_i >= 2                     # tasks 20-39: the v3 step engine
    seed_i %= 2
    if v3 and k not in (1, 8, 9):
        raise ValueError(f"task {task_idx}: tasks 20-39 exist only for k=1 (gate-point "
                         f"forward), k=8 (smoke) and k=9 (toy)")
    mod = SEEDS[seed_i]
    out_dir = os.path.join(config.RESULTS_DIR,
                           f"validate_te_s{seed_i + 1}" + ("_v3" if v3 else ""))
    os.makedirs(out_dir, exist_ok=True)
    L = eng.layout(mod.SPEC.n_free)

    if v3 and k == 1:                            # tasks 21 / 31: forward at the GATE point
        # ★WHY (2026-10-05, jobs 168910/169360): the S2 field-adjoint FD came back
        # [5.9e-4, 9.9e-4, 1.03e-2] at ±4 nm and [-2.0e-4, 1.6e-4, 1.00e-2] at ±1 nm —
        # not step-converged, and no complex C fits either. The gates sample softW at
        # SCAN_CENTER (the SEED's resonance) while te_point() detunes the device
        # (shifts 5 nm, cavity +10 nm). On S2 the line is 0.20 nm wide; if the gate
        # point's resonance sits a linewidth away, both gates measured an
        # off-resonance profile. This one forward measures where it is.
        spec = _forward_spec(mod, seed_i, "_gatepoint", scan_center_nm=_centre(mod),
                             scan_width_nm=8.0, n_wl_points=(161, 321)[seed_i],
                             recenter_nm=100.0, seed_override=tuple(te_point(mod.SPEC)))
        row = eng.run_canary(spec, out_dir)
        print(f"[te-s{seed_i+1} gate point] λ {row.get('lam_pk_nm')} vs gate λ "
              f"{_centre(mod)} | linewidth {row.get('fwhm_nm')} nm | T {row.get('t_pk')} "
              f"W {row.get('fwhm_env_um')} softW {row.get('softw_um')}")
        return
    if k == 0:                                   # λ finder, wide window
        spec = _forward_spec(mod, seed_i, "_lamfind",
                             scan_center_nm=STORED_CONFORMAL_LAM[seed_i] + EXPECTED_PVA_SHIFT_NM,
                             # 50 pm grid for S1 (Q~1700: 18 pts/linewidth); 25 pm for
                             # S2 (Q~7680, 0.2 nm linewidth: 8 pts). 161/321 λ × 40-48
                             # MB ≈ 6.5 / 15 GB on the shrunken region.
                             scan_width_nm=8.0, n_wl_points=(161, 321)[seed_i],
                             recenter_nm=100.0)
        row = eng.run_canary(spec, out_dir)
        print(f"[te-s{seed_i+1} k0] PASTE SCAN_CENTER_NM = {row.get('lam_pk_nm')}  "
              f"(T {row.get('t_pk')}, conformal stored {STORED_CONFORMAL_LAM[seed_i]})")
    elif k == 1:                                 # production-window anchors
        spec = _forward_spec(mod, seed_i, "_anchor", scan_center_nm=_centre(mod))
        row = eng.run_canary(spec, out_dir)
        print(f"[te-s{seed_i+1} k1] PASTE FWHM0_UM = {row.get('fwhm_env_um')}  "
              f"SOFTW0_UM = {row.get('softw_um')}  | T0 {row.get('t_pk')} "
              f"λ {row.get('lam_pk_nm')} Q {row.get('q_loaded')} "
              f"sigma {row.get('sigma_um')} | spectral pts/FWHM check: "
              f"{row.get('fwhm_nm')} nm at {spec.scan_width_nm/(spec.n_wl_points-1):.4f} nm grid")
    elif k in (2, 3):                            # noise floor
        p = eng.seed_params(mod.SPEC).copy()
        tooth = L.SL_CORR.start + (59 if k == 2 else 58)      # outermost free teeth
        p[tooth] += 0.5
        spec = _forward_spec(mod, seed_i, f"_nf{k-1}", scan_center_nm=_centre(mod),
                             seed_override=tuple(p))
        row = eng.run_canary(spec, out_dir)
        print(f"[te-s{seed_i+1} k{k}] noise-floor point: T {row.get('t_pk')} "
              f"λ {row.get('lam_pk_nm')} W {row.get('fwhm_env_um')} — compare with "
              f"the k=1 row; |ΔT| spread = the TE jitter floor → wgp_fom_slack")
    elif k == 4:                                 # C_port Re + FD
        idx, labels = _indices(mod.SPEC, 6)
        spec = _port_spec(mod, seed_i, "_cport_fd")
        print(f"[te-s{seed_i+1} k4] indices {dict(zip(labels, idx))} — FD FIRST in the "
              f"printout; Re = the adjoint half. Pair with k=5 (Im) in fit_port.")
        eng.run_validate_gradient(spec, out_dir, idx, perturbation=2.0,
                                  point=te_point(spec))
    elif k == 5:                                 # C_port Im
        idx, labels = _indices(mod.SPEC, 6)
        spec = _port_spec(mod, seed_i, "_cport_im", adj_phase_fix=True,
                          adj_fix_re=0.0, adj_fix_im=1.0)
        eng.run_adjoint_only(spec, out_dir, idx, point=te_point(spec))
        print(f"[te-s{seed_i+1} k5] Im{{Z}} for {labels}")
    elif k == 6:                                 # C_field FD + Re
        idx, labels = _indices(mod.SPEC, 3)
        # ★FD step per seed (2026-10-05, MEASURED 168910/168911): on S2 the ±4 nm
        # legs gave Re/FD = 1.42 / 1.27 / 1.08 (corr_1 / shift_1 / wcav) and no
        # single C fits. Hypothesis: S2's linewidth is 0.20 nm (S1 1.0) and softW
        # is sampled at a FIXED λ, so a 4 nm move of a cavity-adjacent tooth shifts
        # the resonance by a sizeable fraction of the line → nonlinear FD. The S2
        # rerun uses ±1 nm (fresh label; the ±4 nm vectors are kept in memory/logs).
        pert = (4.0, 1.0)[seed_i]
        spec = _field_spec(mod, seed_i, ("_cfield_fd", "_cfield_fd_p1")[seed_i])
        print(f"[te-s{seed_i+1} k6] indices {dict(zip(labels, idx))} — FD FIRST "
              f"(±{pert} nm); Re = adjoint half at C_field=(1,0). Pair with k=7.")
        eng.run_validate_gradient(spec, out_dir, idx, perturbation=pert,
                                  point=te_point(spec))
    elif k == 7:                                 # C_field Im
        idx, labels = _indices(mod.SPEC, 3)
        spec = _field_spec(mod, seed_i, "_cfield_im",
                           adj_fix_field_re=0.0, adj_fix_field_im=1.0)
        eng.run_adjoint_only(spec, out_dir, idx, point=te_point(spec))
        print(f"[te-s{seed_i+1} k7] Im{{Z_field}} for {labels}")
    elif k == 8:                                 # pipeline smoke
        spec = _campaign_spec(mod, "_smoke", v3=v3, n_periods_side=70,
                              two_kl_floor=0.0, fwhm0_um=None, max_iter=2,
                              max_feval=4)
        best = eng.run_campaign(spec, out_dir)
        rows = [json.loads(l) for l in open(os.path.join(
            out_dir, f"{spec.label}_proj.jsonl"), encoding="utf-8")]
        ran = [r for r in rows if r.get("rho_T") is not None and not r.get("ns2_fallback")]
        side = os.path.exists(os.path.join(out_dir, f"{spec.label}_optstate.json"))
        print(f"[te-s{seed_i+1} smoke] best_fom {best['fom']:.5f} | ns2 ran on "
              f"{len(ran)}/{len(rows)} iterates | sidecar {side} — NOT physics (N=70)")
        if not ran or not side:
            raise RuntimeError("TE SMOKE FAIL: ns2 law never executed or no sidecar")
        _upgrade_markers(out_dir, spec.label, require_reuse=False)
    elif k == 9:                                 # toy
        # 4 iterates (not 3): the width-row REUSE first becomes eligible on the
        # 2nd accepted iterate and the Broyden update (U2) only fires on a reused
        # row, so 3 iterates could structurally never exercise it (CLAUDE.md §5
        # engagement-conditions corollary). U1 (noise reject) cannot be forced —
        # its counter is reported, not asserted.
        spec = _campaign_spec(mod, "_toy", v3=v3, max_iter=4, max_feval=8)
        best = eng.run_campaign(spec, out_dir)
        print(f"[te-s{seed_i+1} toy] best_fom {best['fom']:.5f} — read "
              f"{spec.label}_proj.jsonl: rho_T / rLam_nm / dW per iterate")
        _upgrade_markers(out_dir, spec.label, require_reuse=True)
    else:
        raise ValueError(task_idx)


# Deploy requires a top-level SPEC (build_sweep_list); tasks never use it.
SPEC = S1M.SPEC

if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "fit":
        vec = [json.loads(a) for a in sys.argv[2:5]]
        labels = sys.argv[5].split(",") if len(sys.argv) > 5 else \
            [f"p{i}" for i in range(len(vec[0]))]
        fit_port(*vec, labels)
    else:
        main(int(sys.argv[1]))
