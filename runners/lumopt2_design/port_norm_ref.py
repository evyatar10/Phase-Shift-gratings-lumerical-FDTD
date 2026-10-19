"""Port-normalization fix for prod_confirm_te: straight-guide references + full port spectra.

Study dir: runners/lumopt2_design/ | Created 2026-10-19 | Job(s): see memory
Why: the new Itai Nt80 overshoot (prod_confirm_te row 16, 250781_16) read T+R = 1.00097.
Root cause (subagent, MEASURED from the raw port-plane DFT fields of 250781_16): the logged
T = |S21|^2 = T_out(Port_2) / T_in(Port_1), and Lumerical's source-port T_in is contaminated
by the reflected wave (Port_1's monitor sits one cell inside the source plane), so it moves
with the device (0.942-0.954 across 13 stored rows at nominally identical source/ports).
Net Poynting flux on the same fields gives T ~0.993, T+R ~0.995 for row 16.
Fix: normalize by a STRAIGHT 1000 nm guide (= the feed width) at IDENTICAL numerics:
    T = T_out2(device) / T_out2(ref),   R = |S11|^2 * T_in(device) / T_out2(ref).
The ref keeps each row's corr0/avg/pitch/region_dx (so the mesh override, dy = width_narrow/13
and the port grids are the device's), only the drawn teeth + cavity become 1000 nm. All 98
teeth are free (n_free=98) so the frozen bulk is straight too; the lumopt2 region's dx equals
the global pitch-locked dx, so the longer region does not move a grid line.
Tasks 0-1 dump the full port spectra (T_in, T_out, S of both ports) of already-SOLVED rows
(no solve); tasks 2-5 solve the four references (20 ps: the pulse clears the 116 um guide in
~2 ps; 1001 or 501 points exactly as the row they normalize).

Dispatch (Athena):
    SBATCH_MEM=160G LUMOPT2_QOS=24h_1g LUMOPT2_TIME=02:00:00 bash athena/deploy_athena.sh \
        --lumopt2-design=runners.lumopt2_design.port_norm_ref --gpu=a100 --max-concurrent=4
"""
import dataclasses
import os

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import prod_confirm as pc_tm
from runners.lumopt2_design import prod_confirm_te as pc

ROWS = {"nt80": 16, "s2": 5, "s2idx": 17, "s1": 3}          # prod_confirm_te row per family
FEED_NM = 1000.0                                             # width_port_m (simulation_config)
REF_SIM_TIME_PS = 20.0
# Dumps only where the solved _output.h5 survives (s2/s1 were roll-cleaned; their logged rows
# already carry tout_at_modal_peak, source_port_tin_at_peak and r_pk, enough to renormalize).
TASKS = ([("dump", "nt80"), ("dump", "s2idx")]                  # task 1: after 256556_17
         + [("ref", k) for k in ("nt80", "s2", "s2idx", "s1")])
# tasks 6-7 (user 2026-10-19, "check plain TM too"): no stored TM row logs T_out/T_in (that logging
# is from 2026-10-07; the TM rows are August), so plain TM = prod_confirm row 0 ("bare", corr-325
# N=100, conformal, y8.0/z-mult 5.42, 20 nm/4001 pts, production auto-shutoff) re-run at its EXACT
# stored numerics under a new label, plus its straight reference.
TASKS += [("fwd", "tm"), ("ref", "tm"), ("fwd", "tm_best")]   # tm_best = row 3, BEST_T9636 (comb)
N_TASKS = len(TASKS)
_PC_ROOT = os.path.join(os.path.dirname(os.path.dirname(config.RESULTS_DIR)),
                        "prod_confirm_te", "results")


def row_spec(key):
    if key in ("tm", "tm_best"):                 # prod_confirm rows 0 (bare) / 3 (BEST_T9636)
        return pc_tm.spec_of(0 if key == "tm" else 3)
    return pc.spec_of(ROWS[key])


def ref_spec(key):
    """Row spec with every tooth + cavity at the feed width (a straight guide)."""
    row = row_spec(key)
    n = row.n_periods_side
    base = dataclasses.replace(
        row, label=f"portref_{key}", n_free=n, seed_override=None,
        corr_seed_nm=(0.0,) * n, avg_seed_nm=(FEED_NM,) * n,
        corr_min_nm=0.0, corr_max_nm=750.0,              # bounds only; S1's floor is >0
        avg_bounds_nm=(900.0, 1100.0), wcav_bounds_nm=(850.0, 1250.0), trust_nm=None,
        sim_time_ps=REF_SIM_TIME_PS, auto_shutoff_min=1e-12, two_kl_floor=-1.0,
        # TM rows record 4001 points: with the n_free=100 region that is 223 GB of GPU memory
        # (258000_7 died on a 40 GB A100). A straight guide's launched power and leak are flat
        # in lambda (TE refs: launched constant to 3e-5), so 401 points + interpolation suffice.
        n_wl_points=401 if key == "tm" else row.n_wl_points)
    p = eng.seed_params(base)
    p[eng.layout(n).I_CAV] = FEED_NM
    return dataclasses.replace(base, seed_override=tuple(float(v) for v in p))


def dump_ports(fsp, out_npz):
    """Full port spectra of a solved .fsp (the engine's own getresult calls)."""
    from bragg_device import lumapi
    with lumapi.FDTD(hide=True) as fdtd:
        fdtd.load(fsp)
        d = {}
        for k in (1, 2):
            r = fdtd.getresult(f"FDTD::ports::Port_{k}", "expansion for port monitor")
            d["lam_nm"] = np.squeeze(r["lambda"]) / eng.NM
            for q in ("S", "T_in", "T_out", "T_total", "T_net"):
                if q in r:
                    d[f"p{k}_{q}"] = np.squeeze(r[q])
    np.savez_compressed(out_npz, **d)
    print(f"[port_norm_ref] {fsp} -> {out_npz}: keys {sorted(d)}")


def main(task_idx=0):
    kind, key = TASKS[int(task_idx)]
    out_dir = os.path.join(config.RESULTS_DIR, f"portref_{key}")
    os.makedirs(out_dir, exist_ok=True)
    if kind == "dump":
        label = pc.spec_of(ROWS[key]).label
        fsp = os.path.join(_PC_ROOT, label, f"{label}_files", "fwd_default.fsp")
        dump_ports(fsp, os.path.join(out_dir, f"{label}_ports.npz"))
        return
    spec = (dataclasses.replace(row_spec(key), label=f"portref_{key}_device") if kind == "fwd"
            else ref_spec(key))
    try:
        eng.run_canary(spec, out_dir)        # forward solve via the engine's own path
    except Exception as e:                   # a flat spectrum may upset the peak logger;
        print(f"[port_norm_ref] canary logger: {e!r}")   # the solved .fsp is what we need
    dump_ports(os.path.join(out_dir, f"{spec.label}_files", "fwd_default.fsp"),
               os.path.join(out_dir, f"{spec.label}_ports.npz"))


SPEC = ref_spec("nt80")      # deploy contract: build_sweep_list needs a top-level SPEC


# ── Applying a reference (local, no Lumerical). One reference serves EVERY device at the same
# numerics (pitch, corr0/avg -> mesh override, window/points, box, polarization): reuse it. ──

def incident(ref_npz, lam_nm):
    """True launched mode power (fraction of source power) at lam_nm, from a reference dump."""
    r = np.load(ref_npz)
    lam, t = r["lam_nm"], np.abs(r["p2_T_out"])
    order = np.argsort(lam)
    lam_nm = np.asarray(lam_nm, float)
    assert np.all((lam_nm >= lam.min()) & (lam_nm <= lam.max())), "lambda outside the reference window"
    assert np.all(np.isfinite(t)) and t.min() > 0.5, "reference launched power not sane"
    return np.interp(lam_nm, lam[order], t[order])


def corrected_row(row, ref_npz):
    """(lambda_nm, T) at the peak of Port_2's clean T_out for an eval row that has no stored
    spectrum: T = tout_pk / launched power. R cannot be corrected from a row (it needs the
    complex S11 -> corrected_spectrum); a logged r_pk below ~0.006 is inside the port's
    measurement floor. FWHM/Q are not re-derived here."""
    lam = row["lam_tout_pk_nm"]
    return lam, row["tout_pk"] / float(incident(ref_npz, lam))


def corrected_spectrum(dev_npz, ref_npz):
    """(lam_nm, T, R) over the whole window from a device dump and its reference dump.

    The port's forward/backward split leaks each wave into the other (MEASURED: a straight
    guide reads |S11|^2 = 0.0063, i.e. |eps| = 0.079, -22 dB directivity). At Port_1:
    b_m = b + eps*a and a_m = a + conj(eps)*b, with eps = the reference's complex S11
    (b = 0 there). So r = (S11 - eps)/(1 - conj(eps)*S11) -- a VNA-style directivity
    correction. VALIDATED on 250781_16 with eps taken from the reference, nothing fitted: stop-band T+R
    0.9946-0.9965 (flat, std 3e-4), the predicted Port_1 T_in reproduces the measured
    0.913..1.094 swing to rms 0.005, and the resonance T 0.99451 matches an independent net
    Poynting-flux reading of the raw port fields (0.9947). The other phase conventions fail
    (stop-band T+R 0.73-1.36). T needs no such term: Port_2 sees no backward wave, so its
    T_out is clean; only its denominator (the launched power) comes from the reference."""
    d, r = np.load(dev_npz), np.load(ref_npz)
    lam_r = r["lam_nm"]
    assert d["lam_nm"].min() >= lam_r.min() - 1e-9 and d["lam_nm"].max() <= lam_r.max() + 1e-9,         "device window outside the reference window"
    o = np.argsort(lam_r)                          # eps is smooth in lambda (straight guide)
    eps = (np.interp(d["lam_nm"], lam_r[o], r["p1_S"].real[o])
           + 1j * np.interp(d["lam_nm"], lam_r[o], r["p1_S"].imag[o]))
    s11 = d["p1_S"]
    R = np.abs((s11 - eps) / (1.0 - np.conj(eps) * s11)) ** 2
    return d["lam_nm"], np.abs(d["p2_T_out"]) / incident(ref_npz, d["lam_nm"]), R
