"""Dump T(λ) + the mode profile from an ALREADY-SOLVED lumopt2 .fsp.

Study dir: runners/lumopt2_design/  |  Created 2026-08-23  |  Job(s): TBD
Why this exists: the eval log stores only the SCALARS (t_pk, λ_pk, fwhm) —
the spectrum array is never persisted, so a T(λ) figure had no source. This
re-reads a solved forward .fsp (no re-solve, ~minutes) using the SAME
convention as the engine callback: |S|² from the Port_2 expansion, and
sim_helpers' envelope for the profile. Point FSP at any solved forward file.

Dispatch: SBATCH_MEM=64G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=00:25:00 \\
  bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.extract_spectrum
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng

# the retrimmed best, as re-evaluated by campaign 136248 at iteration 0.
# config.RESULTS_DIR is PER-STUDY (.../results/<study>/results), so walk up
# two levels to the shared results root before crossing into the campaign's
# tree — job 136303 died 'Error: line -1' on the naive relative path.
_ROOT = os.path.dirname(os.path.dirname(config.RESULTS_DIR))
# Use the STABLE canary file (job 136077 t16 = mx_retrim: the retrimmed best
# at the pitch-locked mesh, T 0.95941 / λ 1566.377 / FWHM 17.8530). The live
# campaign's fwd_default_iter0.fsp is REWRITTEN every iteration — job 136304
# hit it mid-write and found no file. Candidates are tried in order.
# ORDER MATTERS: a .fsp alone carries no results — the solver data lives in
# the sibling <name>/<name>_output.h5, and the roll-cleaner deletes those for
# all but the newest two per directory. The live campaign's iter-0 pair is
# intact (job 136305 failed on the cleaned mx_retrim canary: the port had no
# 'expansion for port monitor'). So prefer files whose _output.h5 still
# exists, and say so loudly if none do.
_CANDIDATES = [
    os.path.join(_ROOT, "campaign_v2_projection", "results",
                 "lumopt2_v2proj_px", "lumopt2_v2proj_px_files",
                 "fwd_default_iter0.fsp"),
    os.path.join(_ROOT, "validate_c325", "results", "lumopt2_val_c325",
                 "lumopt2_val_c325_mx_retrim_files", "fwd_default.fsp"),
]


def _has_results(fsp):
    stem = fsp[:-4]
    return os.path.exists(os.path.join(stem, os.path.basename(stem) + "_output.h5"))
FSP = next((p for p in _CANDIDATES if os.path.exists(p) and _has_results(p)),
           None)
OUT = os.path.join(config.RESULTS_DIR, "spectrum_best.npz")
# the dispatcher requires a top-level SPEC; this task never optimizes, it
# only re-reads a solved file, so the spec is a label carrier only.
SPEC = eng.CampaignSpec(label="extract_spectrum")
N_TASKS = 1


def main(task_idx=0):
    from bragg_device import lumapi          # the repo's own lumapi shim
    if FSP is None:
        raise SystemExit("[extract_spectrum] no candidate .fsp still has its "
                         "_output.h5 (roll-cleaner). Candidates:\n  " +
                         "\n  ".join(_CANDIDATES))
    print(f"[extract_spectrum] reading {FSP}")
    with lumapi.FDTD(hide=True) as fdtd:
        fdtd.load(FSP)
        r = fdtd.getresult("FDTD::ports::Port_2", "expansion for port monitor")
        T = np.abs(np.squeeze(r["S"])) ** 2
        wl = np.squeeze(r["lambda"]) / eng.NM
        r1 = fdtd.getresult("FDTD::ports::Port_1", "expansion for port monitor")
        R = np.abs(np.squeeze(r1["S"])) ** 2
        lam_pk, t_pk, fwhm_nm = eng.measure_peak(wl, T)
        px, pI = eng.profile_line(fdtd, lam_pk, SPEC.n_periods_side)
    np.savez_compressed(OUT, wl_nm=wl, T=T, R=R, x_um=px, I=pI,
                        lam_pk_nm=lam_pk, t_pk=t_pk, fwhm_nm=fwhm_nm)
    print(f"[extract_spectrum] wrote {OUT}: {len(wl)} λ points, "
          f"peak T {t_pk:.5f} at {lam_pk:.4f} nm, spectral FWHM {fwhm_nm:.4f} nm, "
          f"mode FWHM {eng.fwhm_env_of_line(px, pI):.4f} µm")


if __name__ == "__main__":
    print("reads a solved .fsp; dispatch via deploy")
