"""Recompute mode width from a STORED .fsp — no solve, no new physics.

Study dir: runners/lumopt2_design/   |   Created 2026-08-18   |   Job(s): TBD
Purpose: on 2026-08-18 the engine's profile extraction was found to have never
integrated over y (it always returned y-row 0 — see lumopt2_design.profile_line),
so EVERY sigma and FWHM this campaign ever logged is void. The forward .fsp
files still hold the complete field_profile monitor data, so the correct widths
can be recovered by re-reading them — a ~2 min load per case instead of a ~25
min re-solve, and it re-uses the same engine functions the campaign now logs
with, so what this prints IS what a fresh run would log.

This is also the general tool for "what is the real width of design X?" whenever
a stored .fsp exists. Answers, for the designs that reached T ~ 0.96, the
question nobody can currently answer: what was their FWHM?

Dispatch (chain behind the in-flight audit so the .fsp files are not being
rewritten while we read them):
    SBATCH_MEM=160G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=00:55:00 \\
        bash athena/deploy_athena.sh \\
        --lumopt2-design=runners.lumopt2_design.fsp_width \\
        --max-concurrent=1 --after=<audit jobid>
Output -> results/fspw_*/  (+ the corrected profile as .npz, kept forever).
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from bragg_device import lumapi
from runners.lumopt2_design import lumopt2_design as eng

N_SIDE = 100                      # the surrogate every one of these was run at

# (label, substring that identifies the stored forward .fsp)
# APPEND-ONLY: tasks 0-2 were already dispatched as job 134334 against this
# exact ordering, and a pending array task re-reads the sweep list at start
# (CLAUDE.md section 6) — so new cases go at the END, never inserted.
# 3-6 are the sigma-neutral probe rows, i.e. the devices that actually reached
# T ~ 0.96: d+20 T 0.96587 and d+40 T 0.9667 are the two highest-transmission
# designs the program has ever measured, and NOBODY knows their real width.
CASES = [("fspw_origin",  "fwhm_origin"),
         ("fspw_best",    "fwhm_best"),
         ("fspw_noshift", "fwhm_noshift"),
         ("fspw_d020",    "signeut_d020"),
         ("fspw_d040",    "signeut_d040"),
         ("fspw_d060",    "signeut_d060"),
         ("fspw_d080",    "signeut_d080")]

SPEC = eng.CampaignSpec(label=CASES[0][0])   # deploy contract needs a top-level SPEC
N_TASKS = len(CASES)


def find_fsp(tag):
    """The forward .fsp lumopt2 wrote for this case, wherever the job put it."""
    roots = [config.RESULTS_DIR, os.path.dirname(config.RESULTS_DIR), "/work/results"]
    for r in roots:
        hits = sorted(glob.glob(os.path.join(r, "**", f"*{tag}*_files", "fwd_*.fsp"),
                                recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no stored forward .fsp for {tag}")


def main(task_idx=0):
    label, tag = CASES[int(task_idx)]
    fsp = find_fsp(tag)
    out_dir = os.path.join(config.RESULTS_DIR, label)
    os.makedirs(out_dir, exist_ok=True)

    fdtd = lumapi.FDTD(hide=True)                 # user rule: never open a window
    fdtd.load(fsp)
    t2 = fdtd.getresult("FDTD::ports::Port_2", "expansion for port monitor")
    T = np.abs(np.squeeze(t2["S"])) ** 2
    wl = np.squeeze(t2["lambda"]) / eng.NM
    lam_pk, t_pk, _ = eng.measure_peak(wl, T)

    x, I = eng.profile_line(fdtd, lam_pk, N_SIDE)      # y-INTEGRATED, cropped
    fwhm = eng.fwhm_env_of_line(x, I)                  # == project fwhm_m
    sig = eng.sigma_of_line(x, I)
    np.savez_compressed(os.path.join(out_dir, f"{label}_profile.npz"),
                        x_um=x, I=I, lam_pk_nm=lam_pk, fwhm_um=fwhm, sigma_um=sig)
    print(f"[fsp_width {label}] src {os.path.basename(fsp)} | T {t_pk:.5f} "
          f"lam {lam_pk:.3f} nm | FWHM {fwhm:.4f} um  sigma {sig:.4f} um "
          f"| bare-N100 reference 19.2448 um")


if __name__ == "__main__":
    for i, (label, tag) in enumerate(CASES):
        print(f"task {i}: {label:14s} <- *{tag}*_files/fwd_*.fsp")
