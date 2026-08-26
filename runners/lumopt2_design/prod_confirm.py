"""Production confirm: the inverse-designed device at the REGULAR project numerics.

Study dir: runners/lumopt2_design/   |   Created 2026-08-25   |   Job(s): TBD
Purpose (user 2026-08-25, and HANDOFF "WHAT THE USER FORGOT" item 1): 0.96361 is
an OPTIMIZER number — surrogate N=100, PVA mesher, campaign box. This study
re-measures it with the mesher every other study in this project uses
("conformal variant 0") at the q3db family box/window, so the design can finally
be compared with the regular devices on equal terms.

★It also discharges the user's binding MESHER RULE (2026-08-24): a PVA winner is
only a winner if it also wins in conformal, and validation is a conformal-vs-
conformal PAIR (final AND its own initial), never a cross-mesher subtraction.
Hence rows 1-3 below: origin, see-saw rung, best — one batch, identical numerics.

WAVE 1 (this file) — 4 forwards at N=100/side, ~1.5 h each:
  0 bare    corr-325, no comb    IDENTITY GATE vs the stored regular-pipeline
                                 anchor (IGUM 51736, NOT re-run: T 0.9104 /
                                 λ 1559.006 / FWHM 19.2448). Bridges "lumopt2
                                 built the scene" to "the regular device".
  1 origin  uniform corr-325 + winner comb   PVA: T 0.90120 / FWHM 18.3460
  2 seesaw  d090 rung (inner-8 −90, outer-17 +68)   PVA: T 0.93836 / FWHM 18.331
  3 best    BEST_T9636                       PVA: T 0.96361 / FWHM 18.35309
  Reading it: (a) does the PVA ORDERING 0.901 < 0.938 < 0.964 survive? (b) what
  is the conformal T-gain of the design over its own initial? (c) FWHM: PVA→
  conformal is ×1.049 EXPECTED, so ~19.25 µm — the ~20 µm class the q3db family
  works at. Every number below is EXPECTED until wave 1 measures it.

WAVE 2 (separate dispatch, rows added here once wave 1 reports) — the q3db
operating point: ladder n_periods_side until peak T crosses −3 dB, then Q at
that N. Stored crossings to compare against, same numerics, NOT re-run:
  ctrl corr-325 N=165  T 0.4906 (−3.09 dB)  Q 13930   (job 130458)
  winner comb  N=169   −3.04 dB             Q 16203   (comb_q3db wave 2)
N is deliberately NOT guessed here: the crossing rides the measured T(N=100),
and wave 1 is what measures it at this mesher.

NUMERICS — the q3db family line exactly (comb_q3db / tm_nladder_c325): box
y 8.0 µm, z-mult 5.42 (z ≈ 8.8 µm), window 20 nm @ 1559.5 / 4001 pts, z-sym,
TM h350, pitch 516.83, W800, n 1.97/1.444. The one lumopt2 residue that cannot
be removed is the optimization-region mesh override, kept pitch-locked
(dx = 516.83/10 nm) per the HANDOFF's cheap-closer recipe — task 0 is what
prices it. free_comb=False on every row: the v2 campaigns froze the comb, so
the stored vectors' comb slots are inert and replay_params resets them to the
winner lattice (Λ 531 / dx 401 / r 80 / d 1900) — that IS the measured device.

Dispatch:
    SBATCH_MEM=160G LUMOPT2_TIME=06:00:00 \
        bash igum/deploy_igum.sh \
        --lumopt2-design=runners.lumopt2_design.prod_confirm --max-concurrent=4
"""

import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9636
from runners.lumopt2_design.seesaw_ladder import rung_params

BOX_Y_UM, BOX_Z_MULT = 8.0, 5.42      # q3db family box (_common defaults + y8)
SCAN_CENTER_NM, SCAN_WIDTH_NM, N_WL = 1559.5, 20.0, 4001   # family window

BASE = eng.CampaignSpec(
    label="prodconf",
    n_periods_side=100,
    box_y_um=BOX_Y_UM, box_z_mult=BOX_Z_MULT,
    scan_center_nm=SCAN_CENTER_NM, scan_width_nm=SCAN_WIDTH_NM, n_wl_points=N_WL,
    mesh_refinement="conformal variant 0",
    region_dx_nm=eng.DX_PITCHLOCK_NM,
    free_comb=False,                  # v2 froze the comb — stored comb slots are inert
)

# (label, bare, design vector or None for the uniform origin, PVA reference)
ROWS = [
    ("bare",   True,  None,                 "no comb — vs stored 0.9104/1559.006/19.2448"),
    ("origin", False, None,                 "PVA 0.90120 / 18.3460"),
    ("seesaw", False, rung_params(90, 68),  "PVA 0.93836 / 18.331"),
    ("best",   False, BEST_T9636,           "PVA 0.96361 / 18.35309"),
]
N_TASKS = len(ROWS)
SPEC = BASE          # deploy contract: build_sweep_list needs a top-level SPEC


def spec_of(idx):
    name, bare, p, _ = ROWS[idx]
    spec = dataclasses.replace(BASE, label=f"prodconf_{name}", bare=bare)
    if p is not None:
        spec.seed_override = tuple(eng.replay_params(spec, np.asarray(p, float)))
    return spec


def main(task_idx=0):
    idx = int(task_idx)
    spec = spec_of(idx)
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    log = os.path.join(out_dir, f"{spec.label}_evals.jsonl")
    if os.path.exists(log):           # preemption resume: one forward, never twice
        row = json.loads(open(log).readlines()[-1])
    else:
        row = eng.run_canary(spec, out_dir)
    print(f"[prodconf {idx} {spec.label}] conformal N=100 q3db-box: "
          f"T {row.get('t_pk')}  λ {row.get('lam_pk_nm')}  Q {row.get('q_loaded')}  "
          f"Q_i {row.get('q_i')}  FWHM {row.get('fwhm_env_um')} µm  "
          f"| PVA ref: {ROWS[idx][3]}")


if __name__ == "__main__":
    for i, (name, bare, p, ref) in enumerate(ROWS):
        s = spec_of(i)
        c = eng.seed_params(s)[eng.SL_CORR]
        print(f"  task {i}: {s.label:16s} bare={bare!s:5s} mcorr={c.mean():7.2f} "
              f"e={2 * eng.seed_params(s)[eng.SL_SHIFT].sum():6.1f} nm "
              f"wcav={eng.seed_params(s)[eng.I_CAV]:7.2f} | {ref}")
    print(f"mesh={BASE.mesh_refinement} region dx={BASE.region_dx_nm:.3f} nm "
          f"box y{BOX_Y_UM}/zmult{BOX_Z_MULT} window {SCAN_WIDTH_NM} nm @"
          f"{SCAN_CENTER_NM}/{N_WL} pts")
