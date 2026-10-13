"""Production confirm, TE: the N=98 inverse-designed devices at the CONFORMAL mesher.

Study dir: runners/lumopt2_design/ | Created 2026-10-09 | Job(s): see memory
project_te_inverse_design_lane.md
Purpose: every TE campaign number is a PVA-mesher number. The binding mesher
rule (user 2026-08-24) says a PVA winner is a winner only if it also wins in
conformal, measured as a conformal-vs-conformal PAIR. Three forwards, identical
numerics (TE far-field box 6.8/6.81, conformal variant 0, N=98):
  0 s2_seed  Itai's overshoot seed = IDENTITY GATE vs the stored conformal rows
             63722 (T 0.97498 / λ 1559.860 / Q 7680 / fwhm 19.633) and 164893
             (0.97308 / 1559.867 / 19.630). Also prices the open 0.004 offset
             between the PVA campaign (0.9691) and the PVA anchor (0.9731).
  1 s2_best  TE_S2_N98_T9739   PVA 0.97386 / W 19.620 / λ 1560.423
  2 s1_best  TE_S1_N98_T9711   PVA 0.97112 / W 19.395 / λ 1561.04
             (stored conformal plain S1: 164893 T 0.912 / λ 1559.990 — not re-run)
Reading: (a) conformal gain of each best over its own seed/control; (b) PVA→
conformal offsets (TM memory: λ −5 nm, FWHM +8 %, T −0.008 — TE EXPECTED smaller,
S1 λ-finder measured +0.95 nm). Numbers here are EXPECTED until the rows land.

Dispatch (Athena, one at a time, behind the converged S1 lane so it adds no seats):
    SBATCH_MEM=160G LUMOPT2_QOS=24h_1g LUMOPT2_TIME=06:00:00 \
        bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.prod_confirm_te \
        --array-tasks=0-2%1 --gpu=a100 --after=172603_0
"""
import dataclasses
import json
import os

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import campaign_te_s1 as s1
from runners.lumopt2_design import campaign_te_s2 as s2
from runners.lumopt2_design.best_designs import (TE_S1_N98_T9711, TE_S2_N98_T9739, TE_S1_N98_T9775,
                                                 TE_S2_N98_STEP19)

MESH = "conformal variant 0"

# (label, base spec, design vector, scan centre nm, width nm, points, PVA reference)
ROWS = [
    ("s2_seed", s2.SPEC_V3, s2.SEED,        1559.9, 4.0, 1001, "PVA 0.96909 / 19.636; conformal stored 0.97498 / 1559.860"),
    ("s2_best", s2.SPEC_V3, TE_S2_N98_T9739, 1559.9, 4.0, 1001, "PVA 0.97386 / 19.620 / 1560.423"),
    ("s1_best", s1.SPEC_V3, TE_S1_N98_T9711, 1560.1, 10.0, 501, "PVA 0.97112 / 19.395 / 1561.04"),
    # rows 3-7 (2026-10-12): the FINAL tooth-space devices at conformal AND fixed
    # sim time (S1 50 ps, S2 150 ps, shutoff off; rows 0-2 ran at the 63 ps
    # auto-shutoff, so they are NOT comparable). Each seed has its own control.
    ("s1_seed_fixt",   s1.SPEC_V3, None,            1560.1, 10.0, 501, "PVA 50 ps: seed not re-run at PVA fixed time (T9711 0.97434)"),
    ("s1_t9775_fixt",  s1.SPEC_V3, TE_S1_N98_T9775, 1560.6, 10.0, 501, "PVA 50 ps 0.97752 / 19.468 / 1561.964"),
    ("s2_seed_fixt",   s2.SPEC_V3, s2.SEED,         1559.9, 4.0, 1001, "PVA 150 ps 0.98069 / 19.636"),
    ("s2_t9739_fixt",  s2.SPEC_V3, TE_S2_N98_T9739, 1559.9, 4.0, 1001, "PVA 150 ps 0.98605 / 19.620"),
    ("s2_dir025_fixt", s2.SPEC_V3, TE_S2_N98_T9739 + 0.25 * (TE_S2_N98_STEP19 - TE_S2_N98_T9739),
                                                    1559.9, 4.0, 1001, "PVA 150 ps 0.98635 / 19.613"),
    ("s2_seed_n110",  s2.SPEC_V3, s2.SEED,         1559.8, 4.0, 1001, "conformal N98 150 ps 0.98939"),
    ("s2_seed_n124",  s2.SPEC_V3, s2.SEED,         1559.8, 4.0, 1001, "conformal N98 150 ps 0.98939"),
    ("s2_t9739_n110", s2.SPEC_V3, TE_S2_N98_T9739, 1559.7, 4.0, 1001, "conformal N98 150 ps 0.98995"),
    ("s2_t9739_n124", s2.SPEC_V3, TE_S2_N98_T9739, 1559.7, 4.0, 1001, "conformal N98 150 ps 0.98995"),
]
SIM_TIME_PS = {"s1": 50.0, "s2": 150.0}      # converged (s2_simtime_ladder 174406 / 174823)
# rows 8-11 (2026-10-13, user: "confirm the overshoot is really ~2e5 at -3 dB"): LENGTH LADDER.
# Q3dB = 0.293 Q_i assumes Q_i does not change with length; a direct -3 dB device (N~165,
# Q~4e5) needs ~8 ns of ring-down. N = 110 / 124 test the assumption at affordable cost.
# Run length = ~12 amplitude decay times 2Q_L/omega (EXPECTED Q_L ~15k / ~33k from the bulk
# kappa 0.0572/um; 150 ps = 12 tau at N=98 was converged to 7e-5).
LADDER = {"s2_seed_n110": (110, 300.0), "s2_seed_n124": (124, 700.0),
          "s2_t9739_n110": (110, 300.0), "s2_t9739_n124": (124, 700.0)}
N_TASKS = len(ROWS)


def spec_of(idx):
    name, base, p, lam, width, npts, _ = ROWS[idx]
    extra = {}
    if name.endswith("_fixt"):
        extra = dict(sim_time_ps=SIM_TIME_PS[name[:2]], auto_shutoff_min=1e-12)
    if name in LADDER:
        n_side, t_ps = LADDER[name]
        extra = dict(n_periods_side=n_side, sim_time_ps=t_ps, auto_shutoff_min=1e-12)
    spec = dataclasses.replace(
        base, label=f"prodconf_te_{name}", mesh_refinement=MESH,
        seed_override=(None if p is None else tuple(float(v) for v in p)), **extra,
        scan_center_nm=lam, scan_width_nm=width, n_wl_points=npts,
        fwhm0_um=None, wg_anchor={"softw": 19.0, "fwhm": 19.0},   # forward only; no band
        two_kl_floor=1.0)
    assert spec.n_periods_side == LADDER.get(name, (98,))[0]
    return spec


SPEC = spec_of(0)      # deploy contract: build_sweep_list needs a top-level SPEC


def main(task_idx=0):
    spec = spec_of(int(task_idx))
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    log = os.path.join(out_dir, f"{spec.label}_evals.jsonl")
    row = eng.last_logged_row(log)    # preemption resume: one forward, never twice
    if row is None:
        row = eng.run_canary(spec, out_dir)
    print(f"[prodconf_te {task_idx} {spec.label}] conformal N=98: "
          f"T {row.get('t_pk')}  λ {row.get('lam_pk_nm')}  Q {row.get('q_loaded')}  "
          f"Q_i {row.get('q_i')}  FWHM {row.get('fwhm_env_um')} µm  | PVA ref: {ROWS[int(task_idx)][6]}")
