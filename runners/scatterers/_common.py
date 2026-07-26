"""Shared knobs + base configs for the scatterer response-matrix (Green's) program.

Study dir: runners/scatterers/   |   Created 2026-07-12   |   Jobs: (fill in per stage)
Purpose: measure the complex far-field response of one weak scatterer at each of
N candidate positions, then choose OFFLINE the COMBINATION of positions whose summed
responses best anti-phase-cancel the device's own radiation leak (max recycling),
and validate that combination in FDTD.

Device: TM corr-400 standalone (pitch 516.83 nm, corr 400 nm, height 350 nm,
n_core 1.97 / n_clad 1.444). Converged baseline (jobs 116854/116870):
T = 0.886, loss = 0.110, Q ~ 1320, lambda_res ~ 1558.5 nm, box y 6.8 / z 8.8 um.

RUNBOOK — stages run one after the other; serialize per CLAUDE.md section 6
(check `bash athena/deploy_athena.sh --status` before every dispatch):

  0 (optional) re-pick RADIUS_NM / Y0_NM:
      bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_stage0_scan
  A baseline (prelim finds lambda_res -> sidecar -> lambda-locked complex-far-field run):
      PRELIM_TIME=00:30:00 bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_a_baseline
      download, then:  python runners/scatterers/solve_response_matrix.py gates
      GATE: resonance in window, complex keys present, E2 == |E_c|^2.
  B noise floor + linearity + (r, y) size gate (~30 tasks):
      bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_b_gates
      download, then:  python runners/scatterers/solve_response_matrix.py gates
      GATE: |response| >> floor AND superposition holds, else STOP. The verdict also
      RECOMMENDS the biggest radius / closest y that stays linear -> paste the winning
      RADIUS_NM / Y0_NM back into this file before stage C.
  C response acquisition (control + N_POSITIONS singles; smoke with N_POSITIONS=5 first):
      bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_c_response
  D offline solve (no FDTD, no license; local or Athena login node):
      python runners/scatterers/solve_response_matrix.py solve
      -> prints the LS ceiling FIRST (go/no-go), then the greedy combination and the
         best periodic comb; writes greens_report.json with ready-to-paste lists.
  C2 (2D extension) second-row acquisition at ROW2_Y_NM (control + row-2 singles +
      3 cross-row/in-row gate pairs; row-1 columns are REUSED from stage C):
      bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_c2_row2
      download, then:  python runners/scatterers/solve_response_matrix.py solve2
      -> cross-row gate verdict FIRST (kill criterion), then the 2-row bounded
         ceiling (compare vs the 34.9% single-row bound) and the best (x, y)
         combinations; writes greens2_report.json with ready-to-paste lists.
  E validation (paste CANDIDATE_ARRAYS + MIN_SPACING_NM into scat_e_validate.py):
      bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_e_validate
      download, then:  python runners/scatterers/solve_response_matrix.py validate
      figure:          matlab_plotting/plot_scatterer_greens.m
  G apodized-device probe (known pair [0,270]@700 on apod n=5/n=10 vs own controls;
      ports-only, NO lambda sidecar — apod shifts lambda_res):
      ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_g_apod --max-concurrent=3
  J metal-mirror d-scan ("option B" after the stage-H null) lives in its OWN dir:
      runners/metal_mirror/metal_mirror_dscan.py (reuses build_ff_base + the
      stage-H sidecar; see runners/metal_mirror/README.md)

Any device/geometry change ==> rerun stage A (the lambda sidecar goes stale).

Helper module only — no SPEC here, so the deploy menu grep never picks it up.
"""

from runners.sweeps._tm_base import build_base


# ── User knobs (edit HERE, nowhere else) ────────────────────────────────────────
N_POSITIONS = 97          # stage-C candidate sites; ODD, symmetric about x=0.
                          # 97 fits ONE dispatch under the QOS 100-task cap
                          # (1 prelim + 1 control + 97 = 99). Set 5 for a smoke run.
SPAN_UM     = 12.96       # full span of the candidate grid (um) -> step 135 nm ~ pitch/4
RADIUS_NM   = 80.0        # DEFAULT radius; stage B's (r, y) gate picks the final value
Y0_NM       = 700.0       # DEFAULT lateral offset; closest clean row for r<=100 (wide-tooth
                          # edge at 500 nm + r + 100 nm oxide gap = 2 mesh cells). Stage B
                          # picks the final value. Prior FDTD line was 1000 nm.
X_REF_NM    = 270.0       # stage-B reference site (grid point nearest analytic best x~250 nm)
JITTER_NM   = 25.0        # half a dx=50 nm mesh cell (noise-floor template, CLAUDE.md §2)
SEPARATIONS_NM = [135.0, 270.0, 405.0, 540.0, 1080.0]   # stage-B pair spacings (Born test)

# Stage-B (r, y) linearity grid — "biggest scatterer at the closest y that stays
# linear AND beats the noise floor". Each combo runs the cavity-straddling pair
# GATE_PAIR_X_NM + each of its two singles alone (3 sims per combo).
R_GATE_LIST_NM = [60.0, 80.0, 100.0, 125.0, 150.0]
# Brackets BOTH failure edges: small r risks noise-limited (response ~ r^2; r=60 is
# ~1.2 mesh cells at dx=50), big r risks nonlinear (pair coupling ~ r^4; the 2026-07
# x-scan's mean-offset diagnostic puts the boundary between 100 and 150). Radii whose
# tooth-gap rule y_min(r) exceeds a Y row simply skip that row (125/150 skip y=700).
Y_GATE_LIST_NM = [700.0, 800.0, 900.0, 1000.0]
GATE_PAIR_X_NM = [-135.0, 135.0]          # one scatterer LEFT, one RIGHT of the
                                          # cavity center — strongest drive, worst
                                          # case for linearity; spacing 270 nm
PAIR_GATE_SPACING_NM = 270.0     # pair spacing (= GATE_PAIR span; also stage-E floor)
TOOTH_EDGE_NM  = 500.0           # wide-tooth edge distance from the axis (measured geom)
GAP_MIN_NM     = 100.0           # minimum oxide gap tooth->pillar (= 2 mesh cells at dx=50)

# Far-field monitor x-extent. The old 30 um default (corr-300/TE era) clips the
# corr-400 leak: spatial envelope FWHM 15.5 um => only ~74% of the radiating region
# inside, and the grazing lobe (ux~0.99) exits past +/-15 um. 60 um captures ~93%
# of the envelope and center-launched grazing rays.
FF_X_SPAN_UM = 60.0

MESH_MODE   = "optimization"   # dx=50 nm; flip to "accurate" only if stage B says noise-limited
BOX_Y_UM    = 6.8              # converged transverse box (jobs 116854/116870)
BOX_Z_MULT  = 5.42             # -> z span ~ 8.8 um at 1558.5 nm

# ── Row-2 knobs (2D-grid extension, stage C2 — user decisions 2026-07-14) ───────
# One additional row of r=RADIUS_NM pairs at a larger |y|, to test whether a second
# row adds NEW far-field basis directions (different u_y pattern) beyond row 1's
# measured ceiling (binary ~31%, radius-weighted 34.9%). Gates for r=80 at this y
# were already MEASURED in stage B (gates_report.json: pair err 3.1%, pull 30 pm,
# response 0.43x of y=700).
ROW2_Y_NM       = 900.0        # second-row lateral offset (row 1 = Y0_NM)
ROW2_N_ALIGNED  = 33           # aligned sites, 135 nm step, symmetric about x=0
ROW2_SPAN_UM    = 4.32         # aligned-grid full span -> +/-2.16 um (4x the row-1
                               # useful half-width; row-1 landscape is ~0 beyond that)
ROW2_STAGGER_X_NM = [-472.5, -337.5, -202.5, -67.5, 67.5, 202.5, 337.5, 472.5]
                               # half-step-offset sites near the cavity ("half-phase"
                               # stagger test + off-grid interpolation check)

# One shared lambda_res sidecar for the whole program, written by stage A's prelim
# task and read by every lambda-locked task in stages A/B/C/E (no {idx} template:
# all tasks lock to the SAME baseline resonance).
LAMBDA_SIDECAR = "/work/results/scat_greens_lambda_res.json"


def y_min_nm(r_nm):
    """Closest clean lateral offset for a pillar of radius r: tooth edge + gap + r,
    rounded UP to the 50 nm mesh grid."""
    import math
    return 50.0 * math.ceil((TOOTH_EDGE_NM + GAP_MIN_NM + r_nm) / 50.0)


def positions_nm():
    """The stage-C candidate grid: N_POSITIONS odd, symmetric about x=0, incl. 0."""
    n = int(N_POSITIONS)
    assert n >= 3 and n % 2 == 1, "N_POSITIONS must be odd (symmetric grid incl. x=0)"
    step = SPAN_UM * 1000.0 / (n - 1)
    half = (n - 1) // 2
    return [round((i - half) * step, 1) for i in range(n)]


def row2_positions_nm():
    """Stage-C2 candidate grid at ROW2_Y_NM: ROW2_N_ALIGNED aligned sites on the same
    135 nm step as row 1, plus the half-step staggered sites, sorted, all unique."""
    n = int(ROW2_N_ALIGNED)
    assert n >= 3 and n % 2 == 1, "ROW2_N_ALIGNED must be odd (symmetric grid incl. x=0)"
    step = ROW2_SPAN_UM * 1000.0 / (n - 1)
    half = (n - 1) // 2
    aligned = [round((i - half) * step, 1) for i in range(n)]
    out = sorted(aligned + [round(x, 1) for x in ROW2_STAGGER_X_NM])
    assert len(set(out)) == len(out), "row-2 sites must be unique"
    return out


def build_ports_base():
    """Ports-only config at the program numerics (no far-field monitors).
    Used by the stage-A prelim (resonance find) — same mesh/box as the main runs
    so the locked lambda_res is measured at identical numerics."""
    cfg = build_base()                       # anchored TM base; scatterer arrives ENABLED
    cfg.mesh.simulation_mode = MESH_MODE
    cfg.y_span_override_m = BOX_Y_UM * 1e-6  # converged box (overrides the 4.8 um quirk)
    cfg.span_multiplier_override = BOX_Z_MULT
    return cfg


def build_ff_base():
    """Program base WITH far-field monitors + complex far-field save (stages A/B/C/E).
    Near-field surfaces pinned OFF (~100s of MB/row); complex FF is ~4 MB/row."""
    cfg = build_ports_base()
    cfg.farfield.enabled = True
    cfg.farfield.save_nearfield = False
    cfg.farfield.save_complex = True
    cfg.farfield.farfield_x_span_m = FF_X_SPAN_UM * 1e-6
    # Grazing-capture guard: the ux~0.99 lobe crosses the TOP monitor plane
    # (~3.15 um above the core) ~22 um downstream of its source. Every candidate
    # scatterer (|x| <= SPAN/2) must radiate INSIDE the monitor, else its measured
    # response column is clipped and the inverse solve optimizes the wrong thing.
    assert FF_X_SPAN_UM / 2.0 >= SPAN_UM / 2.0 + 22.5, \
        "far-field monitor too short for the candidate grid: raise FF_X_SPAN_UM"
    return cfg
