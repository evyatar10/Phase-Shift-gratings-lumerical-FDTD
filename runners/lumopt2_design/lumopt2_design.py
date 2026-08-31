"""lumopt2 inverse-design engine — corr-325 pi-shift grating + SiN comb.

Study dir: runners/lumopt2_design/   |   Created 2026-08-13
Plan: ~/.claude/plans/we-have-previously-talked-crispy-wand.md (approved 2026-08-13)

WHAT THIS IS
  Adjoint (gradient) optimization of the innermost 25 periods/side of the
  corr-325 device plus the 57-post SiN comb, driven by Ansys lumopt2 (ships
  inside Lumerical 2026 R1.3). One forward + one adjoint FDTD per gradient,
  regardless of parameter count.

PARAMETER VECTOR p (length 190, ALL VALUES IN NANOMETERS — one unit everywhere)
  p[  0: 25]  corr_d   tooth corrugation (w_wide - w_narrow), d=1 innermost
  p[ 25: 50]  avg_d    tooth average width ((w_wide + w_narrow)/2)
  p[ 50: 75]  shift_d  tooth shift (shortens the narrow segment of tooth d)
  p[ 75:132]  r_j      comb post radius, site j=1..57 (x-ordered, site 29 = center)
  p[132:189]  x_j      comb post x position
  p[189]      d_comb   shared transverse distance |y| of the comb row

CONVENTIONS (read before editing)
  * The same 25 (corr, avg, shift) values drive BOTH arms by tooth index —
    the builder's convention, used by every measured apodization study. The
    pi-shift layout is intrinsically half-pitch staggered between arms
    (the cavity IS the extra narrow half-period), so index-symmetric is
    geometric mirror symmetry up to that built-in stagger.
  * Shifts: the narrow segment of tooth d shrinks by shift_d on both arms;
    the cavity absorbs 2*sum(shift) so the frozen outer 75 periods NEVER
    move. (This differs from bragg_device's inner_shift_nm bookkeeping,
    which displaces the frozen right arm; production confirms of a winner
    therefore go through THIS engine's geometry, not inner_shift_nm.)
  * The comb is NOT index-mirrored in x: the 270-degree winner is a
    traveling lattice; its x-mirror is the measured-losing 90-degree phase.
  * d_comb is ONE shared knob (settled 2026-08-13): per-site distance is
    amplitude-degenerate with radius (e^-1.95*dd rule, measured), and
    per-site amplitude freedom already exists via r_j.

COST FUNCTION (settled 2026-08-10..13, memory project_inverse_design_cost_function)
  FOM  = softmax_p(T) over a +-2.5*FWHM window re-selected every evaluation
         (p=12; reads T_peak, blind to linewidth)  [adjoint through lumopt2]
       - penalty on rho = mean(corr_d)/325, asymmetric deadband +2%/-5%
         (analytic kappa-integral proxy for the mode width; exact autograd
         gradient, zero extra simulations)
  Measured sigma (2nd moment of the field_profile x-envelope) is a per-eval
  TRIPWIRE, never in the adjoint. Q appears nowhere (TCMT identity).
  Diagnostics logged every eval to <out_dir>/<label>_evals.jsonl.

VALIDATION GATES (run before any campaign; validate_c325.py drives these)
  B0 reader on stored .mat spectra   B1 local build smoke + geometry diff
  B2 canary vs stored N=100 anchor   B3 validate_gradient (6 params)
  B4 known-answer mini-opt (comb dx)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import autograd
import autograd.numpy as anp
from dataclasses import dataclass

import config
from sim_helpers import (apply_monitor_overrides, find_bragg_resonance,
                         extract_envelope_peaks, calculate_fwhm_relative)
from runners.scatterers import _common

NM = 1e-9
C0 = 299792458.0

# ── Device family (corr-325, TM h350 — CLAUDE.md §4) ─────────────────────────
PITCH_NM        = 516.83
# ★Pitch-locked region mesh (2026-08-23, plan §24): the standing wave has the
# pitch period, so dx MUST divide the pitch by an integer or fwhm_env's
# sampling phase drifts across the mode (measured: up to 3.9% width error at
# dx=50.0 vs +0.006% pitch-locked). DERIVED from the pitch by user order —
# never hardcode the quotient: if PITCH_NM changes, this follows.
CELLS_PER_PITCH = 10             # dx = pitch/10 ≈ 51.683 nm ≈ optimization mode
DX_PITCHLOCK_NM = PITCH_NM / CELLS_PER_PITCH
CORR_NM         = 325.0          # frozen outer-tooth corrugation
AVG_W_NM        = 800.0          # nominal average width (W800)
EPS_CORE        = 1.97 ** 2      # 3.8809 (SiN)
EPS_CLAD        = 1.444 ** 2     # 2.0851 (SiO2)
N_FREE          = 25             # free innermost periods per side (user 2026-08-13)
KAPPA_PER_UM    = 0.0353         # MEASURED corr-325 (IGUM 51736); kappa ∝ corr
TWO_KL_FLOOR    = 3.5            # surrogate rule: 2*kappa*L ≥ 3.5 (hard floor 3.2)

# ── Comb seed = measured q3db winner (job 130458): Λ531/δx401(270°)/r80/d1.9 ─
COMB_LAM_NM     = 531.0
COMB_DX_NM      = 401.0
COMB_R_NM       = 80.0
COMB_D_NM       = 1900.0
COMB_N_HALF     = 28             # 57 posts
N_COMB          = 2 * COMB_N_HALF + 1

# ── Cost function (settled + signed off 2026-08-13) ──────────────────────────
P_SOFTMAX       = 12.0
WIN_FWHM_MULT   = 2.5            # FOM window = ±2.5 × measured FWHM
BETA_UP         = 18.0           # widening side of the rho deadband (the T-cheat)
BETA_DN         = 5.0            # narrowing side
RHO_UP          = 1.02           # +2 % deadband edge (user 2026-08-17: program-wide consistent 2% — the live gen-5 drivers loaded 1.02, and a retro-tightening to 1.01 would discard their >1.01 bests at the next reload; supersedes the 08-16 1.01 tightening. Width honesty is enforced at readout: Q_i/sigma^2 decomposition + fixed-width production re-trim)
RHO_DN          = 0.98           # ★SYMMETRIC −2 % (user 2026-08-23, was −5 %).
# The spec is a FIXED interaction width for acousto-optic sensing: a narrower
# mode does not help, so an asymmetric band let the optimizer spend width it
# has no use for — and width is worth transmission (MEASURED payback 0.0030
# T/µm, jobs 136296 vs mx_retrim). Symmetric ±2 % keeps the tolerance that
# exists for measurement noise (the 0.002-T / 0.03-% width floor, §20) while
# removing the free narrowing. Applies to the fwhm_env band (the authority),
# the fwhm_wall hinge, and the retired rho penalty alike.
DEAD_T_FLOOR    = 0.02           # dead-device guard (dead ≈ 0.0008, healthy ≥ 0.5)
BETA_ELONG      = 1e-5           # cavity-elongation guard (per nm²; see kappa_penalty)
ELONG_DEADBAND_NM = 120.0        # allowed |2Σshift| before the guard engages
RECENTER_NM     = 2.0            # |λ_peak − scan center| trip → rebuild + restart
MAX_RESTARTS    = 12             # recenter/width-trip restarts per campaign (λ walks redward ~1-2 nm per early accepted step — measured 2026-08-16)

# ── Parameter vector layout ──────────────────────────────────────────────────
SL_CORR  = slice(0, N_FREE)
SL_AVG   = slice(N_FREE, 2 * N_FREE)
SL_SHIFT = slice(2 * N_FREE, 3 * N_FREE)
SL_R     = slice(3 * N_FREE, 3 * N_FREE + N_COMB)
SL_X     = slice(3 * N_FREE + N_COMB, 3 * N_FREE + 2 * N_COMB)
I_DCOMB  = 3 * N_FREE + 2 * N_COMB
I_CAV    = I_DCOMB + 1           # cavity WIDTH (y span) — user scope addition
                                 # 2026-08-15: measured strong knob (rect-cavity
                                 # study W1050 −29.9 % loss at corr-400; old PSO
                                 # optimum ≈854) and the most boundary-sensitive
                                 # wall of all (y-normal at the field maximum)
N_PARAMS = I_CAV + 1             # 191


@dataclass
class CampaignSpec:
    """Everything a runner may vary. Defaults = the settled campaign values."""
    label: str = "lumopt2_c325"
    n_periods_side: int = 100        # surrogate N (2*kappa*L = 3.65)
    box_y_um: float = 6.8            # gate-A0 decides 6.8 vs 8.0
    box_z_mult: float = 4.14         # z = 0.35 + mult*1.5595 → 6.8 µm (3.50 would
                                     # be z=5.8, the REJECTED span-study rung)
    scan_center_nm: float = 1564.21  # MEASURED at PVA (job 132654) — the
                                     # precise-volume-average mesher shifts λ
                                     # +5.2 nm vs the family's 1559.0
    scan_width_nm: float = 6.0
    n_wl_points: int = 301           # 20 pm grid
    corr_seed_nm: tuple = (CORR_NM,) * N_FREE   # seed B overrides with a dip profile
    corr_max_nm: float = 500.0       # tightened 5 % on a width trip
    max_iter: int = 60
    max_feval: int = 100
    dp_tooth_nm: float = 1.0         # mesher-FD step (auto ≈0.1 nm is too small)
    dp_comb_nm: float = 1.0          # raise to 2-5 if gate B3 shows comb noise
    free_comb: bool = True           # False → comb frozen at the seed geometry
    bare: bool = False               # True → NO comb at all (B2 anchor canary)
    # ── gradient-fix experiments (2026-08-15, all three FD-gated) ──────────
    # Option 3: co-locate the staggered Yee components on the gradient monitor
    # ("nearest mesh cell") — removes the discrete half of the tooth deficit.
    colocate_fields: bool = False
    # Option 2: E∥/D⊥ boundary reweighting (Johnson PRE 65 066611) — scale the
    # NORMAL-component dEps slice of each wall-moving param by
    # R = -Δ(1/ε)/Δε · ε_eval² (harmonic weighting expressed through the same
    # sparse dEps). ε_eval = the ε "at" the recorded fields — genuinely
    # uncertain between clad (0.537), cell-average (1.10) and core (1.86)
    # weights, so it is a spec knob the B3 gate arbitrates empirically.
    bc_patch: bool = False
    bc_eps_eval: float = 2.983       # (ε_core+ε_clad)/2 — first guess
    # Option 1: per-class gradient calibration factors applied to the lumopt2
    # gradient before the optimizer (penalty gradient stays exact). Keys:
    # corr/avg/shift/r/x/d. None → off. Fill from measured B3 α values.
    # ★DEAD 2026-08-16: α measured operating-point-dependent (comb-x flips
    # sign between detune points) — kept only as a code path for experiments.
    grad_cal: dict = None
    # ★Option 4 — THE ROOT-CAUSE FIX (found 2026-08-16, offline recon):
    # lumopt2's adjoint scaling e_adj·(1j·ω/4)·conj(am)/P carries a spurious
    # 90° phase — the adjoint returns the QUADRATURE of the true response
    # (symmetric lobe instead of the antisymmetric resonance-translation
    # spectrum; measured r=+0.991 vs FD dT(λ) with the 1j dropped, ~0.02
    # with it). adj_phase_fix multiplies lumopt2's scaled adjoint fields by
    # complex(adj_fix_re, adj_fix_im). ★MEASURED 2026-08-16 (both quadratures
    # at detune-1: Re from 132657/task5, Im from task 8 = the +1j quadrature
    # run): ONE global complex factor C = 0.8685+0.1022i fits FD on ALL 7
    # params to 1.7% — the pipeline's phase is off by just 6.7° (≈ the
    # quarter-cell Yee offset 0.25·k·dx = 6.2°), catastrophic because Z sits
    # 83-99° off the real axis with |Z| 30-240× the true gradient. C is
    # mesh/device-specific — recalibrate for a new device from one stored
    # (Re, Im, FD) triple. (1,0) = no-op; (0,1) = quadrature probe.
    adj_phase_fix: bool = False
    adj_fix_re: float = 1.0
    adj_fix_im: float = 0.0
    # ── Stage-3: single sigma-hat wall (2026-08-17, tangent probe 133512) ──
    # Replaces the two independent block walls (elongation + rho deadband)
    # with ONE hinge on a calibrated linear width surrogate, so sigma-NEUTRAL
    # cross-block trades (shift up + corr up / w_cav down) become traversable.
    # Coefficients MEASURED by the probe; anchor re-zeroed at each restart on
    # the best row's MEASURED sigma (stage-wise trust region handles the
    # measured curvature). Tripwire + restart filters unchanged (guarantee).
    sigma_wall: bool = False
    sig_anchor: dict = None          # {"sigma","elong","rho","wcav"} — set by
                                     # the runner (initial) and run_campaign
                                     # (each restart)
    # ── Stage-2 refinement (2026-08-17, skill item 19) ─────────────────────
    # seed_override: full 191-vector (nm) to start from instead of the built
    # seed — used to warm-start a refinement campaign from a prior best.
    # freeze_shifts: sliver-bound the shift block at the seed values so the
    # optimizer spends no solves on the exhausted (and width-coupled)
    # elongation direction; corr/avg/comb/cavity stay free.
    seed_override: tuple = None
    freeze_shifts: bool = False
    # ── Trust-region bounds (2026-08-17, skill item 22) ────────────────────
    # lumopt2 scales every param to [-1,1] by its OWN bounds, and L-BFGS-B's
    # first probe is a unit-norm step in that scaled space — so on a warm
    # start, wide-bounds blocks get slammed by a fraction of their FULL
    # physical range regardless of seed quality (MEASURED: 133541 eval 3,
    # 2*Sig_s 130.6->504.2 nm, sigma +13.7%, FOM -7.92; bare 55343 ditto).
    # trust_nm = {"shift": 20, ...} clamps each named block to p0 +/- r,
    # CENTERED on p0 (which also makes the scaler round-trip exact and kills
    # the duplicate-x0 eval tax, item 21d). The physical cap stays enforced
    # by the box intersection; the width limit lives in the penalty.
    trust_nm: dict = None
    # ── Comb post COUNT (2026-08-17, user: "considerably increase or decrease
    # the number of circles") ──────────────────────────────────────────────
    # The count is a SCENE property, not one of the 191 params (which carry a
    # fixed 57 r + 57 x). comb_n_half=K builds 2K+1 posts on the seed lattice
    # and REQUIRES free_comb=False so make_func never addresses scatterer
    # objects that no longer match the param slots. Evaluation-only knob.
    comb_n_half: int = None
    # ── FWHM anchor (2026-08-18, the proxy-trap fix) ───────────────────────
    # The acoustic spec is SPATIAL FWHM; sigma-compliance measurably does NOT
    # imply FWHM-compliance (audit 134217: sigma +1.8% while FWHM +27%). With
    # an anchor set, accepted-best designs must also keep fwhm_env_um (the
    # project fwhm_m convention) inside the same +2%/-5% deadband, and
    # _best_from_log filters restarts/final selection on it. None = sigma-only
    # (legacy). Rows logged before fwhm_env_um existed PASS the filter —
    # whether an old best is spec-compliant is a per-campaign decision.
    fwhm0_um: float = None
    # ── V2 width GRADIENT (2026-08-21, V2_FWHM_PLAN.md; skill item 28) ─────
    # width_grad=True adds a second FOM entry: softW, the soft level-set width
    # of the y-integrated resonance profile (VALIDATED offline: tracks the
    # measured fwhm_env to ≤2 pp where sigma errs 24 pp). Its gradient flows
    # through ONE extra adjoint sim with a WEIGHTED source on field_profile
    # (stock FieldFom only supports plain conj(E) sources — see MixedFom).
    # The fct then subtracts an augmented-Lagrangian band penalty on the
    # anchor-mapped FWHM prediction. The MEASURED fwhm_env stays the authority
    # (WidthTrip + restart filter, unchanged); softW only steers gradients.
    # wg_anchor = {"softw": µm, "fwhm": µm} measured on the SAME evaluation —
    # set by the runner at start and re-anchored by run_campaign per restart
    # (per-stage, not per-eval, so the objective stays stationary under
    # L-BFGS-B; the per-eval residual alarm below catches drift).
    # ★The field-adjoint path is UNVALIDATED: gate W3 must measure its own
    # C_field (adj_fix_field_re/im) before any campaign — never assume the
    # port C transfers.
    width_grad: bool = False
    wg_pure: bool = False            # W3 gate: fct = −softW alone, so FD
                                     # measures d(softW)/dp through the field
                                     # adjoint with dJ/dsoftW = −1 exactly
    wg_anchor: dict = None
    # ★The GPU engine REJECTS FieldRegion-source adjoint scenes ("ERROR:
    # invalid configuration argument", CUDA-level — measured job 136026, all
    # 3 tasks; docs list only TFSF/BFAST as GPU-unsupported, the FieldRegion
    # object is too new to be documented). Width-adjoint jobs are therefore
    # routed to this resource; everything else stays GPU. W1r MEASURES the
    # CPU adjoint solve time — if it exceeds ~1.5 h/solve the campaign
    # iteration cost needs a decision (Ansys bug report / accept / rethink).
    wg_adj_resource: str = "CPU"
    # wg_source: "fieldregion" = lumopt2's designed object (GPU-REJECTED,
    # measured 136026; CPU 8.7 h). "import" = standard addimportedsource
    # injection — NOT on the GPU-unsupported list (only TFSF/BFAST are);
    # user challenge 2026-08-22: test before accepting the CPU tax.
    # Normalization differences are absorbed by C_field (that is its job).
    wg_source: str = "fieldregion"
    # ★wg_src_tiles (2026-08-24) — the GPU-rejection workaround. MEASURED
    # (jobs 136799/136826/136869): the GPU engine accepts a FieldRegion
    # ADJOINT SOURCE at x=528 cells (3/3 rungs, 3,696-45,936 total cells) and
    # rejects it at x>=1056 (4/4 rungs, 29,568-183,744 cells) with "ERROR:
    # invalid configuration argument". Total cells OVERLAP across the split,
    # so the X extent alone is the variable — suspected CUDA 1024-threads/
    # block ceiling (lumcudafdtd.dll carries "Total threads per block %u
    # exceeds device limit of %d"). The same region is fine as a full-size
    # MONITOR (2112 x 29 cells ran 22 min of forwards).
    # >1 keeps field_profile_adj as the pure MONITOR (source mode never on)
    # and injects the SAME weighted dataset W(x,y)*conj(E_fwd) through this
    # many narrow FieldRegion tiles, x-partitioned with no gap and no
    # overlap, all enabled in ONE adjoint run. FDTD sources superpose
    # linearly, and both the adjoint problem and the gradient assembly are
    # linear in the adjoint field, so the gradient is EXACT — not an
    # approximation, not a sum of independent solves.
    # 1 = the legacy single-source path, unchanged.
    wg_src_tiles: int = 1
    # Safe per-tile x-cell ceiling, asserted at build time.
    # ★MEASURED 2026-08-25 (jobs 136799/136826/136869/136907): x = 528 and
    # x = 1000 PASS; x = 1056, 1080, 2112 REJECT ⇒ the ceiling is CUDA's 1024
    # threads/block, bracketed to (1000, 1056]. 1000 is the highest MEASURED
    # pass, so that is the cap. (It was 528 — a placeholder from before the
    # bracket existed — which failed task 37 at 529 cells/tile, off by one.)
    wg_tile_max_xcells: int = 1000
    # ★wg_track_resonance (2026-08-23 review; §23 user ruling "softW ON
    # RESONANCE, always" — hard precondition for exact mode): when True, the
    # parametrization func ALSO emits "field_profile_adj::wavelength center"
    # = the latest measured lam_pk (previous eval; scan center until the
    # first eval lands), so the single-λ twin — and the import-source λ-pin,
    # which follows the twin's RECORDED λ (setup_adjoint_simulation reads
    # sim_result.wavelengths, set by get_results from the twin data) —
    # tracks the CURRENT resonance instead of the static scan center.
    # Mechanism: the documented parametrization-props channel, re-applied by
    # lumopt2 at EVERY project (re)generation — unlike a setnamed after
    # generate(), which is wiped. Restarts already rebuild the twin at the
    # recentered scan_center (build_base_fsp); this closes the WITHIN-segment
    # gap: RECENTER_NM=2.0 nm = 2.7 linewidths worst case before a restart,
    # ~1.4 linewidths after one accepted step. With tracking, the offset is
    # one eval's λ drift (≲0.3 linewidths at accepted cadence; a big
    # line-search probe mis-samples only its own eval and self-corrects).
    # Same-eval exactness is impossible without a second forward — lam_pk is
    # not known until the forward has run. Default False (nothing running
    # changes); exact mode must re-gate with this ON (§23).
    wg_track_resonance: bool = False
    # ★2026-08-22 (user caught it): the DEVICE's own mesh override is
    # PITCH-LOCKED (dx = pitch/10 = 51.683 nm for TM — bragg_device puts mesh
    # edges exactly on the tooth transitions). The region's historical 50.0
    # was a TE-era assumption that broke that alignment. Default stays 50.0
    # (every existing anchor/log is at 50.0 — REQUEUE-safe); v2.1 specs set
    # 51.683 as a NAMED §2 change with fresh anchors (validate tasks 14-17).
    region_dx_nm: float = 50.0
    mesh_refinement: str = "precise volume average"
    # v2-projection: the rho deadband is RETIRED (item-27 hole; it also
    # blocks the legitimate width-payback direction the re-trim uses). The
    # measured fwhm_env guard + stage re-trim own width; the elongation wall
    # stays (cheat channel). Default True = every legacy spec unchanged.
    rho_band: bool = True
    # ★fwhm_wall (2026-08-22, user: "the gradients must go in the direction
    # of less FWHM too"): a delta-anchored LINEAR width term in the FOM,
    # slopes MEASURED on corrected y-integrated data THIS DAY (retrim curve
    # 136051: dFWHM/d(mean corr) = -0.0470 um/nm; corrected noshift->best:
    # +0.01355 um/nm of elongation 2*sum(s)). Re-anchored to the MEASURED
    # fwhm_env at every ACCEPTED-BEST eval (item-24 prescription) - the
    # anchor tracks truth, the slopes only steer between measurements.
    # Zero extra solves. NOT the deleted void-era FWHM_A_* slopes.
    fwhm_wall: bool = False
    # ★2026-08-23: True → the wall's elongation term is the CONVEX FW_C_ELONG
    # fit (6 corrected fspw points) instead of the [0,130.6] nm FW_A_ELONG
    # secant. Default False so in-flight campaigns survive a REQUEUE unchanged;
    # turn it on in the next campaign's spec. See make_fwhm_wall.
    fw_convex: bool = False
    # ★2026-08-23: True → the wall's elongation term is the MEASURED 6-point
    # threshold curve (_fw_elong_curve), which is the only form consistent with
    # jobs 61742/61782. Supersedes fw_convex; both default False so campaigns
    # already in flight are bit-identical after a REQUEUE.
    fw_curve: bool = False
    # ★2026-08-24: saturate the width hinge at this many FOM units (tanh), so a
    # steep fw_curve cannot hand L-BFGS-B an unnavigable cliff and abort the
    # line search (the 136640 failure). None = uncapped = legacy behaviour, so
    # campaigns already in flight are unaffected by a REQUEUE. ~2.0 is ample:
    # FOM is O(0.7) here, so a capped out-of-band score still loses to every
    # in-band one. Small-excess behaviour is unchanged (tanh x ~ x).
    fw_pen_cap: float = None
    # ★2026-08-24 (Fable audit): per-tooth width price for the corr block — a
    # tuple of N_FREE slopes (um FWHM per nm of that tooth's corr, negative =
    # raising narrows), normally FW_TOOTH_W. Replaces the rank-1 mean-corr
    # term whose gradient was identical on all 25 teeth and hid the see-saw
    # direction from the optimizer. Requires fw_anchor["corr_vec"] (loud
    # KeyError otherwise — no silent fallthrough, V2 plan bug-D class).
    # None = legacy mean-corr path, bit-identical for campaigns in flight.
    fw_tooth_w: tuple = None
    # ★2026-08-24: elongation-curve coefficient for THIS spec's device class.
    # None = FW_CURVE_C (fitted on the uniform seed — correct for uniform-seeded
    # campaigns). Apodized-seeded campaigns must pass FW_CURVE_C_APOD or the
    # wall over-taxes elongation by 1.87x (MEASURED, shiftw ladder 136710).
    # Constants carry their device class — see skill item 35.
    fw_curve_c: float = None
    fw_anchor: dict = None      # {"fwhm","mcorr","elong"[,"corr_vec"]} set by
                                # the runner (initial) + callback (re-anchor)
    wg_mu: float = 8.0               # per µm²: 0.05 µm over-band ⇒ 0.01 FOM
    wg_lam_hi: float = 0.0           # AL multipliers, updated per restart on
    wg_lam_lo: float = 0.0           # the MEASURED best-row violation
    adj_fix_field_re: float = 1.0
    adj_fix_field_im: float = 0.0
    # ★wg_project (2026-08-25, HANDOFF "THE FORMULATION, DECIDED 2026-08-24
    # 23:00"): ceiling-riding projected gradient. Replaces ScipyOptimizer with
    # run_projected: PHASE A predictive step-clipping while W < target, PHASE B
    # null-space steps on the width manifold, restoration on MEASURED fwhm_env,
    # shadow price logged every iterate. Requires width_grad=True. Default
    # False = every existing path byte-identical (REQUEUE-safe).
    wg_project: bool = False
    wgp_target_um: float = None   # ride level W_tgt = W_hi − margin (18.613
                                  # for the 18.346 family) — NOT the benchmark
    wgp_margin_um: float = 0.10   # drift margin; restoration beyond margin/2
    wgp_step: float = 0.25        # step scale in the D metric; gate P2 calibrates
    # ★two_kl_floor (2026-08-28): None = the module TWO_KL_FLOOR (3.5).
    # ONLY the pipeline smoke may lower it (weak surrogate, numbers never
    # quoted); a physics spec must never touch this.
    two_kl_floor: float = None
    # ★wgp_rgp (2026-08-28): relaxed gradient projection (Antonau SMO 2021) —
    # replaces the CLIMB/RIDE/RESTORE branches with one continuous formula
    # (buffer-ramped projection weight + bounded rotating correction). Their
    # measured 1.7× objective gain vs pure-restore stepping. Default False =
    # every existing spec keeps the proven 3-phase law; adopt after the first
    # projected campaign, or immediately on restore-thrash.
    # See notes_relaxed_projection.md.
    wgp_rgp: bool = False
    wgp_rgp_bsf: float = 2.0   # their BSF_init; the buffer auto-sizes from
                               # measured per-iterate |ΔW| history (eq 14)
    # ★wg_dwdlam_fit (2026-08-28): re-fit the path-dependent dW/dλ ONLINE from
    # this run's own accepted (λ_pk, W) points — the ±20% coefficient error is
    # the measured dominant residual of the ride test (leak +0.0032 µm/it).
    # Engages only at n ≥ 5 unique points AND λ-span ≥ 0.5 nm (the detrend
    # audit's short-arm trap: below ~25 grid steps the own-slope fit flips
    # sign/magnitude); per-refit change clamped ±30%, absolute [0.10, 0.70].
    wg_dwdlam_fit: bool = False
    # ★wgp_lam_step_nm (2026-08-28, lit item 1): reject any trial step whose
    # measured λ_pk moved more than this from the accepted point (drift is the
    # width leak a fixed-λ gW cannot see; published recipe = recentring PLUS a
    # bandwidth bound). None = inert (every pre-existing spec unchanged).
    wgp_lam_step_nm: float = None
    wgp_step_max_nm: float = 5.0  # hard per-param step cap (C_field magnitude
    # ★wgp_autogain (2026-08-25): STOP RELYING ON |C_field|. The projection's
    # direction is EXACTLY scale-invariant (verified: scaling |C| changes the
    # null space by 0.00 deg), so the magnitude of C only enters step-length
    # PREDICTION and restoration GAIN — and both can be MEASURED instead of
    # fitted. Each iterate compares the predicted width change with the
    # measured one and updates a running gain, so the method self-calibrates
    # and survives a mesh change without a new FD reference. Only the PHASE of
    # C (a Yee-staggering artifact, ~quarter to half cell) still matters.
    wgp_autogain: bool = False
                                  # uncertainty guard; scalar cap keeps ∇W·d=0)
    # ★wg_lam_chain (DEFECT #19, 2026-08-25): the width twin's λ-pin is emitted
    # as a CONSTANT to autograd (make_func, "zero Jacobian row, zero dEps"), so
    # ∇W is ∂W/∂p at FIXED λ — while W is SPECced at the device's own MOVING
    # resonance. The chain term dW/dλ · dλ_pk/dp is therefore unpriced, and the
    # projection nulls the wrong gradient. MEASURED on job 137075_41: the whole
    # +0.0110 µm width change of iterate 0→1 is explained by the +0.04 nm λ
    # drift (0.3655 × 0.04 = +0.0146), and the T-per-µm exchange rate did not
    # improve over the unprojected baseline (0.097 vs 0.091). Unfixed, restore
    # pushes along a fixed-λ ∇W that cannot undo λ-mediated growth while climb
    # re-drifts λ — a treadmill.
    # dλ_pk/dp comes from the SAME solved fields (zero extra adjoint solves):
    # two selector passes give ∂T/∂p either side of the peak, then the implicit
    # function theorem on ∂T/∂λ = 0 gives  gλ = −(∂²T/∂λ∂p)/(∂²T/∂λ²).
    # ∇T needs no such term: at the peak ∂T/∂λ = 0, so its chain term vanishes.
    wg_lam_chain: bool = False
    wg_dwdlam: float = 0.3655     # µm/nm; MEASURED r=0.984 n=9 over the uniform
                                  # baseline's in-band evals. Refreshed online
                                  # from the eval log once ≥4 points accrue.
    # ★wgp_ns2 (2026-08-30, d1 generation): TWO-constraint null+range-space
    # step — hold the measured width W AND the resonance λ_pk (Feppon/Allaire/
    # Dapogny, ESAIM:COCV 26:90). Why: measured ΔW ≈ 0.3655·Δλ_pk on every
    # gradient path (c1 live: the first two production RIDE evals grew W
    # +0.031/+0.050 µm — λ drift, not envelope reshaping), while task 49
    # proved T can rise at fixed λ. Constraint rows use the RAW fixed-λ gW
    # plus gLam: once gLam·d = 0, the fitted wg_dwdlam coefficient drops out
    # of the feasible directions entirely (chain-correcting gW would make the
    # two rows near-collinear). Default False = every existing spec
    # bit-identical.
    wgp_ns2: bool = False
    wgp_lam_target_nm: float = None  # λ hold target; None ⇒ latch the first
                                     # measured λ_pk (persisted in the sidecar)
    wgp_lam_margin_nm: float = 0.05  # λ restoration deadband, nm (grid 20 pm)
    # ★wgp_cap_adapt (2026-08-30): the step cap becomes trust-region STATE.
    # The old _cap(a)=cap0·min(1,a/a0) made the delivered step a CONSTANT
    # cap0 whenever the raw step exceeded it (~73 nm measured ⇒ always):
    # alpha and wgp_step cancel out of the realised move, and nothing ever
    # grows cap0 — the measured pace ceiling of c1/b1. With this flag on:
    # accept whose measured |Δλ|<0.10 nm AND |ΔW|<0.020 µm ⇒ cap ×wgp_cap_grow
    # (ceiling wgp_cap_max_nm); reject ⇒ cap ×0.5 (floor 2 nm). Retry halvings
    # stay real (the property the old coupling existed for).
    wgp_cap_adapt: bool = False
    wgp_cap_max_nm: float = 60.0
    wgp_cap_grow: float = 1.5
    # ★wgp_reuse_k (2026-08-31, iterate-cost cut): reuse the WIDTH constraint
    # row for up to k-1 iterates between width-adjoint refreshes — skips the
    # ~45 min width-adjoint solve + its ~25 min assembly pass (~35% of an
    # iterate) on reuse iterates. Justified by MEASURED row constancy on d1:
    # gW_n 0.361→0.362, gLam_n 0.2413→0.2395, dTp −2.5361→−2.5369 across
    # iterates. gλ stays FRESH every iterate (selector passes ride the port
    # adjoint — zero extra solves), so the λ-hold is never stale; only gW
    # ages. Refresh early on: any filter reject, restoration active, or
    # |ΔW_meas| > wgp_reuse_dw_um. 0 = off (default; existing specs
    # bit-identical). reuse_age persists in the optstate sidecar.
    wgp_reuse_k: int = 0
    wgp_reuse_dw_um: float = 0.025


def seed_params(spec):
    """Seed vector: uniform grating (or spec's corr profile) + winner comb."""
    if spec.seed_override is not None:
        p = np.asarray(spec.seed_override, dtype=float).copy()
        assert p.shape == (N_PARAMS,)
        return p
    p = np.empty(N_PARAMS)
    p[SL_CORR]  = np.asarray(spec.corr_seed_nm, dtype=float)
    p[SL_AVG]   = AVG_W_NM
    p[SL_SHIFT] = 0.0
    p[SL_R]     = COMB_R_NM
    p[SL_X]     = [k * COMB_LAM_NM + COMB_DX_NM
                   for k in range(-COMB_N_HALF, COMB_N_HALF + 1)]
    p[I_DCOMB]  = COMB_D_NM
    p[I_CAV]    = AVG_W_NM        # cavity width, seed = builder's "avg" option
    return p


def param_bounds(spec):
    """Box bounds (nm). Comb d upper bound derives from the y-PML clearance;
    frozen comb params get sliver bounds (inert — func never touches them)."""
    p0 = seed_params(spec)
    b = ([(150.0, spec.corr_max_nm)] * N_FREE +   # corr: dip and overshoot allowed
         [(775.0, 825.0)] * N_FREE)               # avg width: ±25 nm n_eff drift cap
    if spec.freeze_shifts:
        # stage-2: pin shifts at the (evolved) seed — the elongation direction
        # is exhausted and width-coupled; sliver bounds match the frozen-comb
        # mechanism so func still applies the values, optimizer can't move them
        b += [(v - 1e-3, v + 1e-3) for v in p0[SL_SHIFT]]
    else:
        b += [(0.0, 200.0)] * N_FREE              # shift (repo convention, don't tighten)
    if spec.free_comb and not spec.bare:
        d_max = spec.box_y_um * 1000.0 / 2.0 - 240.0 - 1200.0  # y_half − r_max − clear
        b += [(70.0, 240.0)] * N_COMB             # radius: 70 ≈ dx=50 mesh floor
        b += [(x - 100.0, x + 100.0) for x in p0[SL_X]]
        b += [(1500.0, d_max)]
    else:
        b += [(v - 1e-3, v + 1e-3) for v in p0[3 * N_FREE:I_CAV]]
    b += [(750.0, 1150.0)]        # cavity width: covers the measured 1050
    assert len(b) == N_PARAMS
    if spec.trust_nm:
        # Clamp named blocks to p0 +/- r, centered (see CampaignSpec note).
        # r shrinks near a physical edge to preserve centering; a seed ON the
        # edge (fresh campaign, shifts at 0) keeps the plain box clamp.
        blocks = {"corr": SL_CORR, "avg": SL_AVG, "shift": SL_SHIFT,
                  "r": SL_R, "x": SL_X, "d": slice(I_DCOMB, I_DCOMB + 1),
                  "wcav": slice(I_CAV, I_CAV + 1)}
        for name, r in spec.trust_nm.items():
            for i in range(*blocks[name].indices(N_PARAMS)):
                lo, hi = b[i]
                # ★FIX 2026-08-24: was min(r, p0-lo, hi-p0) + a symmetry
                # requirement, which silently did NOTHING whenever the seed sat
                # ON a bound — i.e. exactly when the clamp matters most. The
                # uniform seed's shifts are 0.0 = the lower bound, so r_eff
                # collapsed to 0 and the shift block kept its full 0-200 box.
                # MEASURED consequence (136709 ev2): L-BFGS-B's unit-norm
                # scaled probe is ~0.0575 per param, which over a 100 nm
                # half-range is 5.75 nm/tooth = elongation 287.4 nm, width
                # 32.27 um — 13.6 um out of band, one wasted 90-min eval, and
                # the same 0->287 lurch that crippled campaigns 136466/136640.
                # Now clamps ASYMMETRICALLY against the box: the trust region
                # is a SEARCH-STEP control that re-centres on the best design
                # every restart, NOT a tightening of the physical shift_bounds
                # (which stay 0-200 per the standing rule — no region of the
                # design space is made permanently unreachable).
                # Cost of dropping the centring: x0 no longer maps to scaled 0,
                # so a duplicate-x0 evaluation can reappear (item 21d) — one
                # eval, against the many an out-of-band lurch burns.
                lo_n, hi_n = max(lo, p0[i] - r), min(hi, p0[i] + r)
                if hi_n - lo_n < hi - lo:
                    b[i] = (lo_n, hi_n)
    return b


def replay_params(spec, p):
    """Adapt an EVOLVED param vector for evaluation under `spec` (A/B rows,
    replays, re-anchors). Frozen/bare specs pin the comb slots to SEED values
    with sliver bounds (func never touches them), so evolved comb values are
    rejected by lumopt2's bounds check before any solve — that killed job
    133395 task 1 in 34 s (skill item 17). Resets the inert comb block to
    seed (physics-neutral: the comb is absent or frozen in the scene),
    keeps the grating block untouched, and asserts bounds compliance."""
    p = np.asarray(p, dtype=float).copy()
    assert p.shape == (N_PARAMS,)
    if spec.bare or not spec.free_comb:
        seed = seed_params(spec)
        p[SL_R], p[SL_X], p[I_DCOMB] = seed[SL_R], seed[SL_X], seed[I_DCOMB]
    for i, (lo, hi) in enumerate(param_bounds(spec)):
        assert lo <= p[i] <= hi, f"replay param {i}={p[i]} outside [{lo},{hi}]"
    return p


# ═══════════════════════════════════════════════════════════════════════════
# Geometry: parameter vector → Lumerical object properties
# ═══════════════════════════════════════════════════════════════════════════

def tooth_names(n_side):
    """Builder object names for the free teeth (asserted against the .fsp).

    bragg_device numbers rects with one global seg counter: wg_left_inf_1,
    then the left arm walks d=n..1 emitting (L_narrow_d, L_wide_d), then the
    cavity, then the right arm d=1..n emitting (R_narrow_d, R_wide_d).
    """
    names = {}
    for d in range(1, N_FREE + 1):
        names[d] = (f"L_narrow_{d}_{2 + 2 * (n_side - d)}",
                    f"L_wide_{d}_{3 + 2 * (n_side - d)}",
                    f"R_narrow_{d}_{2 * n_side + 3 + 2 * (d - 1)}",
                    f"R_wide_{d}_{2 * n_side + 4 + 2 * (d - 1)}")
    return names, f"cavity_{2 * n_side + 2}"


def comb_count(spec):
    """Number of posts along x in the BUILT scene (2K+1). Default = N_COMB."""
    if getattr(spec, "comb_n_half", None) is None:
        return N_COMB
    assert not spec.free_comb, \
        "comb_n_half requires free_comb=False (param slots are fixed at 57)"
    return 2 * spec.comb_n_half + 1


def scatterer_names(n=N_COMB):
    """(+y, −y) object pairs, site j=1..n (builder draws both y-mirror copies)."""
    return [(f"scatterer_{j}_1", f"scatterer_{j}_2") for j in range(1, n + 1)]


def make_func(spec):
    """Parametrization func: p (nm) → {"object::property": SI value}.

    All arithmetic is autograd.numpy so lumopt2 can differentiate the map.
    Both free regions are walked from their FIXED outer edges toward the
    cavity; the cavity absorbs 2*sum(shift) (see module docstring).
    """
    hp = PITCH_NM / 2.0
    cav_l0 = PITCH_NM / 2.0                       # builder default cavity length
    x_out = cav_l0 / 2.0 + N_FREE * PITCH_NM      # |x| of the free-region outer edge
    names, cavity = tooth_names(spec.n_periods_side)
    comb_free = spec.free_comb and not spec.bare
    track_twin = (getattr(spec, "width_grad", False)
                  and getattr(spec, "wg_track_resonance", False))

    def func(p):
        corr, avg, shift = p[SL_CORR], p[SL_AVG], p[SL_SHIFT]
        w_n, w_w = (avg - corr / 2.0) * NM, (avg + corr / 2.0) * NM
        props = {}
        xl = -x_out                                # left walk, d = 25 .. 1
        for i in range(N_FREE - 1, -1, -1):
            s = shift[i]
            ln, lw = names[i + 1][0], names[i + 1][1]
            props[f"{ln}::x"], props[f"{ln}::x span"] = (xl + (hp - s) / 2.0) * NM, (hp - s) * NM
            props[f"{lw}::x"], props[f"{lw}::x span"] = (xl + (hp - s) + hp / 2.0) * NM, hp * NM
            props[f"{ln}::y span"], props[f"{lw}::y span"] = w_n[i], w_w[i]
            xl = xl + 2.0 * hp - s
        xr = cav_l0 / 2.0 + anp.sum(shift)         # right walk, d = 1 .. 25
        for i in range(N_FREE):
            s = shift[i]
            rn, rw = names[i + 1][2], names[i + 1][3]
            props[f"{rn}::x"], props[f"{rn}::x span"] = (xr + (hp - s) / 2.0) * NM, (hp - s) * NM
            props[f"{rw}::x"], props[f"{rw}::x span"] = (xr + (hp - s) + hp / 2.0) * NM, hp * NM
            props[f"{rn}::y span"], props[f"{rw}::y span"] = w_n[i], w_w[i]
            xr = xr + 2.0 * hp - s
        props[f"{cavity}::x"] = 0.0
        props[f"{cavity}::x span"] = (cav_l0 + 2.0 * anp.sum(shift)) * NM
        props[f"{cavity}::y span"] = p[I_CAV] * NM  # free cavity width (seed 800
                                                    # = the builder's "avg" option)
        if comb_free:
            d_c = p[I_DCOMB] * NM
            for j, (top, bot) in enumerate(scatterer_names()):
                props[f"{top}::radius"] = props[f"{bot}::radius"] = p[SL_R][j] * NM
                props[f"{top}::x"] = props[f"{bot}::x"] = p[SL_X][j] * NM
                props[f"{top}::y"], props[f"{bot}::y"] = d_c, -d_c
        if track_twin:
            # wg_track_resonance: pin the single-λ twin to the last measured
            # resonance (read PER CALL — the log callback advances it every
            # eval). A CONSTANT to autograd (no p dependence): zero Jacobian
            # row, zero dEps, survives regeneration because func props are
            # re-applied at every eval — see the CampaignSpec note.
            props["field_profile_adj::wavelength center"] = float(
                getattr(spec, "_wg_lam_track", 0.0)
                or spec.scan_center_nm) * NM
        return props

    return func


# ═══════════════════════════════════════════════════════════════════════════
# Cost function: soft-max reader + kappa-ratio penalty
# ═══════════════════════════════════════════════════════════════════════════

def _plain(x):
    """Peel autograd boxes → plain numpy (stop-gradient for index decisions)."""
    while hasattr(x, "_value"):
        x = x._value
    return np.asarray(x)


def measure_peak(wl_nm, T):
    """(λ_peak_nm, T_peak, fwhm_nm) from a plain spectrum.

    Resonance via the scored peak finder (NEVER argmax — CLAUDE.md §2);
    FWHM from half-max crossings by linear interpolation. fwhm_nm is None
    when a crossing leaves the recorded window (recenter condition).
    """
    wl_nm, T = np.asarray(wl_nm, dtype=float), np.asarray(T, dtype=float)
    if wl_nm[0] > wl_nm[-1]:
        wl_nm, T = wl_nm[::-1], T[::-1]
    i_pk = int(find_bragg_resonance(wl_nm, T))
    lam_pk, t_pk = wl_nm[i_pk], T[i_pk]
    half = t_pk / 2.0
    lo = hi = None
    for i in range(i_pk, 0, -1):
        if T[i - 1] <= half:
            f = (T[i] - half) / (T[i] - T[i - 1])
            lo = wl_nm[i] - f * (wl_nm[i] - wl_nm[i - 1])
            break
    for i in range(i_pk, len(T) - 1):
        if T[i + 1] <= half:
            f = (T[i] - half) / (T[i] - T[i + 1])
            hi = wl_nm[i] + f * (wl_nm[i + 1] - wl_nm[i])
            break
    fwhm = (hi - lo) if (lo is not None and hi is not None) else None
    return float(lam_pk), float(t_pk), fwhm


class RecenterNeeded(Exception):
    """Resonance drifted off the recorded grid — rebuild base + restart."""


class WidthTrip(Exception):
    """Measured sigma left the deadband while rho stayed inside — proxy failure."""


def make_fct(wl_nm):
    """FOM fct for lumopt2: T(λ) → windowed soft-max (autograd-differentiable).

    Window indices are picked on the DETACHED spectrum (stop-gradient), so
    the gradient flows only through the T values inside the window.
    """
    wl_nm = np.asarray(wl_nm, dtype=float)

    def fct(T):
        Tp = np.abs(_plain(T))
        lam_pk, t_pk, fwhm = measure_peak(wl_nm, Tp)
        if t_pk < DEAD_T_FLOOR:
            raise RuntimeError(f"dead device: peak T {t_pk:.4g} < {DEAD_T_FLOOR}")
        if fwhm is None:
            # Peak clipped at the band edge (measured: line-search probes jump
            # λ by up to +2.6 nm — jobs 54309/133016/54421 all died here when
            # this raised RecenterNeeded for MID-SEARCH probes). Degraded
            # fallback instead: soft-max over the FULL recorded band. A
            # clipped peak's visible maximum understates the true peak, so
            # the probe scores WORSE and L-BFGS-B backtracks naturally.
            # Recentering is handled by the log callback, and only when the
            # BEST design migrates (in-window evals are bit-identical).
            return anp.mean(anp.abs(T) ** P_SOFTMAX) ** (1.0 / P_SOFTMAX)
        idx = np.where(np.abs(wl_nm - lam_pk) <= WIN_FWHM_MULT * fwhm)[0]
        return anp.mean(anp.abs(T)[idx] ** P_SOFTMAX) ** (1.0 / P_SOFTMAX)

    return fct


def kappa_penalty(p):
    """Width anchor: asymmetric deadband on rho = mean(corr)/325 (autograd),
    PLUS the cavity-elongation guard (added 2026-08-16 after seed B walked
    the width-cheat): Σshift collectively elongates the cavity by 2Σs — the
    exact cavity-LENGTH knob the user excluded ("pure λ-tuner"). Measured
    violator: 2Σs=+255 nm → resonance toward the stopband edge → mirror
    penetration up → σ +9.6% while ρ stayed compliant. Deadband 120 nm of
    total elongation keeps the per-tooth DIFFERENTIAL freedom (the wanted
    physics); beyond it the quadratic overwhelms any measured FOM gain
    (β=1e-5/nm²: at the violator's 255 nm → penalty 0.18 vs gain 0.015)."""
    rho = anp.mean(p[SL_CORR]) / CORR_NM
    elong = 2.0 * anp.sum(p[SL_SHIFT])           # nm of cavity elongation
    return (BETA_UP * anp.maximum(0.0, rho - RHO_UP) ** 2
            + BETA_DN * anp.maximum(0.0, RHO_DN - rho) ** 2
            + BETA_ELONG * anp.maximum(0.0, anp.abs(elong) - ELONG_DEADBAND_NM) ** 2)


_kappa_penalty_grad = autograd.grad(kappa_penalty)


def elong_penalty(p):
    """Cheat-channel wall ONLY (rho band retired for v2-projection specs)."""
    elong = 2.0 * anp.sum(p[SL_SHIFT])
    return BETA_ELONG * anp.maximum(0.0, anp.abs(elong) - ELONG_DEADBAND_NM) ** 2


_elong_penalty_grad = autograd.grad(elong_penalty)


FW_A_MCORR = -0.0470     # um FWHM per nm of MEAN free-tooth corr (MEASURED 136051)
# ★2026-08-23 audit (trigger: 136466 eval 2 predicted 22.29 um, MEASURED 32.27):
# width-vs-elongation is CONVEX, not linear. The old FW_A_ELONG = 0.01355 was
# the [0, 130.6] nm secant of ONE pair (fspw_noshift -> fspw_best) applied as
# a local slope. Six corrected fspw points (noshift/best/d020-d080, corr-
# corrected via FW_A_MCORR) show the local slope rising 0.0135 -> 0.061 um/nm
# across elong 0 -> 211 nm; the uniform seed's own 0 -> 287.5 nm secant is
# 0.0484. dW = FW_C_ELONG * elong^2 fits all six to <= 0.25 um (best LINEAR
# fit leaves 0.96 um = 2.6 half-bands); anchored form is exact at each
# re-anchor. Still under-predicts the far tail (8.9 vs 13.9 um at 287.5) —
# acceptable: that regime is grossly over-band and self-rejecting either way.
FW_C_ELONG = 1.07e-4     # um FWHM per nm^2 of (2*sum(shift))^2 (6-pt fspw fit)
FW_A_ELONG = 0.01355     # um FWHM per nm of 2*sum(shift) — the OLD secant, kept
# as the DEFAULT so this file stays REQUEUE-safe for campaigns already in flight
# (136465/136466/136468): every Athena partition preempts, a requeued driver
# cold-restarts against whatever engine is deployed, and swapping the wall
# formula mid-campaign would be a silent numerics change (CLAUDE.md §6). Opt in
# per spec with fw_convex=True on the NEXT campaign, not by editing this line.
BETA_FW = 4.0            # per um^2 — 0.05 um over-band => ~0.01 FOM

# ★MEASURED elongation curve (IGUM 61742 + 61782, 2026-08-23) — 6 rungs on the
# UNIFORM corr-325 seed, pure common mode, pitch-locked mesh:
#   e (nm)   0       60       120      180      240      287.5
#   fwhm  18.3452  18.3108  20.4825  24.0150  28.7675  32.6984
# The response is FLAT to ~65 nm (e=60 measured -0.034 um, i.e. NO cost) and
# then knees into a steep rise. A threshold power law fits ALL SIX to
# <=0.106 um (half-band 0.367). Both earlier forms are refuted by this data:
# FW_A_ELONG over-taxes small shifts ~10x (predicts +0.813 um at e=60 where the
# truth is zero — this is what paralysed campaign 136466), and FW_C_ELONG
# under-taxes the tail (8.84 vs 14.35 at e=287.5).
FW_E0_NM   = 65.0        # knee (nm of 2*sum(shift)); below it shifts are free
FW_CURVE_N = 1.39
FW_CURVE_C = 7.8654e-3   # um per nm^N above the knee

# ★2026-08-24 (Fable audit, 136468 vs the hand see-saw): the wall's corr term
# was RANK-1 — mean(corr) only — so its gradient was identical on all 25 teeth
# and the per-tooth width price (the entire content of the see-saw rule) sat in
# its null space. The optimizer, told every tooth costs the same width per nm,
# rode to the band ceiling at T 0.917 (3 of its last 6 probes out of band)
# while the see-saw reached 0.938 IN band in one solve: the mean-corr model
# mis-scores that exact move by 0.82 um predicted vs 0.015 um measured.
# 3-block per-tooth slopes, um FWHM per nm of ONE tooth's corr (negative =
# raising narrows), MEASURED on the uniform corr-325 seed, pitch-locked mesh:
#   inner-8    -2.925e-3   in265, IGUM 61901: 1.4038 um / 480 tooth*nm
#                          (see-saw buy leg agrees: 0.02335/8 = 2.92e-3)
#   outer-8    -0.268e-3   out385, same job: 0.1285 um / 480 tooth*nm
#   middle-9   -2.384e-3   DERIVED so a uniform move reproduces FW_A_MCORR
#                          exactly (sum w_i = -0.0470)
# Known residual: the see-saw pay leg suggests the outer-17 average is ~30%
# higher than this profile — the per-accepted-best re-anchor and the measured
# fwhm_env guard (WidthTrip / _best_from_log) absorb errors of that scale.
FW_TOOTH_W = (-2.925e-3,) * 8 + (-2.384e-3,) * 9 + (-0.268e-3,) * 8


# ★2026-08-24 — THE COEFFICIENT DOES NOT TRANSFER BETWEEN DEVICE CLASSES
# (skill item 35; the user's point: once the device is not uniform the envelope
# is no longer a single exponential, so no one constant describes the decay).
# FW_CURVE_C above is fitted on the UNIFORM corr-325 seed. MEASURED on the
# APODIZED best device (BEST_T9636, mcorr 357.95) by the shiftw ladder
# (Athena 136710, pitch-locked, v2 numerics):
#     e (nm)    0        66.3      132.6     198.9
#     fwhm_env  16.85373 16.83672  18.35309  20.62662
# The KNEE TRANSFERS (e=66.3 measures 0.017 um NARROWER than e=0, same as the
# uniform seed's e=60), so FW_E0_NM and FW_CURVE_N keep their 6-point support
# and only C is refitted: least squares on the two above-knee points gives
# 4.1993e-3, residuals -0.040/+0.015 um against a 0.367 um half-band.
# The uniform constant over-taxes this device class by 1.87x.
FW_CURVE_C_APOD = 4.1993e-3   # um per nm^N, apodized devices


def _fw_elong_curve(e, c=None):
    """Measured width cost of elongation, um. Zero below the knee; the n>1
    power keeps it C1 there, so the gradient is continuous for L-BFGS-B.
    `c` selects the device-class coefficient (None = uniform FW_CURVE_C)."""
    return ((FW_CURVE_C if c is None else c)
            * anp.maximum(0.0, e - FW_E0_NM) ** FW_CURVE_N)


def make_fwhm_wall(spec):
    def pen(p):
        anc = spec.fw_anchor
        elong = 2.0 * anp.sum(p[SL_SHIFT])
        if spec.fw_curve:
            cc = getattr(spec, "fw_curve_c", None)   # device-class coefficient
            d_elong = (_fw_elong_curve(elong, cc)
                       - _fw_elong_curve(anc["elong"], cc))
        else:
            d_elong = (FW_C_ELONG * (elong ** 2 - anc["elong"] ** 2) if spec.fw_convex
                       else FW_A_ELONG * (elong - anc["elong"]))
        if spec.fw_tooth_w is not None:
            # per-tooth price (see FW_TOOTH_W) — anchor must carry corr_vec
            d_corr = anp.sum(anp.asarray(spec.fw_tooth_w)
                             * (p[SL_CORR] - anp.asarray(anc["corr_vec"])))
        else:
            d_corr = FW_A_MCORR * (anp.mean(p[SL_CORR]) - anc["mcorr"])
        fhat = anc["fwhm"] + d_corr + d_elong
        hi, lo = RHO_UP * spec.fwhm0_um, RHO_DN * spec.fwhm0_um
        band = (BETA_FW * anp.maximum(0.0, fhat - hi) ** 2
                + BETA_FW * anp.maximum(0.0, lo - fhat) ** 2)
        cap = getattr(spec, "fw_pen_cap", None)
        if cap:
            # ★2026-08-24: SATURATE the hinge. Campaign 136640 died at eval ~6
            # with ABNORMAL_TERMINATION_IN_LNSRCH because fw_curve makes the
            # out-of-band penalty enormous (e=287 -> 793) while the shift block's
            # wide bounds (0-200 nm/tooth) put L-BFGS-B's unit-norm scaled probe
            # right there — no acceptable decrease exists along the ray, so the
            # line search aborts. tanh keeps the SMALL-excess behaviour
            # identical (tanh x = x - x^3/3 ...; at the band edge the penalty is
            # ~1e-3 of the cap, so in-band and near-band gradients are
            # unchanged) and saturates far out, leaving a bounded, navigable
            # landscape. Rejection is preserved: any FOM here is O(0.7), so a
            # cap of ~2 still scores an out-of-band design well below every
            # in-band one. The MEASURED fwhm_env guard (WidthTrip) and the
            # fail-closed _best_from_log remain the actual authority.
            # Rational saturation, not tanh: identical as band->0 (cap*b/(cap+b)
            # = b - b^2/cap + ...), saturates at `cap`, and has NO exponential,
            # so it cannot overflow the way tanh's cosh^2 derivative does at
            # large excess (measured: RuntimeWarning at e>=163).
            band = cap * band / (cap + band)
        return band + elong_penalty(p)
    return pen, autograd.grad(pen)

# ── Stage-3 sigma-hat wall constants (MEASURED, tangent probe job 133512 +
#    12-row regression R²=0.997; see campaign-state memory 2026-08-17) ──────
# ★RECALIBRATED 2026-08-18 by least squares on FOUR measured points taken AT
# the current frontier (shift ladder x1.5 = shifts only, job 134033; plus the
# three sigma-neutral trade rows d+20/40/60, job 134107). Residuals <=0.007 um.
# The old pair came from a 2-point probe at the STAGE-1 operating point and was
# wrong where it mattered: it predicted the trade rows were EXACTLY width-
# neutral (0.0000 um) when they actually widened by 0.023/0.056/0.096 um, i.e.
# the model said "free" to a move that spent a third of the remaining band.
SIG_A_SHIFT = 0.00409    # um sigma per nm of 2*sum(shift)  (was 0.00368, +11%)
SIG_A_RHO   = -2.693     # um sigma per unit rho_free       (was -3.85,  -30%:
                         # corrugation has far LESS width authority than assumed,
                         # which is why every "payback" under-compensated)
SIG_A_WCAV  = 0.01       # um sigma per nm of cavity y-width. ★CORRECTED
                         # 2026-08-17 ~15:00: the 12-row fit said 0.109 but was
                         # collinearity-contaminated (probe rows moved wcav+
                         # shifts+corr together); stage-2 eval 2 measured the
                         # CLEAN experiment — wcav +13.4 nm at frozen shifts,
                         # sigma FLAT (+0.0016 um) ⇒ true coefficient ~0.
                         # Kept small positive as safety margin. (Job 133541
                         # launched with the LOADED 0.109 — conservative
                         # direction, over-taxes a good lever; restart to
                         # adopt only if stage-3 plateaus while stage-2 wins
                         # on wcav.)
SIG_CEIL_UM = 17.795     # 1.02*sigma0(17.493) − 0.045 safety for model error
SIG_FLOOR_UM = 16.62     # 0.95*sigma0
BETA_SIG    = 4.0        # per um²: 0.01 FOM at 0.05 um over — old-wall stiffness


SIG_RESID_WARN = 0.020   # um: surrogate error that must never pass silently

# ★2026-08-18, user instruction: the raw-line FWHM metric and the linear
# FWHM_A_* slopes fitted to it, and the coupled-mode-theory width model, were
# ALL DELETED. The slopes were calibrated on profiles sampled at the transverse
# BOX EDGE (see profile_line's bug note), so every number derived from them was
# contaminated by radiation rather than measuring the guided mode; the CMT model
# was then "validated" against those same bad numbers. The width observable is
# now exactly one thing: fwhm_env_of_line == the project's fwhm_m. Do not
# reintroduce a second convention or a fitted slope without measuring it on
# y-INTEGRATED profiles first.


def sigma_hat_of(spec, p):
    """The width SURROGATE's prediction for p (None if no anchor is set).

    Same expression the wall penalises on — exposed so every evaluation can
    log PREDICTED next to MEASURED. See check_sigma_surrogate.
    """
    anc = getattr(spec, "sig_anchor", None)
    if not anc:
        return None
    p = np.asarray(_plain(p), dtype=float)
    return float(anc["sigma"]
                 + SIG_A_SHIFT * (2.0 * p[SL_SHIFT].sum() - anc["elong"])
                 + SIG_A_RHO * (p[SL_CORR].mean() / CORR_NM - anc["rho"])
                 + SIG_A_WCAV * (p[I_CAV] - anc["wcav"]))


def check_sigma_surrogate(row):
    """★NEVER-AGAIN GUARD (2026-08-18, user: "think how we don't have this
    problem in the future").

    TWICE in one campaign a FITTED SURROGATE silently replaced a MEASURABLE
    quantity and nobody compared them: (1) the sigma-hat wall's coefficients,
    fitted at the stage-1 operating point, over-stated corrugation's width
    authority by 30% — it called the trade rows "width-neutral" while they
    actually spent a third of the remaining band, and it falsely rejected an
    in-band T 0.9591 design on seed B; (2) sigma itself stands in for the
    ACOUSTIC SPEC's FWHM, and the two were never measured together.

    The general rule this encodes: **if a surrogate can be checked against a
    measurement on the very same evaluation, check it there, every time, and
    make the disagreement loud.** Costs nothing (both numbers are already in
    hand), and turns a silent modelling error into a visible one.
    """
    sh, sg, fw = (row.get("sigma_hat_um"), row.get("sigma_um"),
                  row.get("fwhm_env_um"))
    # FWHM/sigma logged unconditionally — it needs no anchor, and gating it
    # behind one made the audit's own rows log None (found 2026-08-18, minutes
    # after shipping this guard). A shape-tracking diagnostic must never depend
    # on whether a wall happens to be configured.
    # NOTE: no alarm threshold here yet. The old one keyed off a reference
    # (0.978) measured with the deleted raw-line metric on box-edge profiles.
    # Re-derive it from y-integrated fwhm_env rows before re-arming.
    if fw and sg:
        row["fwhm_over_sigma"] = round(fw / sg, 4)
    if sh is None or sg is None:
        return
    row["sigma_resid_um"] = round(sh - sg, 6)
    if abs(sh - sg) > SIG_RESID_WARN:
        print(f"[WIDTH-SURROGATE OFF] predicted {sh:.4f} um, measured {sg:.4f} um "
              f"(resid {sh-sg:+.4f}) — the wall is steering on a stale model; "
              f"refit SIG_A_* on the logged rows before trusting further steps.")


def make_sigma_wall(spec):
    """(penalty, grad) closures for the single sigma-hat hinge.

    ★DORMANT + CARRIES THE RANK-DEFICIENCY BUG (audit 2026-08-24). No live spec
    sets sigma_wall (sigma was retired as the width authority — HANDOFF §0), and
    this surrogate prices corr by anp.mean(p[SL_CORR]) and shifts by
    anp.sum(p[SL_SHIFT]) — the exact collapse that made fwhm_wall blind to the
    see-saw (skill item 34). Do NOT enable it without porting the per-tooth
    price (see FW_TOOTH_W / spec.fw_tooth_w); a scalar block summary gives
    L-BFGS-B one identical gradient across 25 teeth and converges to the wrong
    fixed point, not slowly to the right one.
    """
    anc = spec.sig_anchor

    def pen(p):
        rho = anp.mean(p[SL_CORR]) / CORR_NM
        elong = 2.0 * anp.sum(p[SL_SHIFT])
        sig_hat = (anc["sigma"] + SIG_A_SHIFT * (elong - anc["elong"])
                   + SIG_A_RHO * (rho - anc["rho"])
                   + SIG_A_WCAV * (p[I_CAV] - anc["wcav"]))
        return (BETA_SIG * anp.maximum(0.0, sig_hat - SIG_CEIL_UM) ** 2
                + BETA_SIG * anp.maximum(0.0, SIG_FLOOR_UM - sig_hat) ** 2)

    return pen, autograd.grad(pen)


def two_kappa_L(p, n_side):
    """2*kappa*L over one side (the surrogate-rule quantity), kappa ∝ corr."""
    per_um = PITCH_NM * 1e-3
    frozen = 2.0 * KAPPA_PER_UM * (n_side - N_FREE) * per_um
    free = 2.0 * KAPPA_PER_UM * float(np.sum(_plain(p)[SL_CORR]) / CORR_NM) * per_um
    return frozen + free


def attach_penalty(project, spec=None):
    """Subtract the analytic width penalty (and its exact gradient) from the
    project's FOM — Optimization reads both through these two methods.

    Option-1 hook: spec.grad_cal (per-class factors from measured B3 α) is
    applied to the LUMOPT2 gradient only — the penalty gradient is analytic
    and exact, so it is added after calibration, uncalibrated."""
    fom0, grad0 = project.compute_fom, project.compute_gradient
    cal = getattr(spec, "grad_cal", None) if spec is not None else None
    cal_slices = {"corr": SL_CORR, "avg": SL_AVG, "shift": SL_SHIFT,
                  "r": SL_R, "x": SL_X, "d": slice(I_DCOMB, I_DCOMB + 1),
                  "cav": slice(I_CAV, I_CAV + 1)}
    if spec is not None and getattr(spec, "wg_project", False):
        # ★PROJECTION MODE: NO analytic width penalty at all (audit S4,
        # 2026-08-25). Width is steered by the null-space projection using the
        # EXACT grad-W, so any wall here is double-pricing — and the
        # elongation wall in particular would cap e near 120 nm, structurally
        # below BEST_T9636's own e = 132.6, forbidding the from-uniform run
        # from reaching the basin it exists to test. WidthTrip still
        # fail-closes the DELIVERED design, so the spec is still enforced.
        pen = lambda p: 0.0
        pen_grad = lambda p: np.zeros_like(np.asarray(p, dtype=float))
    elif spec is not None and getattr(spec, "sigma_wall", False) and spec.sig_anchor:
        pen, pen_grad = make_sigma_wall(spec)     # stage-3: one wall on sigma-hat
    else:
        if spec is not None and getattr(spec, "fwhm_wall", False):
            # fail LOUD: a missing anchor silently removed all width steering
            # from the gradient before this guard (proactive audit 2026-08-23)
            assert spec.fw_anchor, "fwhm_wall=True requires fw_anchor"
            pen, pen_grad = make_fwhm_wall(spec)
        elif spec is not None and not getattr(spec, "rho_band", True):
            pen, pen_grad = elong_penalty, _elong_penalty_grad
        else:
            pen, pen_grad = kappa_penalty, _kappa_penalty_grad

    def compute_fom(params=None):
        pp = params if params is not None else project.parametrization.get_initial_params()
        return fom0(params) - float(pen(np.asarray(pp, dtype=float)))

    def compute_gradient(params=None):
        pp = params if params is not None else project.parametrization.get_initial_params()
        g = np.asarray(grad0(params), dtype=float)
        if cal:
            for name, factor in cal.items():
                g[cal_slices[name]] *= float(factor)
        return g - pen_grad(np.asarray(pp, dtype=float))

    # Keep the UNPENALIZED handles for the gate entry points: lumopt2's
    # validate_gradient FDs through fom.calculate_fom (RAW — no penalty) but
    # takes its adjoint from project.compute_gradient (wrapped) — an
    # asymmetry that is invisible while pen_grad = 0 at the probe point, and
    # a contamination whenever it isn't (found 2026-08-23 review: at the
    # detune-1 gate point elong = 1000 nm > the 120 nm deadband, so the
    # kappa/elong guard adds 0.0352 per shift param — same order as
    # d(softW)/dshift itself).
    project.compute_fom_raw, project.compute_gradient_raw = fom0, grad0
    project.compute_fom, project.compute_gradient = compute_fom, compute_gradient


# ── Option 2: E∥/D⊥ boundary correction (Johnson PRE 65 066611) ────────────
# lumopt2 stores dEps per FIELD COMPONENT (eps_x/eps_y/eps_z sparse arrays),
# and our walls are axis-aligned, so the correct physics — tangential E keeps
# the plain Δε weight, normal E must be re-expressed as D with the harmonic
# Δ(1/ε) weight — reduces to scaling the normal-component slice of each
# wall-moving parameter by R = -Δ(1/ε)/Δε · ε_eval². Width params (corr/avg)
# move y-normal walls; shift params move x-normal walls; comb cylinders are
# left untouched (measured healthy, curved boundary).

def bc_normal_weight(eps_eval):
    """R factor for the normal component at our material pair."""
    d_eps = EPS_CORE - EPS_CLAD
    d_inv = 1.0 / EPS_CORE - 1.0 / EPS_CLAD
    return -d_inv / d_eps * eps_eval ** 2


def boundary_weights(spec):
    """param index → (wx, wy, wz) dEps-component weights for the bc patch."""
    r = bc_normal_weight(spec.bc_eps_eval)
    w = {}
    for sl, wxyz in ((SL_CORR, (1.0, r, 1.0)),      # y-normal walls
                     (SL_AVG, (1.0, r, 1.0)),
                     (SL_SHIFT, (r, 1.0, 1.0))):    # x-normal walls
        for i in range(sl.start, sl.stop):
            w[i] = wxyz
    w[I_CAV] = (1.0, r, 1.0)                        # cavity wall is y-normal
    return w


def make_bc_parametrization(lmpt):
    """Parametrization subclass applying boundary_weights in the contraction."""
    from lumopt2.parametrization.d_eps_calculator import DEpsCalculator

    class BoundaryCorrected(lmpt.Parametrization):
        bc_weights = {}                     # set per instance in make_project

        def compute_gradient_from_fields(self, gradient_fields, fdtd_session,
                                         params, polygon_name=None):
            if not self.bc_weights:
                return super().compute_gradient_from_fields(
                    gradient_fields, fdtd_session, params)
            # The reweight below skips the parent's dP_dp chain rule, which is
            # only the identity in the directly_on_params=True path.
            assert self.directly_on_params, \
                "bc_patch supports directly_on_params=True only"
            d_eps, global_shape, _ = (
                self.compute_opt_params_direct_to_permittivity_jacobian(
                    use_jac=self.use_jac, fdtd_session=fdtd_session,
                    params=params))
            for i, (wx, wy, wz) in self.bc_weights.items():
                entry = d_eps[f"params[{i}]::params[{i}]"]
                for comp, wgt in (("eps_x", wx), ("eps_y", wy), ("eps_z", wz)):
                    entry[comp]["data"] = entry[comp]["data"] * wgt
            dfom, _ = DEpsCalculator.calculate_dF_dPi(
                d_eps_data=d_eps,
                dF_deps=np.expand_dims(gradient_fields, axis=3),
                global_shape=global_shape)
            return np.asarray(dfom).reshape(-1)   # dP_dp = identity here

    return BoundaryCorrected


# ═══════════════════════════════════════════════════════════════════════════
# Base simulation (.fsp) — reuses the production builder unchanged
# ═══════════════════════════════════════════════════════════════════════════

def build_base_cfg(spec):
    """SimulationConfig for the campaign base scene (comb_q3db numerics family)."""
    cfg = _common.build_ports_base()
    cfg.y_span_override_m = spec.box_y_um * 1e-6
    cfg.span_multiplier_override = spec.box_z_mult
    cfg.geometry.corrugation_depth_m = CORR_NM * NM   # NOT cfg.grating — silent no-op
    cfg.grating.n_periods_each_side = spec.n_periods_side
    cfg.spectral.center_wavelength_m = spec.scan_center_nm * NM
    cfg.spectral.scan_width_nm = spec.scan_width_nm
    cfg.spectral.n_wl_points = spec.n_wl_points
    if spec.bare:
        cfg.scatterer.enabled = False
        cfg.scatterer.radius_m = 0.0
    else:
        p0 = seed_params(spec)
        cfg.scatterer.enabled = True
        cfg.scatterer.radius_m = COMB_R_NM * NM
        if spec.comb_n_half is None:
            x_nm = list(p0[SL_X])
        else:
            # Count study: regenerate the lattice at the requested half-count.
            # (The optimized vector's own x sit within 0.73 nm of this lattice,
            # so the grating sees the same comb, just longer or shorter.)
            k = spec.comb_n_half
            x_nm = [j * COMB_LAM_NM + COMB_DX_NM for j in range(-k, k + 1)]
        cfg.scatterer.x_list_m = [x * NM for x in x_nm]
        cfg.scatterer.y_list_m = [p0[I_DCOMB] * NM] * len(x_nm)
        cfg.scatterer.height_m = 350.0 * NM
    assert cfg.symmetry.use_z_symmetry, "comb is z-symmetric — keep the 2x z saving"
    return cfg


def build_base_fsp(spec, out_path):
    """Build the seed scene with the production builder and save it (no run).

    Returns the port wavelength grid in nm (uniform in FREQUENCY, matching
    the monitor sampling, so PortResults keys land exactly on the samples).
    """
    from bragg_device import PiShiftBraggFDTD
    cfg = build_base_cfg(spec)
    sim = PiShiftBraggFDTD(**cfg.to_device_kwargs())
    try:
        sim.build()
        sim.update_scan(center_lambda_m=cfg.spectral.center_wavelength_m,
                        width_nm=cfg.spectral.scan_width_nm,
                        n_points=cfg.spectral.n_wl_points)
        apply_monitor_overrides(sim, cfg)
        if getattr(spec, "width_grad", False):
            # Single-λ twin of field_profile for the width-FOM entry. lumopt2's
            # generate() REJECTS broadband FieldRegion monitors outright
            # (fdtd_session._verify_not_broadband — measured, job 135954 all
            # tasks, 35 s). The broadband field_profile stays untouched: it is
            # the measured-fwhm_env authority. The twin samples at the scan
            # CENTER (recenter policy keeps the resonance within ±2 nm); the
            # off-peak λ error is absorbed by C_field + per-restart anchoring.
            # ★Must be a FieldRegion object, NOT a copied DFT monitor: in this
            # build NO monitor type has 'source mode' (measured 2026-08-22 —
            # cost job 135986's three forwards). lumopt2's FieldFom is written
            # for the addfieldregion object, which is monitor + adjoint-source
            # in one. Geometry copied from field_profile; single λ at the scan
            # center; 'use source limits' off (broadband-validator trap,
            # job 135954); source mode OFF in the forward.
            f = sim.fdtd
            geo = {k: float(np.squeeze(f.getnamed("field_profile", k)))
                   for k in ("x", "x span", "y", "y span", "z")}
            f.addfieldregion()
            f.set("name", "field_profile_adj")
            f.set("monitor type", "2D Z-normal")
            for k, v in geo.items():
                f.set(k, v)
            f.set("override global monitor settings", 1)
            f.set("use source limits", 0)
            f.set("frequency points", 1)
            lam_c = cfg.spectral.center_wavelength_m
            f.set("wavelength center", lam_c)
            f.set("wavelength span", 0.0)
            f.set("source mode", 0)
            n_tiles = int(getattr(spec, "wg_src_tiles", 1) or 1)
            if n_tiles > 1:
                # x-tiled adjoint sources — see CampaignSpec.wg_src_tiles.
                # Clones of the twin, narrowed in x. What is built here is a
                # PLACEHOLDER partition of the nominal span: the final x
                # geometry is set at adjoint time from the monitor's own
                # recorded x samples (import_tiled_source), once the mesh is
                # frozen. Source mode OFF in the forward, so lumopt2's
                # get_active_sources (fdtd_session.py:483-489) does not see
                # them and its per-config `sources` list is untouched; they
                # are not in sim_results either, so _verify_not_broadband
                # (fdtd_session.py:806) and the symmetry-factor check
                # (fdtd_session.py:699) never look at them.
                dx = getattr(spec, "region_dx_nm", 50.0) * NM
                cap = int(getattr(spec, "wg_tile_max_xcells", 528))
                cells = int(np.ceil(geo["x span"] / dx / n_tiles))
                assert cells <= cap, (
                    f"wg_src_tiles={n_tiles} leaves ~{cells} x cells per tile "
                    f"> {cap} (measured GPU FieldRegion-source ceiling) — "
                    f"raise wg_src_tiles")
                edges = np.linspace(geo["x"] - geo["x span"] / 2.0,
                                    geo["x"] + geo["x span"] / 2.0, n_tiles + 1)
                for t in range(n_tiles):
                    f.addfieldregion()
                    f.set("name", f"field_profile_adj_t{t}")
                    f.set("monitor type", "2D Z-normal")
                    f.set("y", geo["y"])
                    f.set("y span", geo["y span"])
                    f.set("z", geo["z"])
                    f.set("x", 0.5 * (edges[t] + edges[t + 1]))
                    f.set("x span", edges[t + 1] - edges[t])
                    f.set("override global monitor settings", 1)
                    f.set("use source limits", 0)   # broadband-validator trap
                    f.set("frequency points", 1)
                    f.set("wavelength center", lam_c)
                    f.set("wavelength span", 0.0)
                    f.set("source mode", 0)         # forward: pure monitor
            if getattr(spec, "wg_source", "fieldregion") == "import":
                # geometry/span props are INACTIVE on an import source until
                # a dataset is imported (measured, local W1.6) — the sheet's
                # geometry arrives WITH the dataset at adjoint-setup time.
                f.addimportedsource()
                f.set("name", "width_adj_src")
                f.set("injection axis", "z")
                f.set("z", geo["z"])
                f.set("enabled", 0)            # forward: OFF
        # lumopt2's dEps is FD-through-the-mesher; with the family default
        # "conformal variant 0" the grid-aligned TOOTH edges give staircased
        # dEps — MEASURED (job 132637): tooth adjoint/FD scale 0.07-0.26 while
        # the comb cylinders were fine (0.77-0.98). "precise volume average"
        # is the lumopt2-recommended refinement for exactly this. LUMOPT2 PATH
        # ONLY — production SweepSpec studies keep the family default; this is
        # a NAMED §2 numerics change, in-study anchors re-measured at PVA.
        # ★spec.mesh_refinement (2026-08-22): "conformal variant 0" here runs
        # the IDENTITY TEST — lumopt2 stack at the family mesher, to be
        # compared against the stored regular-pipeline anchors.
        sim.fdtd.setnamed("FDTD", "mesh refinement",
                          getattr(spec, "mesh_refinement",
                                  "precise volume average"))
        # PortResults matches wavelengths within 1e-9 m (docs audit risk #2):
        # our λ list assumes FREQUENCY-uniform monitor sampling — assert the
        # global monitor is not wavelength-spaced, or the outer window points
        # could silently snap to neighbors.
        assert not int(sim.fdtd.getglobalmonitor("use wavelength spacing")), \
            "global monitor is wavelength-spaced; PortResults λ list assumes frequency spacing"
        _assert_name_map(sim.fdtd, spec)
        freqs = np.linspace(C0 / sim.lam_max, C0 / sim.lam_min, sim.n_wl_points)
        sim.fdtd.save(out_path)
    finally:
        sim.close()
    return (C0 / freqs) / NM      # nm, ascending in frequency (descending in λ)


def _assert_name_map(fdtd, spec):
    """Every object the func will touch must exist in the scene — never trust
    the name formula (the job-130145 class of bug)."""
    names, cavity = tooth_names(spec.n_periods_side)
    wanted = [n for quad in names.values() for n in quad] + [cavity]
    if not spec.bare:
        wanted += [n for pair in scatterer_names(comb_count(spec)) for n in pair]
    missing = [n for n in wanted if not fdtd.getnamednumber(n)]
    if missing:
        raise RuntimeError(f"object-name map mismatch, missing {len(missing)}: "
                           f"{missing[:8]} ...")


# ═══════════════════════════════════════════════════════════════════════════
# lumopt2 wiring
# ═══════════════════════════════════════════════════════════════════════════

def import_lumopt2():
    """Import lumopt2 from the same install tree as lumapi (works on all 3 envs).

    Also applies the SlurmRunner fix unconditionally (user 2026-08-13, "fix it
    for always"): the shipped R1.3 SlurmRunner imports lumopt2.utils.lumslurm,
    which does not exist — the real module is api/python/lumslurm.py one level
    up. Registering it in sys.modules makes SlurmRunner constructible without
    touching the container image.
    """
    api_dir = os.path.dirname(config.LUMAPI_PATH)
    if api_dir not in sys.path:
        sys.path.insert(0, api_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")     # never open windows (§5 silent rule)
    import lumopt2
    try:
        import lumslurm
        sys.modules["lumopt2.utils.lumslurm"] = lumslurm
    except ImportError:
        pass                     # no lumslurm on this install → LocalRunner only

    # lumopt2 BUG (dev246, measured jobs 132730/132735): with n_params == 1 the
    # dp validator crashes (length-1 dp.squeeze() is 0-d → shape[0] IndexError).
    # Faithful replacement, differing only in atleast_1d after the squeeze.
    from lumopt2.parametrization.d_eps_calculator import DEpsCalculator as _DEC

    def _validate_and_normalize_dp(self, params, dp):
        if (params.ndim != 1) or (params.shape[0] != self.parametrization.n_params):
            raise ValueError(
                f"params must be a 1D array with length {self.parametrization.n_params}")
        bounds = np.array(self.parametrization.get_bounds()).reshape(-1, 2)
        if dp is None:
            dp = np.sqrt(np.finfo(np.float32).eps) * np.abs(bounds[:, 1] - bounds[:, 0])
        elif isinstance(dp, np.ndarray):
            dp = np.atleast_1d(dp.squeeze())
            if dp.shape[0] != self.parametrization.n_params:
                raise ValueError(
                    f"If dp is an array, it must have length {self.parametrization.n_params}.")
        elif isinstance(dp, float):
            dp = np.full((params.shape[0],), dp)
        else:
            raise ValueError(
                "dp must be None, a float, or a 1D array of the correct length. "
                f"Got type(dp)={type(dp)} with value {dp} instead.")
        return dp

    _DEC._validate_and_normalize_dp = _validate_and_normalize_dp

    # lumopt2 BUG (dev246, measured jobs 132730/132735): when ANY exception
    # escapes mid-run, Optimization.run calls on_optimization_end with
    # final_fom=None and the auto-installed FileLogger crashes formatting it —
    # the TypeError then MASKS the original exception (it would swallow our
    # RecenterNeeded/WidthTrip and break the campaign restart loop). Guard the
    # logger so the ORIGINAL exception propagates; this except hides only the
    # known cosmetic logging crash, never simulation errors.
    from lumopt2.utils.file_logger import FileLogger as _FL
    _orig_end = _FL.on_optimization_end

    def _safe_end(self, success, final_fom, final_params, num_iterations, **kw):
        try:
            return _orig_end(self, success, final_fom, final_params,
                             num_iterations, **kw)
        except Exception as e:
            print(f"[lumopt2 patch] FileLogger.on_optimization_end suppressed "
                  f"(final_fom={final_fom!r}): {e!r}")

    _FL.on_optimization_end = _safe_end
    return lumopt2


def make_project(spec, out_dir, lmpt=None):
    """Base .fsp + Parametrization + soft-max PortFom + GPU runner → Project.

    Returns (project, wl_nm) with the width penalty already attached.
    """
    lmpt = lmpt or import_lumopt2()
    if getattr(spec, "width_grad", False):
        # fail BEFORE the scene build — a misconfigured spec must not cost a
        # cluster minute (or a campaign) to discover.
        # fwhm0_um may be None ONLY on an explicit smoke spec (signalled by
        # two_kl_floor being set) — a physics campaign without a width band
        # must still die here, before the scene build.
        _smoke = getattr(spec, "two_kl_floor", None) is not None
        assert (spec.fwhm0_um or _smoke) and spec.wg_anchor and \
            {"softw", "fwhm"} <= set(spec.wg_anchor), \
            "width_grad requires fwhm0_um (or smoke spec) + wg_anchor {'softw','fwhm'}"
    os.makedirs(out_dir, exist_ok=True)
    fsp = os.path.join(out_dir, f"{spec.label}_base.fsp")
    wl_nm = build_base_fsp(spec, fsp)

    dp = np.full(N_PARAMS, spec.dp_tooth_nm)       # dp lives in PARAM units (nm)
    dp[SL_R] = dp[SL_X] = dp[I_DCOMB] = spec.dp_comb_nm
    # Explicit uniform mesh over the region is REQUIRED by lumopt2 (the dEps
    # grid locks to it; dx=None crashes addmesh). 50 nm = the optimization-mesh
    # cell, so the override matches the global mesh; B2 quantifies any residual
    # numerics shift vs the stored anchor.
    region = lmpt.Box(
        x_span=2.0 * (COMB_N_HALF * COMB_LAM_NM + COMB_DX_NM + 240.0 + 500.0) * NM,
        y_span=2.0 * (spec.box_y_um * 500.0 - 900.0) * NM,
        z_span=0.8e-6,
        dx=getattr(spec, "region_dx_nm", 50.0) * NM, dy=50e-9, dz=50e-9,
    )
    # Containment (docs audit risk #5 — "the docs' single loudest warning"):
    # the optimization region must contain every geometry excursion at the
    # BOUNDS EXTREMES, not just the seed layout.
    y_half = spec.box_y_um * 500.0 - 900.0                        # region, nm
    x_half = COMB_N_HALF * COMB_LAM_NM + COMB_DX_NM + 240.0 + 500.0
    assert (825.0 + 500.0 / 2.0) / 2.0 < y_half, \
        "widest tooth edge (avg_max + corr_max/2)/2 exceeds region y half-span"
    assert 1960.0 + 240.0 <= y_half, "comb d_max + r_max exceeds region y half-span"
    if spec.comb_n_half is None:
        assert COMB_N_HALF * COMB_LAM_NM + COMB_DX_NM + 100.0 + 240.0 < x_half, \
            "comb x bound + r_max exceeds region x half-span"
    # else: count study. The comb is FROZEN (comb_count asserts free_comb=False),
    # so nothing differentiable lives out there and containment does not apply.
    # The region deliberately keeps its DEFAULT span: its explicit mesh matches
    # the 50 nm global mesh, so posts outside it are meshed identically — and
    # holding the span fixed keeps runtime/memory equal to the other scan tasks.

    # FD step must fit inside each param's bounds: lumopt2's dEps calculator
    # raises when 2*dp exceeds the bound range (killed stage-2 job 133499 on
    # the frozen-shift slivers; the bare campaign's frozen-comb slivers were
    # the same latent crash, never exercised because B2 computed no gradient).
    # Clamp dp per-param to a quarter of the range — frozen blocks then get a
    # ~5e-4 nm step whose dEps is below mesher resolution, i.e. gradient 0,
    # which is exactly what "frozen" means.
    bnds = param_bounds(spec)
    for _i, (_lo, _hi) in enumerate(bnds):
        if 2.0 * dp[_i] >= (_hi - _lo):
            dp[_i] = (_hi - _lo) / 4.0
    par_cls = make_bc_parametrization(lmpt) if spec.bc_patch else lmpt.Parametrization
    parametrization = par_cls(
        func=make_func(spec), bounds=bnds,
        optimization_region=region, initial_params=seed_params(spec),
        dp=list(dp),
    )
    if spec.bc_patch:
        parametrization.bc_weights = boundary_weights(spec)
    if getattr(spec, "wg_project", False):
        assert getattr(spec, "width_grad", False), "wg_project needs width_grad=True"
        assert spec.wgp_target_um, "wg_project needs wgp_target_um (= W_hi - margin)"
        assert not spec.bc_patch, "wg_project memo assumes stock Parametrization"
        # Single-slot dEps memo keyed on params: grad_W needs a SECOND
        # compute_gradient_from_fields at the SAME params, and lumopt2 re-runs
        # the mesher-FD jacobian on every call (parametrization.py:520-530).
        for _name in ("compute_lumerical_to_permittivity_jacobian",
                      "compute_opt_params_direct_to_permittivity_jacobian"):
            _meth, _c = getattr(parametrization, _name), {}

            def _memo(*a, __m=_meth, __c=_c, **kw):
                p_kw = kw.get("params")
                if p_kw is None:                      # positional call → no memo
                    return __m(*a, **kw)
                k = np.asarray(p_kw, dtype=float).tobytes()
                if __c.get("k") != k:
                    __c["k"], __c["v"] = k, __m(*a, **kw)
                return __c["v"]

            setattr(parametrization, _name, _memo)
    port_res = lmpt.PortResults("Port_2", "transmission", [w * NM for w in wl_nm])
    if getattr(spec, "width_grad", False):
        # V2 (skill item 28): second FOM entry = softW with a weighted
        # monitor-source adjoint; fct = softmax T − AL width-band penalty.
        # (spec validity asserted at the top of make_project, pre-build.)
        WidthResults, MixedFom = make_width_classes(lmpt, spec)
        fom = MixedFom([port_res,
                        WidthResults(spec.scan_center_nm * NM)],
                       fct=make_fct_v2(wl_nm, spec))
        # port-entry C is applied inside MixedFom per entry — do NOT also
        # wrap _compute_adjoint_fields_phased below (would double-apply).
    else:
        fom = lmpt.Fom(port_res, fct=make_fct(wl_nm))
    if spec.adj_phase_fix and not getattr(spec, "width_grad", False):
        # Root-cause fix (2026-08-16): the adjoint's gradient projection is a
        # small real part of a large ~90°-rotated complex sum; a measured
        # global complex factor C (see CampaignSpec.adj_fix_re/im) corrects
        # the ~6.7° phase error + amplitude in one multiplication of the
        # scaled adjoint fields.
        _orig_adj = fom._compute_adjoint_fields_phased
        _fac = complex(spec.adj_fix_re, spec.adj_fix_im)

        def _phased_fixed(fdtd_session, entry, fwd_aux, _o=_orig_adj, _f=_fac):
            return [e * _f for e in _o(fdtd_session, entry, fwd_aux)]

        fom._compute_adjoint_fields_phased = _phased_fixed
    if getattr(spec, "width_grad", False):
        # width-adjoint solves must NOT go to the GPU (spec.wg_adj_resource
        # note) — partition them to their own resource, all else unchanged.
        class WidthAwareRunner(lmpt.LocalRunner):
            def _is_width_job(self, job):
                return "field_profile_adj" in str(getattr(job, "task", ""))

            def _with_resource(self, res, fn, *a):
                # athena_run_one patches resource row 1 to device type GPU,
                # leaving ZERO active CPU rows — runjobs(...,"CPU") then dies
                # with "no active CPU resources" (measured, job 136035). Flip
                # row 1's device type for the CPU lane and restore after.
                flip = (res != "GPU")
                f = self.fdtd_session.fdtd if flip else None
                if flip:
                    f.setresource("FDTD", 1, "device type", res)
                keep, self.resource = self.resource, res
                try:
                    return fn(*a)
                finally:
                    self.resource = keep
                    if flip:
                        f.setresource("FDTD", 1, "device type", "GPU")

            def run(self, job):
                res = spec.wg_adj_resource if self._is_width_job(job) else self.resource
                return self._with_resource(res, super().run, job)

            def run_jobs(self, jobs):
                gpu = [j for j in jobs if not self._is_width_job(j)]
                cpu = [j for j in jobs if self._is_width_job(j)]
                if gpu:
                    super().run_jobs(gpu)
                if cpu:
                    self._with_resource(spec.wg_adj_resource,
                                        super().run_jobs, cpu)

        runner = WidthAwareRunner(resource="GPU")
    else:
        runner = lmpt.LocalRunner(resource="GPU")
    project = lmpt.Project(setup=fsp, parametrization=parametrization,
                           fom=fom, runner=runner,
                           project_name=os.path.join(out_dir, spec.label))
    # lumopt2 BUG (dev246): Project(project_name=...) sets config_map.project_name,
    # but the folder actually used is config_map.project_folder, which defaults to
    # a RELATIVE lumopt2_project_<timestamp> under CWD — inside the Athena
    # container that is the ephemeral overlay, so every fwd/adj file vanishes at
    # job exit (measured, job 132624). Point it at persistent storage explicitly.
    project.fom.config_map.project_folder = os.path.join(out_dir,
                                                         spec.label + "_files")
    if spec.colocate_fields:
        # Option 3: co-locate the staggered Yee components on the gradient
        # monitor (research 2026-08-15: necessary but not sufficient alone).
        project.generate()
        project.fdtd_session.fdtd.setnamed(
            "optimization_dft", "spatial interpolation", "nearest mesh cell")
    attach_penalty(project, spec)
    return project, wl_nm


# ═══════════════════════════════════════════════════════════════════════════
# Per-evaluation diagnostics + guards (one callback, one jsonl)
# ═══════════════════════════════════════════════════════════════════════════

def make_log_callback(spec, out_dir, sigma0_um=None, lmpt=None, fwhm0_um=None):
    """Log (T, λ, Q_L, Q_i, R, 1−T−R, σ, FWHM both conventions, ρ, 2κL,
    params) each eval; enforce the recenter, 2κL, σ-deadband, and FWHM-
    deadband guards (raise RecenterNeeded/WidthTrip).

    ★fwhm0_um (2026-08-18, user: "we do not want to cheat in any way"): the
    acoustic spec is written in spatial FWHM, and the audit proved sigma-
    compliance does NOT imply FWHM-compliance (sigma +1.8% while raw-line
    FWHM +27%, jobs 134217 t0/t1). When an anchor is given, accepted-best
    designs must ALSO keep fwhm_env_um/fwhm0_um inside the same deadband —
    the spec observable gates directly, sigma stays as the smooth tripwire."""
    lmpt = lmpt or import_lumopt2()
    from lumopt2.utils.callbacks import BaseCallback
    path = os.path.join(out_dir, f"{spec.label}_evals.jsonl")
    fwhm0_um = fwhm0_um or getattr(spec, "fwhm0_um", None)

    class CampaignLog(BaseCallback):
        def on_function_eval(self, project, eval_num, params, fom_value,
                             gradient=None, **kw):
            p = np.asarray(params, dtype=float)
            row = {"eval": int(eval_num), "t": time.time(), "fom": float(fom_value),
                   "rho": float(np.mean(p[SL_CORR]) / CORR_NM),
                   "two_kL": two_kappa_L(p, spec.n_periods_side),
                   "params": p.tolist()}
            try:
                project.load_forward_results()
                fdtd = project.fdtd_session.fdtd
                t2 = fdtd.getresult("FDTD::ports::Port_2", "expansion for port monitor")
                s11 = np.abs(np.squeeze(
                    fdtd.getresult("FDTD::ports::Port_1", "expansion for port monitor")["S"])) ** 2
                T = np.abs(np.squeeze(t2["S"])) ** 2   # modal |S21|²; expansion
                                                       # results carry no "T" key
                wl = np.squeeze(t2["lambda"]) / NM
                lam_pk, t_pk, fwhm = measure_peak(wl, T)
                if getattr(spec, "wg_track_resonance", False):
                    # advance the twin's λ-pin for the NEXT eval (make_func
                    # reads it per call; every eval, probes included, so
                    # consecutive line-search probes chain correctly)
                    spec._wg_lam_track = float(lam_pk)
                i_pk = int(np.argmin(np.abs(wl - lam_pk)))
                if (getattr(spec, "wg_lam_chain", False) and fwhm
                        and 1 < i_pk < len(wl) - 2):
                    # i_pk within 1 of a band edge would make i_lo-1 / i_hi+1
                    # wrap (numpy negative indexing) and SILENTLY return a
                    # gradient built from the wrong end of the spectrum. In
                    # practice measure_peak already returns fwhm=None for a
                    # clipped peak, but a wrap is too quiet to leave unguarded.
                    # ★defect #19: hand the gradient pass the two λ indices that
                    # straddle the peak by ~half a linewidth, plus the measured
                    # spectral curvature there. Read per call, same contract as
                    # _wg_lam_track. Curvature is free — it is the spectrum we
                    # already solved for, and FDTD is deterministic (repeated
                    # evals are bit-identical), so there is no noise floor here.
                    # ★MATCHED-STENCIL pair. For any translating lineshape
                    # T = A(p)·S(λ−λ₀(p)) with S even, the amplitude part is
                    # EVEN and cancels in both antisymmetric differences:
                    #   g_hi − g_lo    = −A·C·[S'(h) − S'(−h)]
                    #   T'(λ_hi) − T'(λ_lo) =  A·[S'(h) − S'(−h)]
                    # so gλ = −(g_hi−g_lo)/(T'_hi−T'_lo) is EXACT for any h and
                    # any symmetric lineshape — the stencil truncation cancels
                    # instead of merely being small. (The naive pair — central
                    # difference of ∂T/∂p over a SECOND difference of T — does
                    # NOT cancel: its error is 1/(1+x²), x = h/g, which cost a
                    # 49.4% low bias at x≈1 before the gate caught it.)
                    # Because truncation is gone, a WIDE stencil is now better:
                    # bigger difference signal, less float cancellation.
                    # ★wl is λ-DESCENDING (Lumerical stores freq-ascending;
                    # MEASURED live: 137853_41 it0 skipped on dTp = +0.394 at
                    # a genuine max because "hi" sat on the BLUE side). |dl|
                    # for k, and swap so wl[i_lo] < wl[i_hi] ALWAYS — then
                    # the ascending-convention guard dTp<0 and the driver's
                    # gLam formula hold untouched for either ordering.
                    dl = abs(float(wl[1] - wl[0]))
                    k = max(1, min(int(round(0.5 * fwhm / dl)),
                                   i_pk - 1, len(wl) - 2 - i_pk))
                    i_lo, i_hi = i_pk - k, i_pk + k
                    if wl[i_hi] < wl[i_lo]:
                        i_lo, i_hi = i_hi, i_lo
                    # T' from the measured spectrum (free). Spacing is uniform
                    # in FREQUENCY, so take it from the actual wl values.
                    tp_lo = float(T[i_lo + 1] - T[i_lo - 1]) / float(
                        wl[i_lo + 1] - wl[i_lo - 1])
                    tp_hi = float(T[i_hi + 1] - T[i_hi - 1]) / float(
                        wl[i_hi + 1] - wl[i_hi - 1])
                    spec._wg_lam_idx = (i_lo, i_hi)
                    # denom < 0 ⟺ the stencil straddles a MAXIMUM (T' falls
                    # through zero); this replaces the curvature sign check.
                    spec._wg_dTp = tp_hi - tp_lo
                q_l = lam_pk / fwhm if fwhm else None
                try:
                    px, pI = profile_line(fdtd, lam_pk, spec.n_periods_side)
                except Exception:
                    px = pI = None
                if pI is not None:
                    prof_dir = os.path.join(out_dir, "profiles")
                    os.makedirs(prof_dir, exist_ok=True)
                    np.savez_compressed(
                        os.path.join(prof_dir, f"{spec.label}_ev{int(eval_num):04d}.npz"),
                        x_um=px, I=pI, lam_pk_nm=lam_pk)
                row.update(lam_pk_nm=lam_pk, t_pk=t_pk, fwhm_nm=fwhm, q_loaded=q_l,
                           q_i=(q_l / (1.0 - np.sqrt(t_pk)) if q_l else None),
                           r_pk=float(s11[i_pk]),
                           loss=float(1.0 - t_pk - s11[i_pk]),
                           sigma_um=sigma_of_line(px, pI) if pI is not None else None,
                           fwhm_env_um=fwhm_env_of_line(px, pI) if pI is not None else None,
                           sigma_hat_um=sigma_hat_of(spec, params))
                if pI is not None:
                    # V2 dual measurement (item 28): the gradient carrier and
                    # the spec observable, side by side on EVERY eval.
                    row["softw_um"] = round(float(soft_width_of_line(px, pI)), 6)
                if getattr(spec, "width_grad", False):
                    # the FOM carrier's OWN sample (single-λ twin monitor) —
                    # this is what wg_anchor must be measured through, or the
                    # delta-anchored penalty is offset at its own anchor point.
                    try:
                        ra = fdtd.getresult("field_profile_adj", "E")
                        ax, aI, _ = _line_from_res(
                            ra, float(np.squeeze(ra["lambda"])) / NM,
                            spec.n_periods_side)
                        row["softw_adj_um"] = round(
                            float(soft_width_of_line(ax, aI)), 6)
                    except Exception:
                        pass
                    anc = getattr(spec, "wg_anchor", None)
                    if anc and row.get("fwhm_env_um"):
                        fhat = anc["fwhm"] + (row["softw_um"] - anc["softw"])
                        row["fwhm_hat_um"] = round(fhat, 6)
                        resid = fhat - row["fwhm_env_um"]
                        row["wg_resid_um"] = round(resid, 6)
                        if abs(resid) > WG_RESID_WARN:
                            print(f"[WIDTH-GRAD SURROGATE OFF] softW-anchored "
                                  f"prediction {fhat:.4f} µm vs measured "
                                  f"{row['fwhm_env_um']:.4f} (resid {resid:+.4f})"
                                  f" — re-anchor before trusting further steps.")
                check_sigma_surrogate(row)   # predicted vs measured, every eval
            except Exception as e:                # diagnostics must never kill an eval
                row["diag_error"] = repr(e)
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            # spec override (2026-08-28): the PIPELINE SMOKE runs a deliberate
            # weak-grating surrogate (N=60, 2κL≈2.2) — an honesty guard for
            # optimization results must not block a plumbing exercise whose
            # numbers are never quoted (137868_47 died on exactly this).
            _floor = getattr(spec, "two_kl_floor", None)
            _floor = TWO_KL_FLOOR if _floor is None else float(_floor)
            if row["two_kL"] < _floor:
                raise RuntimeError(f"2kL {row['two_kL']:.3f} < floor {_floor}")
            # Recenter ONLY when the BEST design migrates (2026-08-16: λ moves
            # redward ~1-2.6 nm per early step; recentering on every wandering
            # line-search probe would reset L-BFGS-B constantly. A probe that
            # drifts far but scores worse just keeps its degraded score — see
            # make_fct's clipped-peak fallback.)
            # BOTH guards fire only on ACCEPTED-BEST designs (2026-08-16, twice
            # measured): a line-search PROBE that violates gets its degraded/
            # penalized score and L-BFGS-B backtracks on its own — tripping on
            # probes restarts the optimizer and ERASES the lesson, so it
            # re-walks the same direction every cycle (measured: 54896 re-walked
            # the width-cheat to 2Σs=339 right after the penalized 0.213 eval
            # would have taught it). Guards protect accepted progress only.
            if float(fom_value) >= self.best_fom:
                self.best_fom = float(fom_value)
                if getattr(spec, "fwhm_wall", False) and row.get("fwhm_env_um"):
                    # item-24 fix, live: anchor tracks the measurement at
                    # accepted-best cadence (probes never move it)
                    spec.fw_anchor = {"fwhm": float(row["fwhm_env_um"]),
                                      "mcorr": float(np.mean(p[SL_CORR])),
                                      "elong": float(2.0 * p[SL_SHIFT].sum()),
                                      "corr_vec": tuple(float(v)
                                                        for v in p[SL_CORR])}
                if row.get("lam_pk_nm") and abs(row["lam_pk_nm"] - spec.scan_center_nm) > RECENTER_NM:
                    raise RecenterNeeded(f"peak {row['lam_pk_nm']:.3f} vs center {spec.scan_center_nm}")
                s = row.get("sigma_um")
                if sigma0_um and s and not (RHO_DN <= s / sigma0_um <= RHO_UP):
                    raise WidthTrip(f"sigma {s:.2f} µm vs ctrl {sigma0_um:.2f} "
                                    f"(ratio {s / sigma0_um:.3f})")
                fw = row.get("fwhm_env_um")
                if fwhm0_um and fw and not (RHO_DN <= fw / fwhm0_um <= RHO_UP):
                    raise WidthTrip(f"fwhm_env {fw:.2f} µm vs ctrl {fwhm0_um:.2f} "
                                    f"(ratio {fw / fwhm0_um:.3f}) — spec observable")

    CampaignLog.best_fom = -np.inf
    return CampaignLog()


def profile_line(fdtd, lam_pk_nm, n_side):
    """(x_um, I_x) — the PROJECT's canonical 1D mode profile.

    ★REWRITTEN 2026-08-18 on the user's instruction ("just use my model with
    integral and envelope"). This now replicates
    sim_helpers.extract_and_process_field_profile EXACTLY, step for step:
    pick the resonance lambda, |Ex|^2+|Ey|^2+|Ez|^2, **INTEGRATE OVER Y**, then
    crop to the grating.

    ★THE BUG THIS FIXES (my error, caught by the user): the previous version
    never integrated over y. It flattened (y, lambda) into one axis and indexed
    with the LAMBDA index — and since that index is always < n_lambda, the
    result was ALWAYS y-row 0, for every design in every campaign. "field_profile"
    is a 2D Z-normal plane of y span 1.5*width_wide (~1.5 um), so row 0 sits
    ~0.75 um off the guide axis, in the evanescent skirt, rather than across the
    mode. Not the transverse box edge, and NOT necessarily radiation-dominated —
    what is certain is only that every sigma and FWHM ever logged came from one
    off-axis y-row instead of the y-integral the project defines. How much that
    moves the numbers is unmeasured; treat all pre-fix width values as suspect
    until re-measured, and do not attribute a mechanism without evidence.
    """
    res = fdtd.getresult("field_profile", "E")
    x, I, _ = _line_from_res(res, lam_pk_nm, n_side)
    return x, I


def _line_from_res(res, lam_pk_nm, n_side):
    """(x_um_cropped, I_cropped, aux) from a raw field_profile result dict.

    aux carries what the width-adjoint source needs: the resonance λ index,
    the crop mask on the full x grid, and the y-trapz weights (dI/dI_xy)."""
    lam = np.squeeze(res["lambda"])
    i = int(np.argmin(np.abs(lam - lam_pk_nm * NM)))
    E = res["E"]
    if E.ndim == 5:                     # (x, y, z, lambda, comp)
        E_res = E[:, :, 0, i, :]
    elif E.ndim == 4:                   # (x, y, lambda, comp)
        E_res = E[:, :, i, :]
    else:
        E_res = E[..., i, :]
    I_xy = (np.abs(E_res[..., 0]) ** 2 + np.abs(E_res[..., 1]) ** 2
            + np.abs(E_res[..., 2]) ** 2)
    x = np.squeeze(res["x"]) * 1e6
    if I_xy.ndim > 1:
        y = np.squeeze(res["y"])
        I = np.trapz(I_xy, y, axis=1)
        # trapz weights: dI(x)/dI_xy(x, y_j) — edges half, interior mean gap
        wy = np.empty_like(y)
        wy[0], wy[-1] = (y[1] - y[0]) / 2.0, (y[-1] - y[-2]) / 2.0
        wy[1:-1] = (y[2:] - y[:-2]) / 2.0
    else:
        I, wy = I_xy, np.ones(1)
    keep = np.abs(x) <= n_side * PITCH_NM / 1000.0      # crop to the grating
    return x[keep], I[keep], {"i_lam": i, "keep": keep, "wy": wy}


def sigma_of_line(x, I):
    """Mode half-width (µm): sqrt of the centered 2nd moment of the profile.
    Compared only as a RATIO to the same-N control."""
    w = np.trapz(I, x)
    mu = np.trapz(x * I, x) / w
    return float(np.sqrt(np.trapz((x - mu) ** 2 * I, x) / w))


def fwhm_env_of_line(x, I):
    """SPATIAL FWHM (µm) — THE width observable, and the only one.

    Cubic envelope through the standing-wave peaks, half-max RELATIVE TO THE
    PROFILE FLOOR, via the sim_helpers functions themselves (not a copy), so
    this is identical to post_processing's fwhm_m by construction. Comparable
    to every stored fwhm_m (nladder corr-325 N=100 bare: 19.24 µm) and to the
    ~20 µm acoustic spec.
    ★USER RULE 2026-08-18: this convention ONLY. The raw-line variant I had
    written is deleted and must not come back."""
    try:
        env = extract_envelope_peaks(x, I)
        v = calculate_fwhm_relative(x, env)
        return float(v) if v else None
    except Exception:                 # <4 peaks breaks the cubic interp
        return None


# ── V2 soft level-set width (2026-08-21, V2_FWHM_PLAN.md §2; skill item 28) ─
# The DIFFERENTIABLE stand-in for fwhm_env: boxcar(one 258 nm fringe) +
# Gaussian(0.25 µm) smoothing → softmax peak → fixed-edge-window floor →
# sigmoid superlevel-set integral. VALIDATED offline (zero GPU): tracks the
# measured fwhm_env growth to ≤2.2 pp across +4.9%..+26.6% designs where
# sigma errs 24 pp; autograd ≡ FD to 1e-8. LOCAL surrogate — the anchor maps
# softW-µm to fwhm-µm and is re-zeroed per restart; the measured fwhm_env
# guard stays the authority.

WG_FRINGE_UM = 0.258     # standing-wave period ≈ pitch/2 — boxcar kills it
WG_GAUSS_UM = 0.25       # kills harmonics; 0.25 measured best of .15/.25/.40
WG_EPS = 0.05            # sigmoid temperature, fraction of (peak−floor)
WG_BETA_PK = 60.0        # softmax-peak sharpness (on I/scale ∈ [0,1])
WG_RESID_WARN = 0.05     # µm: |Δpredicted − Δmeasured| that must not pass silently

_WSMOOTH = {}


def _wsmooth_matrix(n, dx_um):
    """Dense boxcar+Gaussian smoothing matrix with edge replication (cached).
    A constant to autograd — the smoothing becomes one differentiable dot."""
    key = (n, round(dx_um, 9))
    if key not in _WSMOOTH:
        def mat(k):
            m = len(k) // 2
            M = np.zeros((n, n))
            for i in range(n):
                for j, kv in enumerate(k):
                    M[i, min(max(i + j - m, 0), n - 1)] += kv
            return M
        nb = max(1, int(round(WG_FRINGE_UM / dx_um)))
        ng = int(np.ceil(4 * WG_GAUSS_UM / dx_um))
        t = np.arange(-ng, ng + 1) * dx_um
        kg = np.exp(-0.5 * (t / WG_GAUSS_UM) ** 2)
        _WSMOOTH[key] = mat(kg / kg.sum()) @ mat(np.ones(nb) / nb)
    return _WSMOOTH[key]


def soft_width_of_line(x, I):
    """softW (µm) of a y-integrated profile line — autograd-differentiable in I."""
    x = np.asarray(x, dtype=float)
    Is = anp.dot(_wsmooth_matrix(len(x), float(np.mean(np.diff(x)))), I)
    # scale stays IN the autograd graph: detaching it puts a 2e-3 relative
    # error in the gradient (measured, gate W0.3) because the softmax weights
    # depend on it — a "safe constant" that silently isn't.
    scale = anp.max(Is) - anp.min(Is)
    w = anp.exp(WG_BETA_PK * (Is / scale))
    P = anp.sum(w * Is) / anp.sum(w)                    # softmax peak
    m = max(3, int(0.05 * len(x)))                      # fixed edge windows
    F = 0.5 * (anp.mean(Is[:m]) + anp.mean(Is[-m:]))    # floor
    h = F + 0.5 * (P - F)
    z = anp.clip((Is - h) / (WG_EPS * (P - F)), -60.0, 60.0)
    sig = 1.0 / (1.0 + anp.exp(-z))
    return anp.sum(0.5 * (sig[1:] + sig[:-1]) * (x[1:] - x[:-1]))


_soft_width_grad = autograd.grad(lambda I, x: soft_width_of_line(x, I))


def softw_and_weight(x, I):
    """(softW value, dsoftW/dI) — the weight is the adjoint-source profile."""
    I = np.asarray(I, dtype=float)
    return float(soft_width_of_line(x, I)), np.asarray(_soft_width_grad(I, x))


def width_band_penalty(spec, softw):
    """Augmented-Lagrangian band penalty on the anchor-mapped FWHM prediction
    (autograd in softw). Band = the standing +2%/−5% deadband on fwhm0_um."""
    anc, f0 = spec.wg_anchor, spec.fwhm0_um
    fhat = anc["fwhm"] + (softw - anc["softw"])         # µm, delta-anchored
    g_hi = fhat - RHO_UP * f0
    g_lo = RHO_DN * f0 - fhat
    return (spec.wg_lam_hi * anp.maximum(0.0, g_hi)
            + 0.5 * spec.wg_mu * anp.maximum(0.0, g_hi) ** 2
            + spec.wg_lam_lo * anp.maximum(0.0, g_lo)
            + 0.5 * spec.wg_mu * anp.maximum(0.0, g_lo) ** 2)


def make_fct_v2(wl_nm, spec):
    """fct over [T(λ) grid, softW]: windowed soft-max T minus the AL width
    penalty. The softW jacobian entry drives the weighted field adjoint."""
    base = make_fct(wl_nm)
    n_wl = len(wl_nm)

    if getattr(spec, "wg_pure", False):
        # W3 calibration fct: J = −softW. The port slice is untouched (its
        # jacobian is exactly 0 → no port adjoint contribution), so the FD
        # check isolates the weighted field-adjoint path with dJ/dsoftW = −1.
        return lambda x: -x[n_wl]

    if getattr(spec, "wg_project", False):
        # projection mode: objective = PURE windowed softmax T. Width steering
        # lives in the projected step (run_projected), not a penalty. The width
        # entry stays in the FOM list (its adjoint still runs, feeding grad_W)
        # but its jacobian here is exactly 0.
        # ★AND THE ELONGATION WALL MUST NOT BE STACKED ON TOP (audit S4,
        # 2026-08-25): attach_penalty picks elong_penalty when rho_band=False,
        # and run_projected uses the WRAPPED compute_fom/compute_gradient, so
        # the wall would silently re-enter. DERIVED: d(pen)/dshift =
        # 4*BETA_ELONG*(e-120) ~ 5e-4 at e=132.6 vs a measured dT/dshift ~ 4e-5
        # — the wall would DOMINATE the shift block ~12x only 12 nm past its
        # deadband, and cap the run below BEST_T9636's own e=132.6, i.e. it
        # would structurally forbid the campaign from reaching the basin it is
        # meant to test. The projection prices elongation exactly, via grad-W.
        return lambda x: base(x[:n_wl])

    def fct(x):
        return base(x[:n_wl]) - width_band_penalty(spec, x[n_wl])

    return fct


def check_import_src_injects(src_plane):
    """★ROOT CAUSE of the exactly-zero width gradients (136189/136190,
    diagnosed 2026-08-23, plan §27): the twin/source plane sits at z=0 = the
    ANTI-symmetric (PEC-like) BC, where tangential E ≡ 0 by parity — the TM
    field there is Ez-only. A plane import source with injection axis z
    consumes ONLY the tangential (x, y) field components to build its
    equivalence currents; Ez in the dataset injects NOTHING. So the imported
    W·conj(E) sheet was ~empty and the adjoint solved noise → gradient 0.
    (The FieldRegion 'source mode' object is dipole-based — z-polarized
    dipoles allowed — which is why only the import route zeroes out.)
    This guard turns the silent 6-GPU-h zero into a 1-second loud failure."""
    s = np.abs(np.asarray(src_plane))
    tan = float(max(s[..., 0].max(), s[..., 1].max()))
    nrm = float(s[..., 2].max())
    if tan < 1e-3 * max(nrm, 1e-300):
        raise RuntimeError(
            f"width-adjoint import source would inject ~nothing: tangential "
            f"source components max {tan:.3e} vs Ez {nrm:.3e}. The z=0 twin "
            f"plane lies on the anti-symmetric BC (tangential E = 0) and a "
            f"z-injection import source uses ONLY tangential fields — root "
            f"cause of the zero gradients in jobs 136189/136190 (plan §27). "
            f"Use wg_source='fieldregion' (CPU-only) or re-design the twin "
            f"off the mirror plane (re-gate required).")
    return tan, nrm


def tile_x_edges(x_m, n_tiles):
    """(edges, bounds): split a monitor's own x SAMPLES into n_tiles blocks.

    x_m is the 1-D ascending x sample vector in metres. Block k holds samples
    bounds[k]:bounds[k+1] and is bracketed by edges[k]..edges[k+1]. Interior
    edges sit MIDWAY between the two samples that straddle the cut, never on
    a sample: that is what makes each tile's own (frozen-mesh) sampling come
    out as exactly its slice of the parent's samples, with no sample landing
    in two tiles and none falling in the crack.
    """
    n = len(x_m)
    assert n_tiles >= 1 and n >= n_tiles, "more tiles than x samples"
    bounds = [int(round(k * n / n_tiles)) for k in range(n_tiles + 1)]
    assert bounds[0] == 0 and bounds[-1] == n and \
        all(b < a for b, a in zip(bounds, bounds[1:])), "degenerate tile split"
    edges = [x_m[0] - 0.5 * (x_m[1] - x_m[0])]
    edges += [0.5 * (x_m[i - 1] + x_m[i]) for i in bounds[1:-1]]
    edges += [x_m[-1] + 0.5 * (x_m[-1] - x_m[-2])]
    return np.asarray(edges, float), bounds


def split_dataset_x(prof, bounds):
    """The dataset sliced along x at `bounds` — one dict per tile.

    Only the x-indexed arrays move; everything else (y, z, lambda, f, the
    Lumerical_dataset descriptor) is shared. No interpolation and no shared
    edge sample, so concatenating the slices returns the original array
    element for element — this is the no-double-counting guarantee.
    """
    parts = []
    for lo, hi in zip(bounds, bounds[1:]):
        d = dict(prof)
        d["x"] = np.asarray(prof["x"])[lo:hi]
        d["E"] = np.asarray(prof["E"])[lo:hi]
        parts.append(d)
    return parts


def import_tiled_source(fdtd, prof, n_tiles, lam_m, monitor_name):
    """Inject the weighted adjoint source as n_tiles narrow FieldRegion
    sources instead of one full-width one (CampaignSpec.wg_src_tiles).

    field_profile_adj stays a pure MONITOR — it is what WidthResults reads.
    The tiles carry disjoint x-slices of the SAME globally computed
    W(x,y)*conj(E_fwd), enabled together in ONE adjoint run; sources
    superpose linearly, so the injected field equals the full-region source
    exactly and the gradient is exact.

    Runs in LAYOUT mode with the mesh already frozen (project.generate ->
    fdtd_session.lock_grid, fdtd_session.py:1618), which is why the tile x
    geometry is set HERE, from the monitor's own recorded samples, and only
    placeholder-sized at build time.
    """
    x_m = np.squeeze(np.asarray(prof["x"])).astype(float)
    edges, bounds = tile_x_edges(x_m, n_tiles)
    parts = split_dataset_x(prof, bounds)
    dx = float(np.min(np.diff(x_m)))
    fdtd.setnamed(monitor_name, "source mode", False)   # monitor stays a monitor
    amps, live = [], 0
    for t, part in enumerate(parts):
        name = f"{monitor_name}_t{t}"
        if not fdtd.getnamednumber(name):
            raise RuntimeError(
                f"wg_src_tiles={n_tiles} but '{name}' is not in the scene — "
                f"the base .fsp was built with a different wg_src_tiles")
        # ★Geometry from the SAMPLES the tile must carry, not from the
        # midpoint edges (MEASURED, job 136967: edge-derived spans made the
        # region sample 529 cells while the slice held 528, and the engine
        # rejects a shape mismatch — "imported source profile dataset
        # dimensions ... do not match the field region dimensions").
        # Centre on the tile's own first/last sample and pad 0.45 dx a side:
        # every intended sample sits >=0.45 dx inside, and the nearest
        # EXCLUDED neighbour sits 0.55 dx outside. Fencepost-proof.
        xs_t = np.squeeze(np.asarray(part["x"])).astype(float)
        fdtd.setnamed(name, "x", 0.5 * (xs_t[0] + xs_t[-1]))
        fdtd.setnamed(name, "x span", (xs_t[-1] - xs_t[0]) + 0.9 * dx)
        fdtd.setnamed(name, "wavelength center", lam_m)
        fdtd.setnamed(name, "wavelength span", 0.0)
        a = float(np.abs(part["E"]).max())
        amps.append(a)
        if a == 0.0:            # outside the grating crop -> W is exactly 0
            fdtd.setnamed(name, "source mode", False)
            continue
        fdtd.setnamed(name, "source mode", True)
        fdtd.select(name)
        fdtd.importdataset(part)
        live += 1
    if live == 0:
        raise RuntimeError(
            "tiled width-adjoint source is entirely zero — the adjoint would "
            "solve noise (the 136189/136190 class of silent failure)")
    # Per-tile amplitudes are THE dead-tile detector: read this line in the
    # job log before trusting any tiled gradient.
    print("[wg-tiles] cells/tile "
          f"{[h - l for l, h in zip(bounds, bounds[1:])]} | live {live}/"
          f"{n_tiles} | max|src| per tile "
          f"{['%.3e' % a for a in amps]} | full {np.abs(prof['E']).max():.3e}",
          flush=True)


def make_width_classes(lmpt, spec):
    """(WidthResults, MixedFom) — lazy, like make_bc_parametrization.

    WidthResults extracts ONE value per eval: softW of the y-integrated
    resonance profile (resonance re-picked from Port_2 each eval, same
    convention as profile_line; stop-gradient on the selection). It stashes
    the dsoftW/dI weight for the adjoint source.

    MixedFom dispatches per entry: port entries keep PortFom's machinery
    verbatim; the width entry gets a FieldFom-style monitor-source adjoint
    whose imported source is W(x,y)·conj(E_fwd) at the resonance λ ONLY
    (other λ planes zeroed), W = dsoftW/dI × y-trapz weight. The stock
    FieldFom cannot do this — its source is hard-coded plain conj(E_fwd),
    which is only correct for Σ|E|² FOMs.

    ★W3 MUST verify before any campaign: (a) FD gate on the width-gradient
    components with the field path's own C_field; (b) that the forward PORT
    source is genuinely disabled in the saved width-adjoint .fsp; (c) the
    single-λ source import behaves as intended (other planes zeroed)."""
    from lumopt2.fom.field_fom import FieldFom
    from lumopt2.fom.port_fom import PortFom
    from lumopt2.fom.simulation_results import FieldResults

    class WidthResults(FieldResults):
        # Reads the SINGLE-λ twin monitor (see build_base_fsp): lumopt2
        # rejects broadband FieldRegion monitors at generate() (job 135954).
        def __init__(self, wavelength_m):
            super().__init__("field_profile_adj", "intensity", [wavelength_m])
            self.aux = None                     # set per get_results

        def get_results(self, fdtd_session):
            res = fdtd_session.fdtd.getresult(self.monitor_name, "E")
            lam_nm = float(np.squeeze(res["lambda"])) / NM   # single plane
            x, I, aux = _line_from_res(res, lam_nm, spec.n_periods_side)
            sw, w_x = softw_and_weight(x, I)
            self.values = anp.array([sw])
            self.wavelengths = [lam_nm * NM]
            aux.update(w_x=w_x, softw=sw, x_um=x, I=I, lam_pk_nm=lam_nm)
            self.aux = aux

    class MixedFom(PortFom):
        supports_concurrent_adjoint = False     # field source needs fwd results
        # FieldFom helper needed by its get_scaling_factor when called unbound
        # for width entries (measured: task 11 died on this AFTER its 8.7 h
        # adjoint completed — the least-tested-code lesson, item 23, again).
        dipole_base_amplitude = FieldFom.dipole_base_amplitude

        def _is_width(self, sim_result):
            return isinstance(sim_result, WidthResults)

        # ★PortFom's bookkeeping loops over ALL config-map entries and
        # addresses each as FDTD::ports::<monitor> — the width entry is not a
        # port and crashes getnamed (measured: job 135971 task 12, 53 min in,
        # 'FDTD::ports::field_profile_adj not found'; invisible to the local
        # generate() smoke because it fires in the RUN path). Filter width
        # entries out of every port-side loop.
        def _get_port_monitor_info(self, fdtd_session):
            self._port_aux_cache = {}
            for config_info in self.config_map.config_map.values():
                for adj in config_info["adjoints"]:
                    sr = adj["sim_result"]
                    if self._is_width(sr):
                        continue
                    port_path = f"FDTD::ports::{sr.monitor_name}"
                    raw_axis = fdtd_session.fdtd.getnamed(port_path, "injection axis")
                    axis = raw_axis[0]
                    adj["propagation_axis"] = axis
                    adj["propagation_position"] = float(
                        fdtd_session.fdtd.getnamed(port_path, axis))
                    adj["direction"] = fdtd_session.fdtd.getnamed(port_path,
                                                                  "direction")

        def _update_port_positions(self, fdtd_session, config_info, step):
            ports_only = dict(config_info)
            ports_only["adjoints"] = [a for a in config_info["adjoints"]
                                      if not self._is_width(a["sim_result"])]
            return super()._update_port_positions(fdtd_session, ports_only, step)

        def setup_adjoint_simulation(self, fdtd_session, sources, sim_result):
            if not self._is_width(sim_result):
                return super().setup_adjoint_simulation(
                    fdtd_session, sources, sim_result)
            if sim_result.aux is None:          # fct ran first via compute_fom,
                sim_result.get_results(fdtd_session)   # but be self-sufficient
            aux = sim_result.aux
            prof = fdtd_session.fdtd.getresult(sim_result.monitor_name, "E")
            E = prof["E"]
            w_full = np.zeros(len(aux["keep"]))
            w_full[aux["keep"]] = aux["w_x"]     # zero outside the grating crop
            n_spatial = E.ndim - 2               # E is (spatial..., λ, comp)
            if n_spatial == 1:
                W = w_full
            else:
                W = w_full[:, None] * aux["wy"][None, :]
                if n_spatial == 3:               # (x, y, z=1)
                    W = W[:, :, None]
            src = np.zeros_like(E)
            i = aux["i_lam"]
            src[..., i, :] = np.conj(E[..., i, :]) * W[..., None]
            prof["E"] = src
            fdtd_session.fdtd.switchtolayout()
            for s in sources:
                # ('port', name) entries need the full FDTD::ports:: path —
                # bare setnamed('Port_1',...) fails (measured, job 135986);
                # skip our own fieldregion (source mode is off in forward).
                try:
                    if s[0] == "port":
                        fdtd_session.fdtd.setnamed(f"FDTD::ports::{s[1]}",
                                                   "enabled", False)
                    elif s[0] == "source":
                        fdtd_session.fdtd.setnamed(s[1], "enabled", False)
                except Exception as e:
                    raise RuntimeError(f"disable forward source {s}: {e}")
            if getattr(spec, "wg_source", "fieldregion") == "import":
                # standard import source carries the weighted sheet; the
                # FieldRegion stays a pure monitor (source mode stays off)
                check_import_src_injects(src[..., i, :])   # fail LOUD (§27)
                fdtd_session.fdtd.setnamed("width_adj_src", "enabled", True)
                # pin the source spectrum to the twin's λ (audit 2026-08-22:
                # override=1 with unset wavelengths risks an off-λ spectrum;
                # the DFT at the monitor λ is all the adjoint uses)
                lam_m = float(sim_result.wavelengths[0])
                fdtd_session.fdtd.setnamed("width_adj_src",
                                           "wavelength start", lam_m)
                fdtd_session.fdtd.setnamed("width_adj_src",
                                           "wavelength stop", lam_m)
                fdtd_session.fdtd.select("width_adj_src")
                fdtd_session.fdtd.importdataset(prof)
            elif int(getattr(spec, "wg_src_tiles", 1) or 1) > 1:
                # GPU workaround: same source, split across narrow x tiles.
                import_tiled_source(
                    fdtd_session.fdtd, prof,
                    int(spec.wg_src_tiles), float(sim_result.wavelengths[0]),
                    sim_result.monitor_name)
            else:
                fdtd_session.fdtd.setnamed(sim_result.monitor_name,
                                           "source mode", True)
                fdtd_session.fdtd.select(sim_result.monitor_name)
                fdtd_session.fdtd.importdataset(prof)

        def get_scaling_factor(self, fdtd_session, parametrization, sim_result):
            if self._is_width(sim_result):
                return FieldFom.get_scaling_factor(
                    self, fdtd_session, parametrization, sim_result)
            return super().get_scaling_factor(fdtd_session, parametrization,
                                              sim_result)

        def _collect_fwd_side_aux(self, fdtd_session, entry):
            if self._is_width(entry["sim_result"]):
                return None
            return super()._collect_fwd_side_aux(fdtd_session, entry)

        def _compute_adjoint_fields_phased(self, fdtd_session, entry, fwd_aux):
            if self._is_width(entry["sim_result"]):
                if getattr(spec, "_wg_reuse", False):
                    # ★wgp_reuse_k: no width-adjoint solve ran this iterate —
                    # its fields must not be read (file is stale geometry).
                    # Scalar zeros broadcast in the accumulate loop; the
                    # width entry's jac is 0 under the pure-T fct anyway,
                    # and the driver substitutes the STORED parameter-space
                    # gW instead of an assembly here.
                    return [0.0] * len(entry["sim_result"].wavelengths)
                fac = complex(spec.adj_fix_field_re, spec.adj_fix_field_im)
                e = FieldFom.get_adjoint_fields(self, fdtd_session, entry)
                return [ei * fac for ei in e]
            e = super()._compute_adjoint_fields_phased(fdtd_session, entry, fwd_aux)
            if spec.adj_phase_fix:
                fac = complex(spec.adj_fix_re, spec.adj_fix_im)
                e = [ei * fac for ei in e]
            return e

        # ★wg_project (2026-08-25): split ∇T / ∇W from the SAME solved fields.
        # The assembly (base_fom.py:497-513) is linear in jac_of_fom, so two
        # passes with two fcts give the exact components — zero extra solves.
        # Pass 1: self.fct is already pure-T under wg_project (width jac = 0).
        # Pass 2: fct = x[-1] → raw d(softW)/dEps fields, stashed for the
        # driver. Legacy (wg_project False): first statement = stock behavior.
        def calculate_gradient_fields(self, fdtd_session, parametrization,
                                      symmetry_factors=None):
            sup = super(MixedFom, self).calculate_gradient_fields
            if not getattr(spec, "wg_project", False):
                return sup(fdtd_session, parametrization, symmetry_factors)
            entries = self.config_map.get_filenames_in_order()
            assert self._is_width(entries[-1]["sim_result"]), \
                "wg_project assumes the width entry is LAST (fct = x[-1])"
            keep = self.fct
            idx = (getattr(spec, "_wg_lam_idx", None)
                   if getattr(spec, "wg_lam_chain", False) else None)
            try:
                gT = sup(fdtd_session, parametrization, symmetry_factors)
                # ★defect #19 selector passes — STASH ONLY, convert at the
                # DRIVER (2026-08-27, job 137845_41): calling
                # compute_gradient_from_fields HERE crashed on hardware —
                # dEps does update_structure while the CAD is still in
                # ANALYSIS mode ("use switchtolayout first"). The driver
                # site (right after compute_gradient returns) is where the
                # width conversion provably works (137075, 3 iterates).
                # RAM is a non-issue: sup() returns the already-summed
                # (nx,ny,nz,3) region array (MB), not the per-λ monitor
                # reads (those live and die inside sup regardless).
                # ★x IS FLAT — [T(λ_0) … T(λ_{n_wl−1}), softW] — NOT a list of
                # entry results (make_fct_v2:1850 is `base(x[:n_wl])`, which is
                # exactly why the width selector `x[-1]` works). Writing
                # x[0][i_lo] cost job 137267 2 GPU-h with "IndexError: invalid
                # index to scalar variable" — x[0] is the SCALAR T at λ_0.
                self.gfields_Tlo = self.gfields_Thi = None
                if idx is not None:
                    for attr, i in (("gfields_Tlo", idx[0]),
                                    ("gfields_Thi", idx[1])):
                        self.fct = (lambda j: (lambda x: anp.abs(x[j])))(i)
                        setattr(self, attr,
                                sup(fdtd_session, parametrization,
                                    symmetry_factors))
                if getattr(spec, "_wg_reuse", False):
                    # ★wgp_reuse_k: no fresh width fields this iterate — the
                    # driver reuses the stored parameter-space gW. Selector
                    # passes above still ran (fresh gλ off the port adjoint).
                    self.gfields_W = None
                else:
                    self.fct = lambda x: x[-1]
                    self.gfields_W = sup(fdtd_session, parametrization,
                                         symmetry_factors)
            finally:
                self.fct = keep
            return gT          # combined == ∇T fields here (width jac was 0)

    return WidthResults, MixedFom


# ═══════════════════════════════════════════════════════════════════════════
# Entries: campaign (with recenter/restart loop) and server-side validation
# ═══════════════════════════════════════════════════════════════════════════

def _final_fom(result):
    """FOM out of whatever lumopt2's Optimization.run() handed back.

    R1.3 returns a (params, fom) TUPLE, not an object with .final_fom — and
    the program's FIRST natural completion crashed on exactly that
    (AttributeError, IGUM bare job 55343, 2026-08-17; every earlier campaign
    was stopped or crashed before reaching this line, so the completion path
    had never once been exercised). This value is BOOKKEEPING ONLY — the
    delivered design always comes from the width-filtered log — so an
    unrecognised shape degrades to -inf instead of killing a finished run.
    """
    if hasattr(result, "final_fom"):
        return float(result.final_fom)
    if isinstance(result, (tuple, list)):
        scalars = [v for v in result if isinstance(v, (int, float, np.floating))]
        if scalars:
            return float(scalars[-1])
    return -np.inf


def _rgp_step(gT, gW, D, W, W_tgt, marg, cap_nm, st, bsf_init=2.0):
    """Relaxed gradient projection (Antonau/Hojjat/Bletzinger, SMO 63:1633,
    2021) — opt-in via spec.wgp_rgp. One continuous formula replaces the
    CLIMB/RIDE/RESTORE branches: the projection weight w_r ramps 0→1 through
    a history-sized buffer below the band edge, and past the edge a bounded
    correction w_c ROTATES the direction toward feasible instead of replacing
    it (their measured payoff: 1.7× objective gain vs pure-restore GP).
    Endpoint identity: at w_r=1, w_c=0 this is EXACTLY _proj_step's RIDE
    (∇W·d = 0, same D metric) — asserted by gate_projection_local. Full
    derivation + eq refs: notes_relaxed_projection.md (R1-R4).
    `st` carries the adaptive state {bsf, cbv_hi, cbv_lo, Wh, w_prev}."""
    gT, gW, D = (np.asarray(v, dtype=float) for v in (gT, gW, D))
    lv_hi, lv_lo = W_tgt + marg / 2.0, W_tgt - marg / 2.0
    # buffer size from measured per-iterate width history (eq 14)
    dW_hist = [abs(b - a) for a, b in zip(st["Wh"], st["Wh"][1:])]
    bs = st["bsf"] * max(dW_hist) if dW_hist else max(1e-3, 0.01 * marg)
    # one-sided w (eq 12): ceiling g = W - cbv_hi, floor g = cbv_lo - W;
    # only one side can be in its buffer (bs << marg)
    g_hi, g_lo = W - st["cbv_hi"], st["cbv_lo"] - W
    ceil_side = g_hi >= g_lo
    g = g_hi if ceil_side else g_lo
    w = float(np.clip(1.0 + g / bs, 0.0, 2.0))
    w_r = min(w, 1.0)
    w_c = float(np.clip(bsf_init * (w - 1.0), 0.0, bsf_init))  # w_max = 2
    # zigzag (eq 19): 3 alternating ΔW signs ⇒ widen buffer (factor = 1)
    if len(st["Wh"]) >= 4:
        d1, d2, d3 = (st["Wh"][-1] - st["Wh"][-2],
                      st["Wh"][-2] - st["Wh"][-3],
                      st["Wh"][-3] - st["Wh"][-4])
        if d1 * d2 < 0 and d2 * d3 < 0 and st["w_prev"] is not None:
            st["bsf"] += abs(w - st["w_prev"])
    # infeasible drift (eqs 20-21): two violated iterates, not improving ⇒
    # recentre the working boundary inward (the published answer to "width
    # creeps up every iterate")
    if len(st["Wh"]) >= 2:
        g_prev = (st["Wh"][-1] - st["cbv_hi"] if ceil_side
                  else st["cbv_lo"] - st["Wh"][-1])
        if g > 0 and g_prev > 0 and g >= g_prev:
            if ceil_side:
                st["cbv_hi"] -= (st["Wh"][-1] - lv_hi)
            else:
                st["cbv_lo"] += (lv_lo - st["Wh"][-1])
    st["w_prev"] = w
    # direction (eq 17 with our D metric; note R1)
    DgW = D * gW
    gd = float(gW @ DgW)
    coef = float(gT @ DgW) / gd if gd > 0 else 0.0
    p = D * gT - w_r * coef * DgW              # w_r=1 ⇒ today's RIDE exactly
    sgn = 1.0 if ceil_side else -1.0           # ∇g = ±gW by active side
    pm, um = float(np.max(np.abs(p))), float(np.max(np.abs(DgW)))
    s = (p / pm if pm > 0 else p) - sgn * w_c * (DgW / um if um > 0 else DgW)
    sm = float(np.max(np.abs(s)))
    step = cap_nm * (s / sm) if sm > 0 else s  # cap = max-norm radius (§5)
    nu, nW = np.linalg.norm(DgW), np.linalg.norm(gW)
    lam = float(gT @ DgW) / (nu * nW) if nu > 0 and nW > 0 else 0.0  # same
    return step, f"rgp(w={w:.2f})", lam, w_c        # shadow price as _proj_step


def _proj_step(gT, gW, D, W, W_tgt, marg, alpha, step_max_nm):
    """Pure step math for the ceiling-riding projection (HANDOFF 2026-08-24
    23:00) — tested by gate_projection_local.py (P0/clip/restore, zero GPU).
    û lives in the D^{1/2}-scaled space: the only reading of "D = per-block
    trust scales squared" whose null-space step gives ∇W·d = 0 exactly (the
    literal û = D∇W/|D∇W| does not, since D² ≠ D — P0 arbitrates)."""
    gT, gW, D = (np.asarray(v, dtype=float) for v in (gT, gW, D))
    DgW = D * gW
    nu, nW = np.linalg.norm(DgW), np.linalg.norm(gW)
    lam = float(gT @ DgW) / (nu * nW) if nu > 0 and nW > 0 else 0.0
    if W - W_tgt > marg / 2.0:                    # restoration on MEASURED W
        step = -(W - W_tgt) * gW / max(float(gW @ gW), 1e-300)
        phase = "restore"
    elif W < W_tgt - marg / 2.0:                  # PHASE A: climb + clip
        step = alpha * D * gT
        dw = float(gW @ step)                     # predicted ΔW — free
        if dw > 0.0 and W + dw > W_tgt:
            step = step * ((W_tgt - W) / dw)      # land exactly on the ceiling
        phase = "climb"
    else:                                         # PHASE B: null-space ride
        gd = float(gW @ DgW)                      # = |D^{1/2}∇W|²
        coef = float(gT @ DgW) / gd if gd > 0 else 0.0
        step = alpha * (D * gT - coef * DgW)      # ∇W·step = 0 exactly
        phase = "ride"
    m = float(np.max(np.abs(step)))
    if m > step_max_nm:                           # scalar cap keeps ∇W·d = 0
        step = step * (step_max_nm / m)
    return step, phase, lam


def _ns2_step(gT, gW, gLam, D, W, W_tgt, marg, lam_nm, lam_tgt_nm, lam_marg_nm,
              cap_nm):
    """Two-constraint null+range-space step (d1, 2026-08-30) — Feppon form.

    d = cap·ξ_J/‖ξ_J‖_∞ + ξ_C  with  A = [gW, gLam],  M = AᵀDA:
      ξ_J = D·gT − D·A·M⁻¹·AᵀD·gT   ⇒  gW·ξ_J = gLam·ξ_J = 0  exactly;
      ξ_C = −D·A·M⁻¹·h              ⇒  A·d ≈ −h  (violations decay, first-
                                        order exact; replaces pure-restore).
    gW here is the RAW fixed-λ width gradient — with gLam·d = 0 the fitted
    dW/dλ coefficient cancels out of the feasible directions for ANY value
    (gate section 8 asserts this numerically). ξ_J is normalized to the cap
    (RGP-style), so step length is set by the trust radius, not |∇T|.
    Returns (step, phase, diag); diag carries rho_T (the falsification
    observable: fraction of D^½∇T surviving the projection), condM, lam
    (shadow price vs gW, same definition as _proj_step's), rW/rLam residuals.
    """
    gT, gW, D = (np.asarray(v, dtype=float) for v in (gT, gW, D))
    s = np.sqrt(D)
    # shadow price vs gW — logged with the same meaning as _proj_step's lam
    DgW = D * gW
    nu, nW = np.linalg.norm(DgW), np.linalg.norm(gW)
    lam_sp = float(gT @ DgW) / (nu * nW) if nu > 0 and nW > 0 else 0.0
    # deadbanded residuals (restore only the part beyond the band)
    def _resid(v, tgt, band):
        r = float(v - tgt)
        return 0.0 if abs(r) <= band else r - np.sign(r) * band
    rW = _resid(W, W_tgt, marg / 2.0)
    cols, hs = [gW], [rW]
    degraded = False
    if gLam is not None and lam_nm is not None and lam_tgt_nm is not None:
        cols.append(np.asarray(gLam, dtype=float))
        hs.append(_resid(lam_nm, lam_tgt_nm, lam_marg_nm))
    A = np.stack(cols, axis=1)                    # (n, r)
    M = A.T @ (D[:, None] * A)                    # r×r
    condM = float(np.linalg.cond(M)) if M.shape[0] > 1 else 1.0
    if M.shape[0] > 1 and (not np.isfinite(condM) or condM > 1e8):
        # near-collinear D^½gW ∥ D^½gLam — drop the λ column, keep width
        degraded = True
        A, hs = A[:, :1], hs[:1]
        M = A.T @ (D[:, None] * A)
    h = np.asarray(hs, dtype=float)
    try:
        Minv_AtDgT = np.linalg.solve(M, A.T @ (D * gT))
        Minv_h = np.linalg.solve(M, h)
    except np.linalg.LinAlgError:                 # |gW| ≈ 0: no constraint
        xi_J = D * gT
        mJ = float(np.max(np.abs(xi_J))) or 1.0
        return xi_J * (cap_nm / mJ), "ns2_free", {
            "lam": lam_sp, "rho_T": 1.0, "condM": condM, "rW_um": rW,
            "rLam_nm": hs[1] if len(hs) > 1 else None, "ns2_degraded": degraded}
    xi_J = D * gT - (D[:, None] * A) @ Minv_AtDgT
    xi_C = -(D[:, None] * A) @ Minv_h
    # rho_T in the D^{1/2} metric: ξ_J = s·(projected s·gT), so
    # ‖ξ_J/s‖/‖s·gT‖ is exactly the surviving fraction of the scaled ∇T.
    n_g = float(np.linalg.norm(s * gT))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_T = float(np.linalg.norm(
            np.where(s > 0, xi_J / np.where(s > 0, s, 1.0), 0.0)) / n_g) \
        if n_g > 0 else 0.0
    mJ = float(np.max(np.abs(xi_J)))
    step = xi_J * (cap_nm / mJ) if mJ > 0 else np.zeros_like(xi_J)
    # ★near-null guard (audit #3, 2026-08-31): normalizing to the cap
    # amplifies a numerically-null ξ_J (rho_T→0, the "locked" regime) into a
    # full-size noise-direction step, burning a solve per filter reject.
    # Below rho_T=0.02 scale the step down proportionally instead.
    if rho_T < 0.02:
        step = step * (rho_T / 0.02)
    mC = float(np.max(np.abs(xi_C)))
    if mC > cap_nm:                               # scalar clamp: ξ_J part keeps
        xi_C = xi_C * (cap_nm / mC)               # both orthogonalities
    step = step + xi_C
    phase = "ns2" if not np.any(h) else "ns2+restore"
    if degraded:
        phase = "ns2_degraded"
    return step, phase, {
        "lam": lam_sp, "rho_T": rho_T, "condM": condM, "rW_um": rW,
        "rLam_nm": (float(lam_nm - lam_tgt_nm)
                    if lam_nm is not None and lam_tgt_nm is not None else None),
        "ns2_degraded": degraded}


def _queue_adjoint_jobs_filtered(project, spec):
    """Mirror of Project._queue_adjoint_jobs (R1.3, project.py:500-507) that
    SKIPS the width entry when spec._wg_reuse is set (the wgp_reuse_k solve
    skip). Module-level so the gate can drive it with fakes (audit #6: the
    old closure was only string-grep-testable). The one version-pinned copy
    in this feature — re-diff on any Lumerical version bump."""
    skip_width = bool(getattr(spec, "_wg_reuse", False))
    for ck, ci in project.fom.config_map.config_map.items():
        for adj in ci["adjoints"]:
            if skip_width and project.fom._is_width(adj["sim_result"]):
                continue
            project._verify_simulation_files_exist([adj["adj_filename"]])
            lab = f"run_adj_{ck}" if ck is not None else "run_adj"
            lab += f"_{adj['sim_result'].monitor_name}"
            project.runner.add_job(task=adj["adj_filename"], label=lab)


def _optstate_path(out_dir, label):
    return os.path.join(out_dir, f"{label}_optstate.json")


def _load_opt_state(out_dir, label):
    """Optimizer-state sidecar: cap / wgain / dTp0 / dwdlam / λ target.
    Every restart (REQUEUE or run_campaign's restart loop) re-enters
    run_projected fresh, which used to reset the whole trust state (measured:
    alpha/wgain/dTp0 all lost per restart). Missing/corrupt file ⇒ {}."""
    try:
        with open(_optstate_path(out_dir, label)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_opt_state(out_dir, label, state):
    path = _optstate_path(out_dir, label)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def run_projected(spec, project, cb, out_dir, p0):
    """Ceiling-riding projected-gradient driver — REPLACES ScipyOptimizer when
    spec.wg_project (L-BFGS-B's line search cannot be held to the null-space
    direction). Trust/bounds survive as: (1) the param_bounds box — trust_nm
    clamps included — np.clip'ed every step; (2) D = (bound half-range)² as
    the metric (the 0-200-vs-sliver conditioning fix). Logging/guards/resume
    UNCHANGED: the same CampaignLog writes the same <label>_evals.jsonl, so
    _best_from_log / WidthTrip / RecenterNeeded work as before (raised raw
    here; run_campaign's handler catches the direct instances).
    Returns (params, fom), the R1.3 Optimization.run tuple shape."""
    b = np.asarray(param_bounds(spec), dtype=float)
    lo, hi = b[:, 0], b[:, 1]
    D = ((hi - lo) / 2.0) ** 2
    W_tgt, marg = float(spec.wgp_target_um), float(spec.wgp_margin_um)
    ppath = os.path.join(out_dir, f"{spec.label}_proj.jsonl")
    p = np.clip(np.asarray(p0, dtype=float), lo, hi)
    alpha, acc = float(spec.wgp_step), None       # acc = last ACCEPTED point
    a0, cap0 = float(spec.wgp_step), float(spec.wgp_step_max_nm)
    # ★online width-gain (spec.wgp_autogain): gain ~ measured dW / predicted dW.
    # Starts at 1 (= trust C) and is updated from MEASURED widths, so a wrong
    # |C_field| is corrected within an iterate or two instead of biasing every
    # step. Clamped so one noisy pair cannot run away.
    wgain = 1.0
    n_neg = 0        # consecutive opposite-sign width responses
    dTp0 = None      # first healthy stencil curvature — the floor reference
    # ★RGP state (spec.wgp_rgp): buffer factor, working boundaries (recentred
    # inward by the drift rule), ACCEPTED-width history, last buffer weight.
    rgp = bool(getattr(spec, "wgp_rgp", False))
    st = {"bsf": float(getattr(spec, "wgp_rgp_bsf", 2.0)),
          "cbv_hi": W_tgt + marg / 2.0, "cbv_lo": W_tgt - marg / 2.0,
          "Wh": [], "w_prev": None}
    wc_last = 0.0
    # ★d1 (2026-08-30): ns2 two-constraint law + trust-region cap as STATE,
    # both persisted in a sidecar so REQUEUE / the restart loop does not
    # reset them (they used to: alpha/wgain/dTp0 all reset per restart).
    ns2 = bool(getattr(spec, "wgp_ns2", False))
    cap_adapt = bool(getattr(spec, "wgp_cap_adapt", False))
    lam_marg = float(getattr(spec, "wgp_lam_margin_nm", 0.05))
    ost = _load_opt_state(out_dir, spec.label) if (ns2 or cap_adapt) else {}
    cap_state = float(ost.get("cap_nm", cap0))
    lam_tgt = ost.get("lam_tgt_nm",
                      getattr(spec, "wgp_lam_target_nm", None))
    if ost:
        wgain = float(ost.get("wgain", wgain))
        dTp0 = ost.get("dTp0", dTp0)
        if ost.get("dwdlam"):
            spec.wg_dwdlam = float(ost["dwdlam"])
        print(f"[proj] optstate resumed: cap {cap_state:.3g} nm, wgain "
              f"{wgain:.3f}, lam_tgt {lam_tgt}", flush=True)
    n_acc, n_rej = int(ost.get("n_acc", 0)), int(ost.get("n_rej", 0))
    ns2_diag = {}
    # ★wgp_reuse_k: width-row reuse state. dirty ⇒ force a refresh (any
    # reject invalidates the stored row's trust).
    reuse_k = int(getattr(spec, "wgp_reuse_k", 0) or 0)
    reuse_dw = float(getattr(spec, "wgp_reuse_dw_um", 0.025))
    reuse_age = int(ost.get("reuse_age", 0))
    reuse_W0 = ost.get("reuse_W0")    # W at the last FRESH width solve —
                                      # drift reference (audit #4: per-hop
                                      # |W−acc.W| lets k≥3 accumulate
                                      # (k−1)·reuse_dw of unchecked drift)
    reuse_dirty = False
    spec._wg_reuse = False
    if ns2 and reuse_k >= 2:
        project._queue_adjoint_jobs = (
            lambda: _queue_adjoint_jobs_filtered(project, spec))

    def _save_state():
        if ns2 or cap_adapt:
            _save_opt_state(out_dir, spec.label, {
                "cap_nm": cap_state, "wgain": wgain, "dTp0": dTp0,
                "dwdlam": float(spec.wg_dwdlam), "lam_tgt_nm": lam_tgt,
                "n_acc": n_acc, "n_rej": n_rej, "reuse_age": reuse_age,
                "reuse_W0": reuse_W0})

    def _step_of(gTv, gWv, Wv, a, gLamv=None, lamv=None, mutate=False):
        """One step under the active law; retry calls must not mutate the
        RGP adaptation state (they re-step from the same accepted point)."""
        nonlocal wc_last, ns2_diag
        if ns2 and gLamv is not None and lam_tgt is not None:
            s_, ph, dg = _ns2_step(gTv, gWv, gLamv, D, Wv, W_tgt, marg,
                                   lamv, lam_tgt, lam_marg, _cap(a))
            if mutate:
                ns2_diag = dg
            return s_, ph, dg["lam"]
        if rgp:
            s, ph, lm, wc = _rgp_step(gTv, gWv, D, Wv, W_tgt, marg, _cap(a),
                                      st if mutate else dict(st),
                                      float(getattr(spec, "wgp_rgp_bsf", 2.0)))
            if mutate:
                wc_last = wc
            return s, ph, lm
        return _proj_step(gTv, gWv, D, Wv, W_tgt, marg, a, _cap(a))

    def _cap(a):
        """Effective per-param cap.

        cap_adapt: the cap IS the trust radius (state, persisted): grown on
        verified accepts, halved on rejects — alpha no longer touches it (the
        old lockstep coupling made the delivered step a constant cap0; the
        property it existed for — retry halvings being real — is preserved
        because a reject halves cap_state directly).

        Legacy path — shrinking WITH alpha:
        ★Without this the retry is a no-op (MEASURED offline 2026-08-25): at
        the real gradient magnitudes the raw step is ~73 nm, so the 5 nm cap
        binds for the first FOUR halvings of alpha — the filter would re-propose
        the IDENTICAL 5 nm move and have it rejected again, ~2.7 GPU-h each.
        Scaling the cap by alpha/alpha0 (never ABOVE the base cap) makes a
        halving genuinely halve the move. A scalar factor cannot rotate the
        step, so the null-space property grad-W . d = 0 is preserved.
        """
        if cap_adapt:
            return cap_state
        return cap0 * min(1.0, a / a0)
    for it in range(spec.max_iter):
        fom = float(project.compute_fom(p))
        cb.on_function_eval(project, it, p, fom)  # same jsonl + guards
        row = _row_of_params(spec, out_dir, p, need=("fwhm_env_um",))
        if not row:
            # ★A MISSING WIDTH IS A REJECTED STEP, NOT A DEAD CAMPAIGN (audit
            # S6, 2026-08-25). profile extraction failures are swallowed into
            # diag_error, so one flaky eval used to raise here and end an 81 h
            # / 300 G run with nothing to resume from. We genuinely cannot take
            # a projected step without the measured width — but we CAN back off
            # and retry from the last accepted point, which is exactly what the
            # filter already does for a rejected iterate.
            if acc is None:
                raise RuntimeError(
                    "wg_project: no fwhm_env_um on the FIRST eval — the width "
                    "pipeline is broken, not flaky; fix before rerunning")
            alpha *= 0.5
            if cap_adapt:
                cap_state = max(cap_state * 0.5, 2.0)
                n_rej += 1
            reuse_dirty = True
            step, phase, lam = _step_of(acc["gT"], acc["gW"], acc["W"], alpha,
                                        acc.get("gLam"), acc.get("lam_pk"))
            p = np.clip(acc["p"] + step, lo, hi)
            print(f"[proj {it}] NO WIDTH READ — treating as a rejected step, "
                  f"alpha -> {alpha:.4g}, retrying from the accepted point",
                  flush=True)
            _save_state()
            with open(ppath, "a") as f:
                f.write(json.dumps({"it": it, "t": time.time(), "fom": fom,
                                    "W": None, "alpha": alpha,
                                    "cap_nm": _cap(alpha),
                                    "phase": "no-width-retry"}) + "\n")
            continue
        W = float(row["fwhm_env_um"])
        h = abs(W - W_tgt)
        rec = {"it": it, "t": time.time(), "fom": fom, "W": W, "alpha": alpha,
               "cap_nm": _cap(alpha)}
        # ★λ-drift step bound (lit item 1, 2026-08-28: published recipe is
        # recentring PLUS an explicit bandwidth bound; drift is also how width
        # leaks past a fixed-λ gW). Opt-in knob, default None = inert.
        lam_pk_row = row.get("lam_pk_nm")
        if ns2 and lam_tgt is None and lam_pk_row is not None:
            lam_tgt = float(lam_pk_row)           # latch once, persist
            print(f"[proj {it}] ns2: λ target latched at {lam_tgt:.3f} nm",
                  flush=True)
            _save_state()
        lam_bound = getattr(spec, "wgp_lam_step_nm", None)
        lam_jump = (lam_bound is not None and acc is not None
                    and acc.get("lam_pk") is not None and lam_pk_row is not None
                    and abs(lam_pk_row - acc["lam_pk"]) > float(lam_bound))
        if lam_jump:
            rec["lam_jump_nm"] = lam_pk_row - acc["lam_pk"]
            print(f"[proj {it}] ★λ STEP BOUND: peak moved "
                  f"{rec['lam_jump_nm']:+.3f} nm > {lam_bound} nm — rejecting "
                  f"the step like a filter fail.", flush=True)
        if acc and (lam_jump or
                    not (fom > acc["fom"] - 1e-4 or h < acc["h"])):
            # minimal Fletcher-Leyffer filter. Reject: re-step from the
            # accepted point's STORED gradients at alpha/2 — no re-solve
            # (the trial forward is the bounded loss).
            alpha *= 0.5
            if cap_adapt:
                cap_state = max(cap_state * 0.5, 2.0)
                n_rej += 1
            reuse_dirty = True
            step, phase, lam = _step_of(acc["gT"], acc["gW"], acc["W"], alpha,
                                        acc.get("gLam"), acc.get("lam_pk"))
            p = np.clip(acc["p"] + step, lo, hi)
            rec.update(phase=phase + "-retry", lam=lam)
        else:
            # ★wgp_reuse_k decision — BEFORE compute_gradient so the width
            # adjoint solve is skipped from the queue. Reuse only when the
            # stored row is trustworthy: fresh enough, no reject since the
            # last refresh, W barely moved, and we're inside the deadband
            # (a real restoration needs a fresh gW).
            reuse_now = (ns2 and reuse_k >= 2 and acc is not None
                         and acc.get("gW_raw") is not None
                         and not reuse_dirty
                         and reuse_age < reuse_k - 1
                         # drift measured vs the point of the last FRESH
                         # solve, not per hop — k-proof (audit #4)
                         and abs(W - (reuse_W0 if reuse_W0 is not None
                                      else acc["W"])) <= reuse_dw
                         # within the full margin: light restoration may run
                         # on a reused row (first-order exact vs the row
                         # used; residual re-measured every eval) — only a
                         # REAL excursion forces a fresh solve. marg/2 here
                         # blocked reuse whenever any restoration was active,
                         # which is d1's normal regime (found 2026-08-31).
                         and abs(W - W_tgt) <= marg)
            spec._wg_reuse = bool(reuse_now)
            gT = np.asarray(project.compute_gradient(p), dtype=float)  # fwd cached
            if reuse_now:
                gW = np.asarray(acc["gW_raw"], dtype=float)
                reuse_age += 1
                rec["gw_reused"] = reuse_age
                print(f"[proj {it}] width row REUSED (age {reuse_age}/"
                      f"{reuse_k - 1}) — width adjoint skipped this iterate",
                      flush=True)
            else:
                gW = np.asarray(
                    project.parametrization.compute_gradient_from_fields(
                        project.fom.gfields_W, project.fdtd_session, p),
                    dtype=float)
                if acc is not None and acc.get("gW_raw") is not None:
                    # ★refresh-angle diagnostic (user, 2026-08-31): measured
                    # drift of the width row per refresh — the number that
                    # decides how deep wgp_reuse_k can safely go.
                    a_ = np.asarray(acc["gW_raw"], dtype=float)
                    na, nb = np.linalg.norm(a_), np.linalg.norm(gW)
                    if na > 0 and nb > 0:
                        rec["gW_refresh_cos"] = float(a_ @ gW / (na * nb))
                reuse_age = 0
                reuse_W0 = W          # fresh solve ⇒ new drift reference
            reuse_dirty = False
            gLam_vec = None       # ns2: set by the chain block when healthy
            if getattr(spec, "wg_lam_chain", False):
                # ★DEFECT #19: add the unpriced resonance-shift term, so the
                # projection nulls the gradient of the width we actually SPEC
                # (at the moving resonance) rather than at a frozen λ.
                #   gλ = dλ_pk/dp = −(∂²T/∂λ∂p)/(∂²T/∂λ²)   [implicit function
                #   theorem on the peak condition ∂T/∂λ = 0]
                ffl = getattr(project.fom, "gfields_Tlo", None)
                ffh = getattr(project.fom, "gfields_Thi", None)
                dTp = float(getattr(spec, "_wg_dTp", 0.0))
                # ★curvature floor (lit item 2, 2026-08-28): the IFT gain
                # 1/dTp is UNBOUNDED as the peak flattens — a sign check alone
                # lets a near-zero denominator rotate the null space wildly.
                # Relative floor vs the first healthy iterate; no absolute
                # constant to mis-tune.
                flat = (dTp < 0.0 and dTp0 is not None
                        and abs(dTp) < 0.05 * dTp0)
                if flat:
                    print(f"[proj {it}] ★λ-CHAIN CURVATURE COLLAPSED "
                          f"(|dTp| {abs(dTp):.3g} < 5% of it-0 {dTp0:.3g}) — "
                          f"skipping the chain term this step.", flush=True)
                    rec.update(lam_chain="curvature-floor")
                if ffl is not None and ffh is not None and dTp < 0.0 \
                        and not flat:
                    if dTp0 is None:
                        dTp0 = abs(dTp)
                    elif abs(dTp) < 0.25 * dTp0:
                        print(f"[proj {it}] ★λ-CHAIN CURVATURE THINNING: |dTp| "
                              f"{abs(dTp):.3g} < 25% of it-0 {dTp0:.3g} — gλ "
                              f"gain rising, watch it.", flush=True)
                    # convert HERE, not inside calculate_gradient_fields —
                    # the CAD is back in layout at this point (job 137845_41
                    # died on the in-analysis-mode dEps; this is the same
                    # session state where the gW conversion above works).
                    gfl, gfh = (np.asarray(
                        project.parametrization.compute_gradient_from_fields(
                            f, project.fdtd_session, p), dtype=float)
                        for f in (ffl, ffh))
                    gLam = -(gfh - gfl) / dTp                   # nm per param-nm
                    chain = float(spec.wg_dwdlam) * gLam        # µm per param-nm
                    # ★coefficient-sensitivity audit (lit item 3, 2026-08-28):
                    # wg_dwdlam is PATH-FITTED (±20%+, and meaningless on
                    # shift-frozen paths — detrend audit). Log how far the
                    # projected direction rotates when the coefficient swings
                    # x0.8 -> x1.2; small angle = projection robust to the fit.
                    dirs = []
                    for eta in (0.8, 1.2):
                        u = gW + eta * chain
                        u = u / (np.linalg.norm(u) or 1.0)
                        d = gT - (gT @ u) * u
                        dirs.append(d / (np.linalg.norm(d) or 1.0))
                    rec["proj_rot_deg"] = float(np.degrees(np.arccos(
                        np.clip(dirs[0] @ dirs[1], -1.0, 1.0))))
                    if ns2:
                        # ★d1: keep gLam as its OWN constraint row — do NOT
                        # fold the chain into gW. With gLam·d = 0 enforced,
                        # the fitted wg_dwdlam cancels out of the feasible
                        # directions for any coefficient value; folding it in
                        # would make the two rows near-collinear.
                        gLam_vec = gLam
                    else:
                        gW = gW + chain
                    spec._wg_gLam = gLam    # for dlam_pred once step is known
                    rec.update(gLam_n=float(np.linalg.norm(gLam)),
                               dwdlam=float(spec.wg_dwdlam), dTp=dTp)
                else:
                    # dTp ≥ 0 means the stencil does NOT straddle a maximum —
                    # off-resonance or a clipped band. Loud, because silently
                    # reverting to the fixed-λ gradient is the defect we are
                    # fixing.
                    print(f"[proj {it}] ★λ-CHAIN SKIPPED (dTp {dTp:+.4g}) — "
                          f"falling back to the FIXED-λ ∇W for this step; do "
                          f"not trust its width control.", flush=True)
                    rec.update(lam_chain="skipped")
            if (getattr(spec, "wgp_autogain", False) and acc is not None
                    and (acc.get("phase") == "restore"
                         # RGP has no restore phase; a strong correction
                         # component (wc > 0.5) is the equivalent condition —
                         # only then did the step carry real ∇W signal.
                         or (rgp and acc.get("wc", 0.0) > 0.5))):
                # ★DEFECT #18 (2026-08-25): the old gate was `abs(dW_pred) >
                # 1e-4`, which admits CLIMB steps — and climb is `alpha·D·gT`,
                # whose ∇W·step is whatever ∇T's incidental overlap with ∇W
                # happens to be (iterate 0 of job 137075_41: −2.1e-4, from a
                # shadow price of −2e-5). Dividing a real measured ΔW by that
                # near-zero gave r = −52 and fired the "C_field PHASE error"
                # alarm falsely. Only RESTORE builds its step along ∇W, so only
                # there is the prediction actual signal. (Deeper: until the
                # λ-chain term above is on, dW_meas is dominated by the
                # unpriced resonance drift and r is garbage in EVERY phase —
                # that false alarm was defect #19 announcing itself.)
                dW_pred = float(acc["gW"] @ (p - acc["p"]))
                dW_meas = W - acc["W"]
                if abs(dW_pred) > 1e-4:            # only when the step said something
                    r = dW_meas / dW_pred
                    # ★A NEGATIVE ratio means the width moved OPPOSITE to the
                    # prediction — that is a PHASE error in C_field, which no
                    # scalar gain can repair (audit 2026-08-25). Left unflagged
                    # it would pin wgain at the 0.2 clamp and masquerade as a
                    # harmless small gain while the null space stayed rotated.
                    if r < 0:
                        n_neg += 1
                        print(f"[proj {it}] ★WIDTH MOVED OPPOSITE TO PREDICTION "
                              f"(r={r:+.3f}, {n_neg} so far) — this is a C_field "
                              f"PHASE error, not a gain error; autogain cannot "
                              f"fix it. Re-fit the phase ON RESONANCE if this "
                              f"repeats.", flush=True)
                    else:
                        n_neg = 0
                    if 0.05 < abs(r) < 20.0:       # reject nonsense pairs
                        wgain = float(np.clip(0.7 * wgain + 0.3 * r, 0.2, 5.0))
                        print(f"[proj {it}] autogain: dW pred {dW_pred:+.4f} "
                              f"meas {dW_meas:+.4f} ratio {r:+.3f} -> gain "
                              f"{wgain:.3f}", flush=True)
            gW_eff = gW * wgain if getattr(spec, "wgp_autogain", False) else gW
            # gW_eff (not raw gW) is what the step is BUILT from, so it is also
            # what the retry branch and the dw_pred audit must use — otherwise
            # both silently disagree with the step whenever wgain ≠ 1.
            if cap_adapt and acc is not None:
                # ★trust-region growth: only on a VERIFIED accepted step —
                # both constraints measurably held (absolute bounds; a
                # relative dlam_pred test is ill-posed under ns2 where the
                # prediction is ~0 by construction).
                held = (abs(W - acc["W"]) < 0.020
                        and acc.get("lam_pk") is not None
                        and lam_pk_row is not None
                        and abs(lam_pk_row - acc["lam_pk"]) < 0.10)
                if held:
                    cap_state = min(cap_state * float(spec.wgp_cap_grow),
                                    float(spec.wgp_cap_max_nm))
                    print(f"[proj {it}] cap grown -> {cap_state:.3g} nm "
                          f"(held: dW {W - acc['W']:+.4f} um, dlam "
                          f"{lam_pk_row - acc['lam_pk']:+.3f} nm)", flush=True)
            n_acc += 1
            acc = {"p": p.copy(), "fom": fom, "W": W, "h": h,
                   "gT": gT, "gW": gW_eff, "gW_raw": gW, "gLam": gLam_vec,
                   "lam_pk": lam_pk_row}
            step, phase, lam = _step_of(gT, gW_eff, W, alpha,
                                        gLam_vec, lam_pk_row, mutate=True)
            if ns2 and (gLam_vec is None or lam_tgt is None):
                # chain skipped / curvature floor / no λ read — this step ran
                # the single-constraint law on the FIXED-λ gW. Loud: its width
                # control is the defect ns2 exists to fix.
                rec["ns2_fallback"] = True
                print(f"[proj {it}] ★ns2 FALLBACK: no gLam this iterate — "
                      f"single-constraint step on fixed-λ ∇W.", flush=True)
            st["Wh"].append(W)          # RGP: accepted widths size the buffer
            if getattr(spec, "wg_dwdlam_fit", False) and lam_pk_row is not None:
                st.setdefault("lamh", []).append((lam_pk_row, W))
                pts = sorted(set(st["lamh"]))
                lams = np.array([a for a, _ in pts])
                # np.ptp(): the ndarray METHOD was removed in NumPy 2.0 —
                # killed d1u (139050) at 11:39 h; IGUM's NumPy 1.x masked it.
                if len(pts) >= 5 and np.ptp(lams) >= 0.5:
                    slope = float(np.polyfit(lams,
                                             [w for _, w in pts], 1)[0])
                    cur = float(spec.wg_dwdlam)
                    new = float(np.clip(slope, 0.7 * cur, 1.3 * cur))
                    new = float(np.clip(new, 0.10, 0.70))
                    if abs(new - cur) > 1e-4:
                        spec.wg_dwdlam = new
                        print(f"[proj {it}] dwdlam refit: slope {slope:+.4f} "
                              f"(n={len(pts)}, span {np.ptp(lams):.2f} nm) -> "
                              f"{cur:.4f} => {new:.4f}", flush=True)
                    rec.update(dwdlam_fit=slope, dwdlam_n=len(pts))
            acc["wc"] = wc_last         # RGP: autogain gates on this
            acc["phase"] = phase        # defect #18: autogain gates on this
            p = np.clip(p + step, lo, hi)
            alpha = min(alpha * 1.2, float(spec.wgp_step) * 4.0)
            rec.update(phase=phase, lam=lam, wgain=wgain,
                       dw_pred=float(gW_eff @ (p - acc["p"])),  # post-clip audit
                       # ★dT_pred (2026-08-31, predictive stopping): expected
                       # FOM gain of the step just taken, from the exact
                       # adjoint gradient. Stop rule (checkpoint policy):
                       # dT_pred < the 0.002 T noise floor on 3 consecutive
                       # accepted iterates, OR cap driven to its 2 nm floor
                       # by rejects ⇒ converged-within-noise — detected 1-2
                       # iterates after flattening instead of 5 measured
                       # evals (~8-15 h saved per verdict).
                       dT_pred=float(gT @ (p - acc["p"])),
                       gT_n=float(np.linalg.norm(gT)),
                       gW_n=float(np.linalg.norm(gW_eff)))
            if ns2 and ns2_diag:
                rec.update(rho_T=ns2_diag.get("rho_T"),
                           condM=ns2_diag.get("condM"),
                           rW_um=ns2_diag.get("rW_um"),
                           rLam_nm=ns2_diag.get("rLam_nm"))
                if ns2_diag.get("ns2_degraded"):
                    rec["ns2_degraded"] = True
            if ns2 or cap_adapt:
                # per-block inf-norm shares of the step — the D-anisotropy
                # starvation readout (which blocks actually move)
                rec["step_shares_nm"] = {
                    "corr": float(np.max(np.abs(step[SL_CORR]))),
                    "avg": float(np.max(np.abs(step[SL_AVG]))),
                    "shift": float(np.max(np.abs(step[SL_SHIFT]))),
                    "comb": float(np.max(np.abs(step[SL_R.start:I_CAV]))),
                    "cav": float(abs(step[I_CAV]))}
            gl = getattr(spec, "_wg_gLam", None)
            if gl is not None:
                # predicted per-step resonance move — the DIRECT test of the
                # IFT gradient against the next eval's measured λ_pk (the
                # 137873 toy could not be judged on this: only the norm was
                # logged). nm units.
                rec["dlam_pred_nm"] = float(gl @ (p - acc["p"]))
                spec._wg_gLam = None
        rec["cap_nm"] = _cap(alpha)               # actual cap after any change
        _save_state()
        with open(ppath, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[proj {it}] {rec['phase']} fom {fom:.5f} W {W:.4f} "
              f"lam {rec.get('lam', 0.0):.4g} alpha {alpha:.3g} "
              f"cap {rec['cap_nm']:.3g}", flush=True)
    return (acc["p"] if acc else p), (acc["fom"] if acc else -np.inf)


def run_campaign(spec, out_dir, sigma0_um=None):
    """The main loop. Restarts from the best params on a recenter or width
    trip (rebuilding the base scene at the new λ) — also the crash-recovery
    path, since lumopt2 has no checkpointing. Returns the best (fom, params)."""
    lmpt = import_lumopt2()
    best = {"fom": -np.inf, "params": seed_params(spec)}
    # Cold-start resume: EVERY Athena partition is PreemptMode=REQUEUE (measured
    # 2026-08-14), so the driver can be evicted and rerun from scratch at any
    # time. The eval log persists across restarts — pick up from the best
    # recorded point instead of the seed (a fresh run has no log → seed).
    best["params"], spec.scan_center_nm = _best_from_log(spec, out_dir, best, sigma0_um)
    spec.scan_center_nm = round(float(spec.scan_center_nm), 2)
    for attempt in range(1 + MAX_RESTARTS):
        if getattr(spec, "trust_nm", None):
            # Trust regions must center on THIS attempt's actual start point,
            # not the module seed — on a cold-start resume the module seed can
            # sit on a bound edge (bare: shifts at 0), where the centering
            # fallback silently degrades to the full physical box and the
            # blow-up protection vanishes (the lnsrch death, job 55343).
            # Re-seeding per attempt also keeps the scaler round-trip exact
            # (no duplicate-x0 tax) on every restart, not just the first.
            spec.seed_override = tuple(np.asarray(best["params"], dtype=float))
        if getattr(spec, "sigma_wall", False):
            # re-anchor the sigma-hat wall at the current best's MEASURED sigma
            # (stage-wise trust region: each stage moves a bounded distance
            # where the linear surrogate holds; curvature handled by re-zeroing)
            p_b = np.asarray(best["params"], dtype=float)
            sig_meas = _sigma_of_params(spec, out_dir, p_b)
            if sig_meas is None:
                sig_meas = (spec.sig_anchor or {}).get("sigma")
            assert sig_meas is not None, "sigma_wall needs an initial sig_anchor"
            spec.sig_anchor = {"sigma": float(sig_meas),
                               "elong": float(2.0 * p_b[SL_SHIFT].sum()),
                               "rho": float(p_b[SL_CORR].mean() / CORR_NM),
                               "wcav": float(p_b[I_CAV])}
            print(f"[sigma-wall {attempt}] anchor sigma {sig_meas:.4f} um, "
                  f"elong {spec.sig_anchor['elong']:.1f} nm, "
                  f"rho {spec.sig_anchor['rho']:.4f}")
        if getattr(spec, "width_grad", False):
            # V2 (item 28): re-anchor the softW→fwhm map on the restart point's
            # own MEASURED pair, and update the AL multipliers on its MEASURED
            # band violation — the truth steers the multipliers, the surrogate
            # only steers inner gradients. Attempt 0 with a fresh log keeps the
            # runner-supplied initial anchor.
            rowb = _row_of_params(spec, out_dir, np.asarray(best["params"], float),
                                  need=("softw_um", "fwhm_env_um"))
            if rowb:
                # prefer the FOM carrier's own sample (single-λ twin) so the
                # penalty is exactly zero-offset at the anchor point
                sw_anchor = rowb.get("softw_adj_um") or rowb["softw_um"]
                spec.wg_anchor = {"softw": float(sw_anchor),
                                  "fwhm": float(rowb["fwhm_env_um"])}
                if spec.fwhm0_um is not None:
                    # fwhm0_um=None (pipeline smoke) ⇒ no band ⇒ no AL
                    # multiplier update (137870_47 died here on None*float)
                    g_hi = rowb["fwhm_env_um"] - RHO_UP * spec.fwhm0_um
                    g_lo = RHO_DN * spec.fwhm0_um - rowb["fwhm_env_um"]
                    spec.wg_lam_hi = max(0.0, spec.wg_lam_hi + spec.wg_mu * g_hi)
                    spec.wg_lam_lo = max(0.0, spec.wg_lam_lo + spec.wg_mu * g_lo)
                print(f"[width-grad {attempt}] anchor softW "
                      f"{spec.wg_anchor['softw']:.4f} / fwhm "
                      f"{spec.wg_anchor['fwhm']:.4f} um; lam_hi "
                      f"{spec.wg_lam_hi:.4f} lam_lo {spec.wg_lam_lo:.4f}")
        if getattr(spec, "fwhm_wall", False):
            # ★BUG FIX 2026-08-23 (proactive audit): within one process the
            # callback re-anchors on every accepted-best, but a COLD restart
            # (preemption/requeue) re-imports the module and resets fw_anchor
            # to the SEED literal while _best_from_log resumes elsewhere —
            # the wall would then predict width at the wrong operating point
            # (item-24 failure class). Re-anchor from the resumed row.
            rowf = _row_of_params(spec, out_dir,
                                  np.asarray(best["params"], float),
                                  need=("fwhm_env_um",))
            if rowf:
                pb = np.asarray(best["params"], dtype=float)
                spec.fw_anchor = {"fwhm": float(rowf["fwhm_env_um"]),
                                  "mcorr": float(np.mean(pb[SL_CORR])),
                                  "elong": float(2.0 * pb[SL_SHIFT].sum()),
                                  "corr_vec": tuple(float(v)
                                                    for v in pb[SL_CORR])}
                print(f"[fwhm-wall {attempt}] re-anchored at measured "
                      f"{spec.fw_anchor['fwhm']:.4f} um "
                      f"(mcorr {spec.fw_anchor['mcorr']:.1f})")
        project, _ = make_project(spec, out_dir, lmpt)
        cb = make_log_callback(spec, out_dir, sigma0_um, lmpt)
        try:
            if getattr(spec, "wg_project", False):
                result = run_projected(spec, project, cb, out_dir,
                                       np.asarray(best["params"], dtype=float))
            else:
                optimizer = lmpt.ScipyOptimizer(method="L-BFGS-B", max_iter=spec.max_iter,
                                                max_feval=spec.max_feval, ftol=1e-8,
                                                gtol=1e-6, max_line_search=4,
                                                bounds=param_bounds(spec))
                # FileLogger added EXPLICITLY: passing a custom callbacks list
                # replaces the auto-installed one (docs audit 2026-08-15, risk #1).
                opt = lmpt.Optimization(project, optimizer,
                                        callbacks=[cb, lmpt.FileLogger()],
                                        store_all_simulations=True)
                result = opt.run(initial_params=np.asarray(best["params"], dtype=float))
            # Final selection goes through the SAME width-compliant filter as
            # restarts (platform guarantee: a violating design can never be
            # delivered) — never trust an optimizer-internal optimum directly.
            best["params"], _ = _best_from_log(spec, out_dir, best, sigma0_um)
            rows_best = _final_fom(result)
            if rows_best > best["fom"]:
                best["fom"] = rows_best             # bookkeeping only; params
                                                    # always from filtered log
            break                                   # converged / budget exhausted
        except (RecenterNeeded, WidthTrip, RuntimeError) as e:
            # lumopt2 wraps fct exceptions TWICE: scipy_optimizer.py:583
            # raises RuntimeError WITHOUT `from e` (original only in
            # __context__ — measured job 133016), then optimization.py:852
            # wraps that WITH `from e`. Walk BOTH chains to find our guard
            # exceptions (measured kills: jobs 54309, 133016).
            root = e
            for _ in range(8):
                if isinstance(root, (RecenterNeeded, WidthTrip)):
                    break
                nxt = root.__cause__ or root.__context__
                if nxt is None or nxt is root:
                    break
                root = nxt
            if isinstance(root, RecenterNeeded):
                best["params"], new_center = _best_from_log(spec, out_dir, best, sigma0_um)
                print(f"[recenter {attempt}] {root} -> new center {new_center:.2f} nm")
                spec.scan_center_nm = round(new_center, 2)
                # ★2026-08-31 (adversarial audit #1): a recenter means the
                # operating point MOVED — a persisted λ target / curvature
                # reference from before it would make the ns2 restoration
                # fight the recenter (pull λ back to the old resonance) and
                # mis-floor the chain guard. Drop both; they re-latch on the
                # next iterate at the new point.
                ost_r = _load_opt_state(out_dir, spec.label)
                if ost_r:
                    ost_r["lam_tgt_nm"] = None
                    ost_r["dTp0"] = None
                    _save_opt_state(out_dir, spec.label, ost_r)
                    print(f"[recenter {attempt}] optstate λ-target + dTp0 "
                          f"cleared for re-latch", flush=True)
            elif isinstance(root, WidthTrip):
                best["params"], _ = _best_from_log(spec, out_dir, best, sigma0_um)
                if getattr(spec, "sigma_wall", False) or                         getattr(spec, "fwhm_wall", False):
                    # under the sigma-hat wall, corr UP is the width PAYBACK
                    # direction — capping it would block the trade. The restart
                    # re-anchor (top of loop) already tightens the wall to the
                    # violating stage's measured reality; that is the response.
                    # corr UP is the width PAYBACK direction under a width
                    # wall — capping it would block the very recovery the
                    # trip demands (fix 2026-08-23, same reasoning as the
                    # sigma-wall branch). Re-anchoring IS the response.
                    print(f"[width trip {attempt}] {root} -> re-anchor only")
                else:
                    spec.corr_max_nm *= 0.95
                    best["params"][SL_CORR] = np.minimum(best["params"][SL_CORR], spec.corr_max_nm)
                    print(f"[width trip {attempt}] {root} -> corr cap {spec.corr_max_nm:.0f} nm")
            else:
                raise                               # a real error — let it crash loudly
    out = {"label": spec.label, "best_fom": best["fom"],
           "best_params_nm": np.asarray(best["params"]).tolist(),
           "scan_center_nm": spec.scan_center_nm}
    with open(os.path.join(out_dir, f"{spec.label}_best.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[done] best FOM {best['fom']:.5f} -> {spec.label}_best.json")
    return best


def _row_of_params(spec, out_dir, p, tol=1e-6, need=()):
    """First eval-log row matching params p and carrying every `need` key
    (None if absent) — duplicate evals of the same p may differ in diagnostics."""
    path = os.path.join(out_dir, f"{spec.label}_evals.jsonl")
    if not os.path.exists(path):
        return None
    p = np.asarray(p, dtype=float)
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if all(row.get(k) for k in need) and np.allclose(
                    np.asarray(row["params"], dtype=float), p, atol=tol):
                return row
    return None


def _sigma_of_params(spec, out_dir, p, tol=1e-6):
    """Measured sigma_um of the log row whose params match p (None if absent)."""
    row = _row_of_params(spec, out_dir, p, tol, need=("sigma_um",))
    return float(row["sigma_um"]) if row else None


def _best_from_log(spec, out_dir, fallback, sigma0_um=None):
    """Best WIDTH-COMPLIANT params + their peak λ from the eval log.

    2026-08-16: selection MUST filter on the σ deadband — the width-cheat
    design is by construction the FOM-max row, so an unfiltered max restarts
    the campaign INSIDE the violation (measured burn loop, job 54488). Rows
    without a σ reading are treated as non-compliant when a σ0 exists.
    """
    path = os.path.join(out_dir, f"{spec.label}_evals.jsonl")
    best_fom = fallback["fom"]
    best_p = np.asarray(fallback["params"], dtype=float)
    lam = spec.scan_center_nm
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if sigma0_um and not (row.get("sigma_um") and
                        RHO_DN <= row["sigma_um"] / sigma0_um <= RHO_UP):
                    continue
                # FWHM (spec observable) filter. Rows predating fwhm_env_um
                # PASS here (missing != violating) — see CampaignSpec.fwhm0_um.
                f0 = getattr(spec, "fwhm0_um", None)
                if f0:
                    fw = row.get("fwhm_env_um")
                    if not fw:
                        # ★FAIL-CLOSED 2026-08-23 (proactive audit): a row
                        # whose profile extraction failed carries NO width
                        # evidence. For a width-guarded campaign that row must
                        # not be selectable — the legacy fail-open rule exists
                        # only for pre-fwhm_env logs, so keep it exactly there.
                        if getattr(spec, "fwhm_wall", False) or                                 getattr(spec, "width_grad", False):
                            continue
                    elif not (RHO_DN <= fw / f0 <= RHO_UP):
                        continue
                if row["fom"] > best_fom:
                    best_fom, best_p = row["fom"], np.asarray(row["params"])
                    if row.get("lam_pk_nm"):
                        lam = row["lam_pk_nm"]
    return best_p, lam


def run_canary(spec, out_dir):
    """Gate B2: ONE forward through the full lumopt2 stack at the seed.

    Run it with spec.bare=True to compare against the STORED bare N=100
    anchor (IGUM 51736: T 0.9104 / λ 1559.01 / Q 1760 / mode 19.24 µm —
    never re-run the control itself), and once with the comb to calibrate
    σ0 for the campaign tripwire. Returns the logged diagnostics row.
    """
    lmpt = import_lumopt2()
    project, _ = make_project(spec, out_dir, lmpt)
    cb = make_log_callback(spec, out_dir, lmpt=lmpt)
    p0 = seed_params(spec)
    fom = project.compute_fom(p0)
    try:
        cb.on_function_eval(project, 0, p0, fom)
    except (RecenterNeeded, WidthTrip):
        pass                                       # canary only reports, gates decide
    path = os.path.join(out_dir, f"{spec.label}_evals.jsonl")
    with open(path) as f:
        row = json.loads(f.readlines()[-1])
    print(f"[canary {spec.label}] FOM {fom:.5f}  T {row.get('t_pk')}  "
          f"λ {row.get('lam_pk_nm')}  Q {row.get('q_loaded')}  "
          f"σ {row.get('sigma_um')} µm")
    return row


def run_validate_gradient(spec, out_dir, indices, perturbation=2.0, detune=1):
    """Gate B3: lumopt2's built-in adjoint-vs-FD check on chosen params.

    Runs at a DETUNED point, not the seed (measured 132636): the seed sits ON
    the shift lower bound (central FD steps outside → ValueError) and AT the
    measured comb optimum (comb gradients ~0 there → sign/scale gates are
    noise). Perturbation is physical nm; 2.0 keeps ΔFOM well above the FDTD
    jitter floor while staying in the linear range.

    detune=2 → a second operating point (α-stability measurement for the
    per-class calibration route). Gradient-fix experiments (colocate_fields /
    bc_patch / grad_cal) are driven by the SPEC flags — build variant specs
    with dataclasses.replace.
    """
    lmpt = import_lumopt2()
    project, _ = make_project(spec, out_dir, lmpt)
    p = seed_params(spec)
    if detune == 1:
        p[SL_SHIFT] = 20.0                    # off the (0, 200) lower bound
        p[SL_R.start + COMB_N_HALF] = 100.0   # detune center post r 80→100
        p[SL_X.start + COMB_N_HALF] += 50.0   # detune center post x by +50 nm
        p[I_DCOMB] = 1750.0                   # detune comb distance 1900→1750
    else:                                     # point 2 — different geometry
        p[SL_CORR] = 355.0
        p[SL_AVG] = 795.0
        p[SL_SHIFT] = 40.0
        p[SL_R.start + COMB_N_HALF] = 115.0
        p[SL_X.start + COMB_N_HALF] -= 40.0
        p[I_DCOMB] = 1820.0
    # RAW-vs-RAW (fix 2026-08-23): lumopt2's FD side never sees the attached
    # penalty (it calls fom.calculate_fom), so the adjoint side must not
    # either — at the detune points the elong guard's pen_grad is NONZERO
    # (0.0352/shift) and would contaminate the printed vector / C fit.
    # Vectors printed by jobs ≤136190 predate this fix: correct them offline
    # (fit_c_field.py PEN_GRAD) before fitting.
    project.compute_gradient = project.compute_gradient_raw
    res = lmpt.validate_gradient(project, p, indices, perturbation)
    print(f"[validate_gradient detune={detune} colocate={spec.colocate_fields} "
          f"bc_patch={spec.bc_patch}] {res}")
    return res


def run_adjoint_only(spec, out_dir, indices, detune=1):
    """Adjoint gradient WITHOUT the FD half (2 sims, ~50 min vs ~7 h).

    For gradient-fix iteration: FD is config-independent and already stored
    at detune=1 in THREE reproducible tables (132657 / 132883 tasks 5, 6 —
    agree ≤2%), so a fix experiment only needs the adjoint side. Compare the
    printed vector offline against the stored FD.
    """
    lmpt = import_lumopt2()
    project, _ = make_project(spec, out_dir, lmpt)
    p = seed_params(spec)
    if detune == 1:
        p[SL_SHIFT] = 20.0
        p[SL_R.start + COMB_N_HALF] = 100.0
        p[SL_X.start + COMB_N_HALF] += 50.0
        p[I_DCOMB] = 1750.0
    else:
        p[SL_CORR] = 355.0
        p[SL_AVG] = 795.0
        p[SL_SHIFT] = 40.0
        p[SL_R.start + COMB_N_HALF] = 115.0
        p[SL_X.start + COMB_N_HALF] -= 40.0
        p[I_DCOMB] = 1820.0
    # RAW gradient (fix 2026-08-23): the stored FD tables these prints are
    # compared against are penalty-free (see run_validate_gradient) — the
    # wrapped gradient would subtract a nonzero pen_grad at the detune
    # points (0.0352/shift) and skew the quadrature/C fit.
    grad = project.compute_gradient_raw(params=p)
    out = np.asarray([grad[i] for i in indices])
    # header prints BOTH C knobs (2026-08-23: the old header showed only the
    # PORT C, so a width-path quadrature run at C_field=(0,1) printed
    # "C=(1.0,0.0)" and read like the rotation was never consumed — it is,
    # via MixedFom._compute_adjoint_fields_phased; the print was cosmetic)
    print(f"[adjoint_only detune={detune} adj_phase_fix={spec.adj_phase_fix} "
          f"C_port=({spec.adj_fix_re},{spec.adj_fix_im}) "
          f"C_field=({spec.adj_fix_field_re},{spec.adj_fix_field_im}) "
          f"indices={indices}] {out!r}")
    return out
