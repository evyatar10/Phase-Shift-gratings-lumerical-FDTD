"""Q3dB design tool: predict long-device observables / design (corr, N) with
ONE confirmation run — no tuning ladder.

Reads q3db_calibration.csv (written by calibrate_q3db.py from stored results
only; hold-out backtested there). Model: L0 exact two-port algebra + per-family
Qc exponential + saturating-Qi power law + width truncation fit. Families are
single-polarization by construction (tm_*/te_*/itai_*) — TM and TE anchors are
never mixed; the width<->corr knob lines are per-polarization too.

Edit the knobs below and run (no CLI args — CLAUDE.md par.11):
  MODE="observe" -> predict observables of FAMILY at N
  MODE="design"  -> find N* for TARGET_DB (and optionally TARGET_WIDTH_UM via
                    the corrugation knob) in FAMILY
  MODE="extend"  -> take ONE new measured device (ROW below), anchor the
                    levels on it, borrow the SHAPE from BASE_FAMILY, then
                    predict any N / solve the generalized Q3dB point.
  MODE="compare" -> a run has LANDED (MEASURED below): predict at its N and
                    print PREDICTED vs MEASURED with an INSIDE/OUTSIDE verdict
                    against the empirical hold-out deviation band.
Targets are generalized: TARGET_DB is any dB point (-3 dB default), and
TARGET_WIDTH_UM any mode width (e.g. 14) — width is retuned by CORRUGATION
via the per-pol knob line, which rescales Qc rate (kappa prop. corr, measured
0.1-1.3%), F_inf (measured cross-family exponent -1.11) and Qi (radiative
corr^-2.9, MEASURED for TM at N=150; for TE this exponent is UNMEASURED and
the output is a band, labeled).

Two error bars are printed with every prediction: "model sensitivity" (the fit
parameters wiggled) and "expected deviation" — the MEASURED spread of the
hold-out backtests at the same extrapolation span (family "errband" in the CSV,
written by calibrate_q3db.py). The second one is what a landed run is judged
against; compare mode does that comparison for you.

Every number printed is PREDICTED unless labeled; the confirmation-run spec
is the dispatch note. Validity rules are printed with each output:
  - calibration rows need 2*kappa*L >= ~3.2 (below that the device is too
    short to carry the family shape — prediction refused);
  - T +-0.03 holds to ~30 periods beyond the anchored range; beyond ~45 it is
    a band, not a point (measured boundary, backtest B2-E);
  - a single-row anchor pins LEVELS but trusts the base family's Qi SHAPE —
    quote the band, and one extra row ~30 periods away removes most of it.

EXTENSION SEMANTICS (user rule 2026-09-11): "extending" a device ALWAYS means
adding UNIFORM periods on the outside. Whatever is inside the measured device
(apodization, comb, tooth shifts) stays as measured and is carried ONLY by the
anchored levels; ROW corr_nm is the OUTER (uniform) corrugation. Nothing inside
is modeled and no effort goes there.

SCOPE — which device classes the calibration covers, and at what grade:
  - bare uniform gratings, TM corr 276/325/448 and TE corr 250: calibrated AND
    live-validated -> DESIGN-GRADE (predict, then ONE confirmation run).
  - decorated devices (trench / flush / comb): only via the measured Qi
    multipliers at the -3 dB anchor (backtest B8) and the tm_trench_c325
    family; away from those anchors EXPECTED-grade.
  - the inverse-designed device (comb + per-tooth shifts/widths): only the
    tm_invdesign family AS MEASURED — no generalization to other shift/comb
    settings; for a new setting use extend mode with its own anchor row.
  - apodized devices: mode WIDTH via the CMT kappa(z) engine (backtest B11:
    <1% TM, <5% TE); T and Q only through the itai_* (HH-apodized) families
    as shapes.
  - tooth shifts: NOT modeled by the engine or the fits (a shift is a phase
    perturbation, not a kappa change) — measured families only.
  - the TE corrugation knob line rests on ONE N=80 legacy point (~4%
    truncation bias) plus TM-derived exponents: EXPECTED-grade.

Usage examples (edit the knobs below, then run):
  # observe: what does the stored c325 family give at N=180?
  MODE, FAMILY, N = "observe", "tm_bare_c325", 180
  # design: -3 dB with a 14 um mode (corr knob; bare families only)
  MODE, FAMILY = "design", "tm_bare_c325"
  TARGET_DB, TARGET_WIDTH_UM = -3.0, 14.0
  # extend: anchor on ONE new measured row, borrow the family shape
  MODE, BASE_FAMILY, TARGET_DB = "extend", "tm_bare_c325", -3.0
  ROW = dict(pol="TM", corr_nm=325.0, pitch_nm=516.83, N=100, T=0.9104, ...)
  # compare: did the landed run agree? (COMPARE_FROM_ROW=True anchors on ROW)
  MODE, FAMILY, COMPARE_FROM_ROW = "compare", "tm_bare_c448", False
  MEASURED = dict(N=103, T=0.5097, Q_L=4644.0, width_um=14.19, lam_nm=1557.75)
"""

import os

import numpy as np
from scipy.optimize import brentq

# ------------------------------- knobs -------------------------------------
MODE = "extend"            # "observe" | "design" | "extend"
FAMILY = "tm_bare_c325"    # family key in q3db_calibration.csv
N = 165                    # observe-mode N
TARGET_DB = -3.0           # peak-T target in dB (0.5 = -3.01 dB)
TARGET_WIDTH_UM = None     # e.g. 14.0 -> retune corr via the knob line; None = keep corr
# extend-mode: the NEW measured device (fill from the result .mat / your note).
# mesher matters: the calibration families are ALL conformal (the q3db family
# numerics). "pva" rows (the optimizer's smoother mesher) are a DIFFERENT
# frame: lam +5.3 nm / FWHM -8% / T +0.0079 / Q_L -7% vs conformal (stored
# notes, mesher memory + prod_q3db_ladder header) — never mix the frames.
ROW = dict(pol="TM", corr_nm=325.0, pitch_nm=516.83, N=100, T=0.9104,
           Q_L=1760.0, lam_nm=1559.006, width_um=19.245, mesher="conformal")
BASE_FAMILY = "tm_bare_c325"   # shape priors for extend mode (match pol!)
# compare-mode: the run that LANDED (width_um / lam_nm optional -> None skips them)
MEASURED = dict(N=103, T=0.5097, Q_L=4644.0, width_um=14.19, lam_nm=1557.75)
COMPARE_FROM_ROW = False   # True -> compare against the ROW-anchored extension
# ----------------------------------------------------------------------------

CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q3db_calibration.csv")
Q_ADEQ = 5e4
CORR_QI_EXP_TM = -2.9      # MEASURED (N=150 trench corr ladder, backtest B12)
FINF_CORR_EXP = -1.11      # MEASURED cross-family (F_inf 20.06@c325 vs 24.05@c276)
# Qc corr-transform intercept: lnQc = ... + rate(corr)*N + QC_H_PER_NM*corr.
# MEASURED on the stored N=150 corr ladder (C266-400, residuals <=3.5%);
# post-hoc validated on the C448/N=98 live row: -7.8% (old rate-only
# transform: +31%, the 2026-09-01 knob-test T miss). See memory file.
QC_H_PER_NM = -0.002818
# fallback N_min when a family has no width fit: kappa prop. corr (MEASURED)
KAPPA_C325_TM = 0.0353e6   # 1/m at corr 325 (MEASURED family kappa)
PITCH_TM_M, PITCH_TE_M = 516.83e-9, 500e-9
CORR_BOUNDS_NM = (150.0, 650.0)   # corr* search band (outside = refuse, not extrapolate)
# a corr MOVE is not covered by the length-span hold-outs (they only extrapolate N):
# all we have is the c448 rung-1 live row after the two-term fix and B14's Qc residual
QL_BAND_KNOB, T_BAND_KNOB = 0.10, 0.03
N_SEARCH = (20.0, 3000.0)   # N range the dB-target solver looks in
CORR_MATCH_NM = 2.0         # corr agreement below this = the same corrugation
W_BAND_REL, LAM_BAND_NM = 0.05, 1.0   # standing pass bands for width / lambda
NUMERICS_NOTE = "y8.0/z8.8 box, 20nm window/4001pts (3nm/0.75pm if Q_L>5e4), dx50 conformal, ASL 1e-7"

def load_calib():
    fams = {}
    for line in open(CALIB).read().strip().splitlines()[1:]:
        fam, param, val = line.split(",")[:3]
        fams.setdefault(fam, {})[param] = float(val)
    return fams

def qc_of_N(p, n):
    return np.exp(p["qc_lnQ0"] + p["qc_rate"] * n)

def qi_of_N(p, n):
    qi = np.exp(p["qi_lnA"] + p["qi_p"] * np.log(n))
    if np.isfinite(p.get("qi_sat", np.nan)):
        qi = 1.0 / (1.0 / qi + 1.0 / p["qi_sat"])
    return qi

def width_of_N(p, n):
    if "w_Finf" not in p:
        return np.nan
    return p["w_Finf"] - p["w_B"] * np.exp(-p["w_c"] * n)

def observables(p, n):
    qc, qi = qc_of_N(p, n), qi_of_N(p, n)
    ql = 1.0 / (1.0 / qc + 1.0 / qi)
    T = (qi / (qi + qc)) ** 2
    lam = p.get("lam_nm", np.nan)
    return dict(N=n, T=T, Q_L=ql, Q_c=qc, Q_i=qi, lam_nm=lam,
                spec_fwhm_pm=1e3 * lam / ql if np.isfinite(lam) else np.nan,
                width_um=width_of_N(p, n))

def design_N(p, T_target):
    """N where peak T crosses the target; None if this device never gets there
    (T falls monotonically with N, so no crossing = the target is unreachable)."""
    f = lambda n: (qi_of_N(p, n) / (qi_of_N(p, n) + qc_of_N(p, n))) ** 2 - T_target
    lo, hi = N_SEARCH
    if f(lo) * f(hi) > 0:
        return None
    return brentq(f, lo, hi, xtol=1e-3)

def anchor_on_row(base, row):
    """Extend mode: keep BASE_FAMILY's rates/exponents, shift the Qc and Qi
    LEVELS so the curves pass exactly through the measured row."""
    p = dict(base)
    sqT = np.sqrt(row["T"])
    qc_meas, qi_meas = row["Q_L"] / sqT, row["Q_L"] / (1.0 - sqT)
    p["qc_lnQ0"] += np.log(qc_meas / qc_of_N(base, row["N"]))
    f = qi_meas / qi_of_N(base, row["N"])
    p["qi_lnA"] += np.log(f)
    if np.isfinite(p.get("qi_sat", np.nan)):
        p["qi_sat"] *= f
    p["lam_nm"] = row["lam_nm"]
    if "w_Finf" in p and np.isfinite(row.get("width_um", np.nan)):
        p["w_Finf"] += row["width_um"] - width_of_N(base, row["N"])
    p["n_lo"] = p["n_hi"] = row["N"]
    return p, qc_meas, qi_meas

def rescale_family_to_corr(p, corr_from, corr_to):
    """Move a family's SHAPE to a new corrugation: Qc rate AND level, width fit,
    Qi. TM-measured exponents (CORR_QI_EXP_TM / QC_H_PER_NM / FINF_CORR_EXP), so
    EXPECTED-grade for TE. Shared by the width knob and the extend-mode anchor."""
    corr_now, corr_new = corr_from, corr_to
    r = corr_new / corr_now
    q = dict(p)
    q["qc_rate"] = p["qc_rate"] * r                      # kappa prop. corr (0.1-1.3%)
    q["qc_lnQ0"] = p["qc_lnQ0"] + QC_H_PER_NM * (corr_new - corr_now)
    if "w_Finf" in p:
        scale = r ** FINF_CORR_EXP
        q["w_Finf"], q["w_B"] = p["w_Finf"] * scale, p["w_B"] * scale
        q["w_c"] = p["w_c"] * r                          # 2*kappa*Lambda prop. corr
    fqi = r ** CORR_QI_EXP_TM
    q["qi_lnA"] = p["qi_lnA"] + np.log(fqi)
    if np.isfinite(q.get("qi_sat", np.nan)):
        q["qi_sat"] *= fqi
    return q

def retune_corr(p, corr_now, pol, width_target, T_target):
    """Corrugation knob. The per-pol 1/w-vs-corr knob line is only the initial
    guess: it lives at the knob rows' own lengths, so scaling the ASYMPTOTIC
    width fit with it misses the target (TM ~1%, TE ~9%). So solve corr* so the
    RETUNED family hits width_target at its OWN design length N*(corr).
    Returns (p', corr*, note) — p' None means refused (no root in the band)."""
    kn = load_calib()["knob_tm" if pol == "TM" else "knob_te"]
    corr_line = (1.0 / width_target - kn["iw_a"]) / kn["iw_b"]
    if "w_Finf" not in p:                      # nothing to solve against
        return rescale_family_to_corr(p, corr_now, corr_line), corr_line, \
            "knob line only — no width fit for this family, width NOT solved (EXPECTED)"

    def miss(corr):
        q = rescale_family_to_corr(p, corr_now, corr)
        n_star = design_N(q, T_target)
        if n_star is None:            # this corr is too lossy to reach the dB target
            return -1e3               # push the solver toward smaller corrugations
        return width_of_N(q, n_star) - width_target
    lo, hi = CORR_BOUNDS_NM
    if miss(lo) * miss(hi) > 0:
        return None, corr_line, (f"no corrugation in {lo:.0f}-{hi:.0f} nm gives width"
                                 f" {width_target} um at this dB target"
                                 f" (knob-line guess was {corr_line:.1f} nm)")
    corr_new = brentq(miss, lo, hi, xtol=1e-2)
    return rescale_family_to_corr(p, corr_now, corr_new), corr_new, \
        f"self-consistent at N*(corr); knob-line guess was {corr_line:.1f} nm"

def validity_notes(p, n, corr=np.nan, pol="TM"):
    notes = []
    n_min, src = np.nan, ""
    if "w_c" in p:                                       # w_c == 2*kappa*Lambda
        n_min = 3.2 / p["w_c"]
    elif np.isfinite(corr):                              # no width fit for this family
        kappa = KAPPA_C325_TM * corr / 325.0             # kappa prop. corr (MEASURED)
        n_min = 3.2 / (2 * kappa * (PITCH_TM_M if pol == "TM" else PITCH_TE_M))
        src = " (kappa from corr, EXPECTED)"
    if np.isfinite(n_min):
        if n < n_min:
            notes.append(f"REFUSE: N={n:.0f} < N_min~{n_min:.0f} (2kL<3.2 — device too short to carry the family shape){src}")
        notes.append(f"shortest usable calibration/anchor device: N_min ~ {n_min:.0f} (2kL>=3.2){src}")
    span = n - p.get("n_hi", n)
    if span > 45:
        notes.append(f"EXTRAPOLATION {span:.0f} periods beyond anchored range: T is a BAND not a point (measured boundary ~45)")
    elif span > 30:
        notes.append(f"{span:.0f} periods beyond anchored range: T +-0.03 marginal (rule: <=30)")
    return notes

def uncertainty_band(p, n, dqi=0.07, drate=0.01):
    outs = []
    for si in (-dqi, 0, dqi):
        for sr in (-drate, 0, drate):
            q = dict(p)
            q["qi_lnA"] = p["qi_lnA"] + np.log(1 + si)
            if np.isfinite(q.get("qi_sat", np.nan)):
                q["qi_sat"] = p["qi_sat"] * (1 + si)
            q["qc_rate"] = p["qc_rate"] * (1 + sr)
            outs.append(observables(q, n))
    return (min(o["Q_L"] for o in outs), max(o["Q_L"] for o in outs)), \
           (min(o["T"] for o in outs), max(o["T"] for o in outs))

def deviation_band(bands, span):
    """Empirical hold-out deviation at this extrapolation span: (Q_L rel, T abs,
    n_samples, bucket, few_samples). bands = the CSV's "errband" family, written
    by calibrate_q3db.py from the backtest residuals."""
    key = "le30" if span <= 30 else ("31_45" if span <= 45 else "gt45")
    nq, nt = bands.get(f"ql_n_{key}", 0), bands.get(f"t_n_{key}", 0)
    if nq + nt == 0:
        return np.nan, np.nan, 0, key, True
    few = min(nq, nt) < 3
    stat = "max" if few else "p90"
    return (bands.get(f"ql_{stat}_{key}", np.nan), bands.get(f"t_{stat}_{key}", np.nan),
            nq + nt, key, few)

def knob_band_line():
    return ("expected deviation (corr-knob designs: 1 post-fix live validation"
            " T -0.002 / Q_L -0.5%, plus the B14 Qc residual -7.6%): quote"
            f" Q_L +-{100*QL_BAND_KNOB:.0f}%, T +-{T_BAND_KNOB:.2f} (EXPECTED)")

def print_prediction(fam, p, n, corr_note="", corr=np.nan, pol="TM", bands=None,
                     corr_moved=False):
    obs = observables(p, n)
    (ql_lo, ql_hi), (t_lo, t_hi) = uncertainty_band(p, n)
    print(f"\n{fam} at N={n:.0f} {corr_note}(PREDICTED)")
    print(f"  T      = {obs['T']:.4f}  [{t_lo:.4f} .. {t_hi:.4f}]  ({10*np.log10(obs['T']):.2f} dB)")
    print(f"  Q_L    = {obs['Q_L']:.0f}  [{ql_lo:.0f} .. {ql_hi:.0f}]   Q_c={obs['Q_c']:.0f}  Q_i={obs['Q_i']:.0f}")
    print(f"  lambda = {obs['lam_nm']:.2f} nm   spectral fwhm = {obs['spec_fwhm_pm']:.2f} pm"
          if np.isfinite(obs["lam_nm"]) else "  lambda = n/a (no lambda recorded for this family)")
    print(f"  width  = {obs['width_um']:.2f} um" if np.isfinite(obs["width_um"])
          else "  width  = n/a (no width fit for this family)")
    print("  [ .. ] = model sensitivity (Qi +-7%, Qc rate +-1%)")
    span = max(0.0, n - p.get("n_hi", n))
    dql, dt, nsamp, key, few = deviation_band(bands or {}, span)
    if corr_moved:      # the span buckets only cover LENGTH extrapolation
        print(f"  {knob_band_line()}")
    elif nsamp:
        print(f"  expected deviation (from {nsamp:.0f} hold-outs at span {span:.0f}, bucket"
              f" {key}): Q_L +-{100*dql:.1f}%, T +-{dt:.3f}"
              + ("  (few samples: max, not p90)" if few else ""))
    else:
        print(f"  expected deviation: no hold-outs at span {span:.0f} periods"
              f" (bucket {key}) — model sensitivity only")
    for note in validity_notes(p, n, corr, pol):
        print(f"  ! {note}")
    if obs["Q_L"] > Q_ADEQ:
        print("  ! confirmation run needs the HIGH-Q window (3nm/0.75pm) and"
              " os.environ['TM_SIM_TIME_PS']='4000' inside the runner")
    return obs

def bare_family_at_corr(fams, pol, corr):
    """A calibrated BARE family of this polarization at this corrugation, if the
    CSV holds one — the right base to borrow shapes from."""
    for name in fams:
        if not name.startswith("tm_bare_" if pol == "TM" else "te_q3db_"):
            continue
        tail = name.split("_c")[-1]
        if tail.isdigit() and abs(float(tail) - corr) <= CORR_MATCH_NM:
            return name
    return None

def anchor_with_corr_fix(fams, pol, corr_now, base_corr):
    """Anchor on ROW: move the base family's SHAPE to the ROW's corrugation
    first (both corrugations known), then shift the levels onto the row. Used by
    extend mode and by compare mode's COMPARE_FROM_ROW path — same physics."""
    check_anchor_pol(pol)
    base = fams[BASE_FAMILY]
    if not np.isfinite(base_corr):
        print(f"  ! {BASE_FAMILY} has no corrugation in its name: its shape is"
              " borrowed AS-IS, no corr transform (EXPECTED-grade)")
    elif abs(corr_now - base_corr) > CORR_MATCH_NM:
        better = bare_family_at_corr(fams, pol, corr_now)
        if better and better != BASE_FAMILY:
            print(f"  note: family {better} matches your ROW's corrugation —"
                  f" using it as BASE_FAMILY is better than {BASE_FAMILY}")
        base = rescale_family_to_corr(base, base_corr, corr_now)
        print(f"  ! base shape moved corr {base_corr:.0f} -> {corr_now:.0f} nm"
              " before anchoring (TM-measured exponents, EXPECTED-grade)")
    return anchor_on_row(base, ROW)

def check_anchor_pol(pol):
    """TM and TE anchors are never mixed — enforced, not just documented."""
    base_pol = "TM" if BASE_FAMILY.startswith(("tm_", "itai_tm")) else "TE"
    assert pol == base_pol, (f"polarization mismatch: ROW is {pol} but"
                             f" BASE_FAMILY {BASE_FAMILY} is {base_pol} —"
                             " TM and TE anchors are never mixed")

def compare(fam, p, corr, pol, bands, corr_moved=False):
    """Judge a LANDED run: predict at MEASURED["N"], print PREDICTED vs MEASURED
    and the verdict — T/Q_L against the empirical deviation band, width/lambda
    against the standing pass bands. MEASURED entries left None are skipped."""
    n = MEASURED["N"]
    obs = print_prediction(fam, p, n, corr=corr, pol=pol, bands=bands,
                           corr_moved=corr_moved)
    dql, dt, nsamp, key, few = deviation_band(bands or {}, max(0.0, n - p.get("n_hi", n)))
    if corr_moved:
        dql, dt = QL_BAND_KNOB, T_BAND_KNOB
    print(f"\nMEASURED vs PREDICTED at N={n:.0f}:")
    if not COMPARE_FROM_ROW and p.get("n_lo", np.inf) <= n <= p.get("n_hi", -np.inf):
        print(f"  note: N={n:.0f} is inside this family's fitted range"
              f" [{p['n_lo']:.0f}, {p['n_hi']:.0f}] — a near-zero deviation here is a"
              " consistency check, not a validation")
    for name, pred, meas, kind, band in (
            ("T     ", obs["T"], MEASURED.get("T"), "abs", dt),
            ("Q_L   ", obs["Q_L"], MEASURED.get("Q_L"), "rel", dql),
            ("width ", obs["width_um"], MEASURED.get("width_um"), "rel", W_BAND_REL),
            ("lambda", obs["lam_nm"], MEASURED.get("lam_nm"), "nm", LAM_BAND_NM)):
        if meas is None or not np.isfinite(pred):
            print(f"  {name}  n/a (not measured, or no prediction for this family)")
            continue
        dev = 100 * (pred - meas) / meas if kind == "rel" else pred - meas
        unit = {"rel": "%", "abs": "", "nm": " nm"}[kind]
        b = 100 * band if kind == "rel" else band
        verdict = "(no band)" if not np.isfinite(b) else \
            ("INSIDE" if abs(dev) <= b else "OUTSIDE")
        print(f"  {name}  PREDICTED {pred:>10.4f}   MEASURED {meas:>10.4f}"
              f"   dev {dev:+8.3f}{unit}   band +-{b:.3f}{unit}   {verdict}")
    print("  bands: T/Q_L " + ("from the corr-knob band (EXPECTED)" if corr_moved else
                               f"from {nsamp:.0f} hold-outs (bucket {key}"
                               + (", few samples -> max" if few else "") + ")")
          + f"; width +-{100*W_BAND_REL:.0f}%, lambda +-{LAM_BAND_NM:.1f} nm (standing)")

def main():
    fams = load_calib()
    bands = fams.get("errband", {})
    T_target = 10 ** (TARGET_DB / 10.0)
    from_row = MODE == "extend" or (MODE == "compare" and COMPARE_FROM_ROW)
    # current operating corrugation: from the ROW when anchoring, else from the
    # family name (tm_bare_c325 -> 325; families without a _c<NNN> tag give nan)
    tail = FAMILY.split("_c")[-1]
    corr_now = ROW["corr_nm"] if from_row else \
        (float(tail) if tail.isdigit() else np.nan)
    pol = ROW["pol"] if from_row else \
        ("TM" if FAMILY.startswith(("tm_", "itai_tm")) else "TE")
    if MODE == "observe":
        print_prediction(FAMILY, fams[FAMILY], N, corr=corr_now, pol=pol, bands=bands)
        return
    # did the corrugation MOVE off the family it was fitted on? (width knob, or an
    # anchor row at a different corr) — then the length-span bands do not apply
    base_tail = BASE_FAMILY.split("_c")[-1]
    base_corr = float(base_tail) if base_tail.isdigit() else np.nan
    corr_moved = TARGET_WIDTH_UM is not None or bool(
        from_row and np.isfinite(base_corr) and np.isfinite(corr_now)
        and abs(corr_now - base_corr) > CORR_MATCH_NM)
    if MODE == "compare":
        p, fam = fams[FAMILY], FAMILY
        if COMPARE_FROM_ROW:
            p = anchor_with_corr_fix(fams, pol, corr_now, base_corr)[0]
            fam = f"extend({BASE_FAMILY} shapes)"
        compare(fam, p, corr_now, pol, bands, corr_moved)
        return
    if TARGET_WIDTH_UM is not None and MODE != "extend" and (
            not np.isfinite(corr_now) or not FAMILY.startswith(("tm_bare_", "te_q3db_"))):
        print("REFUSED: the corrugation knob needs a bare family with corr in its"
              f" name (tm_bare_c325, te_q3db_c250...) — got {FAMILY}."
              " Use MODE='extend' with a measured ROW for any other device class.")
        return
    if MODE == "design":
        p, fam, corr_note = fams[FAMILY], FAMILY, ""
    else:  # extend
        p, qc_m, qi_m = anchor_with_corr_fix(fams, pol, corr_now, base_corr)
        fam = f"extend({BASE_FAMILY} shapes)"
        print(f"anchored on the measured row: N={ROW['N']} T={ROW['T']} Q_L={ROW['Q_L']}"
              f" -> Qc={qc_m:.0f} (DERIVED), Qi={qi_m:.0f} (DERIVED)")
        print("  ! single-row anchor: Qi SHAPE borrowed from base family — one more row"
              " ~30 periods away pins it")
        if ROW.get("mesher", "conformal").lower() != "conformal":
            print("  ! ROW is NOT conformal-mesh: predictions stay in the row's own"
                  " mesher frame; comparing them to conformal/spec numbers is"
                  " EXPECTED-grade only (known offsets: lam +5.3 nm, FWHM -8%,"
                  " T +0.008, Q_L -7%)")
        corr_note = ""
    if TARGET_WIDTH_UM is not None:
        p_new, corr_new, how = retune_corr(p, corr_now, pol, TARGET_WIDTH_UM, T_target)
        if p_new is None:
            print(f"REFUSED: {how}")
            return
        p = p_new
        corr_note = f"corr {corr_now:.0f}->{corr_new:.1f} nm (width knob -> {TARGET_WIDTH_UM} um) "
        print(f"\nCORRUGATION RETUNE: corr* = {corr_new:.1f} nm for width {TARGET_WIDTH_UM} um"
              f" — {how}. Qi rescaled by (corr ratio)^{CORR_QI_EXP_TM}"
              + (" [CORR_QI_EXP_TM, QC_H_PER_NM and FINF_CORR_EXP are ALL TM-measured — EXPECTED-grade for TE, treat as a band]" if pol == "TE" else " [MEASURED at N=150]")
              + "; levels now EXPECTED-grade — the confirmation run is the arbiter.")
        corr_now = corr_new
    n_star = design_N(p, T_target)
    if n_star is None:
        lo, hi = N_SEARCH
        print(f"REFUSED: no N reaches T target {T_target:.4f} ({TARGET_DB} dB) for this"
              f" device: at N={lo:.0f} T={observables(p, lo)['T']:.4f}, at N={hi:.0f}"
              f" T={observables(p, hi)['T']:.4f} — pick a different dB target or family")
        return
    obs = print_prediction(fam, p, round(n_star), corr_note, corr_now, pol, bands,
                           corr_moved)
    lam_s = f"{obs['lam_nm']:.2f} nm" if np.isfinite(obs["lam_nm"]) else "n/a"
    w_s = f"{obs['width_um']:.2f} um" if np.isfinite(obs["width_um"]) else "n/a"
    print(f"\nDESIGN: N* = {n_star:.1f} -> run N={round(n_star)} for T target {T_target:.4f} ({TARGET_DB} dB)")
    print(f"CONFIRMATION RUN SPEC (ONE run): {NUMERICS_NOTE}")
    print(f"  EXPECTED: T={obs['T']:.4f}, Q_L={obs['Q_L']:.0f}, lam~{lam_s},"
          f" width~{w_s}")
    print("  pass bands (design-grade): Q_L +-10%, T +-0.03, width +-5%")

if __name__ == "__main__":
    main()
