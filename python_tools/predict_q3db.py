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
Targets are generalized: TARGET_DB is any dB point (-3 dB default), and
TARGET_WIDTH_UM any mode width (e.g. 14) — width is retuned by CORRUGATION
via the per-pol knob line, which rescales Qc rate (kappa prop. corr, measured
0.1-1.3%), F_inf (measured cross-family exponent -1.11) and Qi (radiative
corr^-2.9, MEASURED for TM at N=150; for TE this exponent is UNMEASURED and
the output is a band, labeled).

Every number printed is PREDICTED unless labeled; the confirmation-run spec
is the dispatch note. Validity rules are printed with each output:
  - calibration rows need 2*kappa*L >= ~3.2 (below that the device is too
    short to carry the family shape — prediction refused);
  - T +-0.03 holds to ~30 periods beyond the anchored range; beyond ~45 it is
    a band, not a point (measured boundary, backtest B2-E);
  - a single-row anchor pins LEVELS but trusts the base family's Qi SHAPE —
    quote the band, and one extra row ~30 periods away removes most of it.
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
# extend-mode: the NEW measured device (fill from the result .mat / your note)
ROW = dict(pol="TM", corr_nm=325.0, pitch_nm=516.83, N=100, T=0.9104,
           Q_L=1760.0, lam_nm=1559.006, width_um=19.245)
BASE_FAMILY = "tm_bare_c325"   # shape priors for extend mode (match pol!)
# ----------------------------------------------------------------------------

CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q3db_calibration.csv")
Q_ADEQ = 5e4
CORR_QI_EXP_TM = -2.9      # MEASURED (N=150 trench corr ladder, backtest B12)
FINF_CORR_EXP = -1.11      # MEASURED cross-family (F_inf 20.06@c325 vs 24.05@c276)
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
    f = lambda n: (qi_of_N(p, n) / (qi_of_N(p, n) + qc_of_N(p, n))) ** 2 - T_target
    return brentq(f, 20, 3000, xtol=1e-3)

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

def retune_corr(p, corr_now, pol, width_target):
    """Corrugation knob: corr* for a width target from the per-pol measured
    1/w-vs-corr line, then rescale the family parameters. Returns (p', corr*)."""
    kn = load_calib()["knob_tm" if pol == "TM" else "knob_te"]
    corr_new = (1.0 / width_target - kn["iw_a"]) / kn["iw_b"]
    r = corr_new / corr_now
    q = dict(p)
    q["qc_rate"] = p["qc_rate"] * r                      # kappa prop. corr (0.1-1.3%)
    if "w_Finf" in p:
        scale = r ** FINF_CORR_EXP
        q["w_Finf"], q["w_B"] = p["w_Finf"] * scale, p["w_B"] * scale
        q["w_c"] = p["w_c"] * r                          # 2*kappa*Lambda prop. corr
    fqi = r ** CORR_QI_EXP_TM
    q["qi_lnA"] = p["qi_lnA"] + np.log(fqi)
    if np.isfinite(q.get("qi_sat", np.nan)):
        q["qi_sat"] *= fqi
    return q, corr_new

def validity_notes(p, n):
    notes = []
    if "w_c" in p:                                       # w_c == 2*kappa*Lambda
        n_min = 3.2 / p["w_c"]
        if n < n_min:
            notes.append(f"REFUSE: N={n:.0f} < N_min~{n_min:.0f} (2kL<3.2 — device too short to carry the family shape)")
        notes.append(f"shortest usable calibration/anchor device: N_min ~ {3.2/p['w_c']:.0f} (2kL>=3.2)")
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

def print_prediction(fam, p, n, corr_note=""):
    obs = observables(p, n)
    (ql_lo, ql_hi), (t_lo, t_hi) = uncertainty_band(p, n)
    print(f"\n{fam} at N={n:.0f} {corr_note}(PREDICTED)")
    print(f"  T      = {obs['T']:.4f}  [{t_lo:.4f} .. {t_hi:.4f}]  ({10*np.log10(obs['T']):.2f} dB)")
    print(f"  Q_L    = {obs['Q_L']:.0f}  [{ql_lo:.0f} .. {ql_hi:.0f}]   Q_c={obs['Q_c']:.0f}  Q_i={obs['Q_i']:.0f}")
    print(f"  lambda = {obs['lam_nm']:.2f} nm   spectral fwhm = {obs['spec_fwhm_pm']:.2f} pm")
    print(f"  width  = {obs['width_um']:.2f} um")
    for note in validity_notes(p, n):
        print(f"  ! {note}")
    if obs["Q_L"] > Q_ADEQ:
        print("  ! confirmation run needs the HIGH-Q window (3nm/0.75pm) and"
              " os.environ['TM_SIM_TIME_PS']='4000' inside the runner")
    return obs

def main():
    fams = load_calib()
    T_target = 10 ** (TARGET_DB / 10.0)
    if MODE == "observe":
        print_prediction(FAMILY, fams[FAMILY], N)
        return
    if MODE == "design":
        p, fam, corr_note = fams[FAMILY], FAMILY, ""
        pol = "TM" if fam.startswith(("tm_", "itai_tm")) else "TE"
    else:  # extend
        base = fams[BASE_FAMILY]
        p, qc_m, qi_m = anchor_on_row(base, ROW)
        pol = ROW["pol"]
        fam = f"extend({BASE_FAMILY} shapes)"
        print(f"anchored on the measured row: N={ROW['N']} T={ROW['T']} Q_L={ROW['Q_L']}"
              f" -> Qc={qc_m:.0f} (DERIVED), Qi={qi_m:.0f} (DERIVED)")
        print("  ! single-row anchor: Qi SHAPE borrowed from base family — one more row"
              " ~30 periods away pins it")
        corr_note = ""
    corr_now = ROW["corr_nm"] if MODE == "extend" else float(fam.split("_c")[-1]) \
        if "_c" in fam and fam.split("_c")[-1].isdigit() else np.nan
    if TARGET_WIDTH_UM is not None:
        p, corr_new = retune_corr(p, corr_now, pol, TARGET_WIDTH_UM)
        corr_note = f"corr {corr_now:.0f}->{corr_new:.1f} nm (width knob -> {TARGET_WIDTH_UM} um) "
        print(f"\nCORRUGATION RETUNE: corr* = {corr_new:.1f} nm for width {TARGET_WIDTH_UM} um"
              f" (per-{pol} knob line). Qi rescaled by (corr ratio)^{CORR_QI_EXP_TM}"
              + (" [TM-MEASURED exponent applied to TE — UNMEASURED, treat as band]" if pol == "TE" else " [MEASURED at N=150]")
              + "; levels now EXPECTED-grade — the confirmation run is the arbiter.")
    n_star = design_N(p, T_target)
    obs = print_prediction(fam, p, round(n_star), corr_note)
    print(f"\nDESIGN: N* = {n_star:.1f} -> run N={round(n_star)} for T target {T_target:.4f} ({TARGET_DB} dB)")
    print(f"CONFIRMATION RUN SPEC (ONE run): {NUMERICS_NOTE}")
    print(f"  EXPECTED: T={obs['T']:.4f}, Q_L={obs['Q_L']:.0f}, lam~{obs['lam_nm']:.2f} nm,"
          f" width~{obs['width_um']:.2f} um")
    print("  pass bands (design-grade): Q_L +-10%, T +-0.03, width +-5%")

if __name__ == "__main__":
    main()
