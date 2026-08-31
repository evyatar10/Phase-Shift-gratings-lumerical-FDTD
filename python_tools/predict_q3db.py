"""Q3dB design tool: predict long-device observables / design (corr, N) with
ONE confirmation run — no tuning ladder.

Reads q3db_calibration.csv (written by calibrate_q3db.py from stored results
only; backtested there — 35/37 gated hold-out checks pass, see that file).
Model: L0 exact two-port algebra + per-family Qc exponential + saturating-Qi
power law + width truncation fit. All outputs are PREDICTED (label per
CLAUDE.md par.9) except where marked; the emitted confirmation-run spec is the
dispatch note.

Edit the knobs below and run (no CLI args — CLAUDE.md par.11):
  MODE="design"  -> find N* for TARGET_T in FAMILY, print expected observables
  MODE="observe" -> predict observables of FAMILY at N
Validity: fits are trusted inside [n_lo, n_hi] and to ~+20%% beyond n_hi
(backtested: B1b predicted N=220 from n_hi=200 at 2.3%% on Q_L). Predictions
further out get an EXTRAPOLATION warning. Known boundary (B2-E): T +-0.03 is
NOT reachable >~45 periods beyond a 2-row crossover fit — keep one calibration
row within ~30 periods of the target.
"""

import os

import numpy as np
from scipy.optimize import brentq

MODE = "design"          # "design" | "observe"
FAMILY = "bare_c325"     # family key in q3db_calibration.csv
TARGET_T = 0.5           # -3 dB
N = 165                  # observe-mode N
CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q3db_calibration.csv")

# measured 1/width-vs-corr knob line (tm_match_corr corr_bisect_log.csv, TM,
# fixed N; used only by suggest_corr)
CORR_WIDTH_PTS = [(300.0, 19.25844), (350.0, 17.24287), (400.0, 15.54882), (450.0, 13.85999)]
Q_ADEQ = 5e4             # above this the standard 20nm/4001 window is inadequate

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
    return brentq(f, 20, 2000, xtol=1e-3)

def suggest_corr(width_target_um):
    """corr for a target mode width via the measured 1/width-vs-corr line
    (linearizing coordinate per the lock-target knob table)."""
    c = np.array([x[0] for x in CORR_WIDTH_PTS])
    iw = 1.0 / np.array([x[1] for x in CORR_WIDTH_PTS])
    b, a = np.polyfit(c, iw, 1)
    return (1.0 / width_target_um - a) / b

def uncertainty_band(p, n, dqi=0.07, drate=0.01):
    """Sensitivity band from the backtested error scales: Qi +-7% (B1b/B6/B7),
    Qc rate +-1%/period-slope error over the extrapolation span."""
    outs = []
    for si in (-dqi, 0, dqi):
        for sr in (-drate, 0, drate):
            q = dict(p)
            q["qi_lnA"] = p["qi_lnA"] + np.log(1 + si)
            if np.isfinite(q.get("qi_sat", np.nan)):
                q["qi_sat"] = p["qi_sat"] * (1 + si)
            q["qc_rate"] = p["qc_rate"] * (1 + sr)
            outs.append(observables(q, n))
    ql = [o["Q_L"] for o in outs]
    T = [o["T"] for o in outs]
    return (min(ql), max(ql)), (min(T), max(T))

def confirmation_spec(fam, p, n, obs):
    lines = [f"CONFIRMATION RUN SPEC — family {fam}, N={n:.0f} (ONE run; cites stored",
             "calibration rows, reuses everything else per CLAUDE.md par.6)",
             f"  numerics: {NUMERICS_NOTE}",
             f"  EXPECTED: T={obs['T']:.4f}, Q_L={obs['Q_L']:.0f}, lam~{obs['lam_nm']:.2f} nm,"
             f" width~{obs['width_um']:.2f} um",
             "  pass bands (design-grade): Q_L +-10%, T +-0.03, width +-5%"]
    if obs["Q_L"] > Q_ADEQ:
        lines += ["  HIGH-Q ADEQUACY (predicted Q_L > 5e4): narrow window to 3 nm/0.75 pm",
                  "  AND set os.environ['TM_SIM_TIME_PS']='4000' INSIDE the runner",
                  "  (--option3 deploy does NOT forward env vars)."]
    return "\n".join(lines)

NUMERICS_NOTE = "y8.0/z8.8 box, 20nm window/4001pts (or 3nm if high-Q), dx50 conformal, ASL 1e-7"

def main():
    fams = load_calib()
    p = fams[FAMILY]
    if MODE == "observe":
        n = N
    else:
        n = round(design_N(p, TARGET_T))
    obs = observables(p, n)
    (ql_lo, ql_hi), (t_lo, t_hi) = uncertainty_band(p, n)
    tag = "PREDICTED"
    if n > p["n_hi"] * 1.2 or n < p["n_lo"] * 0.8:
        tag += " [EXTRAPOLATION beyond backtested range — treat as band only]"
    print(f"{FAMILY} {MODE}: N={n:.0f}  ({tag})")
    print(f"  T      = {obs['T']:.4f}   [{t_lo:.4f} .. {t_hi:.4f}]")
    print(f"  Q_L    = {obs['Q_L']:.0f}   [{ql_lo:.0f} .. {ql_hi:.0f}]")
    print(f"  Q_c    = {obs['Q_c']:.0f}   Q_i = {obs['Q_i']:.0f}")
    print(f"  lambda = {obs['lam_nm']:.2f} nm (family line)  spectral fwhm = {obs['spec_fwhm_pm']:.2f} pm")
    print(f"  width  = {obs['width_um']:.2f} um (truncation fit)")
    print(f"  fit range N=[{p['n_lo']:.0f},{p['n_hi']:.0f}]")
    if MODE == "design":
        print()
        print(confirmation_spec(FAMILY, p, n, obs))

if __name__ == "__main__":
    main()
