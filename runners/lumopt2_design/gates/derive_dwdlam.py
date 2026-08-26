"""PROVENANCE of `CampaignSpec.wg_dwdlam` = 0.3655 um/nm (2026-08-25).

This is the constant the defect-#19 chain-rule fix multiplies by:
    gW_total = gW|_fixed_lambda + wg_dwdlam * dlam_pk/dp
so it must be re-derivable, not a magic number. It says: the spatial mode
width is very nearly a LINEAR function of the resonance wavelength.

SOURCES (local, pulled 2026-08-26 so no cluster holds unique state):
  results_from_athena/v2_gpu_gradient_pause/jsonl/lumopt2_v2_uniform_s5_evals.jsonl
  results_from_athena/v2_gpu_gradient_pause/jsonl/lumopt2_v2_seesaw_evals.jsonl
Both are the CANCELLED unprojected baselines (Athena 136753 / 136752 lineage).

"IN-BAND" means: rows with fom > 0 AND a finite fwhm_env_um. The excluded rows
are line-search probes that jumped out of the scan window -- they carry
fwhm_env_um of 25-36 um (vs ~18.4 in band) and would dominate any fit. That
filter IS the definition; it is applied explicitly below.

Run: python derive_dwdlam.py
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "results_from_athena",
    "v2_gpu_gradient_pause", "jsonl"))

STORED = 0.3655        # what CampaignSpec.wg_dwdlam currently holds


def in_band(rows):
    """THE RULE (stated explicitly -- it was hand-applied and undocumented on
    2026-08-25, which is how a 61%-different slope was nearly shipped):

      1. finite `lam_pk_nm` and `fwhm_env_um`;
      2. `fom > 0.5 * max(fom)` -- drops line-search probes that jumped out of
         band. These are not marginal: the excluded uniform_s5 row has
         fom 0.194 vs a run max of 0.691, sits at W 19.53, and is HIGH
         LEVERAGE (including it alone pulls the slope from 0.366 to ~0.59);
      3. DEDUPLICATE identical (lam_pk, fwhm_env_um) pairs. lumopt2 re-logs the
         accepted point at the start of each restart segment, so uniform_s5
         records W 18.5076 three times and 18.4088 twice. FDTD is deterministic
         (repeats are bit-identical), so these carry NO extra information and
         must not be weighted 3x in a least-squares fit.
    """
    fin = [d for d in rows
           if d.get("fom") is not None
           and d.get("fwhm_env_um") is not None and d.get("lam_pk_nm") is not None
           and np.isfinite(d["fwhm_env_um"]) and np.isfinite(d["lam_pk_nm"])]
    if not fin:
        return []
    cut = 0.5 * max(d["fom"] for d in fin)
    seen, out = set(), []
    for d in fin:
        if d["fom"] <= 0 or d["fom"] <= cut:
            continue
        key = (round(float(d["lam_pk_nm"]), 6), round(float(d["fwhm_env_um"]), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(d["lam_pk_nm"]), float(d["fwhm_env_um"])))
    return out


print(f"jsonl dir: {JSONL}")
print(f"exists   : {os.path.isdir(JSONL)}\n")

allpts, report = [], []
for tag in ("lumopt2_v2_uniform_s5", "lumopt2_v2_seesaw"):
    hits = glob.glob(os.path.join(JSONL, f"{tag}_evals.jsonl"))
    if not hits:
        print(f"  MISSING {tag}_evals.jsonl -- cannot re-derive")
        continue
    rows = [json.loads(l) for l in open(hits[0]) if l.strip()]
    pts = in_band(rows)
    if len(pts) < 3:
        print(f"  {tag}: only {len(pts)} in-band points, skipping")
        continue
    lam = np.array([p[0] for p in pts])
    W = np.array([p[1] for p in pts])
    m, b = np.polyfit(lam, W, 1)
    r = float(np.corrcoef(lam, W)[0, 1])
    report.append((tag, m, r, len(pts), np.ptp(lam), np.ptp(W)))
    allpts += pts
    print(f"  {tag:22s} n={len(pts):2d}  dW/dlam = {m:+.4f} um/nm  r = {r:.4f}"
          f"  (lam span {np.ptp(lam):.2f} nm, W span {np.ptp(W):.4f} um)")
    # how much of the width growth is explained by lambda alone?
    print(f"  {'':22s} lambda explains {100*abs(m*np.ptp(lam))/abs(np.ptp(W)):.0f}%"
          f" of this run's raw width growth")

if allpts:
    lam = np.array([p[0] for p in allpts])
    W = np.array([p[1] for p in allpts])
    m_all, _ = np.polyfit(lam, W, 1)
    print(f"\n  POOLED n={len(allpts)}  dW/dlam = {m_all:+.4f} um/nm")
    # the headline number comes from the uniform baseline alone (r = 0.984);
    # seesaw is noisier (it contains near-band-edge rows), so it is reported
    # but NOT averaged into the stored constant.
    uni = [x for x in report if x[0].startswith("lumopt2_v2_uniform")]
    if uni:
        m_u = uni[0][1]
        drift = abs(m_u - STORED) / STORED
        print(f"  stored wg_dwdlam = {STORED} (from the uniform baseline)")
        print(f"  {'OK  ' if drift < 0.02 else 'DRIFT'} re-derived {m_u:+.4f} "
              f"-> {drift*100:.2f}% from stored")

print("\n★If this number ever moves materially, update CampaignSpec.wg_dwdlam "
      "AND re-read the HANDOFF's 93-94% claim, which depends on it.")
