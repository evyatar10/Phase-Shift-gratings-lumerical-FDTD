"""
Build coupling-vs-(gap, stagger) 2D maps from the side_by_side_coupling grid.

Reads every result_*_2pishift_*.mat in a results dir, extracts per-config scalars
at the device-1 cavity resonance, and renders, per polarization:
  (a) coupling into device 2 (|S31|^2+|S41|^2 at resonance) heatmap over (Ygap, Xstag)
  (b) device-1 loss (1-R-T over 4 ports) heatmap
  (c) resonance Q heatmap
  (d) line cuts: coupling vs Xstag for each Ygap, with the lambda0/2nc = 0.544 um
      half-wavelength markers (Fabry-Perot-BIC phase periodicity)

Usage:
    python plot_side_by_side_maps.py <results_dir>
Default results_dir = results_from_athena/side_by_side_coupling/results
"""
import os, sys, glob, re
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes"
DEFAULT_DIR = os.path.join(ROOT, "results_from_athena", "side_by_side_coupling", "results")
OUT_DIR = os.path.join(ROOT, "results_from_athena", "side_by_side_coupling")
LAM0_NM = 1550.0
N_CLAD = 1.444
HALF_WL_UM = (LAM0_NM * 1e-3) / (2.0 * N_CLAD)   # lambda0 / (2 n_c) ~ 0.5367 um

def _s(m, k, default=np.nan):
    return float(np.asarray(m[k]).ravel()[0]) if k in m else default

def load_records(results_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(results_dir, "result_*2pishift*.mat"))):
        base = os.path.basename(p)
        pol = "TM" if "_TM_" in base else "TE"
        m = sio.loadmat(p)
        wl = np.asarray(m["wl_nm"]).ravel()
        T = np.asarray(m["T"]).ravel(); R = np.asarray(m["R"]).ravel()
        ctot = np.asarray(m["coupling_total"]).ravel() if "coupling_total" in m else np.full_like(T, np.nan)
        l4 = np.asarray(m["loss_4port"]).ravel() if "loss_4port" in m else (1 - R - T - ctot)
        res_nm = _s(m, "resonance_wavelength_nm")
        i = int(np.argmin(np.abs(wl - res_nm))) if np.isfinite(res_nm) else int(np.argmax(ctot))
        fwhm = abs(_s(m, "spectral_fwhm_nm"))
        Q = res_nm / fwhm if fwhm > 0 else np.nan
        recs.append(dict(
            pol=pol,
            gap_nm=round(_s(m, "device_gap_m")*1e9),
            stag_nm=round(_s(m, "device_stagger_m")*1e9),
            corr2_nm=round(_s(m, "corrugation_depth_2_m")*1e9),
            res_nm=res_nm,
            coupling_res=float(ctot[i]),
            coupling_peak=float(np.nanmax(ctot)),
            loss4_res=float(l4[i]),
            T_res=float(T[i]),
            Q=Q,
        ))
    return recs

def grid(recs, pol, field):
    rp = [r for r in recs if r["pol"] == pol]
    gaps = sorted(set(r["gap_nm"] for r in rp))
    stags = sorted(set(r["stag_nm"] for r in rp))
    Z = np.full((len(gaps), len(stags)), np.nan)
    for r in rp:
        Z[gaps.index(r["gap_nm"]), stags.index(r["stag_nm"])] = r[field]
    return np.array(gaps), np.array(stags), Z

def heat(ax, gaps, stags, Z, title, cbar_label):
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis",
                   extent=[stags.min()/1000, stags.max()/1000, 0, len(gaps)])
    ax.set_yticks(np.arange(len(gaps))+0.5); ax.set_yticklabels([f"{g/1000:.1f}" for g in gaps])
    ax.set_xlabel("X stagger Δx (µm)"); ax.set_ylabel("Y gap (µm)"); ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label)

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    recs = load_records(results_dir)
    if not recs:
        print(f"No result_*2pishift*.mat found in {results_dir}"); return
    print(f"Loaded {len(recs)} records from {results_dir}")
    os.makedirs(OUT_DIR, exist_ok=True)

    for pol in ("TE", "TM"):
        if not any(r["pol"] == pol for r in recs):
            continue
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        gaps, stags, Zc = grid(recs, pol, "coupling_res")
        _, _, Zl = grid(recs, pol, "loss4_res")
        _, _, Zq = grid(recs, pol, "Q")
        heat(axes[0], gaps, stags, Zc, f"{pol}: coupling→dev2 at resonance", "|S31|²+|S41|²")
        heat(axes[1], gaps, stags, Zl, f"{pol}: dev-1 radiated (1-R-T-C)", "loss_4port")
        heat(axes[2], gaps, stags, Zq, f"{pol}: resonance Q", "Q")
        fig.suptitle(f"Side-by-side coupling — {pol} (n_core 1.977, 500 nm corr, N=80)")
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"maps_{pol}.png"); fig.savefig(out, dpi=140); print("wrote", out)

        # line cuts: coupling vs stagger per gap, with half-wavelength markers
        fig2, ax = plt.subplots(figsize=(9, 5))
        for gi, g in enumerate(gaps):
            ax.plot(stags/1000, Zc[gi, :], "-o", label=f"gap {g/1000:.1f} µm")
        for m_ in range(1, int(stags.max()/1000/HALF_WL_UM)+1):
            ax.axvline(m_*HALF_WL_UM, ls=":", c="gray", lw=0.8)
        ax.set_xlabel("X stagger Δx (µm)"); ax.set_ylabel("coupling→dev2 at resonance")
        ax.set_title(f"{pol}: coupling vs stagger (dotted = m·λ₀/2n_c = {HALF_WL_UM:.3f} µm)")
        ax.legend(); ax.grid(alpha=0.3)
        out2 = os.path.join(OUT_DIR, f"coupling_vs_stagger_{pol}.png"); fig2.savefig(out2, dpi=140); print("wrote", out2)

    # text summary
    print("\npol  gap_nm stag_nm corr2  res_nm   coupl@res  peak   loss4  Q")
    for r in sorted(recs, key=lambda r:(r["pol"], r["gap_nm"], r["stag_nm"])):
        print(f"{r['pol']:3s} {r['gap_nm']:6.0f} {r['stag_nm']:6.0f} {r['corr2_nm']:5.0f}  "
              f"{r['res_nm']:7.1f}  {r['coupling_res']:.4f}  {r['coupling_peak']:.4f}  {r['loss4_res']:.4f}  {r['Q']:.0f}")

if __name__ == "__main__":
    main()
