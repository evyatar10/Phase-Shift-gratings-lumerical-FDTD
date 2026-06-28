"""Verify (a) there is a real Bragg stopband, (b) the SPECIALIZED resonance finder
(find_bragg_resonance, sharpness x dip-depth) locks onto the genuine cavity peak
inside it — NOT max(T). Saves spectra with markers for visual confirmation."""
import os, sys
import numpy as np, scipy.io as sio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes")
from sim_helpers import find_bragg_resonance
from scipy.signal import find_peaks, peak_prominences

B = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\side_by_side_coupling\results"
OUT = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\side_by_side_coupling"
cases = [
    ("TE isolated (gap2000 dx8000)", "result_N80_avg_2pishift_Ygap2000nm_Xstag8000nm_corr2500nm.mat"),
    ("TE coupled (gap1000 dx0)",     "result_N80_avg_2pishift_Ygap1000nm_Xstag0nm_corr2500nm.mat"),
    ("TM isolated (gap2000 dx8000)", "result_N80_TM_avg_2pishift_Ygap2000nm_Xstag8000nm_corr2500nm.mat"),
    ("TM coupled (gap1000 dx0)",     "result_N80_TM_avg_2pishift_Ygap1000nm_Xstag0nm_corr2500nm.mat"),
]
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
for ax, (label, fn) in zip(axes.ravel(), cases):
    m = sio.loadmat(os.path.join(B, fn))
    wl = np.asarray(m["wl_nm"]).ravel(); T = np.asarray(m["T"]).ravel(); R = np.asarray(m["R"]).ravel()
    res_stored = float(np.asarray(m["resonance_wavelength_nm"]).ravel()[0])
    Tres_stored = float(np.asarray(m["resonance_transmission"]).ravel()[0])
    # re-run the specialized finder on T to confirm reproducibility
    idx = find_bragg_resonance(wl, T); res_calc = wl[idx]; Tres_calc = T[idx]
    # stopband = contiguous low-T region containing the resonance
    below = T < 0.5; i0 = int(np.argmin(np.abs(wl - res_stored)))
    lo = i0
    while lo > 0 and below[lo-1]: lo -= 1
    hi = i0
    while hi < len(T)-1 and below[hi+1]: hi += 1
    band_w = wl[hi] - wl[lo] if below[i0] else 0.0
    # naive max(T) for contrast
    imax = int(np.argmax(T))
    # peak count (prominence>0.05)
    pk,_ = find_peaks(T); prom = peak_prominences(T, pk)[0] if len(pk) else np.array([])
    nbig = int(np.sum(prom > 0.05))
    print(f"\n{label}")
    print(f"  stored resonance: {res_stored:.2f} nm  T={Tres_stored:.4f}   (finder re-run: {res_calc:.2f} nm T={Tres_calc:.4f})")
    print(f"  stopband (T<0.5) around resonance: {wl[lo]:.1f}-{wl[hi]:.1f} nm  width={band_w:.1f} nm  floor Tmin={T[lo:hi+1].min():.3f}  Rmax={R[lo:hi+1].max():.3f}")
    print(f"  resonance inside stopband: {below[i0]}   |  naive max(T)={T[imax]:.3f} @ {wl[imax]:.1f}nm (R={R[imax]:.3f}, {'OUT-of-band' if R[imax]<0.2 else 'in-band'})")
    print(f"  # prominent T peaks (>0.05): {nbig}  ({'possible supermode split' if nbig>1 else 'single peak'})")
    ax.plot(wl, T, label="T", color="C0"); ax.plot(wl, R, label="R", color="C1", alpha=0.7)
    ax.axvspan(wl[lo], wl[hi], color="gray", alpha=0.15, label="stopband (T<0.5)")
    ax.axvline(res_stored, color="r", ls="--", lw=1, label=f"resonance {res_stored:.1f}nm")
    ax.plot(res_stored, Tres_stored, "r*", ms=14)
    ax.axvline(wl[imax], color="g", ls=":", lw=1, label=f"naive max(T) {wl[imax]:.1f}nm")
    ax.set_title(f"{label}\nresonance-peak T={Tres_stored:.3f}"); ax.set_xlabel("λ (nm)"); ax.set_ylim(-0.02,1.02)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.suptitle("Band + resonance check: red★ = specialized finder (cavity peak in stopband); green: = naive max(T) [out-of-band]")
fig.tight_layout()
out = os.path.join(OUT, "band_resonance_check.png"); fig.savefig(out, dpi=130); print("\nwrote", out)
