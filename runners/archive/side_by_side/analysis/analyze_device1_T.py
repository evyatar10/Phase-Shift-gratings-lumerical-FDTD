"""Re-analyze the side-by-side grid with the RECYCLING figure of merit:
device-1 transmission T at the device-1 cavity resonance (we want it MAXIMIZED).
The most-decoupled config (largest gap, largest stagger) is the pseudo 'device-1
alone' baseline; does any coupled config beat it?"""
import os, glob
import numpy as np, scipy.io as sio

d = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\side_by_side_coupling\results"
def s(m,k): return float(np.asarray(m[k]).ravel()[0]) if k in m else np.nan

recs=[]
for p in sorted(glob.glob(os.path.join(d,"result_*2pishift*.mat"))):
    b=os.path.basename(p); pol="TM" if "_TM_" in b else "TE"
    m=sio.loadmat(p); wl=np.asarray(m["wl_nm"]).ravel(); T=np.asarray(m["T"]).ravel(); R=np.asarray(m["R"]).ravel()
    ctot=np.asarray(m["coupling_total"]).ravel() if "coupling_total" in m else np.full_like(T,np.nan)
    res=s(m,"resonance_wavelength_nm"); i=int(np.argmin(np.abs(wl-res)))
    recs.append(dict(pol=pol,gap=round(s(m,"device_gap_m")*1e9),stag=round(s(m,"device_stagger_m")*1e9),
                     Tres=float(T[i]),Tmax=float(T.max()),Rres=float(R[i]),cpl=float(ctot[i]),res=res))

for pol in ("TE","TM"):
    rp=[r for r in recs if r["pol"]==pol]
    # pseudo-baseline = most decoupled (max gap, max stagger, min coupling)
    base=min(rp,key=lambda r:r["cpl"])
    print(f"\n===== {pol} : device-1 T at resonance (want HIGH) =====")
    print(f" pseudo-baseline (least coupled): gap{base['gap']} stag{base['stag']}  T={base['Tres']:.4f}  coupling={base['cpl']:.4f}")
    best=max(rp,key=lambda r:r["Tres"])
    print(f" BEST device-1 T: gap{best['gap']} stag{best['stag']}  T={best['Tres']:.4f}  (vs baseline {base['Tres']:.4f}; "
          f"{'+' if best['Tres']>base['Tres'] else ''}{(best['Tres']-base['Tres'])*100:.2f} pts)")
    print(" gap  stag   T_res   R_res  coupling  Tmax")
    for r in sorted(rp,key=lambda r:(r["gap"],r["stag"])):
        flag=" <-- best" if r is best else ""
        print(f" {r['gap']:5d} {r['stag']:5d}  {r['Tres']:.4f}  {r['Rres']:.4f}  {r['cpl']:.4f}  {r['Tmax']:.4f}{flag}")
