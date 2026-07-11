"""Device-1 transmission (|S21|^2 at the cavity resonance) mapped over
Δy (gap) × Δx (stagger) — the figure of merit the user cares about: maximize
the first pi-shift's input->output transmission. One heatmap + line cuts per pol."""
import os, glob
import numpy as np, scipy.io as sio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\side_by_side_coupling\results"
OUT = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\side_by_side_coupling"
def sc(m,k): return float(np.asarray(m[k]).ravel()[0]) if k in m else np.nan

recs=[]
for p in sorted(glob.glob(os.path.join(D,"result_*2pishift*.mat"))):
    b=os.path.basename(p); pol="TM" if "_TM_" in b else "TE"
    m=sio.loadmat(p); wl=np.asarray(m["wl_nm"]).ravel(); T=np.asarray(m["T"]).ravel()
    res=sc(m,"resonance_wavelength_nm"); i=int(np.argmin(np.abs(wl-res)))
    recs.append(dict(pol=pol,gap=round(sc(m,"device_gap_m")*1e9),stag=round(sc(m,"device_stagger_m")*1e9),
                     Tres=float(T[i])))

for pol in ("TE","TM"):
    rp=[r for r in recs if r["pol"]==pol]
    gaps=sorted(set(r["gap"] for r in rp)); stags=sorted(set(r["stag"] for r in rp))
    Z=np.full((len(gaps),len(stags)),np.nan)
    for r in rp: Z[gaps.index(r["gap"]),stags.index(r["stag"])]=r["Tres"]
    best=max(rp,key=lambda r:r["Tres"])

    fig,(a0,a1)=plt.subplots(1,2,figsize=(13,5))
    im=a0.imshow(Z,aspect="auto",origin="lower",cmap="inferno",
                 extent=[min(stags)/1000,max(stags)/1000,0,len(gaps)],vmin=0,vmax=1)
    a0.set_yticks(np.arange(len(gaps))+0.5); a0.set_yticklabels([f"{g/1000:.1f}" for g in gaps])
    a0.set_xlabel("Δx stagger (µm)"); a0.set_ylabel("Δy gap (µm)")
    a0.set_title(f"{pol}: device-1 transmission T at resonance")
    plt.colorbar(im,ax=a0,label="T (|S21|²)")
    # annotate each cell
    for gi,g in enumerate(gaps):
        for si,s_ in enumerate(stags):
            a0.text(s_/1000,gi+0.5,f"{Z[gi,si]:.2f}",ha="center",va="center",
                    color="white" if Z[gi,si]<0.6 else "black",fontsize=7)
    for g in gaps:
        zr=[r["Tres"] for r in sorted([x for x in rp if x["gap"]==g],key=lambda r:r["stag"])]
        a1.plot(np.array(stags)/1000,zr,"-o",label=f"gap {g/1000:.1f} µm")
    a1.set_xlabel("Δx stagger (µm)"); a1.set_ylabel("device-1 T at resonance")
    a1.set_title(f"{pol}: T vs stagger (higher = better)"); a1.set_ylim(0,1); a1.legend(); a1.grid(alpha=0.3)
    fig.suptitle(f"First-π-shift transmission — {pol}  (max T={best['Tres']:.3f} @ gap{best['gap']} Δx{best['stag']})")
    fig.tight_layout()
    out=os.path.join(OUT,f"transmission_{pol}.png"); fig.savefig(out,dpi=140); print("wrote",out)
    print(f"  {pol}: max device-1 T = {best['Tres']:.4f} at gap {best['gap']} nm, Δx {best['stag']} nm")
