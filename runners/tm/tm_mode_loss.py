"""
Local FDE (MODE) calculation of the TM-mode PROPAGATION LOSS (dB/cm) of the
pi-shift Bragg grating's Si3N4-in-oxide waveguide, driven by MATERIAL ABSORPTION.

WHY THIS EXISTS
  The FDTD model holds the material index constant and PURELY REAL
  (n_core=1.9963, n_clad=1.444), so the TM mode is perfectly bound and lossless
  (FDE would report ~0 dB/cm). To get a real propagation loss we give the core
  and the oxide a BULK material absorption (in dB/cm) — a complex index n+ik —
  and let FDE return the MODAL loss, i.e. the bulk loss weighted by how much of
  the TM field actually sits in each region (the confinement factor). Because the
  TM field is less confined to the core than TE, the TM modal loss differs from
  TE for the same bulk material loss; TE is solved too, for comparison.

STACK (per project: Si3N4 core embedded in oxide, NO Si handle)
  - Si3N4 core: 350 nm tall x WIDTH wide, n=1.9963 + i*k_core.
  - SiO2 cladding: 3.8 um below the core, 4.0 um above, n=1.444 + i*k_clad.
    With no high-index handle the mode is fully contained in oxide, so the exact
    3.8/4.0 split is immaterial to the absorption result (both >> the mode tail).
  - 2D Z-normal FDE: x = thickness (vertical), y = width, z = propagation. Under
    this rotation the device TM mode (E vertical) is Ex-dominant; modes are
    classified by transverse E-power, NOT Lumerical's (inverted-here) TE fraction.

METHOD
  bulk loss [dB/cm] --(at lam0)--> extinction coefficient k
      alpha[1/m] = loss_db_cm * 100 * ln(10)/10        (dB/cm -> power 1/m)
      k          = alpha * lam0 / (4*pi)
  Assign n+ik to core and clad via a Lumerical "(n,k) Material", solve the
  fundamental TM (and TE) mode, and read the modal loss from the complex neff:
      loss_modal[dB/cm] = 4*pi*Im(neff)/lam * (10/ln(10)) / 100
  cross-checked against FDE's built-in per-mode "loss" (dB/m) attribute.

Runs LOCALLY — MODE/FDE is CPU and fast, and is a SEPARATE Lumerical product
from FDTD, so this needs a MODE license on the machine you run it on. Reuses the
lumapi import and the TE/TM mode classifier from runners.tm.calibrate_neff.

Usage:
  python -m runners.tm.tm_mode_loss \
      --core-loss-db-cm 1.0 --clad-loss-db-cm 0.1 \
      [--width-nm 1000] [--lam0-nm 1571] \
      [--wl-start 1.50e-6 --wl-stop 1.62e-6 --n-points 100] \
      [--no-sweep] [--plot] [--out <dir>]

Writes <out>/tm_mode_loss.mat with the headline dB/cm numbers, neff, the implied
k values, and the loss-vs-wavelength sweep for both polarizations.
"""

import argparse
import math
import os
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config  # noqa: E402
from runners.tm.calibrate_neff import (  # noqa: E402
    _import_lumapi,
    N_CORE,
    N_CLAD,
    CORE_HEIGHT_M,
)

IS_HELPER = True  # local tool, not a cluster-dispatchable runner

LN10 = math.log(10.0)
C0 = 299792458.0

# ── Stack geometry (Si3N4-in-oxide, no Si handle) ──
OXIDE_BELOW_M = 3.8e-6     # SiO2 below the core
OXIDE_ABOVE_M = 4.0e-6     # SiO2 above the core
DEFAULT_WIDTH_M = 1000e-9  # access/port waveguide width (grating teeth are 650-950 nm)
DEFAULT_LAM0_M = 1.571e-6  # device TM operating wavelength

MAT_CORE = "SiN_lossy"
MAT_CLAD = "SiO2_lossy"


def bulk_dbcm_to_k(loss_db_cm, lam_m):
    """Bulk material loss [dB/cm] -> extinction coefficient k at wavelength lam_m."""
    alpha = loss_db_cm * 100.0 * LN10 / 10.0      # power attenuation, 1/m
    return alpha * lam_m / (4.0 * math.pi)


def imag_neff_to_dbcm(neff_imag, lam_m):
    """Modal loss [dB/cm] from the imaginary part of the effective index."""
    alpha = 4.0 * math.pi * float(neff_imag) / lam_m   # modal power attenuation, 1/m
    return alpha * (10.0 / LN10) / 100.0               # 1/m -> dB/m -> dB/cm


def _setup_lossy_materials(mode, k_core, k_clad):
    """Create constant complex-index materials n+ik for core and cladding."""
    script = f'''
    if (materialexists("{MAT_CORE}")) {{ deletematerial("{MAT_CORE}"); }}
    if (materialexists("{MAT_CLAD}")) {{ deletematerial("{MAT_CLAD}"); }}
    mc = addmaterial("(n,k) Material"); setmaterial(mc, "name", "{MAT_CORE}");
    setmaterial("{MAT_CORE}", "Refractive Index", {N_CORE});
    setmaterial("{MAT_CORE}", "Imaginary Refractive Index", {k_core});
    md = addmaterial("(n,k) Material"); setmaterial(md, "name", "{MAT_CLAD}");
    setmaterial("{MAT_CLAD}", "Refractive Index", {N_CLAD});
    setmaterial("{MAT_CLAD}", "Imaginary Refractive Index", {k_clad});
    '''
    mode.eval(script)


def _build_section(mode, width_m, wl_mid):
    """Build the Si3N4-in-oxide cross-section with lossy materials (2D Z normal).

    x = thickness (vertical), y = width, z = propagation. Core centered at x=0;
    oxide spans -OXIDE_BELOW_M..+OXIDE_ABOVE_M. The FDE x-span stays inside the
    oxide so the (default metal) walls see only oxide, never the background.
    """
    mode.switchtolayout()
    mode.deleteall()

    # FDE region — kept inside the oxide on both sides (±3 um < 3.8/4.0 um).
    mode.addfde()
    mode.set("solver type", "2D Z normal")
    mode.set("x", 0.0); mode.set("x span", 6e-6)     # thickness direction
    mode.set("y", 0.0); mode.set("y span", 6e-6)     # width direction
    mode.set("z", 0.0)
    mode.set("wavelength", wl_mid)

    # Oxide cladding (lossy) — honors the 3.8 um below / 4.0 um above stack.
    ox_span = OXIDE_BELOW_M + OXIDE_ABOVE_M
    ox_center = 0.5 * (OXIDE_ABOVE_M - OXIDE_BELOW_M)
    mode.addrect()
    mode.set("name", "clad")
    mode.set("material", MAT_CLAD)
    mode.set("x", ox_center); mode.set("x span", ox_span)
    mode.set("y", 0.0); mode.set("y span", 10e-6)
    mode.set("z", 0.0); mode.set("z span", 10e-6)

    # Si3N4 core (lossy).
    mode.addrect()
    mode.set("name", "core")
    mode.set("material", MAT_CORE)
    mode.set("x", 0.0); mode.set("x span", CORE_HEIGHT_M)
    mode.set("y", 0.0); mode.set("y span", width_m)
    mode.set("z", 0.0); mode.set("z span", 10e-6)

    # Mesh override around the core (10 nm divides 350 nm cleanly).
    mode.addmesh()
    mode.set("name", "core_mesh")
    mode.set("x", 0.0); mode.set("x span", CORE_HEIGHT_M + 0.5e-6)
    mode.set("y", 0.0); mode.set("y span", width_m + 0.5e-6)
    mode.set("z", 0.0); mode.set("z span", 10e-6)
    mode.set("dx", 10e-9); mode.set("dy", 10e-9); mode.set("dz", 10e-9)


def _ex_fraction(mode, name):
    """|Ex|^2 / (|Ex|^2 + |Ey|^2) for a found mode — the vertical-E (device TM) share."""
    px = float(np.sum(np.abs(np.asarray(mode.getdata(name, "Ex"))) ** 2))
    py = float(np.sum(np.abs(np.asarray(mode.getdata(name, "Ey"))) ** 2))
    return px / (px + py) if (px + py) > 0 else 0.0


def _pick_mode(mode, polarization, n_trial=8):
    """Fundamental DEVICE-convention mode of the requested polarization.

    The cross-section is rotated so FDE-x is the device VERTICAL (thickness) axis and
    FDE-y is the device WIDTH axis. Hence the device TM mode (E vertical) is
    Ex-dominant and the device TE mode (E in-plane) is Ey-dominant. We classify by the
    transverse E-power directly and do NOT use Lumerical's "TE polarization fraction"
    (which equals the |Ex|^2 share and is therefore INVERTED under this rotation —
    the cause of a label swap in calibrate_neff). Returns the matching mode with the
    largest n_eff (the fundamental of that polarization).
    """
    try:
        mode.setanalysis("number of trial modes", n_trial)
    except Exception:
        try:
            mode.set("number of trial modes", n_trial)
        except Exception:
            pass
    mode.findmodes()
    want_tm = (polarization.upper() == "TM")
    best_idx, best_neff = None, -1.0
    for i in range(1, n_trial + 1):
        name = f"FDE::data::mode{i}"
        try:
            neff = float(np.real(np.asarray(mode.getdata(name, "neff")).ravel()[0]))
            is_tm = _ex_fraction(mode, name) > 0.5    # Ex-dominant => device TM
        except Exception:
            break  # no more modes
        if is_tm == want_tm and neff > best_neff:
            best_idx, best_neff = i, neff
    if best_idx is None:
        raise RuntimeError(f"no {polarization} mode found among {n_trial} trial modes")
    return best_idx


def _read_mode_at(mode, polarization, lam_m):
    """Find the fundamental mode of `polarization` at lam_m; return its metrics."""
    mode.setanalysis("wavelength", lam_m)
    k = _pick_mode(mode, polarization)
    mode.selectmode(k)
    name = f"FDE::data::mode{k}"
    neff = np.asarray(mode.getdata(name, "neff")).ravel()[0]
    ex_frac = _ex_fraction(mode, name)   # ~1 for device TM (vertical E), ~0 for device TE
    loss_attr_dbcm = None
    try:  # FDE's built-in modal loss is in dB/m; use it only as a cross-check.
        loss_attr_dbcm = float(np.asarray(mode.getdata(name, "loss")).ravel()[0]) / 100.0
    except Exception:
        pass
    return k, complex(neff), ex_frac, loss_attr_dbcm


def _sweep(mode, k, wl_start, wl_stop, n_points):
    """Frequency-sweep the selected mode -> (wl_nm ascending, loss_db_cm array)."""
    mode.selectmode(k)
    mode.setanalysis("track selected mode", 1)
    mode.setanalysis("detailed dispersion calculation", 0)
    mode.setanalysis("stop wavelength", wl_stop)
    mode.setanalysis("number of points", int(n_points))
    mode.frequencysweep()
    neff = np.squeeze(mode.getdata("frequencysweep", "neff"))
    f = np.squeeze(mode.getdata("frequencysweep", "f"))
    wl_m = C0 / f
    loss = np.array([imag_neff_to_dbcm(np.imag(n), w) for n, w in zip(neff, wl_m)])
    wl_nm = wl_m * 1e9
    if wl_nm[0] > wl_nm[-1]:
        wl_nm = np.flip(wl_nm); loss = np.flip(loss)
    return wl_nm, loss


def solve_polarization(mode, polarization, width_m, lam0, wl_start, wl_stop,
                       n_points, do_sweep):
    """Headline modal loss at lam0 (+ optional loss-vs-wl sweep) for one polarization."""
    _build_section(mode, width_m, 0.5 * (wl_start + wl_stop))
    mode.run()

    k, neff, ex_frac, loss_attr = _read_mode_at(mode, polarization, lam0)
    loss_dbcm = imag_neff_to_dbcm(np.imag(neff), lam0)

    wl_nm, loss_sweep = (None, None)
    if do_sweep:
        # Re-anchor the tracked sweep at the band edge so it spans the full band.
        mode.setanalysis("wavelength", wl_start)
        ks = _pick_mode(mode, polarization)
        wl_nm, loss_sweep = _sweep(mode, ks, wl_start, wl_stop, n_points)

    return {
        "neff": neff, "ex_frac": ex_frac,
        "loss_db_cm": loss_dbcm, "loss_attr_db_cm": loss_attr,
        "wl_nm": wl_nm, "loss_db_cm_sweep": loss_sweep,
    }


def run(core_loss_db_cm=1.0, clad_loss_db_cm=0.0, width_m=DEFAULT_WIDTH_M,
        lam0=DEFAULT_LAM0_M, wl_start=1.50e-6, wl_stop=1.62e-6, n_points=100,
        do_sweep=True, out_dir=None, plot=False):
    k_core = bulk_dbcm_to_k(core_loss_db_cm, lam0)
    k_clad = bulk_dbcm_to_k(clad_loss_db_cm, lam0)

    print("FDE TM/TE modal absorption loss")
    print(f"  width={width_m*1e9:.0f} nm, lam0={lam0*1e9:.1f} nm")
    print(f"  bulk: core={core_loss_db_cm:.4g} dB/cm (k={k_core:.3e}), "
          f"clad={clad_loss_db_cm:.4g} dB/cm (k={k_clad:.3e})")

    lumapi = _import_lumapi()
    mode = lumapi.MODE(hide=True)
    try:
        _setup_lossy_materials(mode, k_core, k_clad)
        tm = solve_polarization(mode, "TM", width_m, lam0, wl_start, wl_stop, n_points, do_sweep)
        te = solve_polarization(mode, "TE", width_m, lam0, wl_start, wl_stop, n_points, do_sweep)
    finally:
        mode.close()

    out = {
        "lam0_nm": lam0 * 1e9, "width_nm": width_m * 1e9,
        "core_loss_db_cm": core_loss_db_cm, "clad_loss_db_cm": clad_loss_db_cm,
        "k_core": k_core, "k_clad": k_clad,
        "n_core": N_CORE, "n_clad": N_CLAD, "core_height_m": CORE_HEIGHT_M,
        "oxide_below_m": OXIDE_BELOW_M, "oxide_above_m": OXIDE_ABOVE_M,
        "tm_modal_loss_db_cm": tm["loss_db_cm"], "te_modal_loss_db_cm": te["loss_db_cm"],
        "tm_neff_real": np.real(tm["neff"]), "tm_neff_imag": np.imag(tm["neff"]),
        "te_neff_real": np.real(te["neff"]), "te_neff_imag": np.imag(te["neff"]),
        "tm_ex_fraction": tm["ex_frac"], "te_ex_fraction": te["ex_frac"],
    }
    if tm["loss_attr_db_cm"] is not None:
        out["tm_modal_loss_db_cm_fde_attr"] = tm["loss_attr_db_cm"]
        out["te_modal_loss_db_cm_fde_attr"] = te["loss_attr_db_cm"]
    if do_sweep and tm["wl_nm"] is not None:
        out["wl_nm"] = tm["wl_nm"]
        out["tm_loss_db_cm_sweep"] = tm["loss_db_cm_sweep"]
        out["te_loss_db_cm_sweep"] = te["loss_db_cm_sweep"]

    out_dir = out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results_from_athena", "run_tm_vs_te", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tm_mode_loss.mat")
    sio.savemat(out_path, out)

    print("\n== FDE modal absorption loss @ lam0 ==")
    print(f"  TM:  neff={np.real(tm['neff']):.4f}, Ex-frac={tm['ex_frac']:.3f} (vertical E), "
          f"modal loss = {tm['loss_db_cm']:.4f} dB/cm"
          + (f"  (FDE attr {tm['loss_attr_db_cm']:.4f})" if tm["loss_attr_db_cm"] is not None else ""))
    print(f"  TE:  neff={np.real(te['neff']):.4f}, Ex-frac={te['ex_frac']:.3f} (in-plane E), "
          f"modal loss = {te['loss_db_cm']:.4f} dB/cm"
          + (f"  (FDE attr {te['loss_attr_db_cm']:.4f})" if te["loss_attr_db_cm"] is not None else ""))
    print(f"  saved: {out_path}")

    if plot and do_sweep and tm["wl_nm"] is not None:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(tm["wl_nm"], tm["loss_db_cm_sweep"], label="TM")
        plt.plot(te["wl_nm"], te["loss_db_cm_sweep"], label="TE")
        plt.xlabel("Wavelength (nm)"); plt.ylabel("Modal loss (dB/cm)")
        plt.title("Modal absorption loss vs wavelength"); plt.grid(True); plt.legend()
        plt.show()

    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console is cp1252
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="FDE TM/TE modal absorption loss (dB/cm)")
    ap.add_argument("--core-loss-db-cm", type=float, default=1.0,
                    help="bulk material loss of the Si3N4 core (dB/cm)")
    ap.add_argument("--clad-loss-db-cm", type=float, default=0.0,
                    help="bulk material loss of the SiO2 cladding (dB/cm)")
    ap.add_argument("--width-nm", type=float, default=DEFAULT_WIDTH_M * 1e9)
    ap.add_argument("--lam0-nm", type=float, default=DEFAULT_LAM0_M * 1e9)
    ap.add_argument("--wl-start", type=float, default=1.50e-6)
    ap.add_argument("--wl-stop", type=float, default=1.62e-6)
    ap.add_argument("--n-points", type=int, default=100)
    ap.add_argument("--no-sweep", action="store_true", help="skip the loss-vs-wavelength sweep")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    return run(
        core_loss_db_cm=args.core_loss_db_cm,
        clad_loss_db_cm=args.clad_loss_db_cm,
        width_m=args.width_nm * 1e-9,
        lam0=args.lam0_nm * 1e-9,
        wl_start=args.wl_start, wl_stop=args.wl_stop, n_points=args.n_points,
        do_sweep=not args.no_sweep, out_dir=args.out, plot=args.plot,
    )


if __name__ == "__main__":
    main()
