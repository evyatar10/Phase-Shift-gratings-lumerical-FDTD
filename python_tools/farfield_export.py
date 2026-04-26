"""
test_farfield_export.py
-----------------------
Far-field analysis for side_monitor and top_monitor.
Reproduces Lumerical's built-in  image(ux, uy, E2, ..., "polar")  display style:
  • Linear-scale, normalised 0→1  (not dB, which causes the all-red look)
  • Hemisphere exterior = 0  → blue corners like Lumerical (not NaN/transparent)
  • Smooth bicubic interpolation  → no pcolormesh pixel artifacts at boundary
  • XY map  : square axes, direction-cosine ticks
  • Polar map: same data + Lumerical-style circular/radial grid, degree ticks

Axis conventions
  top_monitor  (Z-normal) : H = uy (Y-direction),  V = ux (X-direction)
                             Polar x-label "θy [°]"  (bottom), y-label "θx [°]" (left)
  side_monitor (Y-normal) : H = ux (X-direction),   V = uy (Z-direction)
                             Polar x-label "θx [°]"  (bottom), y-label "θz [°]" (left)
"""

import sys, importlib.util, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#0d0d1a",
    "axes.facecolor":   "#0d0d1a",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "axes.edgecolor":   "#555566",
    "axes.titlecolor":  "white",
})

# ─── lumapi ────────────────────────────────────────────────────────────────────
try:
    import lumapi
except ImportError:
    _p = r"C:\Program Files\Lumerical\v252\api\python\lumapi.py"
    spec = importlib.util.spec_from_file_location("lumapi", _p)
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
FSP_PATH   = r"C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles_5_long\layouts\layout_100_periods_CONST.fsp"
SAVE_PATH  = FSP_PATH.replace(".fsp", "_farfield_export.mat")
FF_RES     = 201        # ux/uy grid resolution
IDX_F      = 1          # 1-based frequency index (monitors record 1 point each)
HALF_ANGLE = 30         # degrees for cone-power metric
SCALE_DB         = False  # False = linear 0-1; True = dB scale
DB_FLOOR         = -40    # dB floor (only when SCALE_DB=True)
CUSTOM_ANGLE_DEG = 26.7   # specific cone half-angle to mark (degrees) — orange X

# ─── EXTRACTION ────────────────────────────────────────────────────────────────
def extract(fdtd, monitor_name):
    print(f"\n--- {monitor_name} ---")
    try:
        res = fdtd.getresult(monitor_name, "E")
        lam = float(np.squeeze(res["lambda"]))
        print(f"  Recorded lam = {lam*1e9:.3f} nm")
    except Exception as e:
        print(f"  ERROR: no data  [{e}]"); return None

    script = f"""
    mname = '{monitor_name}';
    idx   = {IDX_F};
    res   = {FF_RES};
    half  = {HALF_ANGLE};
    E2 = farfield3d(mname, idx, res, res);
    ux = farfieldux(mname,  idx, res, res);
    uy = farfielduy(mname,  idx, res, res);
    temp_cone = farfield3dintegrate(E2, ux, uy, half, 0, 0);
    temp_far  = farfield3dintegrate(E2, ux, uy);
    """
    try:
        fdtd.eval(script)
    except Exception as e:
        print(f"  ERROR in eval: [{e}]"); return None

    E2   = np.squeeze(fdtd.getv("E2"))
    ux   = np.squeeze(fdtd.getv("ux"))
    uy   = np.squeeze(fdtd.getv("uy"))
    cone = float(np.squeeze(fdtd.getv("temp_cone")))
    far  = float(np.squeeze(fdtd.getv("temp_far")))
    frac = (cone / far * 100) if far > 0 else 0.0
    print(f"  Power in {HALF_ANGLE}-deg cone: {frac:.1f}%")
    print(f"  E2 {E2.shape},  ux {ux.shape},  uy {uy.shape}")
    return {"E2": E2, "ux": ux, "uy": uy, "lam": lam, "cone_pct": frac}


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def _normalize(E2, ux, uy):
    """
    Prepare E2 for display.
    SCALE_DB=False → linear, normalised 0-1   (like Lumerical's image())
    SCALE_DB=True  → dB,     clipped to DB_FLOOR  (better for side-lobe visibility)
    In both cases, hemisphere exterior = 0 → blue corners, no transparent pixels.
    """
    UX, UY  = np.meshgrid(ux, uy, indexing='ij')
    valid   = (UX**2 + UY**2) <= 1.0
    max_val = np.max(E2[valid])
    if SCALE_DB:
        with np.errstate(divide='ignore', invalid='ignore'):
            E2_dB = 10 * np.log10(E2 / max_val)
        # Map DB_FLOOR→0 and 0 dB→1 linearly for colormap
        E2_norm = np.where(valid,
                           np.clip((E2_dB - DB_FLOOR) / (-DB_FLOOR), 0, 1),
                           0.0)
    else:
        E2_norm = np.where(valid, E2 / max_val, 0.0)
    return E2_norm, UX, UY


def _width_at_thresh(profile, axis_uc, frac):
    """Width in degrees where profile drops to `frac` * max (e.g. frac=0.5 for -3dB)."""
    thresh = np.max(profile) * frac
    above = profile >= thresh
    crossings = np.where(np.diff(above.astype(int)))[0]
    if len(crossings) < 2:
        return None
    def interp(i):
        x0, x1 = axis_uc[i], axis_uc[i+1]
        y0, y1 = profile[i], profile[i+1]
        uc = x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)
        return np.degrees(np.arcsin(np.clip(abs(uc), 0, 1))) * np.sign(uc)
    return abs(interp(crossings[-1]) - interp(crossings[0]))


def _fwhm_deg(profile, axis_uc):
    return _width_at_thresh(profile, axis_uc, 0.5)


def _analyze_lobe(E2, ux, uy):
    """Find peak, compute 3 dB beam width in ux and uy directions."""
    UX, UY = np.meshgrid(ux, uy, indexing='ij')
    E2_v   = np.where((UX**2 + UY**2) <= 1.0, E2, 0.0)

    peak_idx = np.unravel_index(np.argmax(E2_v), E2_v.shape)
    i_ux, i_uy = peak_idx
    peak_ux = ux[i_ux]; peak_uy = uy[i_uy]
    r_peak  = np.sqrt(peak_ux**2 + peak_uy**2)
    theta_peak = np.degrees(np.arcsin(np.clip(r_peak, 0, 1)))
    phi_peak   = np.degrees(np.arctan2(peak_uy, peak_ux))

    fwhm_ux = _fwhm_deg(E2_v[:, i_uy], ux)
    fwhm_uy = _fwhm_deg(E2_v[i_ux, :], uy)

    print(f"  Lobe peak: ux={peak_ux:.3f}, uy={peak_uy:.3f}")
    print(f"  Center angle: {theta_peak:.1f} deg  (azimuth {phi_peak:.1f} deg)")
    if fwhm_ux: print(f"  -3 dB width along UX: {fwhm_ux:.1f} deg")
    if fwhm_uy: print(f"  -3 dB width along UY: {fwhm_uy:.1f} deg  (half-angle {fwhm_uy/2:.1f} deg)")
    return r_peak, theta_peak, fwhm_ux, fwhm_uy

def _draw_x_cone(ax, ux_peak, fwhm_uy_deg, h_rotated=False,
                 color='deepskyblue', lw=1.8, ls='--', label=None):
    """
    Draw an X cone (two crossing lines) at ±half_angle in the UY direction.
    fwhm_uy_deg is the FULL angular width; the half-angle = fwhm_uy_deg/2.
    Label is placed ON the figure near the upper-right arm tip.
    """
    if fwhm_uy_deg is None:
        return
    uy_half = np.sin(np.radians(fwhm_uy_deg / 2))   # uc half-width
    if h_rotated:  # H=uy, V=ux
        ax.plot([-uy_half, uy_half], [-ux_peak, ux_peak], color=color, lw=lw, ls=ls, zorder=6)
        ax.plot([-uy_half, uy_half], [ux_peak, -ux_peak], color=color, lw=lw, ls=ls, zorder=6)
        # place label near upper-right tip, slightly inward
        if label:
            ax.text(uy_half * 0.75, ux_peak * 0.85, label, color=color,
                    fontsize=9, fontweight='bold', ha='center', va='bottom', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.15', fc='#0d0d1a', alpha=0.7, ec='none'))
    else:          # H=ux, V=uy
        ax.plot([-ux_peak, ux_peak], [-uy_half, uy_half], color=color, lw=lw, ls=ls, zorder=6)
        ax.plot([-ux_peak, ux_peak], [uy_half, -uy_half], color=color, lw=lw, ls=ls, zorder=6)
        if label:
            ax.text(ux_peak * 0.75, uy_half * 0.85, label, color=color,
                    fontsize=9, fontweight='bold', ha='center', va='bottom', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.15', fc='#0d0d1a', alpha=0.7, ec='none'))


def _deg_ticks(ax):
    """Set direction-cosine axis ticks and label them in degrees."""
    deg_pts = np.array([-60, -45, -30, 0, 30, 45, 60])
    uc_pts  = np.sin(np.radians(deg_pts))
    labels  = [f"{d}°" for d in deg_pts]
    ax.set_xticks(uc_pts); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(uc_pts); ax.set_yticklabels(labels, fontsize=8)


def _polar_grid(ax, rotated=False, gridcolor='gold'):
    """
    Overlay Lumerical-style circular + radial grid on a Cartesian ux/uy axes.
    rotated=True  → top_monitor convention (H=uy, V=ux): radial 0 points UP
    rotated=False → side_monitor convention (H=ux, V=uy): radial 0 points RIGHT
    """
    phi = np.linspace(0, 2*np.pi, 720)

    # θ-circles at 10°, 20°, ..., 80° (radius in direction-cosine space)
    for td in range(10, 90, 10):
        r = np.sin(np.radians(td))
        ax.plot(r * np.cos(phi), r * np.sin(phi),
                color=gridcolor, lw=0.5, alpha=0.55, zorder=3)
        ax.text(r + 0.02, 0.01, f"{td}°",
                fontsize=6.5, color=gridcolor, alpha=0.9,
                va='bottom', ha='left', zorder=4)

    # hemisphere boundary
    ax.plot(np.cos(phi), np.sin(phi), 'w-', lw=0.9, zorder=3)

    # radial lines every 30°
    for pd in range(0, 360, 30):
        pr = np.radians(pd)
        if rotated:   # (h=uy, v=ux) → endpoint (sin(phi), cos(phi)) in (h, v)
            h_end, v_end = np.sin(pr), np.cos(pr)
        else:         # (h=ux, v=uy) → endpoint (cos(phi), sin(phi))
            h_end, v_end = np.cos(pr), np.sin(pr)
        ax.plot([0, h_end], [0, v_end],
                color=gridcolor, lw=0.4, alpha=0.4, zorder=3)


def _imshow(ax, data, h_axis, v_axis, aspect, cmap='jet'):
    """
    Plot data with bicubic interpolation.
    data[v_idx, h_idx]  (row → v_axis, col → h_axis)
    """
    return ax.imshow(
        data,
        extent=[h_axis.min(), h_axis.max(), v_axis.min(), v_axis.max()],
        origin='lower', aspect=aspect,
        cmap=cmap, interpolation='bicubic',
        vmin=0, vmax=1, zorder=2
    )


# ─── FIGURE BUILDER ────────────────────────────────────────────────────────────
def make_figure(data, monitor_name):
    lam = data["lam"]
    E2  = data["E2"]
    ux  = data["ux"]
    uy  = data["uy"]

    E2_norm, UX, UY = _normalize(E2, ux, uy)

    # ── Lobe analysis (always uses raw E2, not dB) ──────────────────────
    print(f"\n[Lobe analysis — {monitor_name}]")
    r_peak, theta_peak, fwhm_ux, fwhm_uy = _analyze_lobe(E2, ux, uy)
    UX2, UY2 = np.meshgrid(ux, uy, indexing='ij')
    E2_masked = np.where((UX2**2 + UY2**2) <= 1.0, E2, 0.0)
    pi = np.unravel_index(np.argmax(E2_masked), E2_masked.shape)
    peak_ux_val = ux[pi[0]]
    peak_uy_val = uy[pi[1]]

    fig, (ax_xy, ax_pol) = plt.subplots(1, 2, figsize=(15, 6.5))
    fwhm_str = ""
    if fwhm_ux: fwhm_str += f"  |  UX width: {fwhm_ux:.1f}deg"
    if fwhm_uy: fwhm_str += f"  UY width: {fwhm_uy:.1f}deg"
    fig.suptitle(
        f"{monitor_name}   [lam = {lam*1e9:.2f} nm]   "
        f"center {theta_peak:.1f}deg{fwhm_str}",
        fontweight='bold')

    if monitor_name == "top_monitor":
        # H = uy (Y), V = ux (X)
        # imshow data shape must be [V_idx, H_idx] = [ux_idx, uy_idx] = E2_norm as-is
        # ─── XY map ───
        im = _imshow(ax_xy, E2_norm,
                     h_axis=uy, v_axis=ux, aspect='equal')
        ax_xy.set_xlim(-1.05, 1.05); ax_xy.set_ylim(-1.05, 1.05)
        ax_xy.set_xlabel("uy  (Y-direction)", fontsize=10)
        ax_xy.set_ylabel("ux  (X-direction)", fontsize=10)
        ax_xy.set_title("XY Map  (ux/uy)")
        # white dashed – -3 dB (FWHM) half-angle
        _draw_x_cone(ax_xy, peak_ux_val, fwhm_uy, h_rotated=True,
                     color='deepskyblue',
                     label=f"-3 dB: {fwhm_uy/2:.1f}°" if fwhm_uy else None)
        # orange dashed – fixed user angle
        _draw_x_cone(ax_xy, peak_ux_val, CUSTOM_ANGLE_DEG * 2, h_rotated=True,
                     color='orange',
                     label=f"{CUSTOM_ANGLE_DEG}°")

        # ─── Polar map ───
        _imshow(ax_pol, E2_norm,
                h_axis=uy, v_axis=ux, aspect='auto')
        _polar_grid(ax_pol, rotated=True)
        _deg_ticks(ax_pol)
        ax_pol.set_xlim(-1.05, 1.05); ax_pol.set_ylim(-1.05, 1.05)
        ax_pol.set_xlabel("ty [deg]  (Y-direction)", fontsize=10)
        ax_pol.set_ylabel("tx [deg]  (X-direction)", fontsize=10)
        ax_pol.set_title("Polar Map  (tx left, ty bottom)")
        _draw_x_cone(ax_pol, peak_ux_val, fwhm_uy, h_rotated=True,
                     color='deepskyblue',
                     label=f"-3 dB: {fwhm_uy/2:.1f}°" if fwhm_uy else None)
        _draw_x_cone(ax_pol, peak_ux_val, CUSTOM_ANGLE_DEG * 2, h_rotated=True,
                     color='orange',
                     label=f"{CUSTOM_ANGLE_DEG}°")

    else:  # side_monitor
        # H = ux (X), V = uy (Z)
        # imshow data shape must be [uy_idx, ux_idx] = E2_norm.T
        # ─── XY map ───
        im = _imshow(ax_xy, E2_norm.T,
                     h_axis=ux, v_axis=uy, aspect='equal')
        ax_xy.set_xlim(-1.05, 1.05); ax_xy.set_ylim(-1.05, 1.05)
        ax_xy.set_xlabel("ux  (X-direction)", fontsize=10)
        ax_xy.set_ylabel("uy  (Z-direction)", fontsize=10)
        ax_xy.set_title("XY Map  (ux/uy)")
        _draw_x_cone(ax_xy, peak_ux_val, fwhm_uy, h_rotated=False,
                     color='deepskyblue',
                     label=f"-3 dB: {fwhm_uy/2:.1f}°" if fwhm_uy else None)
        _draw_x_cone(ax_xy, peak_ux_val, CUSTOM_ANGLE_DEG * 2, h_rotated=False,
                     color='orange',
                     label=f"{CUSTOM_ANGLE_DEG}°")

        # ─── Polar map ───
        _imshow(ax_pol, E2_norm.T,
                h_axis=ux, v_axis=uy, aspect='auto')
        _polar_grid(ax_pol, rotated=False)
        _deg_ticks(ax_pol)
        ax_pol.set_xlim(-1.05, 1.05); ax_pol.set_ylim(-1.05, 1.05)
        ax_pol.set_xlabel("tx [deg]  (X-direction)", fontsize=10)
        ax_pol.set_ylabel("tz [deg]  (Z-direction)", fontsize=10)
        ax_pol.set_title("Polar Map  (tx horizontal, tz vertical)")
        _draw_x_cone(ax_pol, peak_ux_val, fwhm_uy, h_rotated=False,
                     color='deepskyblue',
                     label=f"-3 dB: {fwhm_uy/2:.1f}°" if fwhm_uy else None)
        _draw_x_cone(ax_pol, peak_ux_val, CUSTOM_ANGLE_DEG * 2, h_rotated=False,
                     color='orange',
                     label=f"{CUSTOM_ANGLE_DEG}°")

    scale_str = f"dB  (floor {DB_FLOOR} dB)" if SCALE_DB else "linear  (0-1)"
    cbar = fig.colorbar(im, ax=[ax_xy, ax_pol], shrink=0.8, pad=0.02)
    cbar.set_label(f"Normalised E2  [{scale_str}]", fontsize=9)
    cbar.ax.tick_params(labelsize=8, colors='white')
    plt.tight_layout()
    return fig


# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading: {FSP_PATH}")
    fdtd = lumapi.FDTD(filename=FSP_PATH, hide=True)

    all_data = {}
    for mname in ["side_monitor", "top_monitor"]:
        d = extract(fdtd, mname)
        if d is not None:
            all_data[mname] = d

    fdtd.close()

    for mname, d in all_data.items():
        make_figure(d, mname)

    if all_data:
        save_dict = {k: {sk: sv for sk, sv in v.items()
                         if isinstance(sv, np.ndarray)}
                     for k, v in all_data.items()}
        sio.savemat(SAVE_PATH, save_dict, do_compression=True)
        print(f"\nSaved: {SAVE_PATH}")

    plt.show()
    print("Done.")
