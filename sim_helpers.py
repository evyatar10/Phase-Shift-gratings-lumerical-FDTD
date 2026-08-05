"""
Shared helper functions for FDTD simulation post-processing.

Extracted from run_simulation.py and run_sweep.py to eliminate duplication.
"""

import numpy as np
import scipy.integrate
import scipy.signal
from scipy.interpolate import interp1d


# ── Far-field extraction ─────────────────────────────────────────────────────

def extract_farfield(fdtd, monitor_name, ff_res=201, idx_f=1, complex_fields=False):
    """
    Pull far-field data from a planar monitor via Lumerical eval().

    Parameters:
        fdtd: Lumerical FDTD session
        monitor_name: Name of the monitor (e.g., "side_monitor", "top_monitor")
        ff_res: Far-field ux/uy grid resolution
        idx_f: 1-based frequency index
        complex_fields: Also return the complex far-field vector components
            Ex_c/Ey_c/Ez_c (response-matrix study — phase carries the
            cancellation; |E|² alone cannot form a linear response).

    Returns:
        dict with E2, ux, uy, lam (+ Ex_c, Ey_c, Ez_c when complex_fields)
        or  None on failure.
    """
    print(f"  Extracting far-field: {monitor_name}")
    try:
        res = fdtd.getresult(monitor_name, "E")
        lam = float(np.squeeze(res["lambda"]))
        print(f"    lam = {lam * 1e9:.3f} nm")
    except Exception as e:
        print(f"    ERROR: no data [{e}]")
        return None

    script = f"""
    mname = '{monitor_name}';
    idx   = {idx_f};
    res   = {ff_res};
    E2 = farfield3d(mname, idx, res, res);
    ux = farfieldux(mname, idx, res, res);
    uy = farfielduy(mname, idx, res, res);
    """
    if complex_fields:
        script += """
    E_vec = farfieldvector3d(mname, idx, res, res);
    Exc = pinch(E_vec, 3, 1);
    Eyc = pinch(E_vec, 3, 2);
    Ezc = pinch(E_vec, 3, 3);
    """
    try:
        fdtd.eval(script)
    except Exception as e:
        print(f"    ERROR in eval [{e}]")
        return None

    E2 = np.squeeze(fdtd.getv("E2"))
    ux = np.squeeze(fdtd.getv("ux"))
    uy = np.squeeze(fdtd.getv("uy"))
    out = {"E2": E2, "ux": ux, "uy": uy, "lam": lam}
    if complex_fields:
        out["Ex_c"] = np.squeeze(fdtd.getv("Exc"))
        out["Ey_c"] = np.squeeze(fdtd.getv("Eyc"))
        out["Ez_c"] = np.squeeze(fdtd.getv("Ezc"))
    return out


def extract_monitor_nearfield(fdtd, monitor_name):
    """
    Extract the 2D E-field recorded on a planar profile monitor surface.

    Returns:
        dict with x, y, z, E_res (complex, all freq pts), lambda_arr,
        or None on failure.
    """
    print(f"  Extracting near-field from monitor surface: {monitor_name}")
    try:
        res = fdtd.getresult(monitor_name, "E")
    except Exception as e:
        print(f"    ERROR: could not read E from {monitor_name} [{e}]")
        return None

    return {
        "x":          np.squeeze(res["x"]),
        "y":          np.squeeze(res["y"]),
        "z":          np.squeeze(res["z"]),
        "E_res":      res["E"],
        "lambda_arr": np.squeeze(res["lambda"]),
    }


def extract_monitor_polarimetry(fdtd, monitor_name, normal):
    """
    Polarization-resolved Poynting flux through a planar profile monitor,
    reduced SERVER-SIDE to scalars + 1D x-profiles (the full complex E/H maps
    on an arm-length monitor are ~100s of MB — never ship those).

    normal='y' (side monitor, plane coords x-z):
        S_y = 0.5*Re(Ez*conj(Hx) - Ex*conj(Hz))
        term_TM = 0.5*Re(Ez*conj(Hx))   — same polarization as the guided TM mode
        term_TE = -0.5*Re(Ex*conj(Hz))  — polarization-converted component
    normal='z' (top monitor, plane coords x-y):
        S_z = 0.5*Re(Ex*conj(Hy) - Ey*conj(Hx))  (no TM/TE split defined here)

    Returns dict with: flux_norm (source-normalized net power, from
    transmission()), P_total / P_tm / P_te (surface-integrated Poynting, W),
    prof_total / prof_tm / prof_te (1D profiles vs x, transverse-integrated),
    x (m), and lam. None on failure.
    """
    print(f"  Polarimetry from monitor: {monitor_name} (normal={normal})")
    # Fail-soft throughout: this runs AFTER the GPU solve — a shape surprise
    # here must degrade to None, never kill the .mat save.
    try:
        rE = fdtd.getresult(monitor_name, "E")
        rH = fdtd.getresult(monitor_name, "H")

        # E/H arrays: (nx, ny, nz, nf, 3) — single recorded frequency
        E = np.squeeze(rE["E"])          # -> (nx, nt, 3), nt = transverse pts
        Hf = np.squeeze(rH["H"])
        if E.ndim != 3 or E.shape[-1] != 3 or Hf.shape != E.shape:
            raise ValueError(f"unexpected field shapes E{E.shape} H{Hf.shape}")
        x = np.atleast_1d(np.squeeze(rE["x"]))
        t_ax = np.atleast_1d(np.squeeze(rE["z"] if normal == "y" else rE["y"]))
        if E.shape[0] != x.size or E.shape[1] != t_ax.size:
            raise ValueError(f"axes mismatch E{E.shape} x{x.size} t{t_ax.size}")
        lam = float(np.atleast_1d(np.squeeze(rE["lambda"]))[0])

        if normal == "y":
            term_tm = 0.5 * np.real(E[..., 2] * np.conj(Hf[..., 0]))
            term_te = -0.5 * np.real(E[..., 0] * np.conj(Hf[..., 2]))
        else:
            term_tm = 0.5 * np.real(E[..., 0] * np.conj(Hf[..., 1]))
            term_te = -0.5 * np.real(E[..., 1] * np.conj(Hf[..., 0]))
        s_tot = term_tm + term_te

        _trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 rename

        def surf_int(a):
            return float(_trapz(_trapz(a, t_ax, axis=1), x, axis=0))

        def prof(a):
            return _trapz(a, t_ax, axis=1)

        out = {
            "lam": lam, "x": x,
            "P_total": surf_int(s_tot), "P_tm": surf_int(term_tm),
            "P_te": surf_int(term_te),
            "prof_total": prof(s_tot), "prof_tm": prof(term_tm),
            "prof_te": prof(term_te),
        }
    except Exception as e:
        print(f"    ERROR: polarimetry failed on {monitor_name} [{e}]")
        return None

    try:
        out["flux_norm"] = float(np.atleast_1d(
            np.squeeze(fdtd.transmission(monitor_name)))[0])
    except Exception as e:
        print(f"    WARN: transmission() failed [{e}]")
        out["flux_norm"] = np.nan
    return out


# ── Resonance detection ──────────────────────────────────────────────────────

def find_bragg_resonance(wl, T):
    """
    Find the cavity resonance peak using a threshold-free combined metric.

    Scores every local maximum by  sharpness × dip_depth:
      - sharpness  = prominence / (width + 1)   →  high for narrow peaks
      - dip_depth  = 1 - base_level             →  high for peaks inside the bandgap

    The cavity resonance wins because it is simultaneously the sharpest
    feature AND sits inside the deepest dip (the stopband floor ≈ 0).
    """
    from scipy.signal import find_peaks, peak_prominences, peak_widths

    peaks, _ = find_peaks(T)

    if len(peaks) == 0:
        print("Warning: No peaks detected. Using global maximum.")
        return np.argmax(T)

    prominences, left_bases, right_bases = peak_prominences(T, peaks)
    widths, _, _, _ = peak_widths(T, peaks, rel_height=0.5)

    sharpness = prominences / (widths + 1)
    base_level = 0.5 * (T[left_bases] + T[right_bases])
    dip_depth = 1.0 - base_level
    score = sharpness * dip_depth

    return peaks[np.argmax(score)]


def peak_t_diagnostic(wl, T):
    """
    Inverse-design diagnostic wrapper around find_bragg_resonance.

    Given a transmission spectrum (wl, T), returns (lambda_peak, T_peak)
    by locating the resonance index via the same sharpness × dip_depth
    scorer used in production. Used by the inverse-design pipeline to
    verify the converged design's peak transmission outside the optimizer
    loop (the optimizer FOM is single-wavelength T at baseline λ_resonance,
    while this wrapper finds the *actual* resonance after convergence).
    """
    idx = find_bragg_resonance(wl, T)
    return float(wl[idx]), float(T[idx])


# ── Field profile processing ─────────────────────────────────────────────────

def calculate_fwhm_relative(x, y):
    """
    Calculate FWHM relative to the minimum (floor) of the signal.
    Uses linear interpolation at the half-maximum crossings.
    """
    y_max = np.max(y)
    y_min = np.min(y)
    target_level = y_min + 0.5 * (y_max - y_min)

    print(f"FWHM Calc -> Max: {y_max:.2e}, Min: {y_min:.2e}, Target: {target_level:.2e}")

    signs = np.sign(y - target_level)
    zero_crossings = np.where(np.diff(signs))[0]

    if len(zero_crossings) >= 2:
        def get_x_interp(idx):
            y1, y2 = y[idx], y[idx + 1]
            x1, x2 = x[idx], x[idx + 1]
            slope = (y2 - y1) / (x2 - x1)
            if slope == 0:
                return x1
            return x1 + (target_level - y1) / slope

        x_left = get_x_interp(zero_crossings[0])
        x_right = get_x_interp(zero_crossings[-1])
        return x_right - x_left
    else:
        return 0.0


def extract_envelope_peaks(x, y):
    """
    Extract the envelope by connecting peaks of the standing wave pattern.
    Uses cubic interpolation with nearest-neighbor extrapolation at edges.
    """
    peaks, _ = scipy.signal.find_peaks(y)

    if len(peaks) < 2:
        return y

    x_peaks = x[peaks]
    y_peaks = y[peaks]

    f_interp = interp1d(
        x_peaks, y_peaks,
        kind='cubic',
        bounds_error=False,
        fill_value=(y_peaks[0], y_peaks[-1])
    )

    return f_interp(x)


def extract_and_process_field_profile(sim, target_wl):
    """
    Retrieve field profile from simulation, integrate over Y, crop to grating,
    extract envelope, and calculate FWHM.

    Returns:
        (f_x, I_x_1D, I_x_envelope, fwhm_val, actual_wavelength)
    """
    field_data = sim.fdtd.getresult("field_profile", "E")
    f_lam = np.squeeze(field_data['lambda'])
    f_x_raw = np.squeeze(field_data['x'])
    f_y = np.squeeze(field_data['y'])
    f_E = field_data['E']

    idx_mon = np.argmin(np.abs(f_lam - target_wl))
    print(f"Extracting 1D core field tracking at monitor wavelength: {f_lam[idx_mon] * 1e9:.3f} nm")

    # Extract E-field at resonance (handle different array dimensions)
    if f_E.ndim == 5:
        E_res = f_E[:, :, 0, idx_mon, :]
    elif f_E.ndim == 4:
        E_res = f_E[:, :, idx_mon, :]
    else:
        E_res = f_E[..., idx_mon, :]

    # Integrate |E|^2 over Y-axis
    I_xy = np.abs(E_res[..., 0]) ** 2 + np.abs(E_res[..., 1]) ** 2 + np.abs(E_res[..., 2]) ** 2
    I_x_1D_raw = scipy.integrate.trapezoid(I_xy, f_y, axis=1)

    # Crop to grating region
    x_limit = sim.x_grating_end
    valid_indices = np.abs(f_x_raw) <= x_limit
    f_x = f_x_raw[valid_indices]
    I_x_1D = I_x_1D_raw[valid_indices]

    # Envelope and FWHM
    I_x_envelope = extract_envelope_peaks(f_x, I_x_1D)
    fwhm_val = calculate_fwhm_relative(f_x, I_x_envelope)

    return f_x, I_x_1D, I_x_envelope, fwhm_val, f_lam[idx_mon]


# ── File naming ───────────────────────────────────────────────────────────────

def generate_file_tag(sim):
    """
    Generate a compact file tag from simulation parameters.

    Format: N{periods}[_A{apod}][_th][_D{detuning}][_S{shift_nm}][_fc][_avg|_avgx][_d]
    See FILE_NAMING.md for the full abbreviation reference.
    """
    N = sim.n_periods_each_side
    Napod = sim.n_apod_periods_each_side
    use_apod = bool(sim.use_apodization) and (Napod is not None) and (Napod > 0)

    cav_tag = ""
    if hasattr(sim, 'cavity_length'):
        detuning_nm = (sim.pitch / 2.0 - sim.cavity_length) * 1e9
        if abs(detuning_nm) > 0.01:
            cav_tag = f"_D{detuning_nm:.2f}".replace(".", "p")

    # Innermost-tooth shift (only annotated when nonzero — keeps default file
    # names unchanged so all existing results / paths still match).
    shift_tag = ""
    fc_tag = ""
    shift_m = float(getattr(sim, 'innermost_tooth_shift_m', 0.0) or 0.0)
    if shift_m > 0.0:
        shift_tag = f"_S{round(shift_m * 1e9):.0f}"
        if not bool(getattr(sim, 'lengthen_cavity', True)):
            fc_tag = "_fc"

    # Polarization (only annotated for TM — keeps all existing TE file names
    # unchanged, and prevents a TM run from overwriting its TE reference).
    pol_tag = "" if getattr(sim, 'polarization', 'TE') == 'TE' else f"_{sim.polarization}"

    mat_tag = "" if sim.use_constant_materials else "_d"
    _cw_m = getattr(sim, 'cavity_width_m', None)
    if _cw_m is not None:
        wgd_tag = f"_W{round(_cw_m * 1e9):.0f}"
    else:
        _cwo = getattr(sim, 'cavity_width_option', 'narrow')
        wgd_tag = "_avgx" if _cwo == "avg_ext" else "_avg" if _cwo == "avg" else ""

    # ── Two side-by-side pi-shift devices: naming convention ──────────────────
    # Single-device (n_devices=1) file names are UNCHANGED — this suffix is only
    # appended for the two-device study, and it carries every distinguishing knob
    # so a two-device .mat name is fully self-documenting:
    #   _2pishift            marks a two-pi-shift (coupled-pair) run
    #   _p{..}               grating pitch in nm (e.g. p500, p518p3) — matters for TM-calibrated runs
    #   _Ygap{..}nm          lateral edge-to-edge distance between the guides  (Y axis)
    #   _Xstag{..}nm         longitudinal stagger of device 2 along the guide  (X axis)
    #   _corr1{..}nm         device-1 corrugation depth
    #   _corr2{..}nm         device-2 corrugation depth
    #   _closed              device 2 is a closed recycler (no feed waveguides / drain ports)
    # Unique per config → concurrent sweep tasks NEVER share a layout/.h5/.mat
    # filename (a shared tag makes same-node array tasks clobber each other's
    # output .h5 mid-run → "expansion for port monitor" crash).
    two_dev_tag = ""
    if getattr(sim, 'n_devices', 1) == 2:
        gap_nm = round(float(getattr(sim, 'device_gap_m', 0.0)) * 1e9)
        stag_nm = round(float(getattr(sim, 'device_stagger_m', 0.0)) * 1e9)
        d1_nm = round((float(sim.width_wide) - float(sim.width_narrow)) * 1e9)
        d2_nm = round((float(getattr(sim, 'width_wide_2', 0.0)) -
                       float(getattr(sim, 'width_narrow_2', 0.0))) * 1e9)
        pitch_str = f"{float(sim.pitch) * 1e9:.1f}".replace(".", "p")  # 500.0->500p0, 518.3->518p3
        closed_tag = "_closed" if getattr(sim, 'device2_closed', False) else ""
        # Device-2 average corrugation width (FW-BIC detuning knob): appended
        # only when device 2's average differs from device 1's, so all existing
        # two-device file names (equal averages) are unchanged. Without this,
        # detuning-only rows share a .fsp/.h5/.mat name and clobber each other.
        avg1_nm = round(0.5 * (float(sim.width_wide) + float(sim.width_narrow)) * 1e9)
        avg2_nm = round(0.5 * (float(getattr(sim, 'width_wide_2', 0.0)) +
                               float(getattr(sim, 'width_narrow_2', 0.0))) * 1e9)
        avg2_tag = f"_avg2W{avg2_nm}" if avg2_nm != avg1_nm else ""
        two_dev_tag = (f"_2pishift_p{pitch_str}_Ygap{gap_nm}nm_Xstag{stag_nm}nm"
                       f"_corr1{d1_nm}nm_corr2{d2_nm}nm{avg2_tag}{closed_tag}")

    # ── Domain-size override (convergence studies): appended only when a y/z box
    #    override is active, so domain sweeps get distinct filenames while every
    #    default-box file name stays exactly as before.
    dom_tag = ""
    if getattr(sim, '_domain_tag_active', False):
        y_um = float(sim.y_span) * 1e6
        z_um = float(sim.z_span) * 1e6
        dom_tag = f"_Ybox{y_um:.1f}_Zbox{z_um:.1f}".replace(".", "p")
    # Auto-shutoff override (convergence study): appended only when the knob is
    # set, so every default-threshold (1e-7) file name is unchanged. Rows
    # differing only in shutoff would otherwise collide (§6).
    _asm = getattr(sim, 'auto_shutoff_min', None)
    if _asm:
        dom_tag += f"_AS{_asm:.0e}".replace("e-0", "em").replace("e-", "em")

    # ── Core width / corrugation (light-line-margin study): appended only when the
    #    average corrugated width differs from the historical 800 nm default, so
    #    every existing file name is unchanged. Carries corr too, because width
    #    rows are swept both at fixed and at proportionally-scaled corrugation.
    wc_tag = ""
    _w_avg_nm = round((float(sim.width_wide) + float(sim.width_narrow)) * 0.5e9)
    _c_nm = round((float(sim.width_wide) - float(sim.width_narrow)) * 1e9)
    if abs(_w_avg_nm - 800) > 0:
        wc_tag = f"_Wavg{_w_avg_nm}_C{_c_nm}"
    elif _c_nm != (400 if getattr(sim, 'polarization', 'TE') == 'TM' else 300):
        # Corr ladders at the historical W800 (q3db studies): corr is otherwise
        # absent from the tag, so same-N ladder rows would clobber each other's
        # .fsp/.h5/.mat (section-6). Historical W800 results are corr-400-only
        # for TM and corr-300-only for TE -> every old name is unchanged.
        wc_tag = f"_C{_c_nm}"

    # ── Distributed pi-shift (per-tooth gap shifts): appended only when the
    #    inner_shift list is set with any nonzero entry, so all legacy names are
    #    unchanged. Carries count, total (m-prefixed if negative) and first
    #    element to keep distribution variants distinct.
    dsh_tag = ""
    _ish_list = getattr(sim, 'inner_shift_nm', None)
    if _ish_list and any(abs(float(v)) > 1e-9 for v in _ish_list):
        _tot = round(sum(float(v) for v in _ish_list))
        _s0 = round(float(_ish_list[0]))
        def _fmt(n):
            return f"m{abs(n)}" if n < 0 else f"{n}"
        dsh_tag = f"_dsh{len(_ish_list)}S{_fmt(_tot)}s{_fmt(_s0)}"

    # ── Explicit per-tooth width arrays (width-envelope / graded-island study):
    #    appended only when set, so every uniform-grating file name is unchanged.
    ptw_tag = ""
    _ptw = getattr(sim, 'width_wide_per_tooth_m', None)
    if _ptw:
        _w0 = round(float(_ptw[0]) * 1e9)
        _w1 = round(float(_ptw[-1]) * 1e9)
        ptw_tag = (f"_ptw{len(_ptw)}W{_w0}" if _w0 == _w1
                   else f"_ptw{len(_ptw)}W{_w0}to{_w1}")
    # Narrow-tooth per-tooth widths (narrow see-saw study): appended only when
    # the narrow list deviates from uniform 600 — legacy names (incl. the
    # width-envelope study, whose narrow lists were uniform) are unchanged.
    _ptn = getattr(sim, 'width_narrow_per_tooth_m', None)
    if _ptn:
        _n0 = round(float(_ptn[0]) * 1e9)
        _n1 = round(float(_ptn[-1]) * 1e9)
        _nom = round(float(sim.width_narrow) * 1e9)
        if any(round(float(v) * 1e9) != _nom for v in _ptn):
            ptw_tag += (f"_ptn{len(_ptn)}W{_n0}" if _n0 == _n1
                        else f"_ptn{len(_ptn)}W{_n0}to{_n1}")

    # ── Sidewall-corrugation phase offset (null-steering study): appended only
    #    when nonzero, so every aligned-wall file name is unchanged.
    wp_tag = ""
    _wp_deg = float(getattr(sim, 'wall_phase_offset_deg', 0.0) or 0.0)
    if abs(_wp_deg) > 1e-9:
        wp_tag = f"_wp{_wp_deg:g}".replace(".", "p")

    # ── Corrugation profile (tooth-shape study): appended only for sin/tri, so
    #    every rectangular-tooth file name is unchanged.
    prof_tag = ""
    _prof = getattr(sim, 'corrugation_profile', 'rect')
    if _prof != 'rect':
        prof_tag = f"_prof{_prof}"

    # ── Inner-tooth shape (center-shape study): appended only for shaped teeth,
    #    so every rectangular-tooth file name is unchanged.
    ish_tag = ""
    _ish = getattr(sim, 'inner_tooth_shape', 'rect')
    if _ish != 'rect' and getattr(sim, 'n_shaped_inner_teeth', 0) > 0:
        ish_tag = f"_ish{_ish}{int(sim.n_shaped_inner_teeth)}"
    # Cavity-segment shape (center-shape study); appended only when non-rect.
    _csh = getattr(sim, 'cavity_shape', 'rect')
    if _csh != 'rect' and float(getattr(sim, 'cavity_shape_depth_m', 0.0)) > 0.0:
        ish_tag += f"_cav{_csh[:4]}{round(sim.cavity_shape_depth_m * 1e9)}"
    # Antisymmetric inner-tooth DW detuning (anti-radiator study); appended only
    # when active. Single-tooth form `_adw{delta}d{tooth}`; multi-tooth joined.
    if getattr(sim, '_has_asym_dw', False):
        _nz = [(i + 1, float(v)) for i, v in enumerate(sim.asym_inner_dw_delta_nm)
               if abs(float(v)) > 1e-9]
        ish_tag += "_adw" + "_".join(f"{v:g}d{d}" for d, v in _nz).replace(".", "p").replace("-", "m")

    # ── Symmetry-off marker: single-device runs historically always used the
    #    y-symmetry BC, so annotate only when it is explicitly OFF (wall-phase
    #    rows and their dedicated controls) to avoid control/legacy collisions.
    nosym_tag = ""
    if not getattr(sim, 'use_symmetry', True) and getattr(sim, 'n_devices', 1) == 1:
        nosym_tag = "_nosym"

    # ── Scatterer (radiation-recycling study): appended only when one is actually
    #    drawn, so all existing file names — and the r=0 in-study control — are
    #    unchanged. Carries radius + position (integer nm; 'm' prefix = minus), so
    #    concurrent array tasks NEVER share a layout/.h5/.mat filename.
    #      _scR{r}_X{x}_Y{y}   cylinder radius / center x / center y in nm
    #      _pair               y-mirrored pair at ±y (the symmetry-compatible form)
    scat_tag = ""
    if getattr(sim, '_has_scatterer', False):
        x_nm = round(sim.scatterer_x_m * 1e9)
        y_nm = round(sim.scatterer_y_m * 1e9)
        x_str = f"m{abs(x_nm)}" if x_nm < 0 else f"{x_nm}"
        y_str = f"m{abs(y_nm)}" if y_nm < 0 else f"{y_nm}"
        # Rect strip form (lateral-reflector study): length x width + center.
        if getattr(sim, 'scatterer_shape', 'cylinder') == 'rect':
            l_nm = round(sim.scatterer_x_span_m * 1e9)
            w_nm = round(sim.scatterer_y_span_m * 1e9)
            head = f"_scRECT_L{l_nm}xW{w_nm}"
        else:
            head = f"_scR{round(sim.scatterer_radius_m * 1e9)}"
        x_list = getattr(sim, 'scatterer_x_list_m', None)
        y_list = getattr(sim, 'scatterer_y_list_m', None)
        if x_list:
            # Array form: count + first/last x (and y for arc/diagonal rows).
            x0 = round(x_list[0] * 1e9)
            x1 = round(x_list[-1] * 1e9)
            if y_list:
                y0 = round(y_list[0] * 1e9)
                y1 = round(y_list[-1] * 1e9)
                scat_tag = f"{head}_arr{len(x_list)}_X{x0}to{x1}_Y{y0}to{y1}"
            else:
                scat_tag = f"{head}_arr{len(x_list)}_X{x0}to{x1}_Y{y_str}"
            # Corrugation depth (array form only): corr-trim rows of a hole-lattice
            # sweep share the same site list and would otherwise collide on the
            # .fsp/.h5/.mat name (corr is not in the base tag at avg width 800).
            scat_tag += f"_C{round((float(sim.width_wide) - float(sim.width_narrow)) * 1e9)}"
        else:
            scat_tag = f"{head}_X{x_str}_Y{y_str}"
        if getattr(sim, 'scatterer_mirrored_y', False) and y_nm != 0:
            scat_tag += "_pair"
        # Flipped-material case: index below the core index marks an oxide HOLE
        # inside the SiN core (vs the default SiN pillar in the oxide cladding).
        if getattr(sim, '_scatterer_n', sim.n_core_const) < sim.n_core_const - 1e-9:
            scat_tag += "_hole"
        # Named-material case (metal-mirror study): first token of the database
        # name, e.g. "_PEC" / "_Al" — dielectric-index file names are unchanged.
        _scat_mat = getattr(sim, 'scatterer_material', None)
        if _scat_mat:
            scat_tag += f"_{_scat_mat.split()[0]}"
        # Height tag — only when the scatterer height differs from the core
        # height, so every historical (full-core / default) file name is stable.
        # Needed since heights became sweepable: rows differing only in height
        # would otherwise share .fsp/.h5/.mat names (§6 clobber).
        _scat_h = getattr(sim, 'scatterer_height_m', None)
        if _scat_h and abs(_scat_h - sim.core_height) > 1e-12:
            scat_tag += f"_H{round(_scat_h * 1e9)}"

    if use_apod:
        tanh_tag = "_th" if getattr(sim, 'apod_method', 'linear') == 'tanh' else ""
        # Annotate center modulation depth only when it differs from the
        # historical default (100 nm) — keeps pre-sweep filenames stable.
        mod_depth_nm = float(getattr(sim, 'center_mod_depth', 100e-9)) * 1e9
        mod_tag = f"_M{round(mod_depth_nm):.0f}" if abs(mod_depth_nm - 100.0) > 0.5 else ""
        return f"N{N}_A{Napod}{tanh_tag}{mod_tag}{cav_tag}{shift_tag}{fc_tag}{pol_tag}{mat_tag}{wgd_tag}{two_dev_tag}{wc_tag}{prof_tag}{ish_tag}{dsh_tag}{ptw_tag}{wp_tag}{nosym_tag}{dom_tag}{scat_tag}"
    else:
        return f"N{N}{cav_tag}{shift_tag}{fc_tag}{pol_tag}{mat_tag}{wgd_tag}{two_dev_tag}{wc_tag}{prof_tag}{ish_tag}{dsh_tag}{ptw_tag}{wp_tag}{nosym_tag}{dom_tag}{scat_tag}"


def apply_monitor_overrides(sim, cfg):
    """
    Override frequency points on large monitors to control memory usage.
    """
    n_2d_pts = cfg.spectral.n_2d_monitor_points

    # Always-on 1D core-tracking monitor (builder default 501 points): cap for
    # mm-scale devices where the 501-point getresult segfaults (N=1300, 127321).
    n_fp = getattr(cfg.monitors, "field_profile_freq_points", None)
    if n_fp:
        sim.fdtd.setnamed("field_profile", "frequency points", int(n_fp))
        print(f"Override: Set field_profile monitor to {n_fp} points.")

    if sim.record_2d_fields_top_and_cross:
        _mons_2d = ("field_profile_2D_XY", "field_profile_2D_YZ_cross",
                    "field_profile_2D_XZ_side")
        for _m in _mons_2d:
            sim.fdtd.setnamed(_m, "frequency points", n_2d_pts)
        print(f"Override: Set 2D monitors to {n_2d_pts} points.")
        # Optional: give the 2D monitors their OWN narrow wavelength window
        # (decoupled from the source scan) so the recorded planes sample densely
        # around a known resonance instead of spreading over the full scan.
        c_nm = cfg.monitors.monitor_2d_center_nm
        s_nm = cfg.monitors.monitor_2d_span_nm
        if c_nm and s_nm:
            for _m in _mons_2d:
                sim.fdtd.setnamed(_m, "use source limits", 0)
                sim.fdtd.setnamed(_m, "wavelength center", c_nm * 1e-9)
                sim.fdtd.setnamed(_m, "wavelength span", s_nm * 1e-9)
            print(f"Override: 2D monitors own window {c_nm} +/- {s_nm / 2} nm.")

    if getattr(sim, 'record_3d_fields', False):
        n_3d_pts = cfg.monitors.n_3d_freq_points
        if n_3d_pts is None:
            n_3d_pts = n_2d_pts
        sim.fdtd.setnamed("field_profile_3D", "frequency points", n_3d_pts)
        print(f"Override: Set 3D monitor to {n_3d_pts} points.")

    if sim.record_farfield:
        # Single far-field frequency point. With "use source limits" it lands at the
        # source band-CENTER frequency — so center the scan on the resonance (narrow
        # band) so the far-field is recorded at the resonance, not the band center
        # of a wide scan (which sits at the band-center FREQUENCY, e.g. 1546 nm for a
        # 1550 nm / 150 nm-wide scan).
        sim.fdtd.setnamed("side_monitor", "frequency points", 1)
        sim.fdtd.setnamed("top_monitor", "frequency points", 1)
        print("Override: Far-field monitors set to 1 frequency point (source band center).")
