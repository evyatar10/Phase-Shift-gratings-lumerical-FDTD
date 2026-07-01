"""
Shared helper functions for FDTD simulation post-processing.

Extracted from run_simulation.py and run_sweep.py to eliminate duplication.
"""

import numpy as np
import scipy.integrate
import scipy.signal
from scipy.interpolate import interp1d


# ── Far-field extraction ─────────────────────────────────────────────────────

def extract_farfield(fdtd, monitor_name, ff_res=201, idx_f=1):
    """
    Pull far-field data from a planar monitor via Lumerical eval().

    Parameters:
        fdtd: Lumerical FDTD session
        monitor_name: Name of the monitor (e.g., "side_monitor", "top_monitor")
        ff_res: Far-field ux/uy grid resolution
        idx_f: 1-based frequency index

    Returns:
        dict with E2, ux, uy, lam  or  None on failure.
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
    try:
        fdtd.eval(script)
    except Exception as e:
        print(f"    ERROR in eval [{e}]")
        return None

    E2 = np.squeeze(fdtd.getv("E2"))
    ux = np.squeeze(fdtd.getv("ux"))
    uy = np.squeeze(fdtd.getv("uy"))
    return {"E2": E2, "ux": ux, "uy": uy, "lam": lam}


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
        two_dev_tag = (f"_2pishift_p{pitch_str}_Ygap{gap_nm}nm_Xstag{stag_nm}nm"
                       f"_corr1{d1_nm}nm_corr2{d2_nm}nm{closed_tag}")

    # ── Scatterer (radiation-recycling study): appended only when one is actually
    #    drawn, so all existing file names — and the r=0 in-study control — are
    #    unchanged. Carries radius + position (integer nm; 'm' prefix = minus), so
    #    concurrent array tasks NEVER share a layout/.h5/.mat filename.
    #      _scR{r}_X{x}_Y{y}   cylinder radius / center x / center y in nm
    #      _pair               y-mirrored pair at ±y (the symmetry-compatible form)
    scat_tag = ""
    if getattr(sim, '_has_scatterer', False):
        r_nm = round(sim.scatterer_radius_m * 1e9)
        x_nm = round(sim.scatterer_x_m * 1e9)
        y_nm = round(sim.scatterer_y_m * 1e9)
        x_str = f"m{abs(x_nm)}" if x_nm < 0 else f"{x_nm}"
        y_str = f"m{abs(y_nm)}" if y_nm < 0 else f"{y_nm}"
        scat_tag = f"_scR{r_nm}_X{x_str}_Y{y_str}"
        if getattr(sim, 'scatterer_mirrored_y', False):
            scat_tag += "_pair"

    if use_apod:
        tanh_tag = "_th" if getattr(sim, 'apod_method', 'linear') == 'tanh' else ""
        # Annotate center modulation depth only when it differs from the
        # historical default (100 nm) — keeps pre-sweep filenames stable.
        mod_depth_nm = float(getattr(sim, 'center_mod_depth', 100e-9)) * 1e9
        mod_tag = f"_M{round(mod_depth_nm):.0f}" if abs(mod_depth_nm - 100.0) > 0.5 else ""
        return f"N{N}_A{Napod}{tanh_tag}{mod_tag}{cav_tag}{shift_tag}{fc_tag}{pol_tag}{mat_tag}{wgd_tag}{two_dev_tag}{scat_tag}"
    else:
        return f"N{N}{cav_tag}{shift_tag}{fc_tag}{pol_tag}{mat_tag}{wgd_tag}{two_dev_tag}{scat_tag}"


def apply_monitor_overrides(sim, cfg):
    """
    Override frequency points on large monitors to control memory usage.
    """
    n_2d_pts = cfg.spectral.n_2d_monitor_points

    if sim.record_2d_fields_top_and_cross:
        sim.fdtd.setnamed("field_profile_2D_XY", "frequency points", n_2d_pts)
        sim.fdtd.setnamed("field_profile_2D_YZ_cross", "frequency points", n_2d_pts)
        sim.fdtd.setnamed("field_profile_2D_XZ_side", "frequency points", n_2d_pts)
        print(f"Override: Set 2D monitors to {n_2d_pts} points.")

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
