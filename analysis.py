import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
import os


def align_phases_at_resonance_peak(wl, S11, S21, target_phase=0.5 * np.pi):
    """
    Aligns phases by finding the resonance peak using a "Stopband Logic".
    """
    # Calculate Transmission scalar
    T = np.abs(S21) ** 2

    # 1. Define Stopband Mask
    is_stopband = T < 0.5
    if not np.any(is_stopband):
        is_stopband = T < 0.8

    # 2. Find Resonance Peak Index
    if np.any(is_stopband):
        stopband_indices = np.where(is_stopband)[0]
        idx_start = stopband_indices[0]
        idx_end = stopband_indices[-1]
        T_roi = T[idx_start: idx_end + 1]
        local_peak_idx = np.argmax(T_roi)
        idx_peak = idx_start + local_peak_idx
    else:
        idx_peak = np.argmax(T)

    # 3. Get Phase at that specific Peak
    current_phase_val = np.angle(S21[idx_peak])

    # 4. Calculate Correction Phasor
    delta_phase = target_phase - current_phase_val
    correction_phasor = np.exp(1j * delta_phase)

    # 5. Apply Correction to S-parameters
    S11_corrected = S11 * correction_phasor
    S21_corrected = S21 * correction_phasor

    return S11_corrected, S21_corrected


def apply_phase_correction(wl, S11_raw, S21_raw, pitch, dist_grating_to_port, x_grating_end,
                           neff_mat_file=None, neff1_internal=None, neff2_internal=None,
                           use_single_neff=False, single_neff_val=None,
                           do_length_correction=True, do_envelope_correction=True):
    """
    1. De-embeds feed waveguides (Controlled by do_length_correction).
    2. Removes theoretical carrier phase (Slope Correction) AND
    3. Tunes Phase to exactly -0.5 * pi (-90 deg) (Both controlled by do_envelope_correction).

    UPDATED: Now supports independent control of length and envelope corrections.
    """
    S11_corr = S11_raw.copy()
    S21_corr = S21_raw.copy()

    # --- A. Standard De-embedding (Length Correction) ---
    if do_length_correction:
        if use_single_neff:
            # For Constant Materials: Use the single scalar value provided
            if single_neff_val is None:
                raise ValueError("use_single_neff is True, but single_neff_val is None.")
            neff1 = single_neff_val
            neff2 = single_neff_val

        elif neff_mat_file and os.path.exists(neff_mat_file):
            print(f"Loading external neff data from: {neff_mat_file}")
            mat_data = sio.loadmat(neff_mat_file)
            wl_fde = np.squeeze(mat_data['wavelengths'])
            neff_fde = np.squeeze(mat_data['neff_complex'])
            interp_real = interp1d(wl_fde, np.real(neff_fde), kind='linear', fill_value="extrapolate")
            interp_imag = interp1d(wl_fde, np.imag(neff_fde), kind='linear', fill_value="extrapolate")
            neff_interp = interp_real(wl) + 1j * interp_imag(wl)
            neff1 = neff_interp
            neff2 = neff_interp
        else:
            print("Using FDTD Port neff (internal) for de-embedding.")
            if neff1_internal is None or neff2_internal is None:
                raise ValueError("Internal neff data required but not provided.")
            neff1 = neff1_internal
            neff2 = neff2_internal

        k0 = 2 * np.pi / wl
        L_feed = dist_grating_to_port

        beta1 = k0 * np.real(neff1)
        beta2 = k0 * np.real(neff2)
        corr_factor_1 = np.exp(-1j * beta1 * L_feed)
        corr_factor_2 = np.exp(-1j * beta2 * L_feed)
        S11_corr = S11_corr * (corr_factor_1 ** 2)
        S21_corr = S21_corr * corr_factor_1 * corr_factor_2

    # --- B & C. Slope Correction & Target Phase Correction ---
    if do_envelope_correction:
        # --- B. Slope Correction ---
        beta_0 = np.pi / pitch
        device_len_m = 2.0 * x_grating_end
        slope_correction = np.exp(-1j * beta_0 * device_len_m)
        S21_corr = S21_corr * slope_correction

        # --- C. Target Phase Correction (-PI/2) ---
        S11_corr, S21_corr = align_phases_at_resonance_peak(
            wl, S11_corr, S21_corr, target_phase=0.5 * np.pi
        )

    return S11_corr, S21_corr


def calculate_physics_matrices(S11, S21):
    """
    Calculates T-matrix, R, T, and Loss from S-parameters.
    """
    S12 = S21  # Reciprocity
    S22 = S11  # Symmetry

    R_modal = np.abs(S11) ** 2
    T_modal = np.abs(S21) ** 2
    Loss_radiation = 1.0 - R_modal - T_modal

    S21_c = S21.astype(complex)
    T11 = np.zeros_like(S11, dtype=complex)
    T12 = np.zeros_like(S11, dtype=complex)
    T21 = np.zeros_like(S11, dtype=complex)
    T22 = np.zeros_like(S11, dtype=complex)

    mask = np.abs(S21_c) > 1e-15

    # T-Matrix (Left-to-Right Propagator)
    # T11 = 1/S21
    T11[mask] = 1.0 / S21_c[mask]
    # T12 = -S22/S21
    T12[mask] = -S22[mask] / S21_c[mask]
    # T21 = S11/S21
    T21[mask] = S11[mask] / S21_c[mask]
    # T22 = S12 - (S11*S22)/S21
    T22[mask] = S12[mask] - (S11[mask] * S22[mask]) / S21_c[mask]

    T_matrix = np.array([
        [T11, T12],
        [T21, T22]
    ])

    return R_modal, T_modal, Loss_radiation, T_matrix