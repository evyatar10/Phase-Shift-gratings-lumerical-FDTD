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
    # Primary threshold: Transmission < 0.5
    is_stopband = T < 0.5

    # Fallback threshold: if no points < 0.5, try < 0.8
    if not np.any(is_stopband):
        is_stopband = T < 0.8

    # 2. Find Resonance Peak Index
    if np.any(is_stopband):
        # Get all indices where T is within the stopband criteria
        stopband_indices = np.where(is_stopband)[0]

        # Define ROI: Span from the *first* detected stopband point to the *last*
        idx_start = stopband_indices[0]
        idx_end = stopband_indices[-1]

        # Extract ROI (add +1 to end because Python slicing is exclusive)
        T_roi = T[idx_start: idx_end + 1]

        # Find the index of the max value relative to the ROI
        local_peak_idx = np.argmax(T_roi)

        # Convert back to global index
        idx_peak = idx_start + local_peak_idx
    else:
        # Fallback: if no stopband detected at all, take global max
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
                           neff_mat_file=None, neff1_internal=None, neff2_internal=None):
    """
    1. De-embeds feed waveguides.
    2. Removes theoretical carrier phase (Slope Correction).
    3. Tunes Phase to exactly -0.5 * pi (-90 deg).
    """
    # --- A. Standard De-embedding ---
    if neff_mat_file and os.path.exists(neff_mat_file):
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
        # We rely on passed arguments now, not self.fdtd
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
    S11_corr = S11_raw * (corr_factor_1 ** 2)
    S21_corr = S21_raw * corr_factor_1 * corr_factor_2

    # --- B. Slope Correction (Keep Enabled) ---
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
    # 3. Physics Conversion (Conjugate for e^-jwt)
    #Ignore this for now
    #S11 = np.conj(S11)
    #S21 = np.conj(S21)

    # 4. Physics Assumptions
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
    #T22[mask] = 1.0 / S21_c[mask]
    #T21[mask] = -S11[mask] / S21_c[mask]
    #T12[mask] = S22[mask] / S21_c[mask]
    #T11[mask] = S12[mask] - (S11[mask] * S22[mask]) / S21_c[mask]

    # 1. T11 (Coefficient of b2 for a1) = 1/S21
    T11[mask] = 1.0 / S21_c[mask]

    # 2. T12 (Coefficient of a2 for a1) = -S22/S21
    T12[mask] = -S22[mask] / S21_c[mask]

    # 3. T21 (Coefficient of b2 for b1) = S11/S21
    T21[mask] = S11[mask] / S21_c[mask]

    # 4. T22 (Coefficient of a2 for b1) = S12 - (S11*S22)/S21
    T22[mask] = S12[mask] - (S11[mask] * S22[mask]) / S21_c[mask]

    T_matrix = np.array([
        [T11, T12],
        [T21, T22]
    ])

    return R_modal, T_modal, Loss_radiation, T_matrix