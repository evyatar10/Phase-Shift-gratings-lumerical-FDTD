% analyze_core_k_space.m
% Analyzes the field profile in the core of a phase-shift Bragg grating
% 1. Finds resonance inside the bandgap.
% 2. Plots the absolute value squared of the electric field |E|^2.
% 3. Converts the field profile to k-space using FFT.
% 4. Highlights radiation region (|kx| < k_clad) vs decaying bound region (|kx| > k_clad).
% Evaluates BOTH XY (Top View) and YZ (Cross Section at Phase Shift).

addpath(fileparts(fileparts(mfilename('fullpath'))));  % Add project root to MATLAB path
clear; clc;
%close all;

%% --- Configuration ---
% Update this to the actual path of your simulation results
result_filepath = "C:\Users\evyat\Lumerical\new_experiment_comparison\p8rc1_tanh\results\result_80_periods_CONST.mat";

pitch = 500e-9; % Grating pitch in meters
n_clad = 1.44;  % Approximate cladding refractive index

% --- Defect Isolation Option ---
isolate_defect = true;    % Set to true to ONLY analyze the radiation near the defect
defect_center_x_um = (pitch / 4) * 1e6;  % Center X coordinate of the defect (quarter pitch offset)
defect_width_um = 30;     % Width of the region to isolate around the defect center (in \mum)
window_type = 'hann';     % Window to apply: 'hann', 'tukey', or 'none' (rectangular)

% --- Extraction Position Option ---
% 'core'     : Evaluates K-space at the center of the core (Y=0 or Z=0)
% 'boundary' : Evaluates K-space at the edge of the simulation domain (far cladding)
extraction_position = 'boundary';
boundary_margin_um = 0.2; % Distance to step back from the simulation boundary (in \mum)

%% --- 1. Load Data & Find Resonance ---
fprintf('Loading data...\n');
if ~exist(result_filepath, 'file')
    error('File not found! Please check result_filepath.');
end
data = load(result_filepath);

if isfield(data, 'T') && isfield(data, 'wl_m')
    T = squeeze(data.T);
    wl = squeeze(data.wl_m);

    % Find resonance inside the bandgap
    stopband_indices = find(T < 0.6);
    if isempty(stopband_indices)
        stopband_indices = find(T < 0.85);
    end

    if isempty(stopband_indices)
        [~, idx_peak] = max(T);
        fprintf('Warning: Stopband not detected. Using global maximum.\n');
    else
        idx_start = stopband_indices(1);
        idx_end = stopband_indices(end);
        [~, local_peak_idx] = max(T(idx_start:idx_end));
        idx_peak = idx_start + local_peak_idx - 1;
    end

    wl_res = wl(idx_peak);
    fprintf('Resonance detected inside bandgap at %.3f nm (T = %.3f)\n', wl_res*1e9, T(idx_peak));
else
    error('Transmission or Wavelength data missing.');
end

%% --- 2. Extract Field ---
if ~isfield(data, 'field_xy') || ~isfield(data, 'field_yz_cross')
    error('Older simulation format detected (missing field_xy/field_yz_cross). Please re-run the Python simulation script first.');
end

d_xy = data.field_xy;
d_yz = data.field_yz_cross;

% For XY plane variables
x_xy = double(d_xy.x); y_xy = double(d_xy.y); z_xy = double(d_xy.z);
if isfield(d_xy, 'lambda_3d'); lam_xy = double(d_xy.lambda_3d);
else; lam_xy = 1.55e-6; end

% Nearest wavelength to resonance
[~, idx_lam] = min(abs(lam_xy - wl_res));
wl_3d_actual = lam_xy(idx_lam);

% Core center indices for XY
[~, idx_z0_xy] = min(abs(z_xy));

% Shape the E fields from Python [Nx, Ny, Nz, Nlam, 3] -> Squeezing out singletons handles 2D automatically
% But to be safe and robust, we explicitly reshape based on axis lengths
Nx_xy = length(x_xy); Ny_xy = length(y_xy); Nz_xy = length(z_xy); Nlam = length(lam_xy);
E_5D_xy = reshape(d_xy.E_res, [Nx_xy, Ny_xy, Nz_xy, Nlam, 3]);

% For YZ plane variables
x_yz = double(d_yz.x); y_yz = double(d_yz.y); z_yz = double(d_yz.z);
Nx_yz = length(x_yz); Ny_yz = length(y_yz); Nz_yz = length(z_yz);
E_5D_yz = reshape(d_yz.E_res, [Nx_yz, Ny_yz, Nz_yz, Nlam, 3]);


planes_to_analyze = {'XY', 'YZ'};

for p_idx = 1:length(planes_to_analyze)
    plane = planes_to_analyze{p_idx};
    fprintf('\n=================================\n');
    fprintf('--- Analyzing %s Plane ---\n', plane);

    % Determine slice index based on option
    if strcmp(plane, 'XY')
        if strcmp(extraction_position, 'boundary')
            [y_max_m, ~] = max(abs(y_xy));
            boundary_y_m = y_max_m - (boundary_margin_um * 1e-6);
            [~, idx_ext] = min(abs(y_xy - boundary_y_m));
        else
            [~, idx_ext] = min(abs(y_xy));
        end
        fprintf('Extracting 1D field profile along propagation axis (X) at Y=%.3fum, Z=%.3fum\n', y_xy(idx_ext)*1e6, z_xy(idx_z0_xy)*1e6);
        E_core_comp = squeeze(E_5D_xy(:, idx_ext, idx_z0_xy, idx_lam, :));
        E_plane_comp = squeeze(E_5D_xy(:, :, idx_z0_xy, idx_lam, :));
        fft_axis = x_xy;
        pos_axis = y_xy;
        pos_name = 'Y';
        fft_name = 'X';
        ext_val_um = y_xy(idx_ext)*1e6;
        fig_offset = 0;
    else % 'YZ'
        % The YZ monitor is ALREADY sliced at X = defect_center_x_um by Python.
        % Its size is [1, Ny, Nz]
        [~, idx_y0_yz] = min(abs(y_yz));
        [~, idx_x0_yz] = min(abs(x_yz)); % Should just be index 1

        if strcmp(extraction_position, 'boundary')
            [z_max_m, ~] = max(abs(z_yz));
            boundary_z_m = z_max_m - (boundary_margin_um * 1e-6);
            [~, idx_ext] = min(abs(z_yz - boundary_z_m));
        else
            [~, idx_ext] = min(abs(z_yz));
        end
        fprintf('Extracting 1D field profile along (Y) at X=%.3fum, Z=%.3fum\n', x_yz(idx_x0_yz)*1e6, z_yz(idx_ext)*1e6);
        E_core_comp = squeeze(E_5D_yz(idx_x0_yz, :, idx_ext, idx_lam, :));
        E_plane_comp = squeeze(E_5D_yz(idx_x0_yz, :, :, idx_lam, :));
        fft_axis = y_yz;
        pos_axis = z_yz;
        pos_name = 'Z';
        fft_name = 'Y';
        ext_val_um = z_yz(idx_ext)*1e6;
        fig_offset = 50;
    end

    % --- Plot 1: 2D Field Profile ---
    % Flatten E_plane_comp for plot
    if ndims(E_plane_comp) >= 3
        E_sq_plane = sum(abs(E_plane_comp).^2, ndims(E_plane_comp));
    else
        E_sq_plane = sum(abs(E_plane_comp).^2, 2);
    end
    E_sq_plane_dB = 10 * log10(E_sq_plane); % Absolute dB, not normalized

    figure('Name', sprintf('%s View Field Profile', plane), 'Color', 'w', 'Position', [150+fig_offset 150+fig_offset 800 400]);
    imagesc(fft_axis*1e6, pos_axis*1e6, E_sq_plane_dB');
    set(gca, 'YDir', 'normal');
    colormap(jet);

    max_dB = max(E_sq_plane_dB(:));
    dB_limit = 80;
    try clim([max_dB - dB_limit, max_dB]); catch; caxis([max_dB - dB_limit, max_dB]); end

    cb = colorbar;
    ylabel(cb, '10*log_{10}(|E|^2) [dB]');
    xlabel(sprintf('Position %s [\\mum]', fft_name));
    ylabel(sprintf('Position %s [\\mum]', pos_name));
    title(sprintf('%s View: Field Profile at \\lambda = %.3f nm', plane, wl_3d_actual*1e9));

    hold on;
    N_fft = length(fft_axis);

    if isolate_defect && strcmp(plane, 'XY')
        x_st = defect_center_x_um - defect_width_um / 2;
        x_en = defect_center_x_um + defect_width_um / 2;
        plot([x_st, x_en], [ext_val_um, ext_val_um], 'w--', 'LineWidth', 1.5);
        text(x_en, ext_val_um - 0.15, sprintf(' Extraction %s=%.2f \\mum', pos_name, ext_val_um), 'Color', 'w', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', 'FontWeight', 'bold');
    else
        yline(ext_val_um, 'w--', sprintf('Extraction %s=%.2f \\mum', pos_name, ext_val_um), 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
    end
    hold off;


    %% --- 3. K-Space Transform (Spatial FFT) ---
    if isolate_defect && strcmp(plane, 'XY')
        fprintf('Isolating phase-shift defect at X=%.3f \\mum (Analysis Width: %g \\mum)\n', defect_center_x_um, defect_width_um);
        center_m = defect_center_x_um * 1e-6;
        half_width_m = (defect_width_um / 2) * 1e-6;
        valid_idx = abs(fft_axis - center_m) <= half_width_m;

        window = zeros(N_fft, 1);
        num_valid = sum(valid_idx);
        if num_valid > 0
            switch lower(window_type)
                case 'hann'
                    window(valid_idx) = hann(num_valid);
                    fprintf('Applying Hann window.\n');
                case 'tukey'
                    window(valid_idx) = tukeywin(num_valid, 0.3);
                    fprintf('Applying Tukey window (0.3 taper).\n');
                case 'none'
                    window(valid_idx) = 1;
                    fprintf('Applying no window (Rectangular).\n');
                otherwise
                    window(valid_idx) = 1;
            end
        else
            window = ones(N_fft, 1);
        end
    else
        fprintf('Analyzing entire axis length.\n');
        switch lower(window_type)
            case 'hann'; window = hann(N_fft);
            case 'tukey'; window = tukeywin(N_fft, 0.3);
            otherwise;   window = ones(N_fft, 1);
        end
    end

    E_xw = E_core_comp(:,1) .* window;
    E_yw = E_core_comp(:,2) .* window;
    E_zw = E_core_comp(:,3) .* window;

    dx = mean(diff(fft_axis));
    zoom_factor = 2;
    N_pad = N_fft * zoom_factor;
    kx = (-N_pad/2 : N_pad/2 - 1)' * (2*pi / (N_pad * dx));

    fft_Ex = fftshift(fft(E_xw, N_pad));
    fft_Ey = fftshift(fft(E_yw, N_pad));
    fft_Ez = fftshift(fft(E_zw, N_pad));

    % K-space Power Profile (Absolute Magnitude)
    P_kx = abs(fft_Ex).^2 + abs(fft_Ey).^2 + abs(fft_Ez).^2;
    P_kx_dB = 10 * log10(P_kx);

    %% --- 4. Physics Threshold Calculations ---
    k0 = 2*pi / wl_3d_actual;
    n_eff = wl_3d_actual / (2 * pitch);
    k_neff = k0 * n_eff;
    k_clad = k0 * n_clad;
    theta_TIR_eff_deg = asind(k_clad / k_neff);

    if p_idx == 1 % Only print once
        fprintf('\n--- Wavevector Analysis ---\n');
        fprintf('Calculated n_eff from Bragg condition: %.4f\n', n_eff);
        fprintf('Effective bound wavevector (k_neff): %.2e rad/m\n', k_neff);
        fprintf('Cladding boundary wavevector (k_clad): %.2e rad/m\n', k_clad);
        fprintf('TIR Critical Angle (theta_c): %.2f degrees from normal\n\n', theta_TIR_eff_deg);
    end

    % --- Plot 2: K-Space Profile ---
    figure('Name', sprintf('K-Space Profile - %s', plane), 'Color', 'w', 'Position', [150+fig_offset 200+fig_offset 800 500]);
    plot(kx/1e6, P_kx_dB, 'b-', 'LineWidth', 1.5, 'DisplayName', 'FFT Power Spectrum');
    hold on;

    y_max = max(P_kx_dB);
    y_min = y_max - 40;
    y_lims = [y_min, y_max + 5];
    ylim(y_lims);

    patch([-k_clad, k_clad, k_clad, -k_clad]/1e6, [y_lims(1) y_lims(1) y_lims(2) y_lims(2)], ...
        'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Radiation Region (|K| < k_{clad})');

    xline(k_clad/1e6, 'r--', 'k_{clad} (Radiation Boundary)', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
    xline(-k_clad/1e6, 'r--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    xline(k_neff/1e6, 'k:', 'k_{neff} (Forward Bound Mode)', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
    xline(-k_neff/1e6, 'k:', 'k_{neff} (Backward Bound Mode)', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');

    xlabel(sprintf('Wavevector K_%s (rad / \\mum)', lower(fft_name)));
    ylabel('Absolute K-Space Power [dB]');
    title(sprintf('1D K-Space Field Profile at %s (%s=%.2f \\mum) - %s', extraction_position, pos_name, ext_val_um, plane));
    xlim([-k_neff*1.3, k_neff*1.3]/1e6);
    legend('Location', 'northeast');
    grid on; hold off;

    %% --- 5. Radiation Angle Plots ---
    idx_clad = abs(kx) <= k_clad;
    theta_cladding_deg = asind(kx(idx_clad) / k_clad);
    P_kx_dB_clad = P_kx_dB(idx_clad);

    % --- Plot 3: Field Profile vs Theta Cladding ---
    figure('Name', sprintf('Profile vs Theta Cladding - %s', plane), 'Color', 'w', 'Position', [150+fig_offset 250+fig_offset 700 400]);
    plot(theta_cladding_deg, P_kx_dB_clad, 'k-', 'LineWidth', 1.5);
    xlabel(sprintf('\\theta_{cladding} = arcsin(k_%s / k_{clad}) [degrees]', lower(fft_name)));
    ylabel('Absolute K-Space Power [dB]');
    title(sprintf('K-Space Power vs \\theta_{cladding} (%s)', plane));
    ylim(y_lims);
    xlim([-90 90]);
    grid on;

    idx_core = abs(kx) <= k_neff;
    theta_core_deg = asind(kx(idx_core) / k_neff);
    P_kx_dB_core = P_kx_dB(idx_core);

    % --- Plot 4: Field Profile vs Theta Core ---
    figure('Name', sprintf('Profile vs Theta Core - %s', plane), 'Color', 'w', 'Position', [200+fig_offset 300+fig_offset 750 450]);
    plot(theta_core_deg, P_kx_dB_core, 'b-', 'LineWidth', 1.5);
    hold on;
    label_str = sprintf('\\theta_{critical} = %.1f^\\circ', theta_TIR_eff_deg);
    xline(theta_TIR_eff_deg, 'r--', label_str, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
    xline(-theta_TIR_eff_deg, 'r--', 'LineWidth', 1.5);
    hold off;

    xlabel(sprintf('\\theta_{core} = arcsin(k_%s / k_{neff}) [degrees]', lower(fft_name)));
    ylabel('Absolute K-Space Power [dB]');
    title(sprintf('K-Space Power vs \\theta_{core} (%s)', plane));
    ylim(y_lims);
    xlim([-90 90]);
    grid on;
end

fprintf('\nAnalysis Complete. Check figures for real space, k-space, and angle mapping.\n');
