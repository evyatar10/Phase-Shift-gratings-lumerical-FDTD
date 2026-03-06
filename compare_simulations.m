% compare_simulations.m
% Compares the field profile and k-space representation of two phase-shift Bragg grating simulations.
% 1. Loads two different .mat simulation results.
% 2. Plots real-space field profiles side-by-side using physical aspect ratios.
% 3. Converts to K-Space and compares absolute dB power profiles at BOTH core and boundary.
% Now does this for both XY (Top View) and YZ (Cross Section at Phase Shift).

clear; close all; clc;

%% --- Configuration ---
% Define the two simulation results to compare
file_sim1 = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles_2\results\result_80_periods_CONST.mat";
label_sim1 = "80 Periods (No Apodization)";

file_sim2 = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles_2\results\result_80_periods_10_apodizations_CONST.mat";
label_sim2 = "80 Periods (10 Apodizations)";

% Global parameters for analysis
params.pitch = 500e-9; % Grating pitch in meters
params.n_clad = 1.44;  % Approximate cladding refractive index

% --- Defect Isolation Option ---
params.isolate_defect = true;    % Set to true to ONLY analyze the radiation near the defect
params.defect_center_x_um = (params.pitch / 4) * 1e6;  % Center X coordinate of the defect (quarter pitch offset)
params.defect_width_um = 30;     % Width of the region to isolate around the defect center (in \mum)
params.window_type = 'hann';     % Window to apply: 'hann', 'tukey', or 'none' (rectangular)
params.boundary_margin_um = 0.2; % Distance to step back from the simulation boundary (in \mum)

%% --- Process Simulations ---

% --- XY Plane (Top View) ---
fprintf('\n=================================\n');
fprintf('--- Analyzing XY Plane (Top View) ---\n');
res1_boundary_xy = extract_sim_data(file_sim1, params, 'boundary', 'XY');
res1_core_xy = extract_sim_data(file_sim1, params, 'core', 'XY');

res2_boundary_xy = extract_sim_data(file_sim2, params, 'boundary', 'XY');
res2_core_xy = extract_sim_data(file_sim2, params, 'core', 'XY');

plot_comparison(res1_boundary_xy, res1_core_xy, res2_boundary_xy, res2_core_xy, label_sim1, label_sim2, 1, 'XY (Top View)', 'Y', 'X');


% --- YZ Plane (Cross Section) ---
fprintf('\n=================================\n');
fprintf('--- Analyzing YZ Plane (Cross Section) ---\n');
res1_boundary_yz = extract_sim_data(file_sim1, params, 'boundary', 'YZ');
res1_core_yz = extract_sim_data(file_sim1, params, 'core', 'YZ');

res2_boundary_yz = extract_sim_data(file_sim2, params, 'boundary', 'YZ');
res2_core_yz = extract_sim_data(file_sim2, params, 'core', 'YZ');

plot_comparison(res1_boundary_yz, res1_core_yz, res2_boundary_yz, res2_core_yz, label_sim1, label_sim2, 2, 'YZ (Cross Section)', 'Z', 'Y');

fprintf('\nComparison Complete.\n');

%% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function plot_comparison(res1_boundary, res1_core, res2_boundary, res2_core, label_sim1, label_sim2, fig_offset, plane_name, axis_name, fft_axis_name)
%% --- Plot 1: Field Profile (Vertical Stack with Physical Aspect Ratio) ---
figure('Name', sprintf('%s Field Profile', plane_name), 'Color', 'w', 'Position', [100+((fig_offset-1)*50) 100 600 800]);
plot_2d_view(res1_boundary, label_sim1, 1, axis_name, fft_axis_name);
plot_2d_view(res2_boundary, label_sim2, 2, axis_name, fft_axis_name);

%% --- Plot 2: K-Space Profile (Overlaid, Absolute dB) ---
figure('Name', sprintf('K-Space Profile Comparison - %s', plane_name), 'Color', 'w', 'Position', [750+((fig_offset-1)*50) 100 800 800]);

k_clad = res1_boundary.k_clad;
k_neff = res1_boundary.k_neff;
xlims_k = [-k_neff*1.3, k_neff*1.3]/1e6;

% 2a. Boundary Extraction
subplot(2, 1, 1);
plot(res1_boundary.kx/1e6, res1_boundary.P_kx_dB, 'b-', 'LineWidth', 1.5, 'DisplayName', label_sim1); hold on;
plot(res2_boundary.kx/1e6, res2_boundary.P_kx_dB, 'r--', 'LineWidth', 1.5, 'DisplayName', label_sim2);

y_max_k1 = max([max(res1_boundary.P_kx_dB), max(res2_boundary.P_kx_dB)]);
y_min_k1 = y_max_k1 - 60;
y_lims_k1 = [y_min_k1, y_max_k1 + 5];
ylim(y_lims_k1); xlim(xlims_k);

patch([-k_clad, k_clad, k_clad, -k_clad]/1e6, [y_lims_k1(1) y_lims_k1(1) y_lims_k1(2) y_lims_k1(2)], ...
    'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Radiation Region (|K_| < k_{clad})');
xline(k_clad/1e6,  'k:', 'k_{clad}', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-k_clad/1e6, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(k_neff/1e6,  'k--', 'k_{neff}', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-k_neff/1e6, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel(sprintf('Wavevector K_%s (rad / \\mum)', lower(fft_axis_name))); ylabel('Absolute K-Space Power [dB]');
title(sprintf('K-Space Profile at BOUNDARY (%s=%.2f \\mum) - %s', axis_name, res1_boundary.ext_val_um, plane_name));
legend('Location', 'northeast'); grid on; hold off;

% 2b. Core Extraction
subplot(2, 1, 2);
plot(res1_core.kx/1e6, res1_core.P_kx_dB, 'b-', 'LineWidth', 1.5, 'DisplayName', label_sim1); hold on;
plot(res2_core.kx/1e6, res2_core.P_kx_dB, 'r--', 'LineWidth', 1.5, 'DisplayName', label_sim2);

y_max_k2 = max([max(res1_core.P_kx_dB), max(res2_core.P_kx_dB)]);
y_min_k2 = y_max_k2 - 60;
y_lims_k2 = [y_min_k2, y_max_k2 + 5];
ylim(y_lims_k2); xlim(xlims_k);

patch([-k_clad, k_clad, k_clad, -k_clad]/1e6, [y_lims_k2(1) y_lims_k2(1) y_lims_k2(2) y_lims_k2(2)], ...
    'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Radiation Region (|K_| < k_{clad})');
xline(k_clad/1e6,  'k:', 'k_{clad}', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-k_clad/1e6, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(k_neff/1e6,  'k--', 'k_{neff}', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-k_neff/1e6, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel(sprintf('Wavevector K_%s (rad / \\mum)', lower(fft_axis_name))); ylabel('Absolute K-Space Power [dB]');
title(sprintf('K-Space Profile at CORE (%s=%.2f \\mum) - %s', axis_name, res1_core.ext_val_um, plane_name));
legend('Location', 'northeast'); grid on; hold off;

%% --- Plot 3: Field Profile vs Theta Core (Overlaid, Absolute dB) ---
figure('Name', sprintf('Profile vs Theta Core Comparison - %s', plane_name), 'Color', 'w', 'Position', [800+((fig_offset-1)*50) 150 800 800]);

theta_tir = res1_boundary.theta_TIR_eff_deg;
label_tir = sprintf('\\theta_{critical} = %.1f^\\circ', theta_tir);
xlims_th = [-90 90];

% 3a. Boundary Extraction
subplot(2, 1, 1);
plot(res1_boundary.theta_core_deg, res1_boundary.P_kx_core_dB, 'b-', 'LineWidth', 1.5, 'DisplayName', label_sim1); hold on;
plot(res2_boundary.theta_core_deg, res2_boundary.P_kx_core_dB, 'r--', 'LineWidth', 1.5, 'DisplayName', label_sim2);

y_max_th1 = max([max(res1_boundary.P_kx_core_dB), max(res2_boundary.P_kx_core_dB)]);
ylim([y_max_th1 - 60, y_max_th1 + 5]); xlim(xlims_th);

xline(theta_tir, 'k:', label_tir, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-theta_tir, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel(sprintf('\\theta_{core} = arcsin(k_%s / k_{neff}) [degrees]', lower(fft_axis_name))); ylabel('Absolute K-Space Power [dB]');
title(sprintf('Theta Core Profile at BOUNDARY (%s=%.2f \\mum) - %s', axis_name, res1_boundary.ext_val_um, plane_name));
legend('Location', 'northeast'); grid on; hold off;

% 3b. Core Extraction
subplot(2, 1, 2);
plot(res1_core.theta_core_deg, res1_core.P_kx_core_dB, 'b-', 'LineWidth', 1.5, 'DisplayName', label_sim1); hold on;
plot(res2_core.theta_core_deg, res2_core.P_kx_core_dB, 'r--', 'LineWidth', 1.5, 'DisplayName', label_sim2);

y_max_th2 = max([max(res1_core.P_kx_core_dB), max(res2_core.P_kx_core_dB)]);
ylim([y_max_th2 - 60, y_max_th2 + 5]); xlim(xlims_th);

xline(theta_tir, 'k:', label_tir, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-theta_tir, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel(sprintf('\\theta_{core} = arcsin(k_%s / k_{neff}) [degrees]', lower(fft_axis_name))); ylabel('Absolute K-Space Power [dB]');
title(sprintf('Theta Core Profile at CORE (%s=%.2f \\mum) - %s', axis_name, res1_core.ext_val_um, plane_name));
legend('Location', 'northeast'); grid on; hold off;
end

function res = extract_sim_data(filepath, params, extract_pos, plane)
if nargin < 4; plane = 'XY'; end

if ~exist(filepath, 'file')
    error('File not found! %s', filepath);
end
data = load(filepath);

% Find Resonance
T = squeeze(data.T); wl = squeeze(data.wl_m);
stopband_indices = find(T < 0.6);
if isempty(stopband_indices); stopband_indices = find(T < 0.85); end
if isempty(stopband_indices)
    [~, idx_peak] = max(T);
else
    idx_start = stopband_indices(1); idx_end = stopband_indices(end);
    [~, local_peak_idx] = max(T(idx_start:idx_end));
    idx_peak = idx_start + local_peak_idx - 1;
end
wl_res = wl(idx_peak);

if ~isfield(data, 'field_xy') || ~isfield(data, 'field_yz_cross')
    error('Older simulation format detected. Please re-run the Python simulation script first.');
end

% Extraction Y/Z
if strcmp(plane, 'XY')
    d_xy = data.field_xy;
    x = double(d_xy.x); y = double(d_xy.y); z = double(d_xy.z);
    if isfield(d_xy, 'lambda_3d'); lam = double(d_xy.lambda_3d); else; lam = 1.55e-6; end
    [~, idx_lam] = min(abs(lam - wl_res)); wl_actual = lam(idx_lam);

    Nx = length(x); Ny = length(y); Nz = length(z); Nlam = length(lam);
    E_5D = reshape(d_xy.E_res, [Nx, Ny, Nz, Nlam, 3]);

    [~, idx_z0] = min(abs(z));

    if strcmp(extract_pos, 'boundary')
        [y_max_m, ~] = max(abs(y));
        boundary_dist_m = y_max_m - (params.boundary_margin_um * 1e-6);
        [~, idx_ext] = min(abs(y - boundary_dist_m));
    else % 'core'
        [~, idx_ext] = min(abs(y));
    end
    E_core_comp = squeeze(E_5D(:, idx_ext, idx_z0, idx_lam, :));
    E_plane_comp = squeeze(E_5D(:, :, idx_z0, idx_lam, :));
    fft_axis = x;
    pos_axis = y;
else % 'YZ'
    d_yz = data.field_yz_cross;
    x = double(d_yz.x); y = double(d_yz.y); z = double(d_yz.z);
    if isfield(d_yz, 'lambda_3d'); lam = double(d_yz.lambda_3d); else; lam = 1.55e-6; end
    [~, idx_lam] = min(abs(lam - wl_res)); wl_actual = lam(idx_lam);

    Nx = length(x); Ny = length(y); Nz = length(z); Nlam = length(lam);
    E_5D = reshape(d_yz.E_res, [Nx, Ny, Nz, Nlam, 3]);

    [~, idx_x0] = min(abs(x)); % Since it is cross section, index is simply minimum/only point

    if strcmp(extract_pos, 'boundary')
        [z_max_m, ~] = max(abs(z));
        boundary_dist_m = z_max_m - (params.boundary_margin_um * 1e-6);
        [~, idx_ext] = min(abs(z - boundary_dist_m));
    else % 'core'
        [~, idx_ext] = min(abs(z));
    end
    E_core_comp = squeeze(E_5D(idx_x0, :, idx_ext, idx_lam, :)); % Line along Y
    E_plane_comp = squeeze(E_5D(idx_x0, :, :, idx_lam, :)); % Plane Y vs Z
    fft_axis = y;
    pos_axis = z;
end

% Flatten E_plane_comp
if size(E_plane_comp, ndims(E_plane_comp)) == 3
    E_sq_plane = sum(abs(E_plane_comp).^2, ndims(E_plane_comp));
else
    % Fallback
    E_sq_plane = sum(abs(E_plane_comp).^2, 2);
end
E_sq_plane_dB = 10 * log10(E_sq_plane);

% Windowing
N_fft = length(fft_axis);
if strcmp(plane, 'XY') && params.isolate_defect
    center_m = params.defect_center_x_um * 1e-6;
    half_width_m = (params.defect_width_um / 2) * 1e-6;
    valid_idx = abs(fft_axis - center_m) <= half_width_m;
    window = zeros(N_fft, 1); num_valid = sum(valid_idx);
    if num_valid > 0
        switch lower(params.window_type)
            case 'hann'; window(valid_idx) = hann(num_valid);
            case 'tukey'; window(valid_idx) = tukeywin(num_valid, 0.3);
            otherwise; window(valid_idx) = 1;
        end
    else; window = ones(N_fft, 1); end
else
    % In YZ, we typically analyze the full non-propagation extent without window isolation of a "defect"
    switch lower(params.window_type)
        case 'hann'; window = hann(N_fft);
        case 'tukey'; window = tukeywin(N_fft, 0.3);
        otherwise; window = ones(N_fft, 1);
    end
end

% FFT
E_xw = E_core_comp(:,1) .* window; E_yw = E_core_comp(:,2) .* window; E_zw = E_core_comp(:,3) .* window;
dx = mean(diff(fft_axis)); N_pad = N_fft * 2;
kx = (-N_pad/2 : N_pad/2 - 1)' * (2*pi / (N_pad * dx));

fft_Ex = fftshift(fft(E_xw, N_pad)); fft_Ey = fftshift(fft(E_yw, N_pad)); fft_Ez = fftshift(fft(E_zw, N_pad));

% Calculate K-Space Absolute Power & dB Scale
P_kx = abs(fft_Ex).^2 + abs(fft_Ey).^2 + abs(fft_Ez).^2;
P_kx_dB = 10 * log10(P_kx);

% Physics
k0 = 2*pi / wl_actual; n_eff = wl_actual / (2 * params.pitch);
k_neff = k0 * n_eff; k_clad = k0 * params.n_clad;
theta_TIR_eff_deg = asind(k_clad / k_neff);

% Angles (Theta Core ONLY)
idx_core = abs(kx) <= k_neff;
theta_core_deg = asind(kx(idx_core) / k_neff);
P_kx_core_dB = P_kx_dB(idx_core);

% Store in struct
res.params = params; res.wl_nm = wl_actual * 1e9;
res.fft_axis = fft_axis; res.pos_axis = pos_axis; res.ext_val_um = pos_axis(idx_ext)*1e6;
res.E_sq_plane_dB = E_sq_plane_dB;
res.kx = kx; res.P_kx_dB = P_kx_dB;
res.k_clad = k_clad; res.k_neff = k_neff; res.theta_TIR_eff_deg = theta_TIR_eff_deg;
res.theta_core_deg = theta_core_deg; res.P_kx_core_dB = P_kx_core_dB;
end

function plot_2d_view(res, title_str, subplot_idx, axis_name, fft_axis_name)
subplot(2, 1, subplot_idx);
imagesc(res.fft_axis*1e6, res.pos_axis*1e6, res.E_sq_plane_dB');
set(gca, 'YDir', 'normal');
colormap(jet);

max_dB = max(res.E_sq_plane_dB(:));
try clim([max_dB - 80, max_dB]); catch; caxis([max_dB - 80, max_dB]); end

cb = colorbar; ylabel(cb, '10*log_{10}(|E|^2) [dB]');
xlabel(sprintf('Position %s [\\mum]', fft_axis_name)); ylabel(sprintf('Position %s [\\mum]', axis_name));
title(sprintf('%s (\\lambda = %.3f nm)', title_str, res.wl_nm));

hold on;
ext_um = res.ext_val_um;

% For XY plane, we use isolation. For YZ plane, we draw across full axis.
if res.params.isolate_defect && size(res.fft_axis,1) > 1000 % Crude check to see if it's the long X axis
    x_st = res.params.defect_center_x_um - res.params.defect_width_um / 2;
    x_en = res.params.defect_center_x_um + res.params.defect_width_um / 2;
else
    x_st = min(res.fft_axis*1e6);
    x_en = max(res.fft_axis*1e6);
end

% Extract midpoint for label alignment
x_mid = (x_st + x_en) / 2;

% Draw boundary line
plot([x_st, x_en], [ext_um, ext_um], 'w--', 'LineWidth', 1.5);
plot(x_st, ext_um, 'w<', 'MarkerFaceColor', 'w', 'MarkerSize', 8);
plot(x_en, ext_um, 'w>', 'MarkerFaceColor', 'w', 'MarkerSize', 8);
text(x_mid, ext_um - 0.15, 'boundary', 'Color', 'w', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% Draw core line (at 0) too
plot([x_st, x_en], [0, 0], 'w--', 'LineWidth', 1.5);
plot(x_st, 0, 'w<', 'MarkerFaceColor', 'w', 'MarkerSize', 8);
plot(x_en, 0, 'w>', 'MarkerFaceColor', 'w', 'MarkerSize', 8);
text(x_mid, 0 - 0.15, 'core', 'Color', 'w', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
hold off;
end
