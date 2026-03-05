% compare_simulations.m
% Compares the field profile and k-space representation of two phase-shift Bragg grating simulations.
% 1. Loads two different .mat simulation results.
% 2. Plots top-view real-space field profiles side-by-side using physical aspect ratios.
% 3. Converts to K-Space and compares absolute dB power profiles at BOTH core and boundary.

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
fprintf('--- Processing Simulation 1: %s ---\n', label_sim1);
res1_boundary = extract_sim_data(file_sim1, params, 'boundary');
res1_core = extract_sim_data(file_sim1, params, 'core');

fprintf('\n--- Processing Simulation 2: %s ---\n', label_sim2);
res2_boundary = extract_sim_data(file_sim2, params, 'boundary');
res2_core = extract_sim_data(file_sim2, params, 'core');

%% --- Plot 1: Top View Field Profile (Vertical Stack with Physical Aspect Ratio) ---
% Using the boundary results just for the top-view spatial data (they share the same 3D spatial field)
figure('Name', 'Top View Comparison (80 periods vs Apodized)', 'Color', 'w', 'Position', [100 100 600 800]);

plot_top_view(res1_boundary, label_sim1, 1);
plot_top_view(res2_boundary, label_sim2, 2);

%% --- Plot 2: K-Space Profile (Overlaid, Absolute dB) ---
figure('Name', 'K-Space Profile Comparison (Boundary vs Core)', 'Color', 'w', 'Position', [750 100 800 800]);

% K-Space boundary region parameters (identical for sim 1 & 2)
k_clad = res1_boundary.k_clad;
k_neff = res1_boundary.k_neff;
xlims_k = [-k_neff*1.3, k_neff*1.3]/1e6;

% 2a. Boundary Extraction
subplot(2, 1, 1);
plot(res1_boundary.kx/1e6, res1_boundary.P_kx_dB, 'b-', 'LineWidth', 1.5, 'DisplayName', label_sim1); hold on;
plot(res2_boundary.kx/1e6, res2_boundary.P_kx_dB, 'r--', 'LineWidth', 1.5, 'DisplayName', label_sim2);

y_max_k1 = max([max(res1_boundary.P_kx_dB), max(res2_boundary.P_kx_dB)]);
y_min_k1 = y_max_k1 - 60; % Plot down 60 dB
y_lims_k1 = [y_min_k1, y_max_k1 + 5];
ylim(y_lims_k1); xlim(xlims_k);

patch([-k_clad, k_clad, k_clad, -k_clad]/1e6, [y_lims_k1(1) y_lims_k1(1) y_lims_k1(2) y_lims_k1(2)], ...
    'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Radiation Region (|K_x| < k_{clad})');
xline(k_clad/1e6,  'k:', 'k_{clad}', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-k_clad/1e6, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(k_neff/1e6,  'k--', 'k_{neff}', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-k_neff/1e6, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('Wavevector K_x (rad / \mum)'); ylabel('Absolute K-Space Power [dB]');
title(sprintf('K-Space Profile at BOUNDARY (Y=%.2f \\mum)', res1_boundary.y_extract_um));
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
    'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Radiation Region (|K_x| < k_{clad})');
xline(k_clad/1e6,  'k:', 'k_{clad}', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-k_clad/1e6, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(k_neff/1e6,  'k--', 'k_{neff}', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-k_neff/1e6, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('Wavevector K_x (rad / \mum)'); ylabel('Absolute K-Space Power [dB]');
title(sprintf('K-Space Profile at CORE (Y=%.2f \\mum)', res1_core.y_extract_um));
legend('Location', 'northeast'); grid on; hold off;


%% --- Plot 3: Field Profile vs Theta Core (Overlaid, Absolute dB) ---
figure('Name', 'Profile vs Theta Core Comparison (Boundary vs Core)', 'Color', 'w', 'Position', [800 150 800 800]);

% TIR limits (assume identical for both)
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

xlabel('\theta_{core} = arcsin(k_x / k_{neff}) [degrees]'); ylabel('Absolute K-Space Power [dB]');
title(sprintf('Theta Core Profile at BOUNDARY (Y=%.2f \\mum)', res1_boundary.y_extract_um));
legend('Location', 'northeast'); grid on; hold off;

% 3b. Core Extraction
subplot(2, 1, 2);
plot(res1_core.theta_core_deg, res1_core.P_kx_core_dB, 'b-', 'LineWidth', 1.5, 'DisplayName', label_sim1); hold on;
plot(res2_core.theta_core_deg, res2_core.P_kx_core_dB, 'r--', 'LineWidth', 1.5, 'DisplayName', label_sim2);

y_max_th2 = max([max(res1_core.P_kx_core_dB), max(res2_core.P_kx_core_dB)]);
ylim([y_max_th2 - 60, y_max_th2 + 5]); xlim(xlims_th);

xline(theta_tir, 'k:', label_tir, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-theta_tir, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('\theta_{core} = arcsin(k_x / k_{neff}) [degrees]'); ylabel('Absolute K-Space Power [dB]');
title(sprintf('Theta Core Profile at CORE (Y=%.2f \\mum)', res1_core.y_extract_um));
legend('Location', 'northeast'); grid on; hold off;

fprintf('\nComparison Complete.\n');

%% =========================================================================
% LOCAL FUNCTION FOR DATA EXTRACTION
% =========================================================================
function res = extract_sim_data(filepath, params, extract_pos)
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

% Unpack Fields
if isfield(data, 'field_3d'); f3d = data.field_3d;
elseif isfield(data, 'E_res'); f3d = data;
else; error('No 3D field data.'); end

x = double(f3d.x); y = double(f3d.y); z = double(f3d.z);
if isfield(f3d, 'lambda_3d'); lam_3d = double(f3d.lambda_3d);
else; lam_3d = 1.55e-6; end

E_5D = reshape(f3d.E_res, [length(x), length(y), length(z), length(lam_3d), 3]);
[~, idx_lam] = min(abs(lam_3d - wl_res)); wl_3d_actual = lam_3d(idx_lam);
[~, idx_z0] = min(abs(z));

% Extraction Y
if strcmp(extract_pos, 'boundary')
    [y_max_m, ~] = max(abs(y));
    boundary_y_m = y_max_m - (params.boundary_margin_um * 1e-6);
    [~, idx_y_extract] = min(abs(y - boundary_y_m));
else % 'core'
    [~, idx_y_extract] = min(abs(y));
end

% Core extraction & Top View
E_core_comp = squeeze(E_5D(:, idx_y_extract, idx_z0, idx_lam, :));
E_top_comp = squeeze(E_5D(:, :, idx_z0, idx_lam, :));
E_sq_top = sum(abs(E_top_comp).^2, 3);
E_sq_top_dB = 10 * log10(E_sq_top);

% Windowing
Nx = length(x);
if params.isolate_defect
    center_m = params.defect_center_x_um * 1e-6;
    half_width_m = (params.defect_width_um / 2) * 1e-6;
    valid_idx = abs(x - center_m) <= half_width_m;
    window = zeros(Nx, 1); num_valid = sum(valid_idx);
    if num_valid > 0
        switch lower(params.window_type)
            case 'hann'; window(valid_idx) = hann(num_valid);
            case 'tukey'; window(valid_idx) = tukeywin(num_valid, 0.3);
            otherwise; window(valid_idx) = 1;
        end
    else; window = ones(Nx, 1); end
else
    switch lower(params.window_type)
        case 'hann'; window = hann(Nx);
        case 'tukey'; window = tukeywin(Nx, 0.3);
        otherwise; window = ones(Nx, 1);
    end
end

% FFT
E_xw = E_core_comp(:,1) .* window; E_yw = E_core_comp(:,2) .* window; E_zw = E_core_comp(:,3) .* window;
dx = mean(diff(x)); N_pad = Nx * 2;
kx = (-N_pad/2 : N_pad/2 - 1)' * (2*pi / (N_pad * dx));

fft_Ex = fftshift(fft(E_xw, N_pad)); fft_Ey = fftshift(fft(E_yw, N_pad)); fft_Ez = fftshift(fft(E_zw, N_pad));

% Calculate K-Space Absolute Power & dB Scale
P_kx = abs(fft_Ex).^2 + abs(fft_Ey).^2 + abs(fft_Ez).^2;
P_kx_dB = 10 * log10(P_kx);

% Physics
k0 = 2*pi / wl_3d_actual; n_eff = wl_3d_actual / (2 * params.pitch);
k_neff = k0 * n_eff; k_clad = k0 * params.n_clad;
theta_TIR_eff_deg = asind(k_clad / k_neff);

% Angles (Theta Core ONLY)
idx_core = abs(kx) <= k_neff;
theta_core_deg = asind(kx(idx_core) / k_neff);
P_kx_core_dB = P_kx_dB(idx_core);

% Store in struct
res.params = params; res.wl_nm = wl_3d_actual * 1e9;
res.x = x; res.y = y; res.y_extract_um = y(idx_y_extract)*1e6;
res.E_sq_top_dB = E_sq_top_dB;
res.kx = kx; res.P_kx_dB = P_kx_dB;
res.k_clad = k_clad; res.k_neff = k_neff; res.theta_TIR_eff_deg = theta_TIR_eff_deg;
res.theta_core_deg = theta_core_deg; res.P_kx_core_dB = P_kx_core_dB;
end

%% --- Local Plot Function ---
function plot_top_view(res, title_str, subplot_idx)
subplot(2, 1, subplot_idx);
imagesc(res.x*1e6, res.y*1e6, res.E_sq_top_dB');
set(gca, 'YDir', 'normal');
colormap(jet);

max_dB = max(res.E_sq_top_dB(:));
try clim([max_dB - 80, max_dB]); catch; caxis([max_dB - 80, max_dB]); end

cb = colorbar; ylabel(cb, '10*log_{10}(|E|^2) [dB]');
xlabel('Position X [\mum]'); ylabel('Position Y [\mum]');
title(sprintf('%s (\\lambda = %.3f nm)', title_str, res.wl_nm));

hold on;
y_ext_um = res.y_extract_um;
if res.params.isolate_defect
    x_st = res.params.defect_center_x_um - res.params.defect_width_um / 2;
    x_en = res.params.defect_center_x_um + res.params.defect_width_um / 2;
else
    x_st = min(res.x*1e6);
    x_en = max(res.x*1e6);
end

% Extract midpoint for label alignment
x_mid = (x_st + x_en) / 2;

% Draw boundary line
plot([x_st, x_en], [y_ext_um, y_ext_um], 'w--', 'LineWidth', 1.5);
plot(x_st, y_ext_um, 'w<', 'MarkerFaceColor', 'w', 'MarkerSize', 8);
plot(x_en, y_ext_um, 'w>', 'MarkerFaceColor', 'w', 'MarkerSize', 8);
text(x_mid, y_ext_um - 0.15, 'boundary', 'Color', 'w', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% Draw core line (y=0) too
plot([x_st, x_en], [0, 0], 'w--', 'LineWidth', 1.5);
plot(x_st, 0, 'w<', 'MarkerFaceColor', 'w', 'MarkerSize', 8);
plot(x_en, 0, 'w>', 'MarkerFaceColor', 'w', 'MarkerSize', 8);
text(x_mid, 0 - 0.15, 'core', 'Color', 'w', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
hold off;
end
