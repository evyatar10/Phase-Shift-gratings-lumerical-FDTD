% plot_mode_profile.m
% Compares mode profiles (|E|^2 envelope + FWHM) side by side.
clear all;
close all;
clc;
addpath(fileparts(fileparts(mfilename('fullpath'))));  % Add project root to MATLAB path

% --- USER CONFIGURATION ---
base_dir = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\leaky_modes_v4\results\comparison";

file_list = {
    fullfile(base_dir, 'result_80_periods_CONST.mat'), ...
    fullfile(base_dir, 'result_80_periods_10_apod_CONST.mat'), ...
    fullfile(base_dir, 'result_80_periods_CONST_shift_100nm.mat'), ...
    fullfile(base_dir, 'result_80_periods_CONST_shift_100.0nm_innersize_100nm.mat'), ...
};

% Human-readable label for each file (shown as subplot title)
% Must match file_list order and length
labels = {
    '80 Periods', ...
    '80 Periods, 10 Apodizations', ...
    '80 Periods, 100 nm Shift', ...
    '80 Periods, 100 nm Shift, Inner Size 100 nm', ...
};

% Run Analysis
compare_profiles(file_list, labels);


% ---------------------------------------------------------
% MAIN: Load each file, extract 2D profile, plot side by side
% ---------------------------------------------------------
function compare_profiles(file_list, labels)
n = numel(file_list);
if n == 0, error('file_list is empty. Please provide at least one file path.'); end
if nargin < 2 || isempty(labels)
    labels = repmat({''}, 1, n);  % no labels if omitted
end

% Layout: 2x2 for 4 files, single row for <=3, 3-column grid for more
if n <= 3
    ncols = n; nrows = 1;
elseif n == 4
    ncols = 2; nrows = 2;
else
    ncols = 3; nrows = ceil(n / 3);
end

fig_w = 520 * ncols;
fig_h = 420 * nrows;
figure('Name', 'Mode Profile Comparison', 'Color', 'w', ...
       'Position', [80, 80, fig_w, fig_h]);

for i = 1:n
    filepath = file_list{i};
    fprintf('\n--- Processing File %d / %d ---\n', i, n);
    fprintf('Path: %s\n', filepath);

    [x, I_x_2d, I_env_2d, fwhm, xL, xR, yHM, wl_res, T_peak] = extract_2d_profile(filepath);

    ax = subplot(nrows, ncols, i);
    plot_profile(ax, x, I_x_2d, I_env_2d, fwhm, xL, xR, yHM, wl_res, labels{i}, T_peak);
end
end


% ---------------------------------------------------------
% EXTRACT 2D PROFILE FROM A SINGLE FILE
% ---------------------------------------------------------
function [x, I_x_2d, I_env_2d, fwhm, xL, xR, yHM, wl_res, T_peak] = extract_2d_profile(filepath)
if ~exist(filepath, 'file')
    error('File not found: %s', filepath);
end
data = load(filepath);

% ---- PATH A: pre-processed file (new format) ----
% Detected by presence of field_energy_density_1D
if isfield(data, 'field_energy_density_1D') && isfield(data, 'field_x')

    x      = double(data.field_x(:));
    I_x_2d = double(data.field_energy_density_1D(:));

    if isfield(data, 'field_envelope_1D')
        I_env_2d = double(data.field_envelope_1D(:));
    else
        [I_env_2d, ~, ~] = calculate_envelope(x, I_x_2d);
    end

    if isfield(data, 'resonance_wavelength_nm')
        wl_res = double(data.resonance_wavelength_nm) * 1e-9;
    elseif isfield(data, 'wl_m') && isfield(data, 'T')
        [~, idx_peak] = max(data.T);
        wl_res = data.wl_m(idx_peak);
    else
        wl_res = NaN;
    end

    % Peak transmission at resonance wavelength
    if isfield(data, 'T') && isfield(data, 'wl_m')
        [~, idx_res] = min(abs(data.wl_m - wl_res));
        T_peak = double(data.T(idx_res));
    else
        T_peak = NaN;
    end

    [fwhm, xL, xR, yHM] = calculate_fwhm_relative(x, I_env_2d);
    fprintf('Pre-processed format detected. FWHM: %.2f um at %.3f nm  |  T_peak = %.4f\n', ...
        fwhm * 1e6, wl_res * 1e9, T_peak);

% ---- PATH B: raw 3D field file (legacy format) ----
else
    if isfield(data, 'T') && isfield(data, 'wl_m')
        [~, idx_peak] = max(data.T);
        wl_res  = data.wl_m(idx_peak);
        T_peak  = double(data.T(idx_peak));
    else
        error('Cannot identify resonance wavelength in: %s', filepath);
    end

    [x, y, z, lam_3d, E_5D] = unpack_data_robust(data);

    [~, idx_wl] = min(abs(lam_3d - wl_res));
    wl_res = lam_3d(idx_wl);
    fprintf('Raw format: resonance at %.3f nm\n', wl_res * 1e9);

    [~, idx_z0] = min(abs(z));
    fprintf('2D slice at Z = %.3f um\n', z(idx_z0) * 1e6);

    E_slice_2d = squeeze(E_5D(:, :, idx_z0, idx_wl, :));
    E_sq_2d    = sum(abs(E_slice_2d).^2, 3);
    I_x_2d     = trapz(y, E_sq_2d, 2);

    [I_env_2d, ~, ~] = calculate_envelope(x, I_x_2d);
    [fwhm, xL, xR, yHM] = calculate_fwhm_relative(x, I_env_2d);
    fprintf('2D Profile FWHM: %.2f um\n', fwhm * 1e6);
end
end


% ---------------------------------------------------------
% PLOT A SINGLE PROFILE (reference-image style)
% ---------------------------------------------------------
function plot_profile(ax, x, I_x_2d, I_env_2d, fwhm, xL, xR, yHM, wl_res, label, T_peak)
axes(ax);
hold on;

% --- Determine scale factor from data so tick labels are clean numbers ---
y_max = max([max(I_x_2d), max(I_env_2d)]);
if y_max > 0
    exp_val = floor(log10(y_max));
else
    exp_val = 0;
end
scale = 10^(-exp_val);

% --- Raw |E|^2 as shaded area (blue, semi-transparent) ---
area(x * 1e6, I_x_2d * scale, ...
    'FaceColor', [0.75 0.75 1], ...
    'EdgeColor', [0.55 0.55 0.85], ...
    'FaceAlpha', 0.55, ...
    'LineWidth', 0.4, ...
    'DisplayName', 'Raw |E|^2');

% --- Field envelope (red solid line) ---
plot(x * 1e6, I_env_2d * scale, ...
    'r-', 'LineWidth', 2, ...
    'DisplayName', 'Field Envelope');

% --- FWHM annotation ---
if fwhm > 0
    % Dashed horizontal line at half-maximum
    plot([xL, xR] * 1e6, [yHM * scale, yHM * scale], ...
        'k--', 'LineWidth', 1.5, ...
        'HandleVisibility', 'off');

    % Text label centred above the line
    text(mean([xL, xR]) * 1e6 - 3, yHM * scale * 1.08, ...
        sprintf('FWHM = %.2f \\mum', fwhm * 1e6), ...
        'HorizontalAlignment', 'center', ...
        'FontWeight', 'bold', ...
        'FontSize', 10, ...
        'Color', 'k');
end

% --- Formatting ---
xlabel('Position x [\mum]', 'FontSize', 10);
if exp_val ~= 0
    ylabel(sprintf('Integrated Energy Density (a.u.) \\times10^{%d}', exp_val), 'FontSize', 10);
else
    ylabel('Integrated Energy Density (a.u.)', 'FontSize', 10);
end
if isnan(T_peak)
    t_str = sprintf('Mode Profile @ %.2f nm', wl_res * 1e9);
else
    t_str = sprintf('Mode Profile @ %.2f nm  |  \\bf\\color[rgb]{0.85,0.40,0.00}T_{peak} = %.4f\\rm\\color{black}', wl_res * 1e9, T_peak);
end
if ~isempty(label)
    title({label, t_str}, 'FontSize', 11, 'Interpreter', 'tex');
else
    title(t_str, 'FontSize', 11, 'Interpreter', 'tex');
end
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([min(x) * 1e6, max(x) * 1e6]);
set(ax, 'Box', 'on', 'Color', 'w', 'GridColor', [0.8 0.8 0.8], ...
    'GridAlpha', 0.7, 'LineWidth', 0.8);
hold off;
end


% ---------------------------------------------------------
% HELPER: ENVELOPE (Hilbert transform)
% ---------------------------------------------------------
function [env, x_pks, y_pks] = calculate_envelope(x, y)
analytic_signal = hilbert(double(y));
env = abs(analytic_signal);
env = smoothdata(env, 'gaussian', 50);
[y_pks, x_pks] = findpeaks(env, double(x));
end


% ---------------------------------------------------------
% HELPER: FWHM (interpolated zero-crossings)
% ---------------------------------------------------------
function [width, x1, x2, y_target] = calculate_fwhm_relative(x, y)
y_max = max(y); y_min = min(y);
y_target = y_min + 0.5 * (y_max - y_min);

above    = y > y_target;
crossings = diff(above);
indices   = find(crossings ~= 0);

if length(indices) < 2, width = 0; x1 = 0; x2 = 0; return; end

x_cross = zeros(length(indices), 1);
for k = 1:length(indices)
    idx  = indices(k);
    slope = (y(idx+1) - y(idx)) / (x(idx+1) - x(idx));
    x_cross(k) = x(idx) + (y_target - y(idx)) / slope;
end

x1    = x_cross(1);
x2    = x_cross(end);
width = x2 - x1;
end


% ---------------------------------------------------------
% HELPER: UNPACK 3D FIELD DATA
% ---------------------------------------------------------
function [x, y, z, lam, E_5D] = unpack_data_robust(data)
% Resolve the struct that holds x/y/z/E_res
if isfield(data, 'field_3d')
    f3d = data.field_3d;
elseif isfield(data, 'field')
    f3d = data.field;
elseif isfield(data, 'monitor')
    f3d = data.monitor;
else
    % Print available top-level fields to help diagnose unknown layouts
    fprintf('\n[unpack_data_robust] Top-level fields in .mat file:\n');
    disp(fieldnames(data));
    % Check if coordinates live directly at the top level
    if isfield(data, 'x') && isfield(data, 'E_res')
        f3d = data;
    else
        error(['Cannot find field data. Expected "field_3d", "field", or "monitor" ' ...
               '(or top-level x/E_res). See field list above.']);
    end
end
x = double(f3d.x);
y = double(f3d.y);
z = double(f3d.z);

if isfield(f3d, 'lambda_3d'),     lam = double(f3d.lambda_3d);
elseif isfield(f3d, 'lambda'),    lam = double(f3d.lambda);
else,                             lam = 1.55e-6;
end
lam = lam(:);

E_raw = f3d.E_res;
dims  = size(E_raw);
if ndims(E_raw) == 4
    if dims(4) == 3 && length(lam) == 1
        E_5D = reshape(E_raw, [dims(1), dims(2), dims(3), 1, dims(4)]);
    else
        E_5D = E_raw;
    end
else
    E_5D = E_raw;
end
end
