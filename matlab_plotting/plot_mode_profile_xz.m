% plot_mode_profile_xz.m
% Compares mode profiles (|E|^2 envelope + FWHM) side by side, computed
% from the XZ (side view) 2D monitor by integrating |E|^2 over z at each x.
%
% This mirrors plot_mode_profile.m but uses field_xz_side instead of the
% pre-computed 1D profile (which comes from the XY z=0 slice integrated in y).
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

labels = {
    '80 Periods', ...
    '80 Periods, 10 Apodizations', ...
    '80 Periods, 100 nm Shift', ...
    '80 Periods, 100 nm Shift, Inner Size 100 nm', ...
};

compare_profiles_xz(file_list, labels);


% ---------------------------------------------------------
% MAIN: Load each file, extract XZ profile, plot side by side
% ---------------------------------------------------------
function compare_profiles_xz(file_list, labels)
n = numel(file_list);
if n == 0, error('file_list is empty. Please provide at least one file path.'); end
if nargin < 2 || isempty(labels)
    labels = repmat({''}, 1, n);
end

if n <= 3
    ncols = n; nrows = 1;
elseif n == 4
    ncols = 2; nrows = 2;
else
    ncols = 3; nrows = ceil(n / 3);
end

fig_w = 520 * ncols;
fig_h = 420 * nrows;
figure('Name', 'Mode Profile Comparison (XZ integral)', 'Color', 'w', ...
       'Position', [80, 80, fig_w, fig_h]);

for i = 1:n
    filepath = file_list{i};
    fprintf('\n--- Processing File %d / %d ---\n', i, n);
    fprintf('Path: %s\n', filepath);

    [x, I_x, I_env, fwhm, xL, xR, yHM, wl_res, T_peak] = extract_xz_profile(filepath);

    ax = subplot(nrows, ncols, i);
    plot_profile(ax, x, I_x, I_env, fwhm, xL, xR, yHM, wl_res, labels{i}, T_peak);
end
end


% ---------------------------------------------------------
% EXTRACT 1D PROFILE FROM THE XZ (Side View) MONITOR
% Integrates |E|^2 over z at each x, at the resonance wavelength.
% ---------------------------------------------------------
function [x, I_x, I_env, fwhm, xL, xR, yHM, wl_res, T_peak] = extract_xz_profile(filepath)
if ~exist(filepath, 'file')
    error('File not found: %s', filepath);
end
data = load(filepath);

if ~isfield(data, 'field_xz_side')
    error(['field_xz_side not found in: %s\n' ...
           'Make sure the simulation was run with record_2d_fields_top_and_cross = True.'], filepath);
end
xz = data.field_xz_side;

% --- Resonance wavelength and peak transmission ---
if isfield(data, 'resonance_wavelength_nm')
    wl_res = double(data.resonance_wavelength_nm) * 1e-9;
elseif isfield(data, 'wl_m') && isfield(data, 'T')
    [~, idx_peak] = max(data.T);
    wl_res = data.wl_m(idx_peak);
else
    wl_res = NaN;
end

if isfield(data, 'T') && isfield(data, 'wl_m') && ~isnan(wl_res)
    [~, idx_res] = min(abs(data.wl_m - wl_res));
    T_peak = double(data.T(idx_res));
else
    T_peak = NaN;
end

% --- Coordinates ---
x   = double(squeeze(xz.x));
z   = double(squeeze(xz.z));
lam = double(squeeze(xz.lambda_3d));
lam = lam(:);

% --- Pick the monitor wavelength closest to resonance ---
if isnan(wl_res)
    idx_wl = 1;
else
    [~, idx_wl] = min(abs(lam - wl_res));
end
fprintf('XZ monitor: resonance at %.3f nm, using monitor wl %.3f nm\n', ...
    wl_res * 1e9, lam(idx_wl) * 1e9);
fprintf('XZ monitor extent: x = [%.2f, %.2f] um, z = [%.2f, %.2f] um (Nz = %d)\n', ...
    min(x)*1e6, max(x)*1e6, min(z)*1e6, max(z)*1e6, numel(z));

% --- Extract |E|^2 at resonance, shape -> (Nx, Nz) ---
% Lumerical XZ (Y-normal) monitor typically returns E with a singleton y-dim.
% Handle 5D (Nx, Ny, Nz, Nwl, 3), 4D (Nx, Nz, Nwl, 3), or 4D (Nx, Ny, Nz, 3) for single-wl.
E = xz.E_res;
E_slice = slice_E_at_wavelength(E, idx_wl, numel(x), numel(z));

E_sq = sum(abs(E_slice).^2, 3);          % (Nx, Nz)

% --- Integrate |E|^2 over z at each x (the requested integral, in MATLAB) ---
% Ensure z is a column vector for trapz along dim 2.
z_row = z(:).';
% trapz needs monotonically increasing z — sort if needed
if any(diff(z_row) < 0)
    [z_row, zord] = sort(z_row);
    E_sq = E_sq(:, zord);
end
I_x = trapz(z_row, E_sq, 2);              % (Nx, 1)
I_x = I_x(:);

% --- Crop x to grating region if possible (match python behaviour) ---
if isfield(data, 'field_x')
    x_limit = max(abs(double(data.field_x(:))));
    mask = abs(x) <= x_limit;
    if nnz(mask) > 10
        x   = x(mask);
        I_x = I_x(mask);
    end
end

% --- Envelope and FWHM ---
[I_env, ~, ~] = calculate_envelope(x, I_x);
[fwhm, xL, xR, yHM] = calculate_fwhm_relative(x, I_env);
fprintf('XZ-integrated profile FWHM: %.2f um\n', fwhm * 1e6);
end


% ---------------------------------------------------------
% HELPER: Extract a (Nx, Nz, 3) slice from the XZ monitor E array
% at a given wavelength index, robust to how Lumerical packs dims.
% ---------------------------------------------------------
function E_slice = slice_E_at_wavelength(E, idx_wl, Nx, Nz)
sz = size(E);
nd = ndims(E);

% Drop singleton y dimension (XZ monitor is Y-normal)
E = squeeze(E);
sz = size(E);
nd = ndims(E);

% Expected after squeeze:
%   (Nx, Nz, Nwl, 3)  — multi-wavelength
%   (Nx, Nz, 3)       — single wavelength
if nd == 4
    % (Nx, Nz, Nwl, 3)
    if sz(1) ~= Nx || sz(2) ~= Nz
        warning('Unexpected E shape %s; trying best-effort reshape.', mat2str(sz));
    end
    E_slice = squeeze(E(:, :, idx_wl, :));    % (Nx, Nz, 3)
elseif nd == 3
    % (Nx, Nz, 3) — single wavelength
    E_slice = E;
else
    error('Unexpected XZ E_res ndims = %d, size = %s', nd, mat2str(sz));
end
end


% ---------------------------------------------------------
% PLOT A SINGLE PROFILE
% ---------------------------------------------------------
function plot_profile(ax, x, I_x, I_env, fwhm, xL, xR, yHM, wl_res, label, T_peak)
axes(ax);
hold on;

y_max = max([max(I_x), max(I_env)]);
if y_max > 0
    exp_val = floor(log10(y_max));
else
    exp_val = 0;
end
scale = 10^(-exp_val);

area(x * 1e6, I_x * scale, ...
    'FaceColor', [0.75 0.75 1], ...
    'EdgeColor', [0.55 0.55 0.85], ...
    'FaceAlpha', 0.55, ...
    'LineWidth', 0.4, ...
    'DisplayName', 'Raw \int|E|^2 dz');

plot(x * 1e6, I_env * scale, ...
    'r-', 'LineWidth', 2, ...
    'DisplayName', 'Field Envelope');

if fwhm > 0
    plot([xL, xR] * 1e6, [yHM * scale, yHM * scale], ...
        'k--', 'LineWidth', 1.5, ...
        'HandleVisibility', 'off');
    text(mean([xL, xR]) * 1e6 - 3, yHM * scale * 1.08, ...
        sprintf('FWHM = %.2f \\mum', fwhm * 1e6), ...
        'HorizontalAlignment', 'center', ...
        'FontWeight', 'bold', ...
        'FontSize', 10, ...
        'Color', 'k');
end

xlabel('Position x [\mum]', 'FontSize', 10);
if exp_val ~= 0
    ylabel(sprintf('\\int|E|^2 dz (a.u.) \\times10^{%d}', exp_val), 'FontSize', 10);
else
    ylabel('\int|E|^2 dz (a.u.)', 'FontSize', 10);
end
if isnan(T_peak)
    t_str = sprintf('XZ Mode Profile @ %.2f nm', wl_res * 1e9);
else
    t_str = sprintf('XZ Mode Profile @ %.2f nm  |  \\bf\\color[rgb]{0.85,0.40,0.00}T_{peak} = %.4f\\rm\\color{black}', wl_res * 1e9, T_peak);
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
