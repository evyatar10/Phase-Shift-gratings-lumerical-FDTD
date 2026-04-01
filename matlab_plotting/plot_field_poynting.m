% plot_field_poynting.m
% Plots 2D field profiles (|E|^2 in dB) with Poynting vector arrows overlaid.
% Generates figures for both XY (top view) and YZ (cross-section) planes.
%
% Uses the same draw_poynting_quiver approach as the microcavity project,
% with per-axis skip to handle non-square domains.

addpath(fileparts(fileparts(mfilename('fullpath'))));
clear; clc;

%% --- Configuration ---
result_filepath = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\leaky_modes\results\result_80_periods_CONST_ff.mat";

pitch = 500e-9;   % Grating pitch [m]
n_clad = 1.44;    % Cladding refractive index

% --- Colormap & Dynamic Range ---
field_colormap = 'hot';
dB_limit       = 60;

% --- Poynting Vector Overlay ---
show_poynting         = true;
arrows_per_axis       = 30;
min_skip              = 2;        % Never sample every single grid point
arrow_color           = 'c';
arrow_linewidth       = 0.9;
arrow_scale           = 0.6;      % MATLAB quiver scale multiplier
poynting_threshold_dB = 55;       % Show arrows where field is within this many dB of max
log_compress_k        = 10;       % log1p compression strength; higher = more uniform sizes
arrow_min_frac        = 0.35;     % Minimum arrow length as fraction of max (0=none, 1=all equal)
jitter_frac           = 0.0;      % Sampling jitter as fraction of skip (0=grid, 0.5=±skip/2)

% --- Geometry Overlay ---
show_geometry         = true;
avg_corrugation_width = 800e-9;   % [m]
core_height           = 350e-9;   % [m]

%% --- 1. Load Data & Find Resonance ---
fprintf('Loading data...\n');
if ~exist(result_filepath, 'file')
    error('File not found! Please check result_filepath.');
end
data = load(result_filepath);

if isfield(data, 'T') && isfield(data, 'wl_m')
    T = squeeze(data.T);
    wl = squeeze(data.wl_m);

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
    fprintf('Resonance at %.3f nm (T = %.3f)\n', wl_res*1e9, T(idx_peak));
else
    error('Transmission or Wavelength data missing.');
end

%% --- 2. Extract Field Data ---
if ~isfield(data, 'field_xy') || ~isfield(data, 'field_yz_cross')
    error('Missing field_xy/field_yz_cross. Please re-run the Python post-processing.');
end

d_xy = data.field_xy;
d_yz = data.field_yz_cross;

% XY plane
x_xy = double(d_xy.x); y_xy = double(d_xy.y); z_xy = double(d_xy.z);
if isfield(d_xy, 'lambda_3d'); lam_xy = double(d_xy.lambda_3d);
else; lam_xy = 1.55e-6; end

[~, idx_lam] = min(abs(lam_xy - wl_res));
wl_3d_actual = lam_xy(idx_lam);
[~, idx_z0_xy] = min(abs(z_xy));

Nx_xy = length(x_xy); Ny_xy = length(y_xy); Nz_xy = length(z_xy); Nlam = length(lam_xy);
E_5D_xy = reshape(d_xy.E_res, [Nx_xy, Ny_xy, Nz_xy, Nlam, 3]);

% YZ plane
x_yz = double(d_yz.x); y_yz = double(d_yz.y); z_yz = double(d_yz.z);
Nx_yz = length(x_yz); Ny_yz = length(y_yz); Nz_yz = length(z_yz);
E_5D_yz = reshape(d_yz.E_res, [Nx_yz, Ny_yz, Nz_yz, Nlam, 3]);

% Poynting vectors
has_poynting_xy = isfield(d_xy, 'P_res') && ~isempty(d_xy.P_res);
if has_poynting_xy
    P_5D_xy = reshape(double(d_xy.P_res), [Nx_xy, Ny_xy, Nz_xy, Nlam, 3]);
end

has_poynting_yz = isfield(d_yz, 'P_res') && ~isempty(d_yz.P_res);
if has_poynting_yz
    P_5D_yz = reshape(double(d_yz.P_res), [Nx_yz, Ny_yz, Nz_yz, Nlam, 3]);
end

if show_poynting && ~has_poynting_xy && ~has_poynting_yz
    fprintf('Warning: Poynting data not found in .mat file. Re-run Python post-processing to include it.\n');
end

% Compute adaptive skip per axis (with minimum to avoid over-dense sampling)
skip_x_xy = max(min_skip, round(Nx_xy / arrows_per_axis));
skip_y_xy = max(min_skip, round(Ny_xy / arrows_per_axis));
skip_y_yz = max(min_skip, round(Ny_yz / arrows_per_axis));
skip_z_yz = max(min_skip, round(Nz_yz / arrows_per_axis));
fprintf('XY quiver grid: skip_x=%d, skip_y=%d (from %dx%d)\n', skip_x_xy, skip_y_xy, Nx_xy, Ny_xy);
fprintf('YZ quiver grid: skip_y=%d, skip_z=%d (from %dx%d)\n', skip_y_yz, skip_z_yz, Ny_yz, Nz_yz);

%% --- 3. Plot XY Plane ---
fprintf('\n--- XY Plane (Top View) ---\n');

E_plane_xy = squeeze(E_5D_xy(:, :, idx_z0_xy, idx_lam, :));  % (Nx, Ny, 3)
I_xy = sum(abs(E_plane_xy).^2, 3);
I_xy_dB = 10 * log10(I_xy);

figure('Name', 'XY Field + Poynting', 'Color', 'w', 'Position', [100 150 900 450]);
imagesc(x_xy*1e6, y_xy*1e6, I_xy_dB');
set(gca, 'YDir', 'normal');
colormap(field_colormap);

max_dB = max(I_xy_dB(:));
try clim([max_dB - dB_limit, max_dB]); catch; caxis([max_dB - dB_limit, max_dB]); end

cb = colorbar;
ylabel(cb, '10\cdotlog_{10}(|E|^2) [dB]');
xlabel('Position X [\mum]');
ylabel('Position Y [\mum]');
title(sprintf('XY View: Field Profile + Poynting at \\lambda = %.3f nm', wl_3d_actual*1e9));

hold on;
if show_poynting && has_poynting_xy
    Px_xy = squeeze(P_5D_xy(:, :, idx_z0_xy, idx_lam, 1));
    Py_xy = squeeze(P_5D_xy(:, :, idx_z0_xy, idx_lam, 2));
    draw_poynting_quiver(x_xy*1e6, y_xy*1e6, Px_xy, Py_xy, ...
        skip_x_xy, skip_y_xy, arrow_scale, arrow_color, arrow_linewidth, ...
        I_xy_dB, max_dB, poynting_threshold_dB, log_compress_k, arrow_min_frac, jitter_frac);
end

if show_geometry
    wg_hw = avg_corrugation_width / 2 * 1e6;
    xl = xlim;
    plot(xl, [ wg_hw  wg_hw], 'w--', 'LineWidth', 1);
    plot(xl, [-wg_hw -wg_hw], 'w--', 'LineWidth', 1);
end
hold off;

%% --- 4. Plot YZ Plane ---
fprintf('--- YZ Plane (Cross Section) ---\n');

[~, idx_x0_yz] = min(abs(x_yz));
E_plane_yz = squeeze(E_5D_yz(idx_x0_yz, :, :, idx_lam, :));  % (Ny, Nz, 3)
I_yz = sum(abs(E_plane_yz).^2, 3);
I_yz_dB = 10 * log10(I_yz);

figure('Name', 'YZ Field + Poynting', 'Color', 'w', 'Position', [150 200 700 500]);
imagesc(y_yz*1e6, z_yz*1e6, I_yz_dB');
set(gca, 'YDir', 'normal');
colormap(field_colormap);

max_dB_yz = max(I_yz_dB(:));
try clim([max_dB_yz - dB_limit, max_dB_yz]); catch; caxis([max_dB_yz - dB_limit, max_dB_yz]); end

cb = colorbar;
ylabel(cb, '10\cdotlog_{10}(|E|^2) [dB]');
xlabel('Position Y [\mum]');
ylabel('Position Z [\mum]');
title(sprintf('YZ View: Field Profile + Poynting at \\lambda = %.3f nm', wl_3d_actual*1e9));

hold on;
if show_poynting && has_poynting_yz
    Py_yz = squeeze(P_5D_yz(idx_x0_yz, :, :, idx_lam, 2));
    Pz_yz = squeeze(P_5D_yz(idx_x0_yz, :, :, idx_lam, 3));
    draw_poynting_quiver(y_yz*1e6, z_yz*1e6, Py_yz, Pz_yz, ...
        skip_y_yz, skip_z_yz, arrow_scale, arrow_color, arrow_linewidth, ...
        I_yz_dB, max_dB_yz, poynting_threshold_dB, log_compress_k, arrow_min_frac, jitter_frac);
end

if show_geometry
    wg_hw = avg_corrugation_width / 2 * 1e6;
    wg_hh = core_height / 2 * 1e6;
    rectangle('Position', [-wg_hw, -wg_hh, 2*wg_hw, 2*wg_hh], ...
              'EdgeColor', 'w', 'LineStyle', '--', 'LineWidth', 1);
end
hold off;

fprintf('\nDone.\n');


%% --- Local Functions ---

function draw_poynting_quiver(coord1, coord2, P1, P2, skip1, skip2, scale, color, lw, I_dB, max_dB, threshold_dB, log_k, min_frac, jitter_frac)
% Overlay Poynting vector arrows with log-compressed, floor-bounded display.
    n1 = numel(coord1);
    n2 = numel(coord2);

    % --- Uniform or jittered sampling ---
    base1 = (1:skip1:n1)';
    base2 = (1:skip2:n2)';
    if jitter_frac > 0
        half1 = floor(skip1 * jitter_frac / 2);
        half2 = floor(skip2 * jitter_frac / 2);
        if half1 > 0; jit1 = randi([-half1, half1], size(base1)); else; jit1 = zeros(size(base1)); end
        if half2 > 0; jit2 = randi([-half2, half2], size(base2)); else; jit2 = zeros(size(base2)); end
        idx1 = unique(max(1, min(n1, base1 + jit1)));
        idx2 = unique(max(1, min(n2, base2 + jit2)));
    else
        idx1 = base1;
        idx2 = base2;
    end

    c1_q = coord1(idx1);
    c2_q = coord2(idx2);
    [C1q, C2q] = meshgrid(c1_q, c2_q);

    P1_q = P1(idx1, idx2).';
    P2_q = P2(idx1, idx2).';

    % --- Magnitude normalization ---
    Pmag = sqrt(P1_q.^2 + P2_q.^2);
    Pmag_max = max(Pmag(:));
    if Pmag_max == 0; return; end
    Pmag_norm = Pmag / Pmag_max;

    % Log compression + minimum floor for balanced arrow sizes
    log_norm = log1p(Pmag_norm * log_k) / log1p(log_k);
    vis_scale = min_frac + (1 - min_frac) * log_norm;

    % --- dB intensity mask ---
    I_dB_q = I_dB(idx1, idx2).';
    mask = I_dB_q >= (max_dB - threshold_dB);

    % Exclude 1% edge margin
    c1_range = max(c1_q) - min(c1_q);
    c2_range = max(c2_q) - min(c2_q);
    mask = mask & (C1q >= min(c1_q) + 0.01*c1_range) & (C1q <= max(c1_q) - 0.01*c1_range) ...
               & (C2q >= min(c2_q) + 0.01*c2_range) & (C2q <= max(c2_q) - 0.01*c2_range);

    vis_scale = vis_scale .* mask;

    % Unit direction vectors scaled by vis_scale; MATLAB auto-scales the longest to a good size
    P1_draw = (P1_q ./ (Pmag + 1e-30)) .* vis_scale;
    P2_draw = (P2_q ./ (Pmag + 1e-30)) .* vis_scale;

    quiver(C1q, C2q, P1_draw, P2_draw, scale, 'Color', color, 'LineWidth', lw);
end
