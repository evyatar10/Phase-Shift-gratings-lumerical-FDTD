% plot_field_poynting.m
% Plots 2D field profiles (|E|^2 in dB) with Poynting vector arrows overlaid.
% Generates figures for both XY (top view) and YZ (cross-section) planes.

addpath(fileparts(fileparts(mfilename('fullpath'))));
clear; clc;

%% --- Configuration ---
result_filepath = "C:\Users\evyat\Lumerical\new_experiment_comparison\p8rc1_tanh\results\result_80_periods_CONST.mat";

pitch = 500e-9;   % Grating pitch [m]
n_clad = 1.44;    % Cladding refractive index

% --- Colormap & Dynamic Range ---
field_colormap = 'hot';    % 'hot' (recommended) or 'jet'
dB_limit       = 60;       % Dynamic range in dB

% --- Poynting Vector Overlay ---
show_poynting   = true;
quiver_skip     = 5;       % Arrow spacing: every Nth grid point
arrow_color     = 'c';     % Cyan — good contrast with hot
arrow_linewidth = 1.0;
arrow_scale     = 0.4;     % quiver auto-scale factor

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

% Poynting vectors (available if post-processing includes P)
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
        quiver_skip, arrow_scale, arrow_color, arrow_linewidth);
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
        quiver_skip, arrow_scale, arrow_color, arrow_linewidth);
end
hold off;

fprintf('\nDone.\n');


%% --- Local Functions ---

function draw_poynting_quiver(coord1, coord2, P1, P2, skip, scale, color, lw)
% Overlay Poynting vector arrows on an imagesc plot.
    idx1 = centered_skip_indices(numel(coord1), skip);
    idx2 = centered_skip_indices(numel(coord2), skip);
    c1_q = coord1(idx1);
    c2_q = coord2(idx2);
    [C1q, C2q] = meshgrid(c1_q, c2_q);

    % Transpose to match imagesc orientation (rows=coord2, cols=coord1)
    P1_q = P1(idx1, idx2).';
    P2_q = P2(idx1, idx2).';

    Pmag = sqrt(P1_q.^2 + P2_q.^2);
    Pmag_max = max(Pmag(:));
    if Pmag_max == 0; return; end
    Pmag_norm = Pmag / Pmag_max;

    % Mask out negligibly weak arrows
    mask = Pmag_norm > 0.005;

    % Mask arrows near edges to prevent clipping
    margin = 0.05;  % um
    c1_min = min(c1_q); c1_max = max(c1_q);
    c2_min = min(c2_q); c2_max = max(c2_q);
    mask = mask & (C1q >= c1_min + margin) & (C1q <= c1_max - margin) ...
               & (C2q >= c2_min + margin) & (C2q <= c2_max - margin);

    % Power-law scaling for visually balanced arrow sizes
    sc = Pmag_norm.^0.35 .* mask;

    % Normalize direction, apply scale
    P1_draw = (P1_q ./ (Pmag + 1e-30)) .* sc;
    P2_draw = (P2_q ./ (Pmag + 1e-30)) .* sc;

    quiver(C1q, C2q, P1_draw, P2_draw, scale, color, 'LineWidth', lw);
end

function idx = centered_skip_indices(n, skip)
% Generate indices centered on the midpoint of 1:n, spaced by skip.
    mid = ceil(n / 2);
    lo = mid:-skip:1;
    hi = mid+skip:skip:n;
    idx = sort([lo, hi]);
end
