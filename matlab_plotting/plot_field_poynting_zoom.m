% plot_field_poynting_zoom.m
% Zoomed field profile (|E|^2 in dB) with Poynting vector arrows,
% restricted to specific spatial regions around the grating center.
%
% Regions:
%   XZ plane — X: [-10, 10] um, Z: [0, max]   (upper half, near defect)
%   YZ plane — full range                       (no restriction)
%   XY plane — X: [-10, 10] um, Y: [-10, 10] um

addpath(fileparts(fileparts(mfilename('fullpath'))));
clear; clc;
close all;
%% --- Configuration ---
result_filepath = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\leaky_modes\results\result_80_periods_CONST_ff.mat";

% Crop bounds [um]
crop_val = 12;
x_range_xz = [-crop_val, crop_val];
z_range_xz = [-inf, Inf];       % 0 to max available
x_range_xy = [-crop_val, crop_val];
y_range_xy = [-inf, inf];

% Visual parameters
field_colormap        = 'hot';
dB_limit              = 60;

% --- Algorithm selection ---
% true  → smoothed overlay (spatial averaging, staggered grid, cleaner on zoomed views)
% false → grid-based with coherence + convergence filters (shows raw Poynting structure)
use_smooth_algorithm  = false;

% Poynting arrows — shared params
arrow_color           = 'c';
arrow_linewidth       = 0.9;
poynting_threshold_dB = 55;

% Grid-based algorithm params (use_smooth_algorithm = false)
arrows_per_axis       = 25;
min_skip              = 2;
arrow_scale           = 0.55;
log_compress_k        = 10;
arrow_min_frac        = 0.60;

% Smoothed overlay algorithm params (use_smooth_algorithm = true)
smooth_opts = struct();
smooth_opts.field_colormap  = 'hot';
smooth_opts.dB_range        = 60;
smooth_opts.arrow_color     = [0 0.9 0.9];
smooth_opts.arrow_linewidth = 0.9;
smooth_opts.arrow_scale     = 0.5;
smooth_opts.arrows_per_span = 30;
smooth_opts.smooth_window   = 11;
smooth_opts.threshold_dB    = 55;
smooth_opts.density_factor  = 0.85;
smooth_opts.power_law_exp   = 0.25;
smooth_opts.base_size       = 0.6;

% Geometry overlay
avg_corrugation_width = 800e-9;   % [m]
core_height           = 350e-9;   % [m]

%% --- 1. Load Data & Find Resonance ---
fprintf('Loading data...\n');
if ~exist(result_filepath, 'file')
    error('File not found! Please check result_filepath.');
end
data = load(result_filepath);

T  = squeeze(data.T);
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
    idx_end   = stopband_indices(end);
    [~, local_peak_idx] = max(T(idx_start:idx_end));
    idx_peak = idx_start + local_peak_idx - 1;
end

wl_res = wl(idx_peak);
fprintf('Resonance at %.3f nm (T = %.3f)\n', wl_res*1e9, T(idx_peak));

%% --- 2. Plot XZ Plane (Zoomed Side View) ---
if isfield(data, 'field_xz_side')
    fprintf('\n--- XZ Plane (Zoomed Side View) ---\n');
    d = data.field_xz_side;
    x = double(d.x); y = double(d.y); z = double(d.z);
    lam = double(d.lambda_3d);
    Nx = length(x); Ny = length(y); Nz = length(z); Nlam = length(lam);

    [~, idx_lam] = min(abs(lam - wl_res));
    [~, idx_y0]  = min(abs(y));
    wl_plot = lam(idx_lam);

    % Crop to zoomed region
    [x_c, ix] = crop_to_range(x, x_range_xz);
    [z_c, iz] = crop_to_range(z, z_range_xz);

    E_5D = reshape(d.E_res, [Nx, Ny, Nz, Nlam, 3]);
    E_plane = squeeze(E_5D(ix, idx_y0, iz, idx_lam, :));   % (Nx_c, Nz_c, 3)
    I_xz = sum(abs(E_plane).^2, 3);
    I_xz_dB = 10 * log10(I_xz);

    Nx_c = length(ix); Nz_c = length(iz);
    skip_x = max(min_skip, round(Nx_c / arrows_per_axis));
    skip_z = max(min_skip, round(Nz_c / arrows_per_axis));
    fprintf('  Cropped grid: %d x %d,  skip_x=%d, skip_z=%d\n', Nx_c, Nz_c, skip_x, skip_z);

    figure('Name', 'XZ Zoomed', 'Color', 'w', 'Position', [200 100 850 500]);
    imagesc(x_c*1e6, z_c*1e6, I_xz_dB');
    set(gca, 'YDir', 'normal');
    colormap(field_colormap);
    max_dB = max(I_xz_dB(:));
    try clim([max_dB - dB_limit, max_dB]); catch; caxis([max_dB - dB_limit, max_dB]); end
    cb = colorbar; ylabel(cb, '10\cdotlog_{10}(|E|^2) [dB]');
    xlabel('Position X [\mum]'); ylabel('Position Z [\mum]');
    title(sprintf('XZ Zoomed: Field + Poynting at \\lambda = %.3f nm', wl_plot*1e9));

    hold on;
    if isfield(d, 'P_res') && ~isempty(d.P_res)
        P_5D = reshape(double(d.P_res), [Nx, Ny, Nz, Nlam, 3]);
        Px = squeeze(P_5D(ix, idx_y0, iz, idx_lam, 1));
        Pz = squeeze(P_5D(ix, idx_y0, iz, idx_lam, 3));
        if use_smooth_algorithm
            draw_smooth_quiver(gca, x_c*1e6, z_c*1e6, Px, Pz, I_xz_dB, smooth_opts); %#ok<UNRCH>
        else
            draw_poynting_quiver(x_c*1e6, z_c*1e6, Px, Pz, ...
                skip_x, skip_z, arrow_scale, arrow_color, arrow_linewidth, ...
                I_xz_dB, max_dB, poynting_threshold_dB, log_compress_k, arrow_min_frac);
        end
    end

    wg_hh = core_height / 2 * 1e6;
    xl = xlim;
    plot(xl, [ wg_hh  wg_hh], 'w--', 'LineWidth', 1);
    plot(xl, [-wg_hh -wg_hh], 'w--', 'LineWidth', 1);
    hold off;
else
    fprintf('XZ side view data not available.\n');
end

%% --- 3. Plot YZ Plane (Full Range Cross Section) ---
if isfield(data, 'field_yz_cross')
    fprintf('--- YZ Plane (Cross Section) ---\n');
    d = data.field_yz_cross;
    x = double(d.x); y = double(d.y); z = double(d.z);
    lam = double(d.lambda_3d);
    Nx = length(x); Ny = length(y); Nz = length(z); Nlam = length(lam);

    [~, idx_lam] = min(abs(lam - wl_res));
    [~, idx_x0]  = min(abs(x));
    wl_plot = lam(idx_lam);

    E_5D = reshape(d.E_res, [Nx, Ny, Nz, Nlam, 3]);
    E_plane = squeeze(E_5D(idx_x0, :, :, idx_lam, :));     % (Ny, Nz, 3)
    I_yz = sum(abs(E_plane).^2, 3);
    I_yz_dB = 10 * log10(I_yz);

    skip_y = max(min_skip, round(Ny / arrows_per_axis));
    skip_z = max(min_skip, round(Nz / arrows_per_axis));
    fprintf('  Grid: %d x %d,  skip_y=%d, skip_z=%d\n', Ny, Nz, skip_y, skip_z);

    figure('Name', 'YZ Full', 'Color', 'w', 'Position', [150 200 700 500]);
    imagesc(y*1e6, z*1e6, I_yz_dB');
    set(gca, 'YDir', 'normal');
    colormap(field_colormap);
    max_dB_yz = max(I_yz_dB(:));
    try clim([max_dB_yz - dB_limit, max_dB_yz]); catch; caxis([max_dB_yz - dB_limit, max_dB_yz]); end
    cb = colorbar; ylabel(cb, '10\cdotlog_{10}(|E|^2) [dB]');
    xlabel('Position Y [\mum]'); ylabel('Position Z [\mum]');
    title(sprintf('YZ Cross Section: Field + Poynting at \\lambda = %.3f nm', wl_plot*1e9));

    hold on;
    if isfield(d, 'P_res') && ~isempty(d.P_res)
        P_5D = reshape(double(d.P_res), [Nx, Ny, Nz, Nlam, 3]);
        Py = squeeze(P_5D(idx_x0, :, :, idx_lam, 2));
        Pz = squeeze(P_5D(idx_x0, :, :, idx_lam, 3));
        if use_smooth_algorithm
            draw_smooth_quiver(gca, y*1e6, z*1e6, Py, Pz, I_yz_dB, smooth_opts); %#ok<UNRCH>
        else
            draw_poynting_quiver(y*1e6, z*1e6, Py, Pz, ...
                skip_y, skip_z, arrow_scale, arrow_color, arrow_linewidth, ...
                I_yz_dB, max_dB_yz, poynting_threshold_dB, log_compress_k, arrow_min_frac);
        end
    end

    wg_hw = avg_corrugation_width / 2 * 1e6;
    wg_hh = core_height / 2 * 1e6;
    rectangle('Position', [-wg_hw, -wg_hh, 2*wg_hw, 2*wg_hh], ...
              'EdgeColor', 'w', 'LineStyle', '--', 'LineWidth', 1);
    hold off;
else
    fprintf('YZ cross-section data not available.\n');
end

%% --- 4. Plot XY Plane (Zoomed Top View) ---
if isfield(data, 'field_xy')
    fprintf('--- XY Plane (Zoomed Top View) ---\n');
    d = data.field_xy;
    x = double(d.x); y = double(d.y); z = double(d.z);
    lam = double(d.lambda_3d);
    Nx = length(x); Ny = length(y); Nz = length(z); Nlam = length(lam);

    [~, idx_lam] = min(abs(lam - wl_res));
    [~, idx_z0]  = min(abs(z));
    wl_plot = lam(idx_lam);

    % Crop to zoomed region
    [x_c, ix] = crop_to_range(x, x_range_xy);
    [y_c, iy] = crop_to_range(y, y_range_xy);

    E_5D = reshape(d.E_res, [Nx, Ny, Nz, Nlam, 3]);
    E_plane = squeeze(E_5D(ix, iy, idx_z0, idx_lam, :));   % (Nx_c, Ny_c, 3)
    I_xy = sum(abs(E_plane).^2, 3);
    I_xy_dB = 10 * log10(I_xy);

    Nx_c = length(ix); Ny_c = length(iy);
    skip_x = max(min_skip, round(Nx_c / arrows_per_axis));
    skip_y = max(min_skip, round(Ny_c / arrows_per_axis));
    fprintf('  Cropped grid: %d x %d,  skip_x=%d, skip_y=%d\n', Nx_c, Ny_c, skip_x, skip_y);

    figure('Name', 'XY Zoomed', 'Color', 'w', 'Position', [100 150 750 650]);
    imagesc(x_c*1e6, y_c*1e6, I_xy_dB');
    set(gca, 'YDir', 'normal');
    colormap(field_colormap);
    max_dB_xy = max(I_xy_dB(:));
    try clim([max_dB_xy - dB_limit, max_dB_xy]); catch; caxis([max_dB_xy - dB_limit, max_dB_xy]); end
    cb = colorbar; ylabel(cb, '10\cdotlog_{10}(|E|^2) [dB]');
    xlabel('Position X [\mum]'); ylabel('Position Y [\mum]');
    title(sprintf('XY Zoomed: Field + Poynting at \\lambda = %.3f nm', wl_plot*1e9));

    hold on;
    if isfield(d, 'P_res') && ~isempty(d.P_res)
        P_5D = reshape(double(d.P_res), [Nx, Ny, Nz, Nlam, 3]);
        Px = squeeze(P_5D(ix, iy, idx_z0, idx_lam, 1));
        Py = squeeze(P_5D(ix, iy, idx_z0, idx_lam, 2));
        if use_smooth_algorithm
            draw_smooth_quiver(gca, x_c*1e6, y_c*1e6, Px, Py, I_xy_dB, smooth_opts); %#ok<UNRCH>
        else
            draw_poynting_quiver(x_c*1e6, y_c*1e6, Px, Py, ...
                skip_x, skip_y, arrow_scale, arrow_color, arrow_linewidth, ...
                I_xy_dB, max_dB_xy, poynting_threshold_dB, log_compress_k, arrow_min_frac);
        end
    end

    wg_hw = avg_corrugation_width / 2 * 1e6;
    xl = xlim;
    plot(xl, [ wg_hw  wg_hw], 'w--', 'LineWidth', 1);
    plot(xl, [-wg_hw -wg_hw], 'w--', 'LineWidth', 1);
    hold off;
else
    fprintf('XY top view data not available.\n');
end

fprintf('\nDone.\n');


%% === Local Functions ===

function [coord_crop, idx] = crop_to_range(coord_m, range_um)
% Return indices and coordinate values within [range_um(1), range_um(2)] microns.
    lo = range_um(1) * 1e-6;
    hi = range_um(2) * 1e-6;
    if isinf(hi); hi = max(coord_m); end
    if isinf(lo); lo = min(coord_m); end
    idx = find(coord_m >= lo & coord_m <= hi);
    coord_crop = coord_m(idx);
end


function draw_smooth_quiver(ax, coord1, coord2, P1, P2, I_dB, opts) %#ok<DEFNU>
% Smoothed Poynting overlay: spatial averaging + staggered interpolated grid.
% Mirrors the algorithm from plot_field_poynting_overlay.m.
    sw     = opts.smooth_window;
    kernel = ones(sw) / sw^2;
    P1s    = imfilter(P1.', kernel, 'replicate');
    P2s    = imfilter(P2.', kernel, 'replicate');

    span1 = max(coord1) - min(coord1);
    span2 = max(coord2) - min(coord2);
    n1    = max(8, round(opts.arrows_per_span * opts.density_factor));
    n2    = max(8, round(opts.arrows_per_span * opts.density_factor * (span2 / span1)));

    c1 = linspace(min(coord1), max(coord1), n1);
    c2 = linspace(min(coord2), max(coord2), n2);
    [C1, C2] = meshgrid(c1, c2);

    d1 = c1(2) - c1(1);
    C1(2:2:end, :) = C1(2:2:end, :) + d1 / 2;
    C1 = max(min(coord1), min(max(coord1), C1));

    [C1o, C2o] = meshgrid(coord1, coord2);
    P1_q = interp2(C1o, C2o, P1s,    C1, C2, 'linear', 0);
    P2_q = interp2(C1o, C2o, P2s,    C1, C2, 'linear', 0);
    I_q  = interp2(C1o, C2o, I_dB.', C1, C2, 'linear', -200);

    Pmag     = sqrt(P1_q.^2 + P2_q.^2);
    Pmag_max = max(Pmag(:));
    if Pmag_max == 0; return; end

    P1_unit   = P1_q ./ (Pmag + 1e-30);
    P2_unit   = P2_q ./ (Pmag + 1e-30);
    Pmag_norm = Pmag / Pmag_max;
    mag_scale = opts.base_size + (1 - opts.base_size) * (Pmag_norm .^ opts.power_law_exp);

    max_dB = max(I_dB(:));
    mask   = I_q >= (max_dB - opts.threshold_dB);
    mask   = mask & (C1 > min(coord1) + 0.01*span1) & (C1 < max(coord1) - 0.01*span1) ...
                  & (C2 > min(coord2) + 0.01*span2) & (C2 < max(coord2) - 0.01*span2);

    vis = mag_scale .* mask;

    d2      = c2(2) - c2(1);
    max_len = sqrt(d1^2 + d2^2) * opts.arrow_scale * 0.85;
    quiver(ax, C1, C2, P1_unit.*vis.*max_len, P2_unit.*vis.*max_len, 0, ...
           'Color', opts.arrow_color, 'LineWidth', opts.arrow_linewidth);
end


function draw_poynting_quiver(coord1, coord2, P1, P2, skip1, skip2, scale, color, lw, I_dB, max_dB, threshold_dB, log_k, min_frac)
% Overlay Poynting vector arrows with log-compressed, floor-bounded display.
    n1 = numel(coord1);
    n2 = numel(coord2);

    % --- Uniform sampling ---
    idx1 = (1:skip1:n1)';
    idx2 = (1:skip2:n2)';

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

    % Log compression + minimum floor
    log_norm = log1p(Pmag_norm * log_k) / log1p(log_k);
    vis_scale = min_frac + (1 - min_frac) * log_norm;

    % --- dB intensity mask ---
    I_dB_q = I_dB(idx1, idx2).';
    mask = I_dB_q >= (max_dB - threshold_dB);

    % Edge margin
    c1_range = max(c1_q) - min(c1_q);
    c2_range = max(c2_q) - min(c2_q);
    mask = mask & (C1q >= min(c1_q) + 0.01*c1_range) & (C1q <= max(c1_q) - 0.01*c1_range) ...
               & (C2q >= min(c2_q) + 0.01*c2_range) & (C2q <= max(c2_q) - 0.01*c2_range);

    % --- Local direction coherence filter ---
    P1_unit = P1_q ./ (Pmag + 1e-30);
    P2_unit = P2_q ./ (Pmag + 1e-30);
    w = double(mask);
    local_mean1  = conv2(P1_unit .* w, ones(3,3)/9, 'same');
    local_mean2  = conv2(P2_unit .* w, ones(3,3)/9, 'same');
    local_mag    = sqrt(local_mean1.^2 + local_mean2.^2) + 1e-30;
    coherence    = (P1_unit .* local_mean1 + P2_unit .* local_mean2) ./ local_mag;
    neighbor_cnt = conv2(w, ones(3,3), 'same') - w;   % valid neighbors (excludes self)
    mask         = mask & (coherence > 0.72) & (neighbor_cnt >= 2);

    % --- Remove converging (intersecting) arrow pairs ---
    % Along X
    rmv = false(size(mask));
    valid_x   = mask(:,1:end-1) & mask(:,2:end);
    conv_x    = (P1_unit(:,1:end-1) > 0.25) & (P1_unit(:,2:end) < -0.25) & valid_x;
    keep_left = Pmag(:,1:end-1) >= Pmag(:,2:end);
    rmv(:,1:end-1) = rmv(:,1:end-1) | (conv_x & ~keep_left);
    rmv(:,2:end)   = rmv(:,2:end)   | (conv_x &  keep_left);
    mask = mask & ~rmv;

    % Along Z
    rmv = false(size(mask));
    valid_z  = mask(1:end-1,:) & mask(2:end,:);
    conv_z   = (P2_unit(1:end-1,:) > 0.25) & (P2_unit(2:end,:) < -0.25) & valid_z;
    keep_top = Pmag(1:end-1,:) >= Pmag(2:end,:);
    rmv(1:end-1,:) = rmv(1:end-1,:) | (conv_z & ~keep_top);
    rmv(2:end,:)   = rmv(2:end,:)   | (conv_z &  keep_top);
    mask = mask & ~rmv;

    vis_scale = vis_scale .* mask;

    % Draw
    P1_draw = (P1_q ./ (Pmag + 1e-30)) .* vis_scale;
    P2_draw = (P2_q ./ (Pmag + 1e-30)) .* vis_scale;
    quiver(C1q, C2q, P1_draw, P2_draw, scale, 'Color', color, 'LineWidth', lw);
end
