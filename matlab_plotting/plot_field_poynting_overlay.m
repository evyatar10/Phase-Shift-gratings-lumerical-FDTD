% plot_field_poynting_overlay.m
% Clean visualization of field intensity with Poynting vector overlay.
% Produces separate figures for XY (top), YZ (cross-section), and XZ (side) planes.
%
% The Poynting vectors are spatially smoothed to reveal macroscopic energy
% flow patterns, filtering out optical vortices and standing-wave noise.

addpath(fileparts(fileparts(mfilename('fullpath'))));
clear; clc;

%% --- Configuration ---
result_filepath = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\leaky_modes\results\result_80_periods_CONST_ff.mat";

pitch = 500e-9;   % Grating pitch [m]

% Visual parameters (struct for clean passing to helper)
opts = struct();
opts.field_colormap       = 'hot';
opts.dB_range             = 60;          % Dynamic range below peak [dB]
opts.arrow_color          = [0 0.9 0.9]; % Cyan arrows
opts.arrow_linewidth      = 0.9;
opts.arrow_scale          = 0.5;         % Arrow length scaling
opts.arrows_per_span      = 35;          % Target arrow count per axis
opts.smooth_window        = 11;          % Spatial averaging kernel size (odd)
opts.threshold_dB         = 55;          % Show arrows within this dB of peak
opts.density_factor       = 0.8;         % Arrow density reduction factor
opts.power_law_exp        = 0.25;        % Gentle magnitude scaling exponent
opts.base_size            = 0.6;         % Minimum arrow size (fraction of max)

% Geometry overlay
show_geometry         = true;
avg_corrugation_width = 800e-9;   % [m]
core_height           = 350e-9;   % [m]

%% --- 1. Load Data & Find Resonance ---
fprintf('Loading data...\n');
if ~exist(result_filepath, 'file')
    error('File not found: %s', result_filepath);
end
data = load(result_filepath);

T = squeeze(data.T);
wl = squeeze(data.wl_m);

% Resonance detection: find transmission peak inside Bragg stopband
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

%% --- 2. Plot XY Plane (Top View) ---
if isfield(data, 'field_xy')
    fprintf('\n--- XY Plane (Top View) ---\n');
    d = data.field_xy;
    x = double(d.x); y = double(d.y); z = double(d.z);
    lam = double(d.lambda_3d);
    Nx = length(x); Ny = length(y); Nz = length(z); Nlam = length(lam);

    [~, idx_lam] = min(abs(lam - wl_res));
    [~, idx_z0]  = min(abs(z));
    wl_plot = lam(idx_lam);

    E_5D = reshape(d.E_res, [Nx, Ny, Nz, Nlam, 3]);
    E_plane = squeeze(E_5D(:, :, idx_z0, idx_lam, :));
    I_dB = 10 * log10(sum(abs(E_plane).^2, 3));

    figure('Name', 'XY: Field + Poynting', 'Color', 'w', 'Position', [80 150 950 450]);
    ax = gca;
    imagesc(ax, x*1e6, y*1e6, I_dB');
    set(ax, 'YDir', 'normal');
    colormap(ax, opts.field_colormap);
    max_dB = max(I_dB(:));
    try clim(ax, [max_dB - opts.dB_range, max_dB]); catch; caxis([max_dB - opts.dB_range, max_dB]); end
    cb = colorbar(ax); ylabel(cb, '|E|^2  [dB]');
    xlabel(ax, 'X [\mum]'); ylabel(ax, 'Y [\mum]');
    title(ax, sprintf('XY (Top View)  \\lambda = %.3f nm', wl_plot*1e9));
    hold(ax, 'on');

    % Poynting overlay
    if isfield(d, 'P_res') && ~isempty(d.P_res)
        P_5D = reshape(double(d.P_res), [Nx, Ny, Nz, Nlam, 3]);
        Px = squeeze(P_5D(:, :, idx_z0, idx_lam, 1));
        Py = squeeze(P_5D(:, :, idx_z0, idx_lam, 2));
        draw_field_with_poynting(ax, x*1e6, y*1e6, Px, Py, I_dB, opts);
    end

    % Geometry: waveguide width boundaries
    if show_geometry
        wg_hw = avg_corrugation_width / 2 * 1e6;
        xl = xlim(ax);
        plot(ax, xl, [ wg_hw  wg_hw], 'w--', 'LineWidth', 1);
        plot(ax, xl, [-wg_hw -wg_hw], 'w--', 'LineWidth', 1);
    end
    hold(ax, 'off');
end

%% --- 3. Plot YZ Plane (Cross Section) ---
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
    E_plane = squeeze(E_5D(idx_x0, :, :, idx_lam, :));
    I_dB = 10 * log10(sum(abs(E_plane).^2, 3));

    figure('Name', 'YZ: Field + Poynting', 'Color', 'w', 'Position', [130 200 700 500]);
    ax = gca;
    imagesc(ax, y*1e6, z*1e6, I_dB');
    set(ax, 'YDir', 'normal');
    colormap(ax, opts.field_colormap);
    max_dB = max(I_dB(:));
    try clim(ax, [max_dB - opts.dB_range, max_dB]); catch; caxis([max_dB - opts.dB_range, max_dB]); end
    cb = colorbar(ax); ylabel(cb, '|E|^2  [dB]');
    xlabel(ax, 'Y [\mum]'); ylabel(ax, 'Z [\mum]');
    title(ax, sprintf('YZ (Cross Section)  \\lambda = %.3f nm', wl_plot*1e9));
    hold(ax, 'on');

    if isfield(d, 'P_res') && ~isempty(d.P_res)
        P_5D = reshape(double(d.P_res), [Nx, Ny, Nz, Nlam, 3]);
        Py = squeeze(P_5D(idx_x0, :, :, idx_lam, 2));
        Pz = squeeze(P_5D(idx_x0, :, :, idx_lam, 3));
        draw_field_with_poynting(ax, y*1e6, z*1e6, Py, Pz, I_dB, opts);
    end

    if show_geometry
        wg_hw = avg_corrugation_width / 2 * 1e6;
        wg_hh = core_height / 2 * 1e6;
        rectangle('Position', [-wg_hw, -wg_hh, 2*wg_hw, 2*wg_hh], ...
                  'EdgeColor', 'w', 'LineStyle', '--', 'LineWidth', 1);
    end
    hold(ax, 'off');
end

%% --- 4. Plot XZ Plane (Side View) ---
if isfield(data, 'field_xz_side')
    fprintf('--- XZ Plane (Side View) ---\n');
    d = data.field_xz_side;
    x = double(d.x); y = double(d.y); z = double(d.z);
    lam = double(d.lambda_3d);
    Nx = length(x); Ny = length(y); Nz = length(z); Nlam = length(lam);

    [~, idx_lam] = min(abs(lam - wl_res));
    [~, idx_y0]  = min(abs(y));
    wl_plot = lam(idx_lam);

    E_5D = reshape(d.E_res, [Nx, Ny, Nz, Nlam, 3]);
    E_plane = squeeze(E_5D(:, idx_y0, :, idx_lam, :));
    I_dB = 10 * log10(sum(abs(E_plane).^2, 3));

    figure('Name', 'XZ: Field + Poynting', 'Color', 'w', 'Position', [180 100 950 500]);
    ax = gca;
    imagesc(ax, x*1e6, z*1e6, I_dB');
    set(ax, 'YDir', 'normal');
    colormap(ax, opts.field_colormap);
    max_dB = max(I_dB(:));
    try clim(ax, [max_dB - opts.dB_range, max_dB]); catch; caxis([max_dB - opts.dB_range, max_dB]); end
    cb = colorbar(ax); ylabel(cb, '|E|^2  [dB]');
    xlabel(ax, 'X [\mum]'); ylabel(ax, 'Z [\mum]');
    title(ax, sprintf('XZ (Side View)  \\lambda = %.3f nm', wl_plot*1e9));
    hold(ax, 'on');

    if isfield(d, 'P_res') && ~isempty(d.P_res)
        P_5D = reshape(double(d.P_res), [Nx, Ny, Nz, Nlam, 3]);
        Px = squeeze(P_5D(:, idx_y0, :, idx_lam, 1));
        Pz = squeeze(P_5D(:, idx_y0, :, idx_lam, 3));
        draw_field_with_poynting(ax, x*1e6, z*1e6, Px, Pz, I_dB, opts);
    end

    if show_geometry
        wg_hh = core_height / 2 * 1e6;
        xl = xlim(ax);
        plot(ax, xl, [ wg_hh  wg_hh], 'w--', 'LineWidth', 1);
        plot(ax, xl, [-wg_hh -wg_hh], 'w--', 'LineWidth', 1);
    end
    hold(ax, 'off');
else
    fprintf('XZ side view data not available. Re-run simulation to include it.\n');
end

fprintf('\nDone.\n');


%% ========================================================================
%  Helper Function
%  ========================================================================
function draw_field_with_poynting(ax, coord1, coord2, P1, P2, I_dB, opts)
% Overlay spatially-smoothed Poynting vector arrows on the current axes.
%
%   coord1, coord2 : 1D coordinate arrays (already in plot units, e.g. um)
%   P1, P2         : 2D Poynting components [N1 x N2] matching coord1 x coord2
%   I_dB           : 2D intensity in dB [N1 x N2] for masking
%   opts           : struct with visualization parameters

    % --- 1. Spatial smoothing ---
    % Averages out optical vortices and standing-wave noise to reveal
    % macroscopic radiation beams and net energy flow direction.
    sw = opts.smooth_window;
    kernel = ones(sw) / sw^2;
    P1s = imfilter(P1.', kernel, 'replicate');
    P2s = imfilter(P2.', kernel, 'replicate');

    % --- 2. Build uniform staggered grid ---
    span1 = max(coord1) - min(coord1);
    span2 = max(coord2) - min(coord2);
    n1 = max(8, round(opts.arrows_per_span * opts.density_factor));
    n2 = max(8, round(opts.arrows_per_span * opts.density_factor * (span2 / span1)));
    n2 = max(8, n2);

    c1 = linspace(min(coord1), max(coord1), n1);
    c2 = linspace(min(coord2), max(coord2), n2);
    [C1, C2] = meshgrid(c1, c2);

    d1 = c1(2) - c1(1);

    % Stagger alternating rows to avoid visual symmetry artifacts
    C1(2:2:end, :) = C1(2:2:end, :) + d1 / 2;
    C1 = max(min(coord1), min(max(coord1), C1));

    % --- 3. Interpolate smoothed data onto grid ---
    [C1_orig, C2_orig] = meshgrid(coord1, coord2);
    P1_q  = interp2(C1_orig, C2_orig, P1s,     C1, C2, 'linear', 0);
    P2_q  = interp2(C1_orig, C2_orig, P2s,     C1, C2, 'linear', 0);
    I_q   = interp2(C1_orig, C2_orig, I_dB.',  C1, C2, 'linear', -200);

    % --- 4. Normalize to unit vectors + magnitude scaling ---
    Pmag = sqrt(P1_q.^2 + P2_q.^2);
    Pmag_max = max(Pmag(:));
    if Pmag_max == 0; return; end

    P1_unit = P1_q ./ (Pmag + 1e-30);
    P2_unit = P2_q ./ (Pmag + 1e-30);

    Pmag_norm = Pmag / Pmag_max;
    mag_scale = opts.base_size + (1 - opts.base_size) * (Pmag_norm .^ opts.power_law_exp);

    % --- 5. Intensity masking ---
    max_dB = max(I_dB(:));
    mask = I_q >= (max_dB - opts.threshold_dB);

    % Small boundary margin to avoid edge artifacts
    margin1 = 0.01 * span1;
    margin2 = 0.01 * span2;
    mask = mask & (C1 >= min(coord1) + margin1) & (C1 <= max(coord1) - margin1) ...
               & (C2 >= min(coord2) + margin2) & (C2 <= max(coord2) - margin2);

    vis = mag_scale .* mask;

    % --- 6. Draw arrows ---
    d2 = c2(2) - c2(1);
    max_len = sqrt(d1^2 + d2^2) * opts.arrow_scale * 0.85;

    P1_draw = P1_unit .* vis .* max_len;
    P2_draw = P2_unit .* vis .* max_len;

    quiver(ax, C1, C2, P1_draw, P2_draw, 0, ...
           'Color', opts.arrow_color, 'LineWidth', opts.arrow_linewidth);
end
