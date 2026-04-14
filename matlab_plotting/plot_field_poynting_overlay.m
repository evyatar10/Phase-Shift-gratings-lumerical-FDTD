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
avg_corrugation_width = 800e-9;   % average corrugation width [m] (XY / YZ views)
corrugation_depth     = 300e-9;   % full corrugation depth [m] (= wide - narrow)
core_height           = 350e-9;   % waveguide slab height [m]
% derived widths:
width_narrow = avg_corrugation_width - corrugation_depth / 2;
width_wide   = avg_corrugation_width + corrugation_depth / 2;
geom_color            = [0.7 0.7 0.7]; % structure overlay color
geom_lw               = 1.5;           % structure line width

% Structure drawing mode for XZ side view: 'uniform' or 'apodized'
geom_mode             = 'uniform';

% Apodization parameters (only used when geom_mode = 'apodized')
center_mod_depth_nm   = 40.0;    % corrugation depth at grating center [nm]
apod_method           = 'linear'; % 'linear' or 'tanh'
tanh_steepness        = 2.0;

% Auto-detection overrides (leave [] for auto)
n_periods_override     = [];
n_apod_override        = [];
cavity_length_override = [];  % [m]; default: pitch/2

%% --- 1. Load Data ---
fprintf('Loading data...\n');
if ~exist(result_filepath, 'file')
    error('File not found: %s', result_filepath);
end
data = load(result_filepath);

wl_res = double(data.resonance_wavelength_nm) * 1e-9;
fprintf('Resonance wavelength: %.3f nm\n', wl_res*1e9);

%% --- 1b. Resolve Geometry Parameters ---
[n_periods_r, n_apod_r, cavity_length_r] = resolve_geometry_params( ...
    result_filepath, data, pitch, n_periods_override, n_apod_override, ...
    cavity_length_override);

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
    clim(ax, [max_dB - opts.dB_range, max_dB]);
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

    % Geometry: corrugated waveguide width boundaries (XY = top view)
    if show_geometry
        xl = xlim(ax); yl = ylim(ax);
        [xp, wp] = make_grating_profile(pitch, width_narrow, width_wide, ...
            n_periods_r, cavity_length_r, core_height, ...
            n_apod_r, center_mod_depth_nm*1e-9, geom_mode, apod_method, tanh_steepness, 'xy');
        plot(ax, xp*1e6,  wp*1e6, '-', 'Color', geom_color, 'LineWidth', geom_lw);
        plot(ax, xp*1e6, -wp*1e6, '-', 'Color', geom_color, 'LineWidth', geom_lw);
        xlim(ax, xl); ylim(ax, yl);
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
    clim(ax, [max_dB - opts.dB_range, max_dB]);
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
                  'EdgeColor', geom_color, 'LineStyle', '-', 'LineWidth', geom_lw);
        xl2 = xlim(ax); yl2 = ylim(ax);
        text(ax, xl2(1) + 0.03*(xl2(2)-xl2(1)), yl2(2) - 0.05*(yl2(2)-yl2(1)), ...
             'Phase-shift defect x-section', 'Color', geom_color, 'FontSize', 8, ...
             'VerticalAlignment', 'top');
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
    clim(ax, [max_dB - opts.dB_range, max_dB]);
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
        xl = xlim(ax); yl = ylim(ax);
        wg_hh = core_height / 2 * 1e6;
        plot(ax, xl, [ wg_hh  wg_hh], '-', 'Color', geom_color, 'LineWidth', geom_lw);
        plot(ax, xl, [-wg_hh -wg_hh], '-', 'Color', geom_color, 'LineWidth', geom_lw);
        plot(ax, [0 0], yl, ':', 'Color', geom_color, 'LineWidth', 1.0);   % defect marker
        xlim(ax, xl); ylim(ax, yl);
    end
    hold(ax, 'off');
else
    fprintf('XZ side view data not available. Re-run simulation to include it.\n');
end

fprintf('\nDone.\n');


%% ========================================================================
%  Helper Functions
%  ========================================================================
function [n_per, n_apod, cav_len] = resolve_geometry_params( ...
        fpath, data, pitch, n_per_manual, n_apod_manual, cav_len_manual)
    n_per   = n_per_manual;
    n_apod  = n_apod_manual;
    cav_len = cav_len_manual;
    if isempty(n_per)   && isfield(data, 'n_periods');            n_per   = double(data.n_periods);            end
    if isempty(n_per)   && isfield(data, 'n_periods_each_side'); n_per   = double(data.n_periods_each_side);  end
    if isempty(n_apod)  && isfield(data, 'n_apod_periods');      n_apod  = double(data.n_apod_periods);       end
    if isempty(cav_len) && isfield(data, 'cavity_length_m'); cav_len = double(data.cavity_length_m); end
    [~, fname] = fileparts(fpath);
    if isempty(n_per)
        tok = regexp(fname, '(\d+)_periods', 'tokens', 'once');
        if ~isempty(tok); n_per = str2double(tok{1}); end
    end
    if isempty(n_apod)
        tok = regexp(fname, '_(\d+)_apod', 'tokens', 'once');
        if ~isempty(tok); n_apod = str2double(tok{1}); end
    end
    if isempty(cav_len)
        tok = regexp(fname, 'L_cav_(\d+)', 'tokens', 'once');
        if ~isempty(tok); cav_len = str2double(tok{1}) * 1e-9; end
    end
    if ~isempty(regexp(fname, '_tanh', 'once'))
        fprintf('  Filename suggests tanh apodization — set apod_method=''tanh'' if drawing apodized.\n');
    end
    if isempty(n_per) && isfield(data, 'L_device')
        cav_try = ternary(~isempty(cav_len), cav_len, pitch / 2);
        n_calc  = (double(data.L_device) / 2 - cav_try / 2) / pitch;
        n_round = round(n_calc);
        if abs(n_calc - n_round) < 0.02
            n_per = n_round;
            fprintf('  n_periods inferred from L_device: %d\n', n_per);
        end
    end
    if isempty(cav_len); cav_len = pitch / 2; end
    if isempty(n_apod);  n_apod  = 0;         end
    if isempty(n_per)
        error('Could not determine n_periods. Set n_periods_override manually.');
    end
    fprintf('Geometry: n_periods=%d, n_apod=%d, cavity_length=%.0f nm\n', n_per, n_apod, cav_len*1e9);
end


function out = ternary(cond, a, b)
    if cond; out = a; else; out = b; end
end


function [x_vec, w_half_vec] = make_grating_profile(pitch, w_narrow, w_wide, ...
        n_periods, cav_length, core_height, n_apod, center_mod_depth, geom_mode, ...
        apod_method, tanh_steepness, view_plane, tooth_shift, lengthen_cav)
% Step-function boundary profile for a Bragg grating with phase-shift defect.
% See plot_field_poynting_zoom.m for full documentation.
    if nargin < 12 || isempty(view_plane),   view_plane   = 'xy'; end
    if nargin < 13 || isempty(tooth_shift),  tooth_shift  = 0;    end
    if nargin < 14 || isempty(lengthen_cav), lengthen_cav = true;  end
    half_pitch      = pitch / 2;
    avg_width       = (w_narrow + w_wide) / 2;
    full_depth_edge = w_wide - w_narrow;
    hw_narrow_arr = zeros(1, n_periods);
    hw_wide_arr   = zeros(1, n_periods);
    for d = 1:n_periods
        if strcmp(geom_mode, 'apodized') && n_apod > 0 && d <= n_apod
            frac = (d - 1) / n_apod;
            if strcmp(apod_method, 'tanh')
                frac = tanh(tanh_steepness * 2 * frac) / tanh(2 * tanh_steepness);
            end
            mod_depth = center_mod_depth + (full_depth_edge - center_mod_depth) * frac;
        else
            mod_depth = full_depth_edge;
        end
        if strcmp(view_plane, 'xy')
            hw_narrow_arr(d) = (avg_width - mod_depth / 2) / 2;
            hw_wide_arr(d)   = (avg_width + mod_depth / 2) / 2;
        else
            hw_narrow_arr(d) = core_height / 2;
            hw_wide_arr(d)   = core_height / 2 + mod_depth / 2;
        end
    end
    if strcmp(view_plane, 'xy')
        hw_cavity = hw_narrow_arr(1);
    else
        hw_cavity = core_height / 2;
    end
    cav_extra = 0;
    if tooth_shift > 0 && lengthen_cav; cav_extra = 2 * tooth_shift; end
    eff_cav_length = cav_length + cav_extra;
    n_segs = 4 * n_periods + 1;
    seg_xl = zeros(1, n_segs); seg_xr = zeros(1, n_segs); seg_hw = zeros(1, n_segs);
    k = 0;
    x = -(n_periods * pitch + eff_cav_length / 2);
    for d = n_periods:-1:2
        k=k+1; seg_xl(k)=x; seg_xr(k)=x+half_pitch; seg_hw(k)=hw_narrow_arr(d); x=x+half_pitch;
        k=k+1; seg_xl(k)=x; seg_xr(k)=x+half_pitch; seg_hw(k)=hw_wide_arr(d);   x=x+half_pitch;
    end
    L_narrow_1_len = half_pitch - tooth_shift;
    k=k+1; seg_xl(k)=x; seg_xr(k)=x+L_narrow_1_len; seg_hw(k)=hw_narrow_arr(1); x=x+L_narrow_1_len;
    k=k+1; seg_xl(k)=x; seg_xr(k)=x+half_pitch;     seg_hw(k)=hw_wide_arr(1);   x=x+half_pitch;
    k=k+1; seg_xl(k)=x; seg_xr(k)=x+eff_cav_length;  seg_hw(k)=hw_cavity;        x=x+eff_cav_length;
    k=k+1; seg_xl(k)=x; seg_xr(k)=x+half_pitch; seg_hw(k)=hw_narrow_arr(1); x=x+half_pitch;
    k=k+1; seg_xl(k)=x; seg_xr(k)=x+half_pitch; seg_hw(k)=hw_wide_arr(1);   x=x+half_pitch;
    if n_periods >= 2
        R_narrow_2_len = half_pitch - tooth_shift;
        k=k+1; seg_xl(k)=x; seg_xr(k)=x+R_narrow_2_len; seg_hw(k)=hw_narrow_arr(2); x=x+R_narrow_2_len;
        k=k+1; seg_xl(k)=x; seg_xr(k)=x+half_pitch;     seg_hw(k)=hw_wide_arr(2);   x=x+half_pitch;
    end
    for d = 3:n_periods
        k=k+1; seg_xl(k)=x; seg_xr(k)=x+half_pitch; seg_hw(k)=hw_narrow_arr(d); x=x+half_pitch;
        k=k+1; seg_xl(k)=x; seg_xr(k)=x+half_pitch; seg_hw(k)=hw_wide_arr(d);   x=x+half_pitch;
    end
    x_vec      = seg_xl(1);
    w_half_vec = seg_hw(1);
    for i = 1:k
        hw = seg_hw(i);
        if i > 1 && abs(seg_hw(i) - seg_hw(i-1)) > 1e-30
            x_vec(end+1)      = seg_xl(i);  %#ok<AGROW>
            w_half_vec(end+1) = hw;         %#ok<AGROW>
        end
        x_vec(end+1)      = seg_xr(i);     %#ok<AGROW>
        w_half_vec(end+1) = hw;            %#ok<AGROW>
    end
end


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
