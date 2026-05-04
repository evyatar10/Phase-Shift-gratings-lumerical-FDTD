% plot_field_3d.m
% Visualization of full 3D field volume (data.field_3d) for 1-4 structures.
%
% For each selected result file produces four figures:
%   1. XZ side view (slice at y=0)
%   2. XY top view  (slice at z=0)
%   3. YZ multi-depth panel: 3x3 = 9 cross-sections at physically meaningful X
%   4. 3D isosurface of |E|^2 with Poynting-vector quiver overlay
%
% All figures across all structures share a single dB color limit anchored to
% the global peak so that amplitude differences between devices are preserved.
%
% Requires that the simulation was run with cfg.monitors.record_3d_fields = True.

addpath(fileparts(fileparts(mfilename('fullpath'))));
clear; clc;
close all;
%% --- Configuration ---
prefs_file = fullfile(fileparts(mfilename('fullpath')), 'plot_prefs.mat');

% Crop bounds [um] for XZ / XY views
crop_val      = 12;
x_range_xz    = [-crop_val, crop_val];
z_range_xz    = [-Inf, Inf];
x_range_xy    = [-crop_val, crop_val];
y_range_xy    = [-Inf, Inf];

% Crop bounds [um] for 3D field-profile figure (wider — show field decay)
x_range_3d    = [-20, 20];
enable_3d_fig = false;          % set true to render the 3D volumetric figure

% YZ panel crop
y_range_yz_um = [-3, 3];
z_range_yz_um = [-2, 2];

% Visual parameters
field_colormap        = 'hot';
dB_limit              = 60;        % dynamic range below global peak

% Poynting overlay (re-uses grid-based algo from plot_field_poynting_zoom.m)
arrow_color           = 'c';
arrow_linewidth       = 0.9;
poynting_threshold_dB = 55;
arrows_per_axis       = 25;
min_skip              = 2;
arrow_scale           = 0.55;
log_compress_k        = 10;
arrow_min_frac        = 0.60;

% YZ subplot Poynting overlay (lower density)
yz_arrows_per_axis    = 14;
yz_threshold_dB       = 70;     % looser intensity gate for YZ arrows

% Geometry overlay
avg_corrugation_width = 800e-9;
corrugation_depth     = 300e-9;
core_height           = 350e-9;
pitch                 = 500e-9;
width_narrow          = avg_corrugation_width - corrugation_depth / 2;
width_wide            = avg_corrugation_width + corrugation_depth / 2;
geom_color            = [0.7 0.7 0.7];
geom_lw               = 1.5;

% Apodization geometry (matches plot_field_poynting_zoom.m defaults)
geom_mode             = 'apodized';
center_mod_depth_nm   = 4.0;
apod_method           = 'linear';
tanh_steepness        = 2.0;
tooth_shift_override  = [];
lengthen_cavity       = true;
n_periods_override    = [];
n_apod_override       = [];
cavity_length_override = [];

% 3D isosurface
iso_levels_dB         = [-10, -25];   % below global peak
iso_alpha             = [0.55, 0.25];
iso_color             = {[1.0 0.4 0.0], [1.0 0.85 0.2]};
quiver3_target_count  = 12;           % per axis (~12^3 candidate points)
quiver3_threshold_dB  = 30;           % below global peak
quiver3_color         = [0 0.9 0.9];
quiver3_lw            = 0.8;
quiver3_scale         = 1.4;

%% --- File selection (1-4 files) ---
result_filepaths = {};
start_path = '*.mat';
if exist(prefs_file, 'file')
    p = load(prefs_file);
    if isfield(p, 'field3d_last_filepaths') && iscell(p.field3d_last_filepaths) ...
            && ~isempty(p.field3d_last_filepaths) ...
            && all(cellfun(@(f) exist(f,'file'), p.field3d_last_filepaths))
        names = '';
        for i = 1:numel(p.field3d_last_filepaths)
            [~,fn,fe] = fileparts(p.field3d_last_filepaths{i});
            names = [names sprintf('  %d) %s%s\n', i, fn, fe)]; %#ok<AGROW>
        end
        ans_q = questdlg(['Use last selection (' num2str(numel(p.field3d_last_filepaths)) ' files)?' newline names], ...
            'Select Files', 'Yes', 'Browse...', 'Yes');
        if strcmp(ans_q, 'Yes')
            result_filepaths = p.field3d_last_filepaths;
        end
    end
    if isfield(p, 'field3d_last_filepaths') && iscell(p.field3d_last_filepaths) ...
            && ~isempty(p.field3d_last_filepaths)
        last_dir = fileparts(p.field3d_last_filepaths{1});
        if isfolder(last_dir); start_path = fullfile(last_dir, '*.mat'); end
    elseif isfield(p, 'field_last_filepath')
        last_dir = fileparts(p.field_last_filepath);
        if isfolder(last_dir); start_path = fullfile(last_dir, '*.mat'); end
    end
end

if isempty(result_filepaths)
    [files, folder] = uigetfile(start_path, ...
        'Select 1 or 2 result .mat files (Ctrl+click for both)', 'MultiSelect', 'on');
    if isequal(files, 0)
        disp('No files selected.'); return;
    end
    if ischar(files); files = {files}; end
    if numel(files) > 2
        warning('More than 2 files selected; using only the first 2 (script is optimized for pair comparison).');
        files = files(1:2);
    end
    result_filepaths = cellfun(@(f) fullfile(folder, f), files, 'UniformOutput', false);
end

field3d_last_filepaths = result_filepaths;
if exist(prefs_file, 'file')
    save(prefs_file, 'field3d_last_filepaths', '-append');
else
    save(prefs_file, 'field3d_last_filepaths');
end

n_struct = numel(result_filepaths);
fprintf('Selected %d structure(s).\n', n_struct);

%% --- Load all structures, compute global dB peak ---
S = cell(1, n_struct);
global_peak_dB = -Inf;
for s = 1:n_struct
    fpath = result_filepaths{s};
    fprintf('\n[%d/%d] Loading %s ...\n', s, n_struct, fpath);
    if ~exist(fpath, 'file'); error('File not found: %s', fpath); end
    data = load(fpath);
    if ~isfield(data, 'field_3d') || isempty(data.field_3d)
        error(['File "%s" does not contain field_3d. Re-run the simulation with ' ...
               'cfg.monitors.record_3d_fields = True.'], fpath);
    end
    d3 = data.field_3d;
    x = double(d3.x); y = double(d3.y); z = double(d3.z);
    lam = double(d3.lambda_3d);
    Nx = numel(x); Ny = numel(y); Nz = numel(z); Nlam = numel(lam);

    wl_res = double(data.resonance_wavelength_nm) * 1e-9;
    [~, idx_lam] = min(abs(lam - wl_res));

    E_5D = reshape(d3.E_res, [Nx, Ny, Nz, Nlam, 3]);
    E_res_lam = squeeze(E_5D(:,:,:,idx_lam,:));         % (Nx,Ny,Nz,3)
    I_3D = sum(abs(E_res_lam).^2, 4);                    % (Nx,Ny,Nz)
    I_3D_dB = 10 * log10(I_3D + 1e-300);

    has_P = isfield(d3, 'P_res') && ~isempty(d3.P_res);
    if has_P
        P_5D = reshape(double(d3.P_res), [Nx, Ny, Nz, Nlam, 3]);
        P_3D = squeeze(P_5D(:,:,:,idx_lam,:));           % (Nx,Ny,Nz,3) - real already from Lumerical
    else
        P_3D = [];
    end

    [~, fn, ~] = fileparts(fpath);
    [n_per, n_apod, cav_len, t_shift] = resolve_geometry_params( ...
        fpath, data, pitch, n_periods_override, n_apod_override, ...
        cavity_length_override, tooth_shift_override);
    if n_apod > 0
        geom_str = sprintf('N=%d, %d apod', n_per, n_apod);
    else
        geom_str = sprintf('N=%d periods', n_per);
    end

    S{s} = struct( ...
        'fpath',     fpath, ...
        'tag',       fn, ...
        'x',         x, 'y', y, 'z', z, ...
        'lam',       lam(idx_lam), ...
        'I_3D',      I_3D, 'I_3D_dB', I_3D_dB, ...
        'P_3D',      P_3D, 'has_P', has_P, ...
        'n_per',     n_per, 'n_apod', n_apod, ...
        'cav_len',   cav_len, 't_shift', t_shift, ...
        'geom_str',  geom_str);

    pk = max(I_3D_dB(:));
    if pk > global_peak_dB; global_peak_dB = pk; end
    fprintf('  peak |E|^2 dB = %.2f\n', pk);
end

clim_global = [global_peak_dB - dB_limit, global_peak_dB];
fprintf('\nGlobal dB clim = [%.2f, %.2f]\n', clim_global(1), clim_global(2));

%% --- Plotting ---
% n_struct == 1 → per-structure figures (4 windows).
% n_struct == 2 → side-by-side comparison figures (XZ, XY, YZ paired in one
%                 window each) + one 3D isosurface figure per structure with
%                 camera linked across the two so they rotate together.
if n_struct == 1
    St = S{1};
    fprintf('\n=== Plotting single structure: %s ===\n', St.tag);
    eff_cav = St.cav_len + ternary(St.t_shift > 0 && lengthen_cavity, 2*St.t_shift, 0);

    plot_xz_view(St, x_range_xz, z_range_xz, clim_global, dB_limit, ...
        field_colormap, geom_color, geom_lw, geom_mode, apod_method, tanh_steepness, ...
        center_mod_depth_nm, lengthen_cavity, pitch, width_narrow, width_wide, ...
        core_height, arrows_per_axis, min_skip, arrow_scale, arrow_color, ...
        arrow_linewidth, poynting_threshold_dB, log_compress_k, arrow_min_frac);

    plot_xy_view(St, x_range_xy, y_range_xy, clim_global, dB_limit, ...
        field_colormap, geom_color, geom_lw, geom_mode, apod_method, tanh_steepness, ...
        center_mod_depth_nm, lengthen_cavity, pitch, width_narrow, width_wide, ...
        core_height, arrows_per_axis, min_skip, arrow_scale, arrow_color, ...
        arrow_linewidth, poynting_threshold_dB, log_compress_k, arrow_min_frac);

    plot_yz_panel(St, eff_cav, pitch, width_narrow, width_wide, core_height, ...
        y_range_yz_um, z_range_yz_um, clim_global, field_colormap, ...
        geom_color, geom_lw, yz_arrows_per_axis, min_skip, arrow_scale, ...
        arrow_color, arrow_linewidth, poynting_threshold_dB, ...
        log_compress_k, arrow_min_frac);

    if enable_3d_fig
        plot_3d_iso([], St, clim_global, iso_levels_dB, iso_alpha, iso_color, ...
            quiver3_target_count, quiver3_threshold_dB, quiver3_color, ...
            quiver3_lw, quiver3_scale, eff_cav, width_narrow, core_height, ...
            pitch, St.n_per, x_range_3d);
    end

else  % n_struct == 2  → comparison layouts
    fprintf('\n=== 2-structure comparison: %s  vs  %s ===\n', S{1}.tag, S{2}.tag);

    plot_xz_compare(S, x_range_xz, z_range_xz, clim_global, ...
        field_colormap, geom_color, geom_lw, lengthen_cavity, core_height, ...
        arrows_per_axis, min_skip, arrow_scale, arrow_color, ...
        arrow_linewidth, poynting_threshold_dB, log_compress_k, arrow_min_frac);

    plot_xy_compare(S, x_range_xy, y_range_xy, clim_global, ...
        field_colormap, geom_color, geom_lw, geom_mode, apod_method, tanh_steepness, ...
        center_mod_depth_nm, lengthen_cavity, pitch, width_narrow, width_wide, ...
        core_height, arrows_per_axis, min_skip, arrow_scale, arrow_color, ...
        arrow_linewidth, poynting_threshold_dB, log_compress_k, arrow_min_frac);

    for kind = {'narrow', 'wide', 'transition'}
        plot_yz_compare(S, pitch, width_narrow, width_wide, core_height, ...
            y_range_yz_um, z_range_yz_um, clim_global, field_colormap, ...
            geom_color, geom_lw, lengthen_cavity, yz_arrows_per_axis, min_skip, ...
            arrow_scale, arrow_color, arrow_linewidth, yz_threshold_dB, ...
            log_compress_k, arrow_min_frac, kind{1});
    end

    plot_yz_cavity_compare(S, width_narrow, core_height, ...
        y_range_yz_um, z_range_yz_um, clim_global, field_colormap, ...
        geom_color, geom_lw, yz_arrows_per_axis, min_skip, ...
        arrow_scale, arrow_color, arrow_linewidth, yz_threshold_dB, ...
        log_compress_k, arrow_min_frac);

    if enable_3d_fig
        plot_3d_iso_compare(S, clim_global, iso_levels_dB, iso_alpha, iso_color, ...
            quiver3_target_count, quiver3_threshold_dB, quiver3_color, ...
            quiver3_lw, quiver3_scale, lengthen_cavity, width_narrow, core_height, ...
            pitch, x_range_3d);
    end
end

fprintf('\nDone.\n');


%% =====================================================================
%  Comparison plots (n_struct == 2)
%% =====================================================================

function plot_xz_compare(S, x_range, z_range, clim_g, cmap, geom_color, geom_lw, ...
        lengthen_cav, core_height, arrows_per_axis, min_skip, arrow_scale, ...
        arrow_color, arrow_lw, thr_dB, log_k, min_frac)
    % Two stacked XZ panels in one window with a shared colorbar.
    fig = figure('Name', sprintf('XZ compare — %s vs %s', display_name(S{1}), display_name(S{2})), ...
        'Color', 'w', 'Position', [120 70 950 900]);
    tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, sprintf('XZ side view — %s  vs  %s', display_name(S{1}), display_name(S{2})), ...
        'FontSize', 14, 'FontWeight', 'bold');

    for s = 1:2
        St = S{s};
        eff_cav = St.cav_len + ternary(St.t_shift > 0 && lengthen_cav, 2*St.t_shift, 0);
        [~, iy0] = min(abs(St.y));
        [x_c, ix] = crop_to_range(St.x, x_range);
        [z_c, iz] = crop_to_range(St.z, z_range);
        I_xz_dB = squeeze(St.I_3D_dB(ix, iy0, iz));

        ax = nexttile;
        imagesc(x_c*1e6, z_c*1e6, I_xz_dB');
        set(ax, 'YDir', 'normal'); colormap(ax, cmap); clim(ax, clim_g);
        xlabel(ax, 'X [\mum]'); ylabel(ax, 'Z [\mum]');
        title(ax, display_name(St), 'FontSize', 13, 'FontWeight', 'bold');
        hold(ax, 'on');

        if St.has_P
            Px = squeeze(St.P_3D(ix, iy0, iz, 1));
            Pz = squeeze(St.P_3D(ix, iy0, iz, 3));
            sx = max(min_skip, round(numel(ix) / arrows_per_axis));
            sz = max(min_skip, round(numel(iz) / arrows_per_axis));
            draw_poynting_quiver(x_c*1e6, z_c*1e6, Px, Pz, sx, sz, ...
                arrow_scale, arrow_color, arrow_lw, ...
                I_xz_dB, max(I_xz_dB(:)), thr_dB, log_k, min_frac);
        end

        xl = xlim(ax); yl = ylim(ax);
        wg_hh = core_height/2*1e6;
        plot(ax, xl, [ wg_hh  wg_hh], '-', 'Color', geom_color, 'LineWidth', geom_lw);
        plot(ax, xl, [-wg_hh -wg_hh], '-', 'Color', geom_color, 'LineWidth', geom_lw);
        draw_cavity_hatch(0, 0, eff_cav*1e6, core_height*1e6, geom_color, geom_lw, 0.15);
        xlim(ax, xl); ylim(ax, yl); hold(ax, 'off');
    end

    cb = colorbar; cb.Layout.Tile = 'east';
    ylabel(cb, '10\cdotlog_{10}(|E|^2) [dB]  (global)');
end


function plot_xy_compare(S, x_range, y_range, clim_g, cmap, geom_color, geom_lw, ...
        geom_mode, apod_method, tanh_steepness, center_mod_depth_nm, lengthen_cav, ...
        pitch, w_narrow, w_wide, core_height, arrows_per_axis, min_skip, arrow_scale, ...
        arrow_color, arrow_lw, thr_dB, log_k, min_frac)
    fig = figure('Name', sprintf('XY compare — %s vs %s', display_name(S{1}), display_name(S{2})), ...
        'Color', 'w', 'Position', [80 50 950 950]);
    tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, sprintf('XY top view — %s  vs  %s', display_name(S{1}), display_name(S{2})), ...
        'FontSize', 14, 'FontWeight', 'bold');

    for s = 1:2
        St = S{s};
        eff_cav = St.cav_len + ternary(St.t_shift > 0 && lengthen_cav, 2*St.t_shift, 0);
        [~, iz0] = min(abs(St.z));
        [x_c, ix] = crop_to_range(St.x, x_range);
        [y_c, iy] = crop_to_range(St.y, y_range);
        I_xy_dB = squeeze(St.I_3D_dB(ix, iy, iz0));

        ax = nexttile;
        imagesc(x_c*1e6, y_c*1e6, I_xy_dB');
        set(ax, 'YDir', 'normal'); colormap(ax, cmap); clim(ax, clim_g);
        xlabel(ax, 'X [\mum]'); ylabel(ax, 'Y [\mum]');
        title(ax, display_name(St), 'FontSize', 13, 'FontWeight', 'bold');
        hold(ax, 'on');

        if St.has_P
            Px = squeeze(St.P_3D(ix, iy, iz0, 1));
            Py = squeeze(St.P_3D(ix, iy, iz0, 2));
            sx = max(min_skip, round(numel(ix) / arrows_per_axis));
            sy = max(min_skip, round(numel(iy) / arrows_per_axis));
            draw_poynting_quiver(x_c*1e6, y_c*1e6, Px, Py, sx, sy, ...
                arrow_scale, arrow_color, arrow_lw, ...
                I_xy_dB, max(I_xy_dB(:)), thr_dB, log_k, min_frac);
        end

        xl = xlim(ax); yl = ylim(ax);
        [xp, wp] = make_grating_profile(pitch, w_narrow, w_wide, ...
            St.n_per, St.cav_len, core_height, ...
            St.n_apod, center_mod_depth_nm*1e-9, geom_mode, apod_method, tanh_steepness, ...
            'xy', St.t_shift, lengthen_cav);
        plot(ax, xp*1e6,  wp*1e6, '-', 'Color', geom_color, 'LineWidth', geom_lw);
        plot(ax, xp*1e6, -wp*1e6, '-', 'Color', geom_color, 'LineWidth', geom_lw);
        [~, i0] = min(abs(xp));
        cav_full_w = 2 * wp(i0);
        draw_cavity_hatch(0, 0, eff_cav*1e6, cav_full_w*1e6, geom_color, geom_lw, 0.15);
        xlim(ax, xl); ylim(ax, yl); hold(ax, 'off');
    end

    cb = colorbar; cb.Layout.Tile = 'east';
    ylabel(cb, '10\cdotlog_{10}(|E|^2) [dB]  (global)');
end


function plot_yz_compare(S, pitch, w_narrow, w_wide, core_height, ...
        y_range, z_range, clim_g, cmap, geom_color, geom_lw, lengthen_cav, ...
        arrows_per_axis, min_skip, arrow_scale, arrow_color, arrow_lw, ...
        thr_dB, log_k, min_frac, slice_kind)
    % 2-row × n_cols flat layout. Row = structure. Columns sample one YZ slice
    % per period, going outward from the cavity on the +x side. Column 1 is
    % always the cavity center. `slice_kind` selects the per-period sample
    % point: 'narrow' (narrow-tooth center), 'wide' (wide-tooth center),
    % 'transition' (narrow→wide sidewall step).
    if nargin < 21 || isempty(slice_kind); slice_kind = 'narrow'; end

    n_periods_show     = 8;                  % cavity + 8 period slices
    x_far_target_um    = 10;                  % last column samples near this x
    far_highlight_color = [0.85 0.33 0.1];    % orange — flags the non-contiguous tile
    n_cols = 1 + n_periods_show;

    fig = figure('Name', sprintf('YZ compare (%s) — %s vs %s', slice_kind, display_name(S{1}), display_name(S{2})), ...
        'Color', 'w', 'Position', [30 80 1700 600]);
    tl = tiledlayout(fig, 2, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, sprintf('YZ mode profile vs distance from cavity  (%s slices)', slice_kind), ...
        'FontSize', 16, 'FontWeight', 'bold');
    subtitle(tl, sprintf('%s    vs    %s', display_name(S{1}), display_name(S{2})), ...
        'FontSize', 12);

    half_pitch = pitch / 2;

    for row = 1:2
        St = S{row};
        L_n1 = half_pitch - St.t_shift;
        cav_edge = (St.cav_len + ternary(St.t_shift > 0 && lengthen_cav, 2*St.t_shift, 0))/2;

        % Sequential periods d=1..(n_periods_show-1), then a far-jump period
        % whose x is closest to x_far_target_um µm. List of d values per col.
        d_seq = 1:(n_periods_show - 1);
        switch slice_kind
            case 'narrow'
                d_far = max(2, round((x_far_target_um*1e-6 - cav_edge - L_n1)/pitch + 1.5));
            case 'wide'
                d_far = max(1, round((x_far_target_um*1e-6 - cav_edge - L_n1 - half_pitch/2)/pitch + 1));
            case 'transition'
                d_far = max(1, round((x_far_target_um*1e-6 - cav_edge - L_n1)/pitch + 1));
            otherwise
                error('Unknown slice_kind: %s', slice_kind);
        end
        d_list = [d_seq, d_far];

        slices = cell(1, n_cols);
        slices{1} = struct('x', 0, 'lab', 'cavity center', 'is_far', false);
        for j = 1:numel(d_list)
            d = d_list(j);
            switch slice_kind
                case 'narrow'
                    if d == 1
                        xc = cav_edge + L_n1/2;
                    else
                        xc = cav_edge + L_n1 + half_pitch + (d-2)*pitch + half_pitch/2;
                    end
                    lab = sprintf('d=%d narrow', d);
                case 'wide'
                    xc = cav_edge + L_n1 + (d-1)*pitch + half_pitch/2;
                    lab = sprintf('d=%d wide', d);
                case 'transition'
                    if d == 1
                        xc = cav_edge + L_n1;
                    else
                        xc = cav_edge + L_n1 + (d-1)*pitch;
                    end
                    lab = sprintf('d=%d trans', d);
            end
            is_far = (j == numel(d_list));
            slices{j+1} = struct('x', xc, 'lab', lab, 'is_far', is_far);
        end

        [y_c, iy] = crop_to_range(St.y, y_range);
        [z_c, iz] = crop_to_range(St.z, z_range);
        sy = max(min_skip, round(numel(iy) / arrows_per_axis));
        sz = max(min_skip, round(numel(iz) / arrows_per_axis));

        for k = 1:n_cols
            sl = slices{k};
            [~, ix] = min(abs(St.x - sl.x));
            I_yz_dB = squeeze(St.I_3D_dB(ix, iy, iz));

            tile_idx = (row - 1) * n_cols + k;
            ax = nexttile(tl, tile_idx);
            imagesc(ax, y_c*1e6, z_c*1e6, I_yz_dB');
            set(ax, 'YDir', 'normal', 'Layer', 'top');
            colormap(ax, cmap); clim(ax, clim_g);
            hold(ax, 'on');

            if St.has_P
                Py = squeeze(St.P_3D(ix, iy, iz, 2));
                Pz = squeeze(St.P_3D(ix, iy, iz, 3));
                draw_poynting_quiver_simple(ax, y_c*1e6, z_c*1e6, Py, Pz, sy, sz, ...
                    arrow_scale, arrow_color, arrow_lw, ...
                    I_yz_dB, max(I_yz_dB(:)), thr_dB, log_k, min_frac);
            end

            % Geometry overlay: cavity tile is always narrow (uniform slab).
            % For periodic tiles, draw the box that matches the local cross-section.
            wg_hh = core_height / 2 * 1e6;
            if k == 1
                widths_to_draw = w_narrow;
            else
                switch slice_kind
                    case 'narrow';     widths_to_draw = w_narrow;
                    case 'wide';       widths_to_draw = w_wide;
                    case 'transition'; widths_to_draw = [w_narrow, w_wide];
                end
            end
            for ww = widths_to_draw
                wg_hw = ww / 2 * 1e6;
                rectangle(ax, 'Position', [-wg_hw, -wg_hh, 2*wg_hw, 2*wg_hh], ...
                          'EdgeColor', geom_color, 'LineStyle', '-', 'LineWidth', geom_lw);
            end

            % Axes labels only on left column / bottom row to reduce clutter.
            if k == 1
                ylabel(ax, 'Z [\mum]', 'FontSize', 12);
            else
                set(ax, 'YTickLabel', []);
            end
            if row == 2
                xlabel(ax, 'Y [\mum]', 'FontSize', 12);
            else
                set(ax, 'XTickLabel', []);
            end
            set(ax, 'FontSize', 11);

            % Top-row tiles get the slice label as a bold title; bottom row only x.
            if row == 1
                title(ax, {sl.lab, sprintf('x = %.2f \\mum', St.x(ix)*1e6)}, ...
                    'FontSize', 13, 'FontWeight', 'bold');
            else
                title(ax, sprintf('x = %.2f \\mum', St.x(ix)*1e6), 'FontSize', 12);
            end

            % Highlight the far-jump tile so viewers don't read it as the
            % period contiguous with d=(n-1).
            if sl.is_far
                set(ax, 'XColor', far_highlight_color, 'YColor', far_highlight_color, ...
                    'LineWidth', 2);
                title(ax, get(ax, 'Title').String, 'Color', far_highlight_color);
            end
            hold(ax, 'off');
        end
    end

    % Per-row structure-tag labels on the far left, large and rotated.
    for row = 1:2
        ax_left = nexttile(tl, (row - 1) * n_cols + 1);
        text(ax_left, -0.42, 0.5, display_name(S{row}), ...
            'Units', 'normalized', 'Rotation', 90, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', 14, 'FontWeight', 'bold');
    end

    cb = colorbar; cb.Layout.Tile = 'east';
    ylabel(cb, '10\cdotlog_{10}(|E|^2) [dB]  (global)', 'FontSize', 12);
    cb.FontSize = 11;
end


function plot_yz_cavity_compare(S, w_narrow, core_height, y_range, z_range, ...
        clim_g, cmap, geom_color, geom_lw, arrows_per_axis, min_skip, ...
        arrow_scale, arrow_color, arrow_lw, thr_dB, log_k, min_frac)
    % Side-by-side YZ slice at the cavity center (x=0) for the two structures.
    fig = figure('Name', sprintf('YZ cavity center — %s vs %s', ...
        display_name(S{1}), display_name(S{2})), ...
        'Color', 'w', 'Position', [200 150 1100 600]);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, 'YZ cross-section at cavity center', ...
        'FontSize', 16, 'FontWeight', 'bold');

    for col = 1:2
        St = S{col};
        [y_c, iy] = crop_to_range(St.y, y_range);
        [z_c, iz] = crop_to_range(St.z, z_range);
        [~, ix0]  = min(abs(St.x));
        I_yz_dB = squeeze(St.I_3D_dB(ix0, iy, iz));

        ax = nexttile(tl);
        imagesc(ax, y_c*1e6, z_c*1e6, I_yz_dB');
        set(ax, 'YDir', 'normal', 'Layer', 'top', 'FontSize', 12);
        colormap(ax, cmap); clim(ax, clim_g);
        hold(ax, 'on');

        if St.has_P
            Py = squeeze(St.P_3D(ix0, iy, iz, 2));
            Pz = squeeze(St.P_3D(ix0, iy, iz, 3));
            sy = max(min_skip, round(numel(iy) / arrows_per_axis));
            sz = max(min_skip, round(numel(iz) / arrows_per_axis));
            draw_poynting_quiver_simple(ax, y_c*1e6, z_c*1e6, Py, Pz, sy, sz, ...
                arrow_scale, arrow_color, arrow_lw, ...
                I_yz_dB, max(I_yz_dB(:)), thr_dB, log_k, min_frac);
        end

        wg_hw = w_narrow / 2 * 1e6;
        wg_hh = core_height / 2 * 1e6;
        rectangle(ax, 'Position', [-wg_hw, -wg_hh, 2*wg_hw, 2*wg_hh], ...
                  'EdgeColor', geom_color, 'LineStyle', '-', 'LineWidth', geom_lw);

        xlabel(ax, 'Y [\mum]', 'FontSize', 13);
        ylabel(ax, 'Z [\mum]', 'FontSize', 13);
        title(ax, display_name(St), 'FontSize', 14, 'FontWeight', 'bold');
        hold(ax, 'off');
    end

    cb = colorbar; cb.Layout.Tile = 'east';
    ylabel(cb, '10\cdotlog_{10}(|E|^2) [dB]  (global)', 'FontSize', 12);
    cb.FontSize = 11;
end


%% =====================================================================
%  XZ side view (slice at y=0 of field_3d)
%% =====================================================================
function plot_xz_view(St, x_range, z_range, clim_g, dB_limit, cmap, geom_color, geom_lw, ...
        geom_mode, apod_method, tanh_steepness, center_mod_depth_nm, lengthen_cav, ...
        pitch, w_narrow, w_wide, core_height, arrows_per_axis, min_skip, arrow_scale, ...
        arrow_color, arrow_lw, thr_dB, log_k, min_frac)
    [~, iy0] = min(abs(St.y));
    [x_c, ix] = crop_to_range(St.x, x_range);
    [z_c, iz] = crop_to_range(St.z, z_range);
    I_xz_dB = squeeze(St.I_3D_dB(ix, iy0, iz));      % (Nx_c, Nz_c)

    figure('Name', sprintf('XZ — %s', display_name(St)), 'Color', 'w', 'Position', [200 100 850 500]);
    imagesc(x_c*1e6, z_c*1e6, I_xz_dB');
    set(gca, 'YDir', 'normal'); colormap(cmap); clim(clim_g);
    cb = colorbar; ylabel(cb, '10\cdotlog_{10}(|E|^2) [dB]');
    xlabel('Position X [\mum]'); ylabel('Position Z [\mum]');
    title(sprintf('XZ — %s', display_name(St)), 'FontSize', 14, 'FontWeight', 'bold');

    hold on;
    if St.has_P
        Px = squeeze(St.P_3D(ix, iy0, iz, 1));
        Pz = squeeze(St.P_3D(ix, iy0, iz, 3));
        Nx_c = numel(ix); Nz_c = numel(iz);
        skip_x = max(min_skip, round(Nx_c / arrows_per_axis));
        skip_z = max(min_skip, round(Nz_c / arrows_per_axis));
        draw_poynting_quiver(x_c*1e6, z_c*1e6, Px, Pz, skip_x, skip_z, ...
            arrow_scale, arrow_color, arrow_lw, ...
            I_xz_dB, max(I_xz_dB(:)), thr_dB, log_k, min_frac);
    end

    xl = xlim; yl = ylim;
    wg_hh = core_height/2*1e6;
    plot(xl, [ wg_hh  wg_hh], '-', 'Color', geom_color, 'LineWidth', geom_lw);
    plot(xl, [-wg_hh -wg_hh], '-', 'Color', geom_color, 'LineWidth', geom_lw);
    eff_cav = St.cav_len + ternary(St.t_shift > 0 && lengthen_cav, 2*St.t_shift, 0);
    draw_cavity_hatch(0, 0, eff_cav*1e6, core_height*1e6, geom_color, geom_lw, 0.15);
    xlim(xl); ylim(yl); hold off;
end


%% =====================================================================
%  XY top view (slice at z=0 of field_3d)
%% =====================================================================
function plot_xy_view(St, x_range, y_range, clim_g, dB_limit, cmap, geom_color, geom_lw, ...
        geom_mode, apod_method, tanh_steepness, center_mod_depth_nm, lengthen_cav, ...
        pitch, w_narrow, w_wide, core_height, arrows_per_axis, min_skip, arrow_scale, ...
        arrow_color, arrow_lw, thr_dB, log_k, min_frac)
    [~, iz0] = min(abs(St.z));
    [x_c, ix] = crop_to_range(St.x, x_range);
    [y_c, iy] = crop_to_range(St.y, y_range);
    I_xy_dB = squeeze(St.I_3D_dB(ix, iy, iz0));      % (Nx_c, Ny_c)

    figure('Name', sprintf('XY — %s', display_name(St)), 'Color', 'w', 'Position', [100 150 750 650]);
    imagesc(x_c*1e6, y_c*1e6, I_xy_dB');
    set(gca, 'YDir', 'normal'); colormap(cmap); clim(clim_g);
    cb = colorbar; ylabel(cb, '10\cdotlog_{10}(|E|^2) [dB]');
    xlabel('Position X [\mum]'); ylabel('Position Y [\mum]');
    title(sprintf('XY — %s', display_name(St)), 'FontSize', 14, 'FontWeight', 'bold');

    hold on;
    if St.has_P
        Px = squeeze(St.P_3D(ix, iy, iz0, 1));
        Py = squeeze(St.P_3D(ix, iy, iz0, 2));
        Nx_c = numel(ix); Ny_c = numel(iy);
        skip_x = max(min_skip, round(Nx_c / arrows_per_axis));
        skip_y = max(min_skip, round(Ny_c / arrows_per_axis));
        draw_poynting_quiver(x_c*1e6, y_c*1e6, Px, Py, skip_x, skip_y, ...
            arrow_scale, arrow_color, arrow_lw, ...
            I_xy_dB, max(I_xy_dB(:)), thr_dB, log_k, min_frac);
    end

    xl = xlim; yl = ylim;
    [xp, wp] = make_grating_profile(pitch, w_narrow, w_wide, ...
        St.n_per, St.cav_len, core_height, ...
        St.n_apod, center_mod_depth_nm*1e-9, geom_mode, apod_method, tanh_steepness, ...
        'xy', St.t_shift, lengthen_cav);
    plot(xp*1e6,  wp*1e6, '-', 'Color', geom_color, 'LineWidth', geom_lw);
    plot(xp*1e6, -wp*1e6, '-', 'Color', geom_color, 'LineWidth', geom_lw);
    [~, i0] = min(abs(xp));
    cav_full_w = 2 * wp(i0);
    eff_cav = St.cav_len + ternary(St.t_shift > 0 && lengthen_cav, 2*St.t_shift, 0);
    draw_cavity_hatch(0, 0, eff_cav*1e6, cav_full_w*1e6, geom_color, geom_lw, 0.15);
    xlim(xl); ylim(yl); hold off;
end


%% =====================================================================
%  YZ multi-depth panel (3x3)
%% =====================================================================
function plot_yz_panel(St, eff_cav, pitch, w_narrow, w_wide, core_height, ...
        y_range, z_range, clim_g, cmap, geom_color, geom_lw, ...
        arrows_per_axis, min_skip, arrow_scale, arrow_color, arrow_lw, ...
        thr_dB, log_k, min_frac)

    half_pitch = pitch/2;
    L_n1 = half_pitch - St.t_shift;
    cav_edge   = eff_cav/2;
    cen_n1     = cav_edge + L_n1/2;
    cen_w1     = cav_edge + L_n1 + half_pitch/2;
    cen_n2     = cav_edge + L_n1 + half_pitch + half_pitch/2;

    % slice list: x [m], label, geometry width to draw
    slices = {
        struct('x', 0,        'lab','cavity center',        'w', w_narrow);
        struct('x', -cav_edge,'lab','cavity edge -',        'w', w_narrow);
        struct('x',  cav_edge,'lab','cavity edge +',        'w', w_narrow);
        struct('x', -cen_n1,  'lab','d=1 narrow center -',  'w', w_narrow);
        struct('x',  cen_n1,  'lab','d=1 narrow center +',  'w', w_narrow);
        struct('x', -cen_w1,  'lab','d=1 wide center -',    'w', w_wide  );
        struct('x',  cen_w1,  'lab','d=1 wide center +',    'w', w_wide  );
        struct('x', -cen_n2,  'lab','d=2 narrow center -',  'w', w_narrow);
        struct('x',  cen_n2,  'lab','d=2 narrow center +',  'w', w_narrow);
    };

    [y_c, iy] = crop_to_range(St.y, y_range);
    [z_c, iz] = crop_to_range(St.z, z_range);

    fig = figure('Name', sprintf('YZ panel — %s', display_name(St)), 'Color', 'w', ...
        'Position', [80 60 1100 900]);
    tl = tiledlayout(fig, 3, 3, 'TileSpacing','compact', 'Padding','compact');
    title(tl, sprintf('YZ cross-sections — %s', display_name(St)), ...
        'FontSize', 14, 'FontWeight', 'bold');

    Ny_c = numel(iy); Nz_c = numel(iz);
    skip_y = max(min_skip, round(Ny_c / arrows_per_axis));
    skip_z = max(min_skip, round(Nz_c / arrows_per_axis));

    for k = 1:numel(slices)
        sl = slices{k};
        [~, ix] = min(abs(St.x - sl.x));
        I_yz_dB = squeeze(St.I_3D_dB(ix, iy, iz));    % (Ny_c, Nz_c)

        nexttile;
        imagesc(y_c*1e6, z_c*1e6, I_yz_dB');
        set(gca, 'YDir', 'normal'); colormap(cmap); clim(clim_g);
        xlabel('Y [\mum]'); ylabel('Z [\mum]');
        title(sprintf('%s   (x=%.3f \\mum)', sl.lab, St.x(ix)*1e6), 'FontSize', 10);

        hold on;
        if St.has_P
            Py = squeeze(St.P_3D(ix, iy, iz, 2));
            Pz = squeeze(St.P_3D(ix, iy, iz, 3));
            draw_poynting_quiver(y_c*1e6, z_c*1e6, Py, Pz, skip_y, skip_z, ...
                arrow_scale, arrow_color, arrow_lw, ...
                I_yz_dB, max(I_yz_dB(:)), thr_dB, log_k, min_frac);
        end

        wg_hw = sl.w / 2 * 1e6;
        wg_hh = core_height / 2 * 1e6;
        rectangle('Position', [-wg_hw, -wg_hh, 2*wg_hw, 2*wg_hh], ...
                  'EdgeColor', geom_color, 'LineStyle', '-', 'LineWidth', geom_lw);
        hold off;
    end

    cb = colorbar(gca);
    cb.Layout.Tile = 'east';
    ylabel(cb, '10\cdotlog_{10}(|E|^2) [dB]  (global)');
end


%% =====================================================================
%  3D field profile (orthogonal slice planes — direct 3D analog of 2D heatmap)
%% =====================================================================
function plot_3d_iso(ax, St, clim_g, ~, ~, ~, ~, ~, ~, ~, ~, eff_cav, ...
        w_narrow, core_height, pitch, n_per, x_range_um)
    if nargin < 17 || isempty(x_range_um); x_range_um = []; end

    if isempty(ax)
        figure('Name', sprintf('3D — %s', display_name(St)), 'Color', 'w', 'Position', [60 40 1000 800]);
        ax = axes;
    end
    hold(ax, 'on');

    % Optional X crop for the central region
    if ~isempty(x_range_um)
        [x_keep, ix_keep] = crop_to_range(St.x, x_range_um);
    else
        x_keep  = St.x;
        ix_keep = 1:numel(St.x);
    end

    I_dB_crop = St.I_3D_dB(ix_keep, :, :);
    % isosurface expects volumes in (Y,X,Z) order via meshgrid
    I_perm = permute(I_dB_crop, [2 1 3]);
    [Xg, Yg, Zg] = meshgrid(x_keep*1e6, St.y*1e6, St.z*1e6);

    % Volumetric rendering via dense isosurface stack (no slice planes).
    % Many nested transparent shells at log-spaced dB levels accumulate to
    % give a true 3D field-volume appearance, with the colormap mapping dB
    % level → color and alpha rising for higher (brighter) levels.
    n_levels   = 16;
    levels_dB  = linspace(clim_g(1) + 3, clim_g(2) - 0.5, n_levels);
    cmap_data  = hot(256);
    for k = 1:n_levels
        lvl  = levels_dB(k);
        frac = (lvl - clim_g(1)) / (clim_g(2) - clim_g(1));   % 0..1
        cidx = max(1, min(256, round(frac * 255) + 1));
        face_color = cmap_data(cidx, :);
        face_alpha = 0.03 + 0.25 * frac^2;                    % outer shells faint
        try
            fv = isosurface(Xg, Yg, Zg, I_perm, lvl);
            if ~isempty(fv.vertices)
                p = patch(ax, fv, 'FaceColor', face_color, 'EdgeColor', 'none', ...
                    'FaceAlpha', face_alpha);
                isonormals(Xg, Yg, Zg, I_perm, p);
            end
        catch ME
            warning('isosurface @ %.1f dB failed: %s', lvl, ME.message);
        end
    end
    lighting(ax, 'gouraud'); camlight(ax, 'headlight');

    % Waveguide bounding box (cavity slab) clipped to the cropped X range
    L_dev_full = (n_per * pitch + eff_cav/2) * 1e6;
    if ~isempty(x_range_um)
        L_lo = max(-L_dev_full, x_range_um(1));
        L_hi = min( L_dev_full, x_range_um(2));
    else
        L_lo = -L_dev_full; L_hi = L_dev_full;
    end
    hh = core_height/2 * 1e6;
    hw = w_narrow/2 * 1e6;
    draw_box(ax, [L_lo L_hi], [-hw hw], [-hh hh], geom_box_color(), 1.0);
    draw_box(ax, [-eff_cav/2 eff_cav/2]*1e6, [-hw hw], [-hh hh], [1 1 0], 1.5);

    xlabel(ax, 'X [\mum]'); ylabel(ax, 'Y [\mum]'); zlabel(ax, 'Z [\mum]');
    title(ax, sprintf('3D |E|^2 field profile — %s', display_name(St)), ...
        'FontSize', 14, 'FontWeight', 'bold');
    axis(ax, 'tight'); axis(ax, 'equal'); grid(ax, 'on'); box(ax, 'on');
    if ~isempty(x_range_um); xlim(ax, x_range_um); end
    view(ax, 40, 25);
    colormap(ax, 'hot'); clim(ax, clim_g);
    hold(ax, 'off');
end


function plot_3d_iso_compare(S, clim_g, iso_levels_dB, iso_alpha, iso_color, ...
        q_target, q_thr_dB, q_color, q_lw, q_scale, lengthen_cav, w_narrow, ...
        core_height, pitch, x_range_um)
    fig = figure('Name', sprintf('3D compare — %s vs %s', display_name(S{1}), display_name(S{2})), ...
        'Color', 'w', 'Position', [40 40 1700 800]);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, sprintf('3D |E|^2 field profile — %s  vs  %s', display_name(S{1}), display_name(S{2})), ...
        'FontSize', 14, 'FontWeight', 'bold');

    ax3d = gobjects(1, 2);
    for s = 1:2
        St = S{s};
        eff_cav = St.cav_len + ternary(St.t_shift > 0 && lengthen_cav, 2*St.t_shift, 0);
        ax = nexttile(tl);
        plot_3d_iso(ax, St, clim_g, iso_levels_dB, iso_alpha, iso_color, ...
            q_target, q_thr_dB, q_color, q_lw, q_scale, eff_cav, w_narrow, ...
            core_height, pitch, St.n_per, x_range_um);
        ax3d(s) = ax;
    end

    cb = colorbar(ax3d(2)); cb.Layout.Tile = 'east';
    ylabel(cb, '|E|^2 isosurface levels [dB] (global clim)');
    try
        linkprop(ax3d, {'CameraPosition','CameraUpVector','CameraTarget','CameraViewAngle'});
    catch
    end
end


%% =====================================================================
%  Local helpers (copied from plot_field_poynting_zoom.m)
%% =====================================================================

function c = geom_box_color()
    c = [0.6 0.6 0.6];
end

function draw_box(ax, xl, yl, zl, color, lw)
    x = [xl(1) xl(2) xl(2) xl(1) xl(1)];
    y = [yl(1) yl(1) yl(2) yl(2) yl(1)];
    plot3(ax, x, y, repmat(zl(1),1,5), '-', 'Color', color, 'LineWidth', lw);
    plot3(ax, x, y, repmat(zl(2),1,5), '-', 'Color', color, 'LineWidth', lw);
    for ix = 1:4
        plot3(ax, [x(ix) x(ix)], [y(ix) y(ix)], zl, '-', 'Color', color, 'LineWidth', lw);
    end
end

function out = ternary(cond, a, b)
    if cond; out = a; else; out = b; end
end

function s = display_name(St)
    % Clean human title: "80 periods" or "80 periods, 100 nm shift".
    if St.t_shift > 0
        s = sprintf('%d periods, %.0f nm shift', St.n_per, round(St.t_shift*1e9));
    else
        s = sprintf('%d periods', St.n_per);
    end
end

function [coord_crop, idx] = crop_to_range(coord_m, range_um)
    lo = range_um(1) * 1e-6;
    hi = range_um(2) * 1e-6;
    if isinf(hi); hi = max(coord_m); end
    if isinf(lo); lo = min(coord_m); end
    idx = find(coord_m >= lo & coord_m <= hi);
    coord_crop = coord_m(idx);
end

function draw_cavity_hatch(x_cen, y_cen, w, h, color, lw, spacing)
    x0 = x_cen - w/2;  x1 = x_cen + w/2;
    y0 = y_cen - h/2;  y1 = y_cen + h/2;
    plot([x0 x1 x1 x0 x0], [y0 y0 y1 y1 y0], '-', 'Color', color, 'LineWidth', lw);
    offsets = (x0 + y0) : spacing : (x1 + y1);
    for off = offsets
        pts = zeros(0, 2);
        yL = -x0 + off;  if yL >= y0 && yL <= y1; pts(end+1,:) = [x0, yL]; end %#ok<AGROW>
        yR = -x1 + off;  if yR >= y0 && yR <= y1; pts(end+1,:) = [x1, yR]; end %#ok<AGROW>
        xB =  off - y0;  if xB >  x0 && xB <  x1; pts(end+1,:) = [xB, y0]; end %#ok<AGROW>
        xT =  off - y1;  if xT >  x0 && xT <  x1; pts(end+1,:) = [xT, y1]; end %#ok<AGROW>
        if size(pts, 1) == 2
            plot([pts(1,1) pts(2,1)], [pts(1,2) pts(2,2)], '-', 'Color', color, 'LineWidth', lw*0.6);
        end
    end
end

function draw_poynting_quiver(coord1, coord2, P1, P2, skip1, skip2, scale, color, lw, I_dB, max_dB, threshold_dB, log_k, min_frac)
    n1 = numel(coord1); n2 = numel(coord2);
    idx1 = (1:skip1:n1)'; idx2 = (1:skip2:n2)';
    c1_q = coord1(idx1); c2_q = coord2(idx2);
    [C1q, C2q] = meshgrid(c1_q, c2_q);
    P1_q = P1(idx1, idx2).';  P2_q = P2(idx1, idx2).';

    Pmag = sqrt(P1_q.^2 + P2_q.^2);
    Pmag_max = max(Pmag(:));
    if Pmag_max == 0; return; end
    Pmag_norm = Pmag / Pmag_max;
    log_norm = log1p(Pmag_norm * log_k) / log1p(log_k);
    vis_scale = min_frac + (1 - min_frac) * log_norm;

    I_dB_q = I_dB(idx1, idx2).';
    mask = I_dB_q >= (max_dB - threshold_dB);
    c1r = max(c1_q) - min(c1_q); c2r = max(c2_q) - min(c2_q);
    mask = mask & (C1q >= min(c1_q) + 0.01*c1r) & (C1q <= max(c1_q) - 0.01*c1r) ...
               & (C2q >= min(c2_q) + 0.01*c2r) & (C2q <= max(c2_q) - 0.01*c2r);

    P1_unit = P1_q ./ (Pmag + 1e-30);
    P2_unit = P2_q ./ (Pmag + 1e-30);
    w = double(mask);
    lm1 = conv2(P1_unit .* w, ones(3,3)/9, 'same');
    lm2 = conv2(P2_unit .* w, ones(3,3)/9, 'same');
    lmag = sqrt(lm1.^2 + lm2.^2) + 1e-30;
    coh = (P1_unit .* lm1 + P2_unit .* lm2) ./ lmag;
    nbr = conv2(w, ones(3,3), 'same') - w;
    mask = mask & (coh > 0.72) & (nbr >= 2);

    rmv = false(size(mask));
    valid_x = mask(:,1:end-1) & mask(:,2:end);
    conv_x  = (P1_unit(:,1:end-1) > 0.25) & (P1_unit(:,2:end) < -0.25) & valid_x;
    keep_l  = Pmag(:,1:end-1) >= Pmag(:,2:end);
    rmv(:,1:end-1) = rmv(:,1:end-1) | (conv_x & ~keep_l);
    rmv(:,2:end)   = rmv(:,2:end)   | (conv_x &  keep_l);
    mask = mask & ~rmv;

    rmv = false(size(mask));
    valid_z = mask(1:end-1,:) & mask(2:end,:);
    conv_z  = (P2_unit(1:end-1,:) > 0.25) & (P2_unit(2:end,:) < -0.25) & valid_z;
    keep_t  = Pmag(1:end-1,:) >= Pmag(2:end,:);
    rmv(1:end-1,:) = rmv(1:end-1,:) | (conv_z & ~keep_t);
    rmv(2:end,:)   = rmv(2:end,:)   | (conv_z &  keep_t);
    mask = mask & ~rmv;

    vis_scale = vis_scale .* mask;
    P1_draw = (P1_q ./ (Pmag + 1e-30)) .* vis_scale;
    P2_draw = (P2_q ./ (Pmag + 1e-30)) .* vis_scale;
    quiver(C1q, C2q, P1_draw, P2_draw, scale, 'Color', color, 'LineWidth', lw);
end

function draw_poynting_quiver_simple(ax, coord1, coord2, P1, P2, skip1, skip2, scale, color, lw, I_dB, max_dB, threshold_dB, log_k, min_frac)
    % Relaxed version of draw_poynting_quiver: no coherence/neighbor pruning,
    % no convergence-pair removal. Keeps log-compressed lengths and a loose
    % per-slice intensity gate so dim-region arrows still appear.
    n1 = numel(coord1); n2 = numel(coord2);
    idx1 = (1:skip1:n1)'; idx2 = (1:skip2:n2)';
    c1_q = coord1(idx1); c2_q = coord2(idx2);
    [C1q, C2q] = meshgrid(c1_q, c2_q);
    P1_q = P1(idx1, idx2).';  P2_q = P2(idx1, idx2).';

    Pmag = sqrt(P1_q.^2 + P2_q.^2);
    Pmag_max = max(Pmag(:));
    if Pmag_max == 0; return; end
    Pmag_norm = Pmag / Pmag_max;
    log_norm = log1p(Pmag_norm * log_k) / log1p(log_k);
    vis_scale = min_frac + (1 - min_frac) * log_norm;

    I_dB_q = I_dB(idx1, idx2).';
    mask = I_dB_q >= (max_dB - threshold_dB);
    c1r = max(c1_q) - min(c1_q); c2r = max(c2_q) - min(c2_q);
    mask = mask & (C1q >= min(c1_q) + 0.01*c1r) & (C1q <= max(c1_q) - 0.01*c1r) ...
               & (C2q >= min(c2_q) + 0.01*c2r) & (C2q <= max(c2_q) - 0.01*c2r);

    vis_scale = vis_scale .* mask;
    P1_draw = (P1_q ./ (Pmag + 1e-30)) .* vis_scale;
    P2_draw = (P2_q ./ (Pmag + 1e-30)) .* vis_scale;
    quiver(ax, C1q, C2q, P1_draw, P2_draw, scale, 'Color', color, 'LineWidth', lw);
end

function [n_per, n_apod, cav_len, t_shift] = resolve_geometry_params( ...
        fpath, data, pitch, n_per_manual, n_apod_manual, cav_len_manual, t_shift_manual)
    n_per   = n_per_manual;
    n_apod  = n_apod_manual;
    cav_len = cav_len_manual;
    t_shift = t_shift_manual;

    if isempty(n_per)   && isfield(data, 'n_periods');             n_per   = double(data.n_periods);            end
    if isempty(n_per)   && isfield(data, 'n_periods_each_side');   n_per   = double(data.n_periods_each_side);  end
    if isempty(n_apod)  && isfield(data, 'n_apod_periods');        n_apod  = double(data.n_apod_periods);       end
    if isempty(cav_len) && isfield(data, 'cavity_length_m');       cav_len = double(data.cavity_length_m);      end
    if isempty(t_shift) && isfield(data, 'innermost_tooth_shift_nm'); t_shift = double(data.innermost_tooth_shift_nm) * 1e-9; end

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
    if isempty(t_shift)
        tok = regexp(fname, '_shift_([\d.]+)nm', 'tokens', 'once');
        if ~isempty(tok); t_shift = str2double(tok{1}) * 1e-9; end
    end
    if isempty(t_shift)
        % Compact filename style: "_S100_" → 100 nm shift.
        tok = regexp(fname, '_S(\d+)(?:_|$)', 'tokens', 'once');
        if ~isempty(tok); t_shift = str2double(tok{1}) * 1e-9; end
    end
    if isempty(n_per) && isfield(data, 'L_device')
        cav_try = ternary(~isempty(cav_len), cav_len, pitch / 2);
        n_calc  = (double(data.L_device) / 2 - cav_try / 2) / pitch;
        n_round = round(n_calc);
        if abs(n_calc - n_round) < 0.02; n_per = n_round; end
    end

    if isempty(cav_len); cav_len = pitch / 2; end
    if isempty(n_apod);  n_apod  = 0;         end
    if isempty(t_shift); t_shift = 0;         end
    if isempty(n_per)
        error(['Could not determine n_periods from data file or filename.\n' ...
               'Set n_periods_override manually in the Configuration section.']);
    end

    fprintf('  Geometry: n_periods=%d, n_apod=%d, cavity_length=%.0f nm, tooth_shift=%.1f nm\n', ...
        n_per, n_apod, cav_len * 1e9, t_shift * 1e9);
end

function [x_vec, w_half_vec] = make_grating_profile(pitch, w_narrow, w_wide, ...
        n_periods, cav_length, core_height, n_apod, center_mod_depth, geom_mode, ...
        apod_method, tanh_steepness, view_plane, tooth_shift, lengthen_cav)
    if nargin < 12 || isempty(view_plane),   view_plane   = 'xy'; end
    if nargin < 13 || isempty(tooth_shift),  tooth_shift  = 0;    end
    if nargin < 14 || isempty(lengthen_cav), lengthen_cav = true; end

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
    k=k+1; seg_xl(k)=x; seg_xr(k)=x+eff_cav_length; seg_hw(k)=hw_cavity;        x=x+eff_cav_length;
    k=k+1; seg_xl(k)=x; seg_xr(k)=x+half_pitch;     seg_hw(k)=hw_narrow_arr(1); x=x+half_pitch;
    k=k+1; seg_xl(k)=x; seg_xr(k)=x+half_pitch;     seg_hw(k)=hw_wide_arr(1);   x=x+half_pitch;
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
