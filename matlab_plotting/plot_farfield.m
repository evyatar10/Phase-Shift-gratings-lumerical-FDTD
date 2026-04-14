% plot_farfield.m
% ---------------
% Two figures (one per monitor), each with:
%   LEFT subplot  : Near field  (E2 = |Ex|²+|Ey|²+|Ez|² at resonance wavelength)
%                   sourced directly from the monitor surface (same monitor as far-field)
%   RIGHT subplot : Far field   (XY map in direction-cosine space, no polar)
%
% Near-field source (monitor surface E-field):
%   top_monitor  (Z-normal) → nearfield_top   : axes = X (along grating), Y (transverse)
%   side_monitor (Y-normal) → nearfield_side  : axes = X (along grating), Z (vertical)
%
% Far-field axis conventions:
%   top_monitor  (Z-normal) : H = uy (Y), V = ux (X)
%   side_monitor (Y-normal) : H = ux (X), V = uy (Z)
%
% All analysis parameters (HALF_ANGLE, CUSTOM_ANGLE_DEG) are defined here only.

addpath(fileparts(fileparts(mfilename('fullpath'))));  % Add project root to MATLAB path
clear; clc;
close all;

% ── FILE SELECTION ─────────────────────────────────────────────────────────
prefs_file = fullfile(fileparts(mfilename('fullpath')), 'plot_prefs.mat');
start_path = '*.mat';
MAT_FILE   = '';

if exist(prefs_file, 'file')
    p = load(prefs_file);
    has_folder = isfield(p, 'farfield_last_folder') && isfolder(p.farfield_last_folder);
    has_file   = isfield(p, 'farfield_last_file')   && ~isempty(p.farfield_last_file);
    if has_folder
        if has_file
            msg = ['Last used:' newline fullfile(p.farfield_last_folder, p.farfield_last_file)];
            answer = questdlg(msg, 'Reuse Previous Selection', ...
                'Same File', 'Latest in Folder', 'Browse...', 'Same File');
        else
            msg = ['Use last folder?' newline p.farfield_last_folder];
            answer = questdlg(msg, 'Reuse Previous Selection', ...
                'Latest in Folder', 'Browse...', 'Latest in Folder');
        end
        if strcmp(answer, 'Same File')
            MAT_FILE = fullfile(p.farfield_last_folder, p.farfield_last_file);
        elseif strcmp(answer, 'Latest in Folder')
            listing = dir(fullfile(p.farfield_last_folder, '*.mat'));
            if ~isempty(listing)
                [~, newest] = max([listing.datenum]);
                MAT_FILE = fullfile(p.farfield_last_folder, listing(newest).name);
            else
                start_path = fullfile(p.farfield_last_folder, '*.mat');
            end
        else
            start_path = fullfile(p.farfield_last_folder, '*.mat');
        end
    end
end

if isempty(MAT_FILE)
    [file, folder] = uigetfile(start_path, 'Select far-field .mat file');
    if isequal(file, 0)
        disp('No file selected.');
        return;
    end
    MAT_FILE = fullfile(folder, file);
end

% Save last folder and file
[farfield_last_folder, farfield_last_file] = fileparts(MAT_FILE);
farfield_last_file = [farfield_last_file, '.mat'];
if exist(prefs_file, 'file')
    save(prefs_file, 'farfield_last_folder', 'farfield_last_file', '-append');
else
    save(prefs_file, 'farfield_last_folder', 'farfield_last_file');
end

% ── CONFIG ─────────────────────────────────────────────────────────────────
SCALE_DB         = false;   % false = linear  |  true = dB
DB_FLOOR         = -40;     % dB floor  (only when SCALE_DB=true)
HALF_ANGLE       = 30;      % degrees — unused currently
CUSTOM_ANGLE_DEG = 22.8;    % analytic critical angle (degrees)
PLOT_X_CONES     = false;    % true = draw the X-cone overlays and labels

NF_SCALE         = 0.85;    % scale factor for the near-field subplot (1 = full default size)
NF_CROP_X_UM     = [];      % crop near-field X to ±N µm  ([] = no crop)

% ── Device overlay (drawn on the near-field subplot) ─────────────────────
% Must match simulation geometry (mirrors plot_field_poynting_zoom.m).
DRAW_DEVICE           = true;
avg_corrugation_width = 800e-9;   % [m]
corrugation_depth     = 300e-9;   % [m]
core_height           = 350e-9;   % [m]
pitch                 = 500e-9;   % [m]
width_narrow          = avg_corrugation_width - corrugation_depth / 2;
width_wide            = avg_corrugation_width + corrugation_depth / 2;
geom_color            = [1 1 1 0.65];   % RGBA — 4th component = alpha (transparency)
geom_lw               = 1.1;
geom_mode             = 'apodized';    % 'uniform' or 'apodized'
center_mod_depth_nm   = 4.0;
apod_method           = 'linear';      % 'linear' or 'tanh'
tanh_steepness        = 2.0;
tooth_shift_override   = [];
lengthen_cavity        = true;
n_periods_override     = [];
n_apod_override        = [];
cavity_length_override = [];

% Parse simulation description from filename (e.g. result_100_periods_10_apodizations_CONST.mat)
[~, fname_stem, ~] = fileparts(MAT_FILE);
SIM_DESC = regexprep(fname_stem, '^result_', '');
SIM_DESC = regexprep(SIM_DESC, '_(CONST|NOCONST)$', '');
SIM_DESC = strrep(SIM_DESC, '_', ' ');
% ───────────────────────────────────────────────────────────────────────────

% Bundle geometry params for the device overlay
geom = struct( ...
    'draw_device',      DRAW_DEVICE, ...
    'pitch',            pitch, ...
    'width_narrow',     width_narrow, ...
    'width_wide',       width_wide, ...
    'core_height',      core_height, ...
    'color',            geom_color, ...
    'lw',               geom_lw, ...
    'mode',             geom_mode, ...
    'center_mod_depth', center_mod_depth_nm * 1e-9, ...
    'apod_method',      apod_method, ...
    'tanh_steepness',   tanh_steepness, ...
    'tooth_shift_override',   tooth_shift_override, ...
    'lengthen_cavity',        lengthen_cavity, ...
    'n_periods_override',     n_periods_override, ...
    'n_apod_override',        n_apod_override, ...
    'cavity_length_override', cavity_length_override);

%% Load
data = load(MAT_FILE);

monitors  = {'top_monitor',  'side_monitor'};    % top first
ff_fields = {'farfield_top', 'farfield_side'};
nf_fields = {'nearfield_top','nearfield_side'};  % E-field on the monitor surface

for k = 1:2
    mname   = monitors{k};
    ff_key  = ff_fields{k};
    nf_key  = nf_fields{k};
    is_top  = strcmp(mname, 'top_monitor');

    if ~isfield(data, ff_key) || isempty(data.(ff_key))
        fprintf('WARNING: %s not found in mat file — skipping.\n', ff_key);
        continue;
    end

    ff   = data.(ff_key);
    E2   = double(ff.E2);
    ux   = double(ff.ux(:));
    uy   = double(ff.uy(:));
    lam  = double(ff.lam);

    % Near-field struct (may be empty if monitor was not recorded)
    nf_struct = [];
    if isfield(data, nf_key) && ~isempty(data.(nf_key))
        nf_struct = data.(nf_key);
    end

    fprintf('\n=== %s ===\n', mname);
    make_figure(E2, ux, uy, lam, is_top, mname, nf_struct, ...
        SCALE_DB, DB_FLOOR, HALF_ANGLE, CUSTOM_ANGLE_DEG, NF_SCALE, SIM_DESC, PLOT_X_CONES, NF_CROP_X_UM, ...
        data, MAT_FILE, geom);
end


%% ═══════════════════════════════════════════════════════════════════════════
function make_figure(E2, ux, uy, lam, is_top, monitor_name, nf_struct, ...
    SCALE_DB, DB_FLOOR, HALF_ANGLE, CUSTOM_ANGLE_DEG, NF_SCALE, SIM_DESC, PLOT_X_CONES, NF_CROP_X_UM, ...
    data, mat_file, geom) %#ok<INUSD>

% Compute hemisphere mask
[UX, UY] = meshgrid(ux, uy);
UX = UX'; UY = UY';
valid    = (UX.^2 + UY.^2) <= 1.0;

% Mask outside hemisphere; keep raw E² values (no normalisation)
E2_ff = zeros(size(E2));
if SCALE_DB
    max_ff  = max(E2(valid));
    E2_dB   = 10*log10(E2 / max_ff);
    E2_ff(valid) = max(DB_FLOOR, E2_dB(valid));
    scale_str = sprintf('dB  (floor %d dB)', DB_FLOOR);
else
    E2_ff(valid) = E2(valid);       % raw E², hemisphere only
    scale_str = 'linear  (raw E²)';
end

% Keep E2_norm (0-1) only for lobe-finding; never displayed
max_ff_   = max(E2(valid));
E2_norm   = zeros(size(E2));
E2_norm(valid) = E2(valid) / max_ff_;

% ── Lobe analysis ─────────────────────────────────────────────────────────
E2_m = E2_norm; E2_m(~valid) = 0;
[~, peak_lin]   = max(E2_m(:));
[pi_ux, pi_uy]  = ind2sub(size(E2_m), peak_lin);
peak_ux_val     = ux(pi_ux);
theta_peak      = asind(min(sqrt(peak_ux_val^2 + uy(pi_uy)^2), 1));
fprintf('  Lobe center angle = %.1f deg\n', theta_peak);

slice_uy  = E2(pi_ux, :);
fwhm_uy   = compute_fwhm_deg(slice_uy, uy');
if ~isnan(fwhm_uy)
    fprintf('  -3 dB width along UY: %.1f deg  (half-angle %.1f deg)\n', ...
        fwhm_uy, fwhm_uy/2);
end

% ── Figure ────────────────────────────────────────────────────────────────
figure('Position', [100 100 1400 620]);

if is_top
    view_name = 'Top View';
else
    view_name = 'Side View';
end
title_str = sprintf('%s  |  \\lambda = %.2f nm  |  %s', view_name, lam*1e9, SIM_DESC);
sgtitle(title_str, 'FontWeight','bold', 'FontSize',16, 'Interpreter','tex');

% ══ LEFT: Near field ══════════════════════════════════════════════════════
ax_nf = subplot(1,2,1);

nf_ok = false;
h_nf  = []; v_nf = [];
if ~isempty(nf_struct)
    try
        [E2_nf, h_nf, v_nf, h_lbl, v_lbl] = get_nf_image(nf_struct, lam, is_top, NF_CROP_X_UM);
        imagesc(ax_nf, h_nf*1e6, v_nf*1e6, E2_nf);  % raw E², no normalisation
        colormap(ax_nf, jet);                          % same colormap as far-field
        set(ax_nf, 'YDir','normal');
        axis(ax_nf, 'tight');
        xlabel(ax_nf, h_lbl);
        ylabel(ax_nf, v_lbl);
        cb_nf = colorbar(ax_nf);
        cb_nf.Label.String = '|E|^2  [V^2/m^2]';
        nf_ok = true;

        % Device overlay (X along grating is VERTICAL here — swap from zoom plot)
        if geom.draw_device
            hold(ax_nf, 'on');
            xl_nf = xlim(ax_nf); yl_nf = ylim(ax_nf);
            [n_per_r, n_apod_r, cav_len_r, t_shift_r] = resolve_geometry_params( ...
                mat_file, data, geom.pitch, geom.n_periods_override, ...
                geom.n_apod_override, geom.cavity_length_override, ...
                geom.tooth_shift_override);
            % Auto-detect apodization on/off and method from filename/resolved n_apod
            [~, fname_only] = fileparts(mat_file);
            if n_apod_r > 0
                geom_mode_eff = 'apodized';
            else
                geom_mode_eff = 'uniform';
            end
            if ~isempty(regexp(fname_only, '_tanh', 'once'))
                apod_method_eff = 'tanh';
            else
                apod_method_eff = geom.apod_method;
            end
            fprintf('  Overlay: mode=%s, apod_method=%s, tooth_shift=%.1f nm\n', ...
                geom_mode_eff, apod_method_eff, t_shift_r*1e9);
            eff_cav = cav_len_r + ternary(t_shift_r > 0 && geom.lengthen_cavity, 2*t_shift_r, 0);
            if is_top
                % Top view: full corrugated grating outline
                [xp, wp] = make_grating_profile(geom.pitch, geom.width_narrow, geom.width_wide, ...
                    n_per_r, cav_len_r, geom.core_height, n_apod_r, geom.center_mod_depth, ...
                    geom_mode_eff, apod_method_eff, geom.tanh_steepness, 'xy', ...
                    t_shift_r, geom.lengthen_cavity);
                % Axes transposed vs zoom plot: horizontal = wp (Y), vertical = xp (X)
                plot(ax_nf,  wp*1e6, xp*1e6, '-', 'Color', geom.color, 'LineWidth', geom.lw);
                plot(ax_nf, -wp*1e6, xp*1e6, '-', 'Color', geom.color, 'LineWidth', geom.lw);
                [~, i0] = min(abs(xp));
                cav_full_w = 2 * wp(i0);
                draw_cavity_hatch(0, 0, cav_full_w*1e6, eff_cav*1e6, geom.color, geom.lw, 0.15);
            else
                % Side view: slab only — two straight lines at z = ±core_height/2
                wg_hh = geom.core_height / 2 * 1e6;
                plot(ax_nf, [ wg_hh  wg_hh], yl_nf, '-', 'Color', geom.color, 'LineWidth', geom.lw);
                plot(ax_nf, [-wg_hh -wg_hh], yl_nf, '-', 'Color', geom.color, 'LineWidth', geom.lw);
                draw_cavity_hatch(0, 0, geom.core_height*1e6, eff_cav*1e6, geom.color, geom.lw, 0.15);
            end
            xlim(ax_nf, xl_nf); ylim(ax_nf, yl_nf);
            hold(ax_nf, 'off');
        end
    catch ME
        fprintf('  Near-field load failed: %s\n', ME.message);
    end
end

if ~nf_ok
    text(ax_nf, 0.5, 0.5, 'Near-field data not available', ...
        'Units','normalized', 'HorizontalAlignment','center');
    set(ax_nf, 'XTick',[], 'YTick',[]);
end

if ~isempty(NF_CROP_X_UM)
    crop_str = sprintf('  [X cropped to \\pm%g µm]', NF_CROP_X_UM);
else
    crop_str = '';
end
if is_top
    title(ax_nf, ['Near Field  — Top View (XY plane, |E|^2)' crop_str], 'FontSize', 11);
else
    title(ax_nf, ['Near Field  — Side View (XZ plane, |E|^2)' crop_str], 'FontSize', 11);
end

% ══ RIGHT: Far field XY map ═══════════════════════════════════════════════
% Both monitors: H = uy (or uz for side), V = ux
% E2_norm is (N_ux × N_uy) → imagesc(uy, ux, E2_norm): cols=uy, rows=ux ✓
ax_ff = subplot(1,2,2);
hold(ax_ff, 'on');

imagesc(ax_ff, uy, ux, E2_ff);
if is_top
    xlabel(ax_ff, 'u_y  (Y-direction)', 'FontSize', 11);
    ylabel(ax_ff, 'u_x  (X-direction)', 'FontSize', 11);
else
    xlabel(ax_ff, 'u_z  (Z-direction)', 'FontSize', 11);
    ylabel(ax_ff, 'u_x  (X-direction)', 'FontSize', 11);
end

colormap(ax_ff, jet);
% clim auto — raw E² values, no forced 0-1
axis(ax_ff, 'equal');
xlim(ax_ff, [-1.05, 1.05]); ylim(ax_ff, [-1.05, 1.05]);
set(ax_ff, 'YDir','normal');
title(ax_ff, sprintf('Far Field  (XY map, %s)', scale_str), 'FontSize', 11);

% Hemisphere circle
phi_c = linspace(0, 2*pi, 720);
plot(ax_ff, cos(phi_c), sin(phi_c), 'k-', 'LineWidth', 0.8);

% X cones and labels
if PLOT_X_CONES
    color_fwhm = [0.90 0.45 0.00];   % darker orange
    color_crit = [1.00 0.45 0.75];   % pink
    draw_x_cone(ax_ff, peak_ux_val, fwhm_uy,           true, color_fwhm, '');
    draw_x_cone(ax_ff, peak_ux_val, CUSTOM_ANGLE_DEG*2, true, color_crit, '');

    % ── Cone labels ON the FF plot — right of centre, lower area
    uy_lbl  = 0.16;    % slightly right of centre
    if ~isnan(fwhm_uy)
        text(ax_ff, uy_lbl, 0.05, sprintf('FWHM half-angle: %.1f°', fwhm_uy/2), ...
            'Color', color_fwhm, 'FontWeight','bold', 'FontSize', 9, ...
            'HorizontalAlignment','left');
    end
    text(ax_ff, uy_lbl, -0.10, sprintf('Analytic critical angle: %.1f°', CUSTOM_ANGLE_DEG), ...
        'Color', color_crit, 'FontWeight','bold', 'FontSize', 9, ...
        'HorizontalAlignment','left');
end

cb_ff = colorbar(ax_ff);
if SCALE_DB
    cb_ff.Label.String = 'E^2  [dB]';
else
    cb_ff.Label.String = 'E^2  [V^2/m^2]';
end
cb_ff.Label.FontSize = 9;

% ── Scale NF subplot uniformly (keeps default aspect ratio, just shrinks it)
drawnow;
pos_nf = ax_nf.Position;   % [l b w h] normalised — default subplot size
new_w = pos_nf(3) * NF_SCALE;
new_h = pos_nf(4) * NF_SCALE;
new_l = pos_nf(1) + (pos_nf(3) - new_w) / 2;
new_b = pos_nf(2) + (pos_nf(4) - new_h) / 2;
ax_nf.Position = [new_l, new_b, new_w, new_h];

end  % make_figure


%% ── HELPERS ────────────────────────────────────────────────────────────────

function [E2_nf, h_axis, v_axis, h_lbl, v_lbl] = get_nf_image(nf, lam_res, is_top, crop_x_um)
% Extract E2 = |Ex|^2 + |Ey|^2 + |Ez|^2 at resonance from the monitor surface.
%
% nf is the nearfield_top or nearfield_side struct saved by run_simulation.py:
%   .x, .y, .z    — coordinate vectors (one per spatial axis)
%   .E_res        — complex E array  (Nx, Ny, Nz, 1, Nf, 3)  from getresult("E")
%   .lambda_arr   — wavelength vector (Nf,)
%
% For top_monitor (Z-normal): the Z axis is a singleton → surface is XY.
% For side_monitor (Y-normal): the Y axis is a singleton → surface is XZ.

lambda_arr = real(squeeze(nf.lambda_arr));
[~, idx_res] = min(abs(lambda_arr - lam_res));

% Select resonance frequency and sum |E|² over components
E_full = nf.E_res;                      % (Nx, Ny, Nz, 1, Nf, 3)
nd     = ndims(E_full);
idx_sel        = repmat({':'}, 1, nd);
idx_sel{end-1} = idx_res;              % 2nd-to-last dim = frequency
E_at_res = E_full(idx_sel{:});         % (Nx, Ny, Nz, 1, 1, 3)
E2_3d    = squeeze(sum(abs(E_at_res).^2, nd));  % → (Nx, Ny, Nz) with singletons squeezed

x_ax = real(squeeze(nf.x));
y_ax = real(squeeze(nf.y));
z_ax = real(squeeze(nf.z));

if is_top
    % top_monitor: Z-normal surface  → spatial dims are X × Y
    % E2_3d should be (Nx, Ny) after squeeze (Z is singleton)
    E2_nf  = squeeze(E2_3d);            % ensure 2D: (Nx, Ny)
    % imagesc(h_axis, v_axis, data) plots columns=h, rows=v
    % We want h = y (transverse), v = x (along grating)
    h_axis = y_ax(:);  v_axis = x_ax(:);
    h_lbl  = 'y  [µm]';
    v_lbl  = 'x  (along grating)  [µm]';
    % E2_nf is (Nx × Ny): imagesc(y, x, E2_nf) → rows=x, cols=y ✓
else
    % side_monitor: Y-normal surface → spatial dims are X × Z
    % User wants: H = z (horizontal), V = x (vertical)
    % imagesc(z, x, E2_nf): rows=x, cols=z → data must be (Nx, Nz) ✓
    E2_nf  = squeeze(E2_3d);            % (Nx, Nz)
    h_axis = z_ax(:);  v_axis = x_ax(:);
    h_lbl  = 'z  (vertical)  [µm]';
    v_lbl  = 'x  (along grating)  [µm]';
end

% Crop X (v_axis) if requested
if ~isempty(crop_x_um)
    mask   = abs(v_axis) <= crop_x_um * 1e-6;
    v_axis = v_axis(mask);
    E2_nf  = E2_nf(mask, :);
end
end


function fwhm = compute_fwhm_deg(profile, axis_uc)
% Compute the -3 dB (FWHM) width in degrees for a 1-D E2 profile.
profile = profile(:)'; axis_uc = axis_uc(:)';
half    = max(profile) / 2;
above   = profile >= half;
cross   = find(diff(above) ~= 0);
if numel(cross) < 2; fwhm = NaN; return; end
    function theta = interp_c(i)
        x0=axis_uc(i); x1=axis_uc(i+1); y0=profile(i); y1=profile(i+1);
        uc = x0 + (half-y0)/(y1-y0)*(x1-x0);
        theta = asind(min(abs(uc),1))*sign(uc);
    end
fwhm = abs(interp_c(cross(end)) - interp_c(cross(1)));
end


function draw_x_cone(ax, ux_peak, fwhm_uy_deg, h_rotated, color, lbl)
% Draw two dashed crossing lines at ±half-angle in the UY direction.
if isnan(fwhm_uy_deg); return; end
uy_half = sin(deg2rad(fwhm_uy_deg / 2));
if h_rotated    % H=uy, V=ux
    line(ax,[-uy_half, uy_half],[-ux_peak, ux_peak],'Color',color,'LineWidth',1.8,'LineStyle','--');
    line(ax,[-uy_half, uy_half],[ ux_peak,-ux_peak],'Color',color,'LineWidth',1.8,'LineStyle','--');
    if ~isempty(lbl)
        text(ax, uy_half*0.75, ux_peak*0.85, lbl, 'Color',color, 'FontSize',9, ...
            'FontWeight','bold','HorizontalAlignment','center','VerticalAlignment','bottom');
    end
else            % H=ux, V=uy
    line(ax,[-ux_peak, ux_peak],[-uy_half, uy_half],'Color',color,'LineWidth',1.8,'LineStyle','--');
    line(ax,[-ux_peak, ux_peak],[ uy_half,-uy_half],'Color',color,'LineWidth',1.8,'LineStyle','--');
    if ~isempty(lbl)
        text(ax, ux_peak*0.75, uy_half*0.85, lbl, 'Color',color, 'FontSize',9, ...
            'FontWeight','bold','HorizontalAlignment','center','VerticalAlignment','bottom');
    end
end
end


function out = ternary(cond, a, b)
    if cond; out = a; else; out = b; end
end


function [n_per, n_apod, cav_len, t_shift] = resolve_geometry_params( ...
        fpath, data, pitch, n_per_manual, n_apod_manual, cav_len_manual, t_shift_manual)
% Mirror of resolve_geometry_params in plot_field_poynting_zoom.m.
    n_per   = n_per_manual;
    n_apod  = n_apod_manual;
    cav_len = cav_len_manual;
    t_shift = t_shift_manual;

    if isempty(n_per)   && isfield(data, 'n_periods');            n_per   = double(data.n_periods);            end
    if isempty(n_per)   && isfield(data, 'n_periods_each_side'); n_per   = double(data.n_periods_each_side);  end
    if isempty(n_apod)  && isfield(data, 'n_apod_periods');      n_apod  = double(data.n_apod_periods);       end
    if isempty(cav_len) && isfield(data, 'cavity_length_m');     cav_len = double(data.cavity_length_m);      end
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

    if isempty(n_per) && isfield(data, 'L_device')
        cav_try  = ternary(~isempty(cav_len), cav_len, pitch / 2);
        n_calc   = (double(data.L_device) / 2 - cav_try / 2) / pitch;
        n_round  = round(n_calc);
        if abs(n_calc - n_round) < 0.02
            n_per = n_round;
        end
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
% Mirror of make_grating_profile in plot_field_poynting_zoom.m.
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
    if tooth_shift > 0 && lengthen_cav
        cav_extra = 2 * tooth_shift;
    end
    eff_cav_length = cav_length + cav_extra;

    n_segs = 4 * n_periods + 1;
    seg_xl = zeros(1, n_segs);
    seg_xr = zeros(1, n_segs);
    seg_hw = zeros(1, n_segs);
    k = 0;
    x = -(n_periods * pitch + eff_cav_length / 2);

    for d = n_periods:-1:2
        k=k+1; seg_xl(k)=x; seg_xr(k)=x+half_pitch; seg_hw(k)=hw_narrow_arr(d); x=x+half_pitch;
        k=k+1; seg_xl(k)=x; seg_xr(k)=x+half_pitch; seg_hw(k)=hw_wide_arr(d);   x=x+half_pitch;
    end
    L_narrow_1_len = half_pitch - tooth_shift;
    k=k+1; seg_xl(k)=x; seg_xr(k)=x+L_narrow_1_len; seg_hw(k)=hw_narrow_arr(1); x=x+L_narrow_1_len;
    k=k+1; seg_xl(k)=x; seg_xr(k)=x+half_pitch;     seg_hw(k)=hw_wide_arr(1);   x=x+half_pitch;

    k=k+1; seg_xl(k)=x; seg_xr(k)=x+eff_cav_length; seg_hw(k)=hw_cavity; x=x+eff_cav_length;

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


function draw_cavity_hatch(x_cen, y_cen, w, h, color, lw, spacing)
% Hatched rectangle marking the cavity region. (x_cen,y_cen) centre [µm],
% w,h full size [µm]. Mirror of helper in plot_field_poynting_zoom.m.
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
