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
%close all;

% ── CONFIG ─────────────────────────────────────────────────────────────────
RESULTS_DIR      = 'C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles_7\results';
MAT_FILE         = fullfile(RESULTS_DIR, 'result_100_periods_CONST.mat');

SCALE_DB         = false;   % false = linear  |  true = dB
DB_FLOOR         = -40;     % dB floor  (only when SCALE_DB=true)
HALF_ANGLE       = 30;      % degrees — unused currently
CUSTOM_ANGLE_DEG = 22.8;    % analytic critical angle (degrees)
PLOT_X_CONES     = true;    % true = draw the X-cone overlays and labels

NF_SCALE         = 0.85;    % scale factor for the near-field subplot (1 = full default size)
NF_CROP_X_UM     = 60;      % crop near-field X to ±N µm  ([] = no crop)

% Parse simulation description from filename (e.g. result_100_periods_10_apodizations_CONST.mat)
[~, fname_stem, ~] = fileparts(MAT_FILE);
SIM_DESC = regexprep(fname_stem, '^result_', '');
SIM_DESC = regexprep(SIM_DESC, '_(CONST|NOCONST)$', '');
SIM_DESC = strrep(SIM_DESC, '_', ' ');
% ───────────────────────────────────────────────────────────────────────────

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
        SCALE_DB, DB_FLOOR, HALF_ANGLE, CUSTOM_ANGLE_DEG, NF_SCALE, SIM_DESC, PLOT_X_CONES, NF_CROP_X_UM);
end


%% ═══════════════════════════════════════════════════════════════════════════
function make_figure(E2, ux, uy, lam, is_top, monitor_name, nf_struct, ...
    SCALE_DB, DB_FLOOR, HALF_ANGLE, CUSTOM_ANGLE_DEG, NF_SCALE, SIM_DESC, PLOT_X_CONES, NF_CROP_X_UM) %#ok<INUSD>

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


