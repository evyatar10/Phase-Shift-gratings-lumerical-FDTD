% analyze_radiation_recycling.m
% Radiation recycling analysis for phase-shift Bragg grating simulations.
%
% Loads 2D field monitor data (XZ side view and XY top view) from a results
% .mat file, then produces diagnostic plots for both vertical and lateral
% radiation leakage:
%   1. Normal Poynting flux vs x at multiple distances from core
%   2. Integrated outward flux vs distance from core
%   3. Lateral flux Px(x) at a fixed distance above/beside core
%   4. |E|^2(x) on log scale with exponential fit
%
% Data sources:
%   - Primary: 2D monitors (field_xz_side, field_xy) with Poynting vectors
%   - Fallback: 3D monitor (field_3d) for intensity-only analysis
%
% Usage:
%   1. Set the result file path below
%   2. Adjust sampling offsets and plot toggles as needed
%   3. Run the script

addpath(fileparts(fileparts(mfilename('fullpath'))));
clear; close all; clc;

%% -- Configuration -----------------------------------------------------------

result_filepath = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\leaky_modes_v2\results\result_80_periods_CONST_ff.mat";

% Geometry overrides (set to NaN to read from file)
override_core_height_m = NaN;
override_avg_width_m   = NaN;

% -- XZ plane (y~0): vertical radiation --------------------------------------
plot_xz_Pz_vs_x   = true;   % Plot 1: Pz(x) at multiple z-heights
plot_xz_Pout_vs_z  = true;   % Plot 2: integrated Pout(z) vs z
plot_xz_Px_at_z    = true;   % Plot 3: Px(x) at fixed z above core
plot_xz_intensity  = true;   % Plot 4: |E|^2(x) at z-heights (log scale)

% Offsets from core surface (um) — absolute z = core_height/2 + offset
xz_offsets_um      = [0.3, 0.5, 1.0, 1.5, 2.0];
xz_fixed_offset_um = 0.5;    % for Plot 3

% -- XY plane (z~0): lateral radiation ----------------------------------------
plot_xy_Py_vs_x    = true;   % Plot 5: Py(x) at multiple y-offsets
plot_xy_Pout_vs_y  = true;   % Plot 6: integrated Pout(y) vs y
plot_xy_Px_at_y    = true;   % Plot 7: Px(x) at fixed y beside core
plot_xy_intensity  = true;   % Plot 8: |E|^2(x) at y-offsets (log scale)

% Offsets from core surface (um) — absolute y = avg_width/2 + offset
xy_offsets_um      = [0.1, 0.25, 0.5, 1.0];
xy_fixed_offset_um = 0.25;   % for Plot 7

% Line colors for multi-curve plots
line_colors = lines(8);

%% -- Load data ---------------------------------------------------------------

fprintf('Loading: %s\n', result_filepath);
if ~exist(result_filepath, 'file')
    error('File not found! Please check result_filepath.');
end
data = load(result_filepath);

% Resonance wavelength
if isfield(data, 'resonance_wavelength_nm')
    wl_res = double(data.resonance_wavelength_nm) * 1e-9;
    fprintf('Resonance wavelength: %.3f nm\n', wl_res * 1e9);
else
    wl_res = 1.55e-6;
    fprintf('Warning: resonance_wavelength_nm not found, defaulting to %.1f nm\n', wl_res*1e9);
end

%% -- Detect available data sources -------------------------------------------

has_xz = isfield(data, 'field_xz_side') && isstruct(data.field_xz_side);
has_xy = isfield(data, 'field_xy') && isstruct(data.field_xy);
has_3d = isfield(data, 'field_3d') && isstruct(data.field_3d);

if ~has_xz && ~has_xy && ~has_3d
    error('No field monitor data found. Need field_xz_side, field_xy, or field_3d.');
end

%% -- Extract XZ side view data -----------------------------------------------

if has_xz
    d_xz = data.field_xz_side;
    x_xz = double(d_xz.x); y_xz = double(d_xz.y); z_xz = double(d_xz.z);
    Nx_xz = length(x_xz); Ny_xz = length(y_xz); Nz_xz = length(z_xz);

    if isfield(d_xz, 'lambda_3d')
        lam_xz = double(d_xz.lambda_3d);
    else
        lam_xz = wl_res;
    end
    Nlam_xz = length(lam_xz);
    [~, idx_lam_xz] = min(abs(lam_xz - wl_res));
    [~, iy0_xz] = min(abs(y_xz));

    E_5D_xz = reshape(double(d_xz.E_res), [Nx_xz, Ny_xz, Nz_xz, Nlam_xz, 3]);
    E_plane_xz = squeeze(E_5D_xz(:, iy0_xz, :, idx_lam_xz, :));  % [Nx, Nz, 3]
    I_xz = sum(abs(E_plane_xz).^2, 3);  % [Nx, Nz]
    I_xz(~isfinite(I_xz)) = 0;

    has_poynting_xz = isfield(d_xz, 'P_res') && ~isempty(d_xz.P_res);
    if has_poynting_xz
        P_5D_xz = reshape(double(d_xz.P_res), [Nx_xz, Ny_xz, Nz_xz, Nlam_xz, 3]);
        Px_xz = real(squeeze(P_5D_xz(:, iy0_xz, :, idx_lam_xz, 1)));  % [Nx, Nz]
        Pz_xz = real(squeeze(P_5D_xz(:, iy0_xz, :, idx_lam_xz, 3)));  % [Nx, Nz]
        Px_xz(~isfinite(Px_xz)) = 0;
        Pz_xz(~isfinite(Pz_xz)) = 0;
        clear P_5D_xz;
    end
    clear E_5D_xz;

    x_xz_um = x_xz * 1e6;
    z_xz_um = z_xz * 1e6;
    fprintf('XZ monitor: [%d x %d], x=[%.2f, %.2f], z=[%.2f, %.2f] um, Poynting=%s\n', ...
            Nx_xz, Nz_xz, min(x_xz_um), max(x_xz_um), min(z_xz_um), max(z_xz_um), ...
            string(has_poynting_xz));
else
    has_poynting_xz = false;
    fprintf('XZ side view monitor not available.\n');
end

%% -- Extract XY top view data ------------------------------------------------

if has_xy
    d_xy = data.field_xy;
    x_xy = double(d_xy.x); y_xy = double(d_xy.y); z_xy = double(d_xy.z);
    Nx_xy = length(x_xy); Ny_xy = length(y_xy); Nz_xy = length(z_xy);

    if isfield(d_xy, 'lambda_3d')
        lam_xy = double(d_xy.lambda_3d);
    else
        lam_xy = wl_res;
    end
    Nlam_xy = length(lam_xy);
    [~, idx_lam_xy] = min(abs(lam_xy - wl_res));
    [~, iz0_xy] = min(abs(z_xy));

    E_5D_xy = reshape(double(d_xy.E_res), [Nx_xy, Ny_xy, Nz_xy, Nlam_xy, 3]);
    E_plane_xy = squeeze(E_5D_xy(:, :, iz0_xy, idx_lam_xy, :));  % [Nx, Ny, 3]
    I_xy = sum(abs(E_plane_xy).^2, 3);  % [Nx, Ny]
    I_xy(~isfinite(I_xy)) = 0;

    has_poynting_xy = isfield(d_xy, 'P_res') && ~isempty(d_xy.P_res);
    if has_poynting_xy
        P_5D_xy = reshape(double(d_xy.P_res), [Nx_xy, Ny_xy, Nz_xy, Nlam_xy, 3]);
        Px_xy = real(squeeze(P_5D_xy(:, :, iz0_xy, idx_lam_xy, 1)));  % [Nx, Ny]
        Py_xy = real(squeeze(P_5D_xy(:, :, iz0_xy, idx_lam_xy, 2)));  % [Nx, Ny]
        Px_xy(~isfinite(Px_xy)) = 0;
        Py_xy(~isfinite(Py_xy)) = 0;
        clear P_5D_xy;
    end
    clear E_5D_xy;

    x_xy_um = x_xy * 1e6;
    y_xy_um = y_xy * 1e6;
    fprintf('XY monitor: [%d x %d], x=[%.2f, %.2f], y=[%.2f, %.2f] um, Poynting=%s\n', ...
            Nx_xy, Ny_xy, min(x_xy_um), max(x_xy_um), min(y_xy_um), max(y_xy_um), ...
            string(has_poynting_xy));
else
    has_poynting_xy = false;
    Ny_xy = 0;
    fprintf('XY top view monitor not available.\n');
end

%% -- Fallback: 3D monitor (intensity only) -----------------------------------

use_3d_fallback_xz = ~has_xz && has_3d;
use_3d_fallback_xy = ~has_xy && has_3d;

if use_3d_fallback_xz || use_3d_fallback_xy
    fprintf('Using 3D monitor as fallback (intensity only, no Poynting).\n');
    d_3d = data.field_3d;
    x_3d = double(d_3d.x); y_3d = double(d_3d.y); z_3d = double(d_3d.z);
    Nx_3d = length(x_3d); Ny_3d = length(y_3d); Nz_3d = length(z_3d);

    if isfield(d_3d, 'lambda_3d')
        lam_3d = double(d_3d.lambda_3d);
    else
        lam_3d = wl_res;
    end
    Nlam_3d = length(lam_3d);
    [~, idx_lam_3d] = min(abs(lam_3d - wl_res));

    E_5D_3d = reshape(double(d_3d.E_res), [Nx_3d, Ny_3d, Nz_3d, Nlam_3d, 3]);
    E_at_lam = squeeze(E_5D_3d(:, :, :, idx_lam_3d, :));  % [Nx, Ny, Nz, 3]
    I_3d_vol = sum(abs(E_at_lam).^2, 4);  % [Nx, Ny, Nz]
    I_3d_vol(~isfinite(I_3d_vol)) = 0;
    clear E_5D_3d E_at_lam;

    if use_3d_fallback_xz
        [~, iy0_3d] = min(abs(y_3d));
        I_xz = squeeze(I_3d_vol(:, iy0_3d, :));  % [Nx, Nz]
        x_xz_um = x_3d * 1e6;
        z_xz_um = z_3d * 1e6;
        Nz_xz = Nz_3d;
        fprintf('3D fallback XZ: sliced at y=%.3f um\n', y_3d(iy0_3d)*1e6);
    end
    if use_3d_fallback_xy
        [~, iz0_3d] = min(abs(z_3d));
        I_xy = squeeze(I_3d_vol(:, :, iz0_3d));  % [Nx, Ny]
        x_xy_um = x_3d * 1e6;
        y_xy_um = y_3d * 1e6;
        Ny_xy = Ny_3d;
        fprintf('3D fallback XY: sliced at z=%.3f um\n', z_3d(iz0_3d)*1e6);
    end
    clear I_3d_vol;
end

%% -- Resolve geometry --------------------------------------------------------

if isnan(override_core_height_m)
    if isfield(data, 'core_height_m')
        core_height_m = double(data.core_height_m);
    else
        core_height_m = 350e-9;
        fprintf('Warning: core_height_m not found, defaulting to %.0f nm\n', core_height_m*1e9);
    end
else
    core_height_m = override_core_height_m;
end

if isnan(override_avg_width_m)
    if isfield(data, 'avg_corrugation_width_m')
        avg_width_m = double(data.avg_corrugation_width_m);
    else
        avg_width_m = 800e-9;
        fprintf('Warning: avg_corrugation_width_m not found, defaulting to %.0f nm\n', avg_width_m*1e9);
    end
else
    avg_width_m = override_avg_width_m;
end

core_hh_um = core_height_m * 1e6 / 2;   % half-height (z)
core_hw_um = avg_width_m   * 1e6 / 2;   % half-width  (y)
fprintf('Core half-height=%.3f um, half-width=%.3f um\n', core_hh_um, core_hw_um);

%% -- Prepare sampling coordinates -------------------------------------------

% XZ plane: z-offsets from core surface
if exist('z_xz_um', 'var')
    [xz_iz_pos, xz_z_pos_um] = compute_sample_indices(z_xz_um, core_hh_um, xz_offsets_um, 'positive');
    [xz_iz_neg, xz_z_neg_um] = compute_sample_indices(z_xz_um, core_hh_um, xz_offsets_um, 'negative');
    [xz_iz_fixed, xz_z_fixed_um] = compute_sample_indices(z_xz_um, core_hh_um, xz_fixed_offset_um, 'positive');
end

% XY plane: y-offsets from core surface
if exist('y_xy_um', 'var') && Ny_xy > 1
    [xy_iy_pos, xy_y_pos_um] = compute_sample_indices(y_xy_um, core_hw_um, xy_offsets_um, 'positive');
    [xy_iy_neg, xy_y_neg_um] = compute_sample_indices(y_xy_um, core_hw_um, xy_offsets_um, 'negative');
    [xy_iy_fixed, xy_y_fixed_um] = compute_sample_indices(y_xy_um, core_hw_um, xy_fixed_offset_um, 'positive');
end

%% -- XZ plane analysis: vertical radiation -----------------------------------

can_plot_xz = exist('I_xz', 'var');

if can_plot_xz && plot_xz_Pz_vs_x && has_poynting_xz && ~isempty(xz_iz_pos)
    figure('Name', 'XZ: Pz(x) at multiple z', 'Position', [50, 550, 900, 450]);
    hold on;
    for k = 1:numel(xz_iz_pos)
        iz = xz_iz_pos(k);
        Pz_line = Pz_xz(:, iz);
        plot(x_xz_um, Pz_line, 'LineWidth', 1.3, 'Color', line_colors(k,:), ...
             'DisplayName', sprintf('z = %.2f \\mum', xz_z_pos_um(k)));
    end
    yline(0, 'k--', 'HandleVisibility', 'off');
    draw_grating_xlines(data);
    hold off;
    xlabel('x (\mum)'); ylabel('P_z (arb. units)');
    title('XZ plane (y\approx0): Vertical Poynting flux P_z(x) at multiple heights');
    legend('Location', 'best'); grid on; box on;
end

if can_plot_xz && plot_xz_Pout_vs_z && has_poynting_xz
    Pout_z = zeros(Nz_xz, 1);
    for iz = 1:Nz_xz
        Pout_z(iz) = trapz(x_xz_um, Pz_xz(:, iz));
    end

    figure('Name', 'XZ: Pout(z)', 'Position', [50, 50, 900, 450]);
    plot(z_xz_um, Pout_z, 'b-', 'LineWidth', 1.5);
    hold on;
    xline(core_hh_um, 'r--', 'core', 'HandleVisibility', 'off', 'LabelOrientation', 'horizontal');
    xline(-core_hh_um, 'r--', '', 'HandleVisibility', 'off');
    yline(0, 'k:', 'HandleVisibility', 'off');
    hold off;
    xlabel('z (\mum)'); ylabel('\int P_z(x, z) dx  (arb. units)');
    title('XZ plane (y\approx0): Integrated vertical flux vs z');
    grid on; box on;
end

if can_plot_xz && plot_xz_Px_at_z && has_poynting_xz && exist('xz_iz_fixed', 'var') && ~isempty(xz_iz_fixed)
    iz_f = xz_iz_fixed(1);
    Px_line = Px_xz(:, iz_f);

    figure('Name', 'XZ: Px(x) at fixed z', 'Position', [1000, 550, 900, 450]);
    plot(x_xz_um, Px_line, 'r-', 'LineWidth', 1.5);
    hold on;
    yline(0, 'k--', 'HandleVisibility', 'off');
    draw_grating_xlines(data);
    hold off;
    xlabel('x (\mum)'); ylabel('P_x (arb. units)');
    title(sprintf('XZ plane (y\\approx0): Lateral flux P_x(x) at z = %.2f \\mum', xz_z_fixed_um(1)));
    grid on; box on;
end

if can_plot_xz && plot_xz_intensity && exist('xz_iz_pos', 'var') && ~isempty(xz_iz_pos)
    n_plot = min(2, numel(xz_iz_pos));
    figure('Name', 'XZ: |E|^2(x) log', 'Position', [1000, 50, 900, 450]);
    hold on;
    for k = 1:n_plot
        iz = xz_iz_pos(k);
        I_line = I_xz(:, iz);
        I_line(I_line <= 0) = NaN;
        semilogy(x_xz_um, I_line, 'LineWidth', 1.3, 'Color', line_colors(k,:), ...
                 'DisplayName', sprintf('z = %.2f \\mum', xz_z_pos_um(k)));
        try
            [alpha, x0, I0] = fit_decay_envelope(x_xz_um, I_line);
            x_fit = x_xz_um(x_xz_um > x0);
            I_fit = I0 * exp(-alpha * abs(x_fit - x0));
            semilogy(x_fit, I_fit, '--', 'Color', line_colors(k,:), 'LineWidth', 1.0, ...
                     'HandleVisibility', 'off');
            fprintf('  XZ z=%.2f um: decay length = %.2f um (alpha=%.2f /um)\n', ...
                    xz_z_pos_um(k), 1/alpha, alpha);
        catch
        end
    end
    draw_grating_xlines(data);
    hold off;
    xlabel('x (\mum)'); ylabel('|E|^2 (arb. units)');
    title('XZ plane (y\approx0): Field intensity at fixed heights (log scale)');
    legend('Location', 'best'); grid on; box on;
end

%% -- XY plane analysis: lateral radiation ------------------------------------

if Ny_xy <= 1
    fprintf('Skipping XY analysis: y-dimension has only %d point (2D simulation).\n', Ny_xy);
else

can_plot_xy = exist('I_xy', 'var');

if can_plot_xy && plot_xy_Py_vs_x && has_poynting_xy && ~isempty(xy_iy_pos)
    figure('Name', 'XY: Py(x) at multiple y', 'Position', [50, 550, 900, 450]);
    hold on;
    for k = 1:numel(xy_iy_pos)
        iy = xy_iy_pos(k);
        Py_line = Py_xy(:, iy);
        plot(x_xy_um, Py_line, 'LineWidth', 1.3, 'Color', line_colors(k,:), ...
             'DisplayName', sprintf('y = %.2f \\mum', xy_y_pos_um(k)));
    end
    yline(0, 'k--', 'HandleVisibility', 'off');
    draw_grating_xlines(data);
    hold off;
    xlabel('x (\mum)'); ylabel('P_y (arb. units)');
    title('XY plane (z\approx0): Lateral Poynting flux P_y(x) at multiple y-offsets');
    legend('Location', 'best'); grid on; box on;
end

if can_plot_xy && plot_xy_Pout_vs_y && has_poynting_xy
    Pout_y = zeros(Ny_xy, 1);
    for iy = 1:Ny_xy
        Pout_y(iy) = trapz(x_xy_um, Py_xy(:, iy));
    end

    figure('Name', 'XY: Pout(y)', 'Position', [50, 50, 900, 450]);
    plot(y_xy_um, Pout_y, 'b-', 'LineWidth', 1.5);
    hold on;
    xline(core_hw_um, 'r--', 'core', 'HandleVisibility', 'off', 'LabelOrientation', 'horizontal');
    xline(-core_hw_um, 'r--', '', 'HandleVisibility', 'off');
    yline(0, 'k:', 'HandleVisibility', 'off');
    hold off;
    xlabel('y (\mum)'); ylabel('\int P_y(x, y) dx  (arb. units)');
    title('XY plane (z\approx0): Integrated lateral flux vs y');
    grid on; box on;
end

if can_plot_xy && plot_xy_Px_at_y && has_poynting_xy && exist('xy_iy_fixed', 'var') && ~isempty(xy_iy_fixed)
    iy_f = xy_iy_fixed(1);
    Px_line = Px_xy(:, iy_f);

    figure('Name', 'XY: Px(x) at fixed y', 'Position', [1000, 550, 900, 450]);
    plot(x_xy_um, Px_line, 'r-', 'LineWidth', 1.5);
    hold on;
    yline(0, 'k--', 'HandleVisibility', 'off');
    draw_grating_xlines(data);
    hold off;
    xlabel('x (\mum)'); ylabel('P_x (arb. units)');
    title(sprintf('XY plane (z\\approx0): Lateral flux P_x(x) at y = %.2f \\mum', xy_y_fixed_um(1)));
    grid on; box on;
end

if can_plot_xy && plot_xy_intensity && exist('xy_iy_pos', 'var') && ~isempty(xy_iy_pos)
    n_plot = min(2, numel(xy_iy_pos));
    figure('Name', 'XY: |E|^2(x) log', 'Position', [1000, 50, 900, 450]);
    hold on;
    for k = 1:n_plot
        iy = xy_iy_pos(k);
        I_line = I_xy(:, iy);
        I_line(I_line <= 0) = NaN;
        semilogy(x_xy_um, I_line, 'LineWidth', 1.3, 'Color', line_colors(k,:), ...
                 'DisplayName', sprintf('y = %.2f \\mum', xy_y_pos_um(k)));
        try
            [alpha, x0, I0] = fit_decay_envelope(x_xy_um, I_line);
            x_fit = x_xy_um(x_xy_um > x0);
            I_fit = I0 * exp(-alpha * abs(x_fit - x0));
            semilogy(x_fit, I_fit, '--', 'Color', line_colors(k,:), 'LineWidth', 1.0, ...
                     'HandleVisibility', 'off');
            fprintf('  XY y=%.2f um: decay length = %.2f um (alpha=%.2f /um)\n', ...
                    xy_y_pos_um(k), 1/alpha, alpha);
        catch
        end
    end
    draw_grating_xlines(data);
    hold off;
    xlabel('x (\mum)'); ylabel('|E|^2 (arb. units)');
    title('XY plane (z\approx0): Field intensity at fixed y-offsets (log scale)');
    legend('Location', 'best'); grid on; box on;
end

end  % Ny_xy > 1

%% -- Summary -----------------------------------------------------------------

fprintf('\n=== Radiation recycling summary ===\n');

% Vertical (from XZ monitor)
if has_poynting_xz
    Pout_z_all = zeros(Nz_xz, 1);
    for iz = 1:Nz_xz
        Pout_z_all(iz) = trapz(x_xz_um, Pz_xz(:, iz));
    end
    z_above = z_xz_um > core_hh_um;
    z_below = z_xz_um < -core_hh_um;
    if any(z_above)
        [peak_up, idx_up] = max(abs(Pout_z_all(z_above)));
        z_a = z_xz_um(z_above);
        fprintf('  Peak |Pout_z| above core: %.3e at z = %.2f um\n', peak_up, z_a(idx_up));
    end
    if any(z_below)
        [peak_dn, idx_dn] = max(abs(Pout_z_all(z_below)));
        z_b = z_xz_um(z_below);
        fprintf('  Peak |Pout_z| below core: %.3e at z = %.2f um\n', peak_dn, z_b(idx_dn));
    end
end

% Lateral (from XY monitor)
if has_poynting_xy && Ny_xy > 1
    Pout_y_all = zeros(Ny_xy, 1);
    for iy = 1:Ny_xy
        Pout_y_all(iy) = trapz(x_xy_um, Py_xy(:, iy));
    end
    y_right = y_xy_um > core_hw_um;
    y_left  = y_xy_um < -core_hw_um;
    if any(y_right)
        [peak_r, idx_r] = max(abs(Pout_y_all(y_right)));
        y_r = y_xy_um(y_right);
        fprintf('  Peak |Pout_y| right of core: %.3e at y = %.2f um\n', peak_r, y_r(idx_r));
    end
    if any(y_left)
        [peak_l, idx_l] = max(abs(Pout_y_all(y_left)));
        y_l = y_xy_um(y_left);
        fprintf('  Peak |Pout_y| left of core: %.3e at y = %.2f um\n', peak_l, y_l(idx_l));
    end

    % Anisotropy ratio (vertical vs lateral)
    if has_poynting_xz && exist('peak_up', 'var') && exist('peak_r', 'var')
        fprintf('  Vertical/lateral ratio: %.2f\n', peak_up / max(peak_r, 1e-30));
    end
end

if ~has_poynting_xz && ~has_poynting_xy
    fprintf('  Poynting data not available. Only intensity decay analysis was performed.\n');
end

fprintf('\nDone.\n');

%% ============================================================================
%  Local functions
%% ============================================================================

function [valid_idx, valid_um] = compute_sample_indices(coord_um, core_half_um, offsets_um, direction)
    % Compute nearest-grid indices for sampling at offsets from core surface.
    % direction: 'positive' (above/right) or 'negative' (below/left)
    if strcmp(direction, 'positive')
        targets = core_half_um + offsets_um;
        coord_max = max(coord_um);
        keep = targets <= coord_max;
    else
        targets = -(core_half_um + offsets_um);
        coord_min = min(coord_um);
        keep = targets >= coord_min;
    end

    if any(~keep)
        dropped = offsets_um(~keep);
        for i = 1:numel(dropped)
            fprintf('Warning: offset %.2f um (%s) exceeds domain, skipping.\n', ...
                    dropped(i), direction);
        end
    end

    targets = targets(keep);
    valid_idx = zeros(1, numel(targets));
    valid_um  = zeros(1, numel(targets));
    for k = 1:numel(targets)
        [~, valid_idx(k)] = min(abs(coord_um - targets(k)));
        valid_um(k) = coord_um(valid_idx(k));
    end
end

function draw_grating_xlines(d)
    % Draw vertical dashed lines at grating region boundaries.
    if ~isfield(d, 'n_periods_each_side'), return; end
    N     = double(d.n_periods_each_side);
    pitch = double(d.pitch_m) * 1e6;  % um

    % Cavity length: default to pitch/2 (quarter-wave phase shift)
    if isfield(d, 'cavity_length_m') && ~isempty(d.cavity_length_m)
        cav = double(d.cavity_length_m) * 1e6;
    else
        cav = pitch / 2;
    end

    x_edge = cav/2 + N * pitch;
    xline(x_edge, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    xline(-x_edge, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    xline(0, ':', 'Color', [0.7 0.3 0.3], 'HandleVisibility', 'off');  % defect center
end

function [alpha, x0, I0] = fit_decay_envelope(x_um, I_line)
    % Fit exponential decay I = I0 * exp(-alpha * |x - x0|) on the positive-x side.
    % Returns decay constant alpha (1/um), peak position x0, and peak intensity I0.
    I_clean = I_line;
    I_clean(isnan(I_clean) | I_clean <= 0) = 0;

    % Work on positive x only, find peak
    mask_pos = x_um > 0;
    x_pos = x_um(mask_pos);
    I_pos = I_clean(mask_pos);

    [I0, ipk] = max(I_pos);
    x0 = x_pos(ipk);

    % Fit in the decay region (beyond peak, intensity > 1% of peak)
    decay_mask = (1:numel(x_pos))' > ipk & I_pos > 0.01 * I0;
    if sum(decay_mask) < 3
        error('Not enough points for exponential fit.');
    end
    x_fit = x_pos(decay_mask) - x0;
    logI  = log(I_pos(decay_mask));
    p = polyfit(x_fit, logI, 1);
    alpha = -p(1);   % positive decay constant
    I0 = exp(p(2));

    if alpha <= 0
        error('Fit returned non-decaying profile.');
    end
end