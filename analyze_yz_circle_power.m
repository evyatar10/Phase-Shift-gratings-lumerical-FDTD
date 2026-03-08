% analyze_yz_circle_power.m
% Extracts the YZ cross-section field profile, draws a circular extraction
% boundary, and plots the field power versus the angle along that circle.

clear; close all; clc;

%% --- Configuration ---
% Update this to the actual path of your simulation results
% Defaulting to the file referenced in previous scripts
result_filepath = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles_5_long\results\result_100_periods_CONST.mat";

circle_radius_um = 3; % Radius of the extraction circle [um]
num_angle_points = 361; % Number of points for angle sweep (0 to 360)

%% --- 1. Load Data ---
fprintf('Loading data...\n');
if ~exist(result_filepath, 'file')
    [file, path] = uigetfile('*.mat', 'Select Simulation Result File');
    if isequal(file, 0)
        error('File selection canceled.');
    end
    result_filepath = fullfile(path, file);
end

data = load(result_filepath);

%% --- 2. Find Resonance ---
if isfield(data, 'T') && isfield(data, 'wl_m')
    T = squeeze(data.T);
    wl = squeeze(data.wl_m);

    stopband_indices = find(T < 0.6);
    if isempty(stopband_indices)
        stopband_indices = find(T < 0.85);
    end

    if isempty(stopband_indices)
        [~, idx_peak] = max(T);
    else
        idx_start = stopband_indices(1);
        idx_end = stopband_indices(end);
        [~, local_peak_idx] = max(T(idx_start:idx_end));
        idx_peak = idx_start + local_peak_idx - 1;
    end
    wl_res = wl(idx_peak);
else
    error('Transmission or Wavelength data missing.');
end

%% --- 3. Extract YZ Field Profile ---
if ~isfield(data, 'field_yz_cross')
    error('The loaded file does not contain field_yz_cross data!');
end

d_yz = data.field_yz_cross;
x_yz = double(d_yz.x);
y_yz = double(d_yz.y);
z_yz = double(d_yz.z);

if isfield(d_yz, 'lambda_3d')
    lam_yz = double(d_yz.lambda_3d);
else
    lam_yz = 1.55e-6;
end

[~, idx_lam] = min(abs(lam_yz - wl_res));
wl_3d_actual = lam_yz(idx_lam);

Nx_yz = length(x_yz); Ny_yz = length(y_yz); Nz_yz = length(z_yz); Nlam = length(lam_yz);
E_5D_yz = reshape(d_yz.E_res, [Nx_yz, Ny_yz, Nz_yz, Nlam, 3]);

[~, idx_x0_yz] = min(abs(x_yz));
E_plane_comp = squeeze(E_5D_yz(idx_x0_yz, :, :, idx_lam, :));

if ndims(E_plane_comp) >= 3
    E_sq_plane = sum(abs(E_plane_comp).^2, ndims(E_plane_comp));
else
    E_sq_plane = sum(abs(E_plane_comp).^2, 2);
end
E_sq_plane_dB = 10 * log10(E_sq_plane);

%% --- 4. Define Circle & Interpolate Power ---
angles_deg = linspace(0, 360, num_angle_points);
angles_rad = deg2rad(angles_deg);

y_circle_um = circle_radius_um * cos(angles_rad);
z_circle_um = circle_radius_um * sin(angles_rad);

[Y_mesh, Z_mesh] = meshgrid(y_yz * 1e6, z_yz * 1e6);
V = E_sq_plane.'; % Transpose to match meshgrid orientation where rows = Y, cols = X
P_circle_lin = interp2(Y_mesh, Z_mesh, V, y_circle_um, z_circle_um, 'linear', NaN);

valid_idx = ~isnan(P_circle_lin);
if sum(~valid_idx) > 0
    fprintf('Warning: Circle exceeds YZ domain limits! NaNs present. Consider reducing circle_radius_um.\n');
end

P_circle_dB = 10 * log10(P_circle_lin);
max_dB = max(E_sq_plane_dB(:));

%% --- 5. Plot 2D Field Profile with Circle Overlay ---
figure('Name', 'YZ View Field Profile (with Circle)', 'Color', 'w', 'Position', [100 100 800 600]);
imagesc(y_yz * 1e6, z_yz * 1e6, E_sq_plane_dB');
set(gca, 'YDir', 'normal');
colormap(jet);
dB_limit = 80;
try clim([max_dB - dB_limit, max_dB]); catch; caxis([max_dB - dB_limit, max_dB]); end

cb = colorbar;
ylabel(cb, '10*log_{10}(|E|^2) [dB]');
xlabel('Position Y [\mum]', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Position Z [\mum]', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('YZ View: Field Profile at \\lambda = %.3f nm (Circle R=%.1f \\mum)', wl_3d_actual*1e9, circle_radius_um), 'FontSize', 14);

hold on;
plot(y_circle_um, z_circle_um, 'w--', 'LineWidth', 2);
hold off;

%% --- 6. Plot Power vs Angle (Linear Plot) ---
figure('Name', 'Power vs Angle on Circle', 'Color', 'w', 'Position', [150 150 700 450]);
plot(angles_deg(valid_idx), P_circle_dB(valid_idx), 'b-', 'LineWidth', 2);
grid on;
xlim([0 180]);
ylim([max_dB - dB_limit, max_dB + 5]);
xlabel('Angle \phi [Degrees] (0^\circ = +Y, 90^\circ = +Z)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Extracted Power [dB]', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Radiation Power vs Angle (Radius = %.1f \\mum)', circle_radius_um), 'FontSize', 14);

%% --- 7. Plot Power vs Angle (Polar Plot) ---
figure('Name', 'Polar Radiation', 'Color', 'w', 'Position', [200 200 600 600]);

noise_floor_dB = max_dB - dB_limit;
polar_P = P_circle_dB - noise_floor_dB;
polar_P(polar_P < 0) = 0; % Ensure non-negative limit for polar plot

polarplot(angles_rad(valid_idx), polar_P(valid_idx), 'r-', 'LineWidth', 2);
ax = gca;
ax.ThetaZeroLocation = 'right';
ax.ThetaDir = 'counterclockwise';

rticks_val = linspace(0, dB_limit, 5);
rticklabels_str = arrayfun(@(val) sprintf('%g dB', val + noise_floor_dB), rticks_val, 'UniformOutput', false);
ax.RTick = rticks_val;
ax.RTickLabel = rticklabels_str;

title(sprintf('Polar Radiation Pattern at R=%.1f \\mum [dB]', circle_radius_um), 'FontSize', 14);
fprintf('Done.\n');
