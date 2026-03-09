% analyze_farfield_radiation.m
% Analyzes the Far-Field radiation pattern projected directly by Lumerical FDTD.

clear; clc; close all;

%% --- 1. Load Data ---
data_file = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles_5_long\results\result_100_periods_CONST.mat";

fprintf('Loading data...\n');
if ~exist(data_file, 'file')
    error('File not found: %s', data_file);
end
data = load(data_file);

if isfield(data, 'side_monitor') && ~isempty(fieldnames(data.side_monitor))
    ff_data = data.side_monitor;
else
    error('The loaded file does not contain native side_monitor data!');
end

lam_m = double(squeeze(ff_data.f_monitor));

% FORCE ux and uy to be strictly real numbers
ux = real(double(squeeze(ff_data.ff_ux)));
uy = real(double(squeeze(ff_data.ff_uy)));

Ex = double(squeeze(ff_data.ff_Ex));
Ey = double(squeeze(ff_data.ff_Ey));
Ez = double(squeeze(ff_data.ff_Ez));

ux = ux(:);
uy = uy(:);

fprintf('--- 3D Spherical Radiation Analysis (Top Monitor, 30um Window) ---\n');
fprintf('Wavelength: %.3f nm\n', lam_m * 1e9);

%% --- 2. Process Math and Physics into 3D Sphere ---

% Create 2D grids for ux and uy
[UX, UY] = ndgrid(ux, uy);

% Calculate raw intensity |E|^2 for the whole 2D array
I_E2_2D = abs(Ex).^2 + abs(Ey).^2 + abs(Ez).^2;

% Filter for the physical hemisphere (ux^2 + uy^2 <= 1)
valid_mask = (UX.^2 + UY.^2) <= 1.0;

% Calculate cos(theta) for the Jacobian projection
% FIX: Clamp the math to 0 to prevent floating-point negative noise
val_inside_sqrt = 1 - UX(valid_mask).^2 - UY(valid_mask).^2;
val_inside_sqrt(val_inside_sqrt < 0) = 0;

cos_theta_2D = zeros(size(UX));
cos_theta_2D(valid_mask) = real(sqrt(val_inside_sqrt));

% Apply the Jacobian scaling logic (Power per solid angle)
P_2D = I_E2_2D .* cos_theta_2D;
P_2D(~valid_mask) = NaN;

% Normalize to 0 dB maximum
% use omitnan to safely find the maximum ignoring the NaN mask
max_P = max(P_2D(:), [], 'omitnan');
P_2D_norm = P_2D / max_P;

% FIX: Force real to avoid complex noise if P_2D_norm hits microscopic negative numbers
P_2D_dB = real(10 * log10(P_2D_norm));

% Apply noise floor
noise_floor_dB = -80;
P_2D_dB(P_2D_dB < noise_floor_dB) = noise_floor_dB;

% --- Calculate Spherical Coordinates for 3D Surface ---
% FIX: Clamp cos to 1 to prevent acos from generating complex numbers
cos_clamp = cos_theta_2D;
cos_clamp(cos_clamp > 1) = 1;

Theta_grid = real(acos(cos_clamp)); % Polar angle (0 to pi/2)
Phi_grid = real(atan2(UY, UX)); % Azimuthal angle (-pi to pi)

% Convert intensity to a 3D physical shape (Radius = Power)
% FIX: Force final matrices to be real
X_3d = real(P_2D_norm .* sin(Theta_grid) .* cos(Phi_grid));
Y_3d = real(P_2D_norm .* sin(Theta_grid) .* sin(Phi_grid));
Z_3d = real(P_2D_norm .* cos(Theta_grid));

%% --- 3. Full 3D Spherical Plot (Hemisphere) ---
figure('Name', 'Full 3D Radiation Hemisphere', 'Color', 'w', 'Position', [100, 100, 800, 600]);

% Plot the top-half of the sphere
surf(X_3d, Y_3d, Z_3d, P_2D_dB);
shading interp;
colormap('jet');
cb = colorbar;
ylabel(cb, 'Normalized Intensity [dB]', 'FontWeight', 'bold');
caxis([noise_floor_dB 0]);

title(sprintf('Top Radiation Pattern (30 \\mu m Window)\n\\lambda = %.2f nm', lam_m * 1e9), 'FontSize', 14);
xlabel('X (Along Grating)', 'FontWeight', 'bold');
ylabel('Y (Across Grating)', 'FontWeight', 'bold');
zlabel('Z (Radiation Upwards)', 'FontWeight', 'bold');
axis equal;
grid on;
view(45, 30);

%% --- 4. Polar Plots (Cross Sections) ---

% 1. XZ Plane Cut (Horizontal length along device)
[~, idx_uy_0] = min(abs(uy));
P_XZ_slice_dB = P_2D_dB(:, idx_uy_0);

% Only calculate asin on valid numbers
valid_xz = abs(ux) <= 1.0;
% FIX: clamp ux to prevent asin complex errors
ux_clamp = ux(valid_xz);
ux_clamp(ux_clamp > 1) = 1;
ux_clamp(ux_clamp < -1) = -1;

theta_XZ_valid = real(asin(ux_clamp));
P_XZ_valid = P_XZ_slice_dB(valid_xz);

% Mirror to the bottom hemisphere correctly to close the loop
theta_full_XZ = [theta_XZ_valid; pi - flipud(theta_XZ_valid)];
P_full_XZ = [P_XZ_valid; flipud(P_XZ_valid)];

r_plot_XZ = P_full_XZ - noise_floor_dB;
r_plot_XZ(r_plot_XZ < 0) = 0;

figure('Name', 'Polar Radiation (XZ Plane)', 'Color', 'w', 'Position', [150, 150, 500, 500]);
polarplot(theta_full_XZ, r_plot_XZ, 'b-', 'LineWidth', 2.0);
ax = gca;
ax.ThetaZeroLocation = 'top';
ax.ThetaDir = 'clockwise';
rticks_val = linspace(0, -noise_floor_dB, 5);
ax.RTick = rticks_val;
ax.RTickLabel = arrayfun(@(val) sprintf('%g dB', val + noise_floor_dB), rticks_val, 'UniformOutput', false);
title('Radiation in XZ Plane (Along Grating Length)', 'FontSize', 14);


% 2. YZ Plane Cut (Vertical width across device)
[~, idx_ux_0] = min(abs(ux));
P_YZ_slice_dB = P_2D_dB(idx_ux_0, :)';

valid_yz = abs(uy) <= 1.0;
% FIX: clamp uy to prevent asin complex errors
uy_clamp = uy(valid_yz);
uy_clamp(uy_clamp > 1) = 1;
uy_clamp(uy_clamp < -1) = -1;

theta_YZ_valid = real(asin(uy_clamp));
P_YZ_valid = P_YZ_slice_dB(valid_yz);

% Mirror to the bottom hemisphere correctly to close the loop
theta_full_YZ = [theta_YZ_valid; pi - flipud(theta_YZ_valid)];
P_full_YZ = [P_YZ_valid; flipud(P_YZ_valid)];

r_plot_YZ = P_full_YZ - noise_floor_dB;
r_plot_YZ(r_plot_YZ < 0) = 0;

figure('Name', 'Polar Radiation (YZ Plane)', 'Color', 'w', 'Position', [200, 200, 500, 500]);
polarplot(theta_full_YZ, r_plot_YZ, 'r-', 'LineWidth', 2.0);
ax = gca;
ax.ThetaZeroLocation = 'top';
ax.ThetaDir = 'clockwise';
ax.RTick = rticks_val;
ax.RTickLabel = arrayfun(@(val) sprintf('%g dB', val + noise_floor_dB), rticks_val, 'UniformOutput', false);
title('Radiation in YZ Plane (Across Grating Width)', 'FontSize', 14);

fprintf('Done plotting.\n');