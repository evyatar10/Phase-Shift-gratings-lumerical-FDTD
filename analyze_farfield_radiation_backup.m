% analyze_farfield_radiation.m
% Analyzes the Far-Field radiation pattern projected directly by Lumerical FDTD.
% It takes the raw complex Electric fields, transforms the math to find
% physical angles and applies the proper power scaling physics.

clear; clc; close all;

%% --- 1. Load Data ---
% Specify the result file to analyze.
data_file = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles_2\results\result_80_periods_10_apodizations_CONST.mat";

fprintf('Loading data...\n');
if ~exist(data_file, 'file')
    error('File not found: %s', data_file);
end
data = load(data_file);

if isfield(data, 'side_monitor') && ~isempty(fieldnames(data.side_monitor))
    ff_data = data.side_monitor;
else
    error('The loaded file does not contain side_monitor data! Please ensure record_farfield=True in your Python script.');
end

lam_m = double(squeeze(ff_data.f_monitor));

% Extract the direction cosines (ux, uy) and complex Electric fields
ux = double(squeeze(ff_data.ff_ux));
uy = double(squeeze(ff_data.ff_uy));
ux = ux(:); % Force into column vector to avoid implicit expansion bugs
uy = uy(:); % Force into column vector to avoid implicit expansion bugs

Ex = double(squeeze(ff_data.ff_Ex));
Ey = double(squeeze(ff_data.ff_Ey));
Ez = double(squeeze(ff_data.ff_Ez));

% Find the index corresponding to the Z=0 plane (uy=0)
[~, idx_uy_0] = min(abs(uy));
ux_1D = ux; % We look along the entire X-axis
Ex_1D = squeeze(Ex(:, idx_uy_0));
Ey_1D = squeeze(Ey(:, idx_uy_0));
Ez_1D = squeeze(Ez(:, idx_uy_0));

fprintf('--- Radiation Analysis ---\n');
fprintf('Wavelength: %.3f nm\n', lam_m * 1e9);

%% --- 2. Process Math and Physics ---

% 1. Convert unit vector 'ux' to physical angle in degrees
%    Note: the Far-Field projection covers hemisphere so |ux| <= 1.0
%    We filter out unphysical evanescent components (|ux| > 1)
valid_idx = abs(ux_1D) <= 1.0;
ux_valid = ux_1D(valid_idx);
theta_rad = asin(ux_valid);
theta_deg = rad2deg(theta_rad);

% 2. Calculate raw intensity |E|^2
Ex_v = Ex_1D(valid_idx);
Ey_v = Ey_1D(valid_idx);
Ez_v = Ez_1D(valid_idx);
I_E2 = abs(Ex_v).^2 + abs(Ey_v).^2 + abs(Ez_v).^2;

% 3. Apply the Jacobian scaling logic required for Angular Projection
%    Power/Angle = Intensity * cos(theta)
P_theta = I_E2 .* cos(theta_rad);

% Normalize to 0 dB maximum for clean visualization
P_theta_norm = P_theta / max(P_theta(:));
P_theta_dB = 10 * log10(P_theta_norm);

% Extract the envelope of the angular pattern using peak detection
[pks, locs] = findpeaks(P_theta_dB);
% Include endpoints to prevent extrapolation artifacts
if ~isempty(locs)
    locs_ext = [1; locs; length(P_theta_dB)];
    pks_ext = [P_theta_dB(1); pks; P_theta_dB(end)];
    % Interpolate the peaks to create a continuous envelope
    P_theta_dB_env = interp1(theta_deg(locs_ext), pks_ext, theta_deg, 'pchip');
else
    P_theta_dB_env = P_theta_dB;
end

% Apply noise floor (cut to -30 dB)
noise_floor_dB = -30;
P_theta_dB(P_theta_dB < noise_floor_dB) = noise_floor_dB;
P_theta_dB_env(P_theta_dB_env < noise_floor_dB) = noise_floor_dB;

%% --- 3. Plot Results ---
% --- Options ---
show_fringes = true;       % Set to false to hide the rapid blue oscillations
show_envelope = true;     % Set to true to show the smooth red envelope
% ---------------

figure('Name', 'Native Lumerical Far-Field Analysis', 'Color', 'w', 'Position', [100, 100, 800, 500]);
hold on;

% Main angular power line (Fringes)
if show_fringes
    plot(theta_deg, P_theta_dB, 'b-', 'LineWidth', 1.0, 'DisplayName', 'Raw Interference Fringes');
end

% Smooth Envelope
if show_envelope
    plot(theta_deg, P_theta_dB_env, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Far-Field Radiation Envelope');
end

% Embellish the plot
title_str = sprintf('Radiation Power vs. Angle (Native Far-Field Math)\n\\lambda = %.2f nm', lam_m * 1e9);
title(title_str, 'FontSize', 14);
xlabel('Radiation Angle \theta (degrees from Core Normal)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Normalized Radiation Power density P(\theta) [dB]', 'FontSize', 12, 'FontWeight', 'bold');
xlim([-90 90]);

% Cut plot to -30 dB floor
ylim([-30 5]);
grid on;
legend('Location', 'south', 'NumColumns', 2);
hold off;

fprintf('Done plotting.\n');
