% analyze_farfield_radiation.m
% Analyzes the Far-Field radiation pattern projected directly by Lumerical FDTD.
% It takes the raw complex Electric fields, transforms the math to find
% physical angles and applies the proper power scaling physics.

clear; clc; close all;

%% --- 1. Load Data ---
% Specify the result file to analyze.
data_file = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles\results\result_80_periods_CONST.mat";

fprintf('Loading data...\n');
if ~exist(data_file, 'file')
    error('File not found: %s', data_file);
end
data = load(data_file);

if isfield(data, 'farfield_data') && ~isempty(fieldnames(data.farfield_data))
    ff_data = data.farfield_data;
elseif isfield(data, 'farfield') && ~isempty(fieldnames(data.farfield))
    ff_data = data.farfield;
else
    error('The loaded file does not contain native farfield_data (or farfield)! Please ensure record_farfield=True in your Python script.');
end

% Handle wavelength field (Python exports either f_monitor or lambda_m depending on the version)
if isfield(ff_data, 'f_monitor')
    lam_m = double(squeeze(ff_data.f_monitor));
elseif isfield(ff_data, 'lambda_m')
    lam_m = double(squeeze(ff_data.lambda_m));
else
    error('farfield data missing wavelength information (f_monitor or lambda_m)');
end

% Handle monitor_dist_wls which is not explicitly exported by the current Python script
if isfield(ff_data, 'monitor_dist_wls')
    dist_wls = double(squeeze(ff_data.monitor_dist_wls));
else
    dist_wls = NaN; % If not in data, set to NaN to avoid making up a number
end

% Extract the direction cosines (ux, uy) and complex Electric fields
ux = double(squeeze(ff_data.ux));
uy = double(squeeze(ff_data.uy));
ux = ux(:); % Force into column vector to avoid implicit expansion bugs
uy = uy(:); % Force into column vector to avoid implicit expansion bugs

Ex = double(squeeze(ff_data.Ex));
Ey = double(squeeze(ff_data.Ey));
Ez = double(squeeze(ff_data.Ez));

% Find the index corresponding to the Z=0 plane (uy=0)
[~, idx_uy_0] = min(abs(uy));
ux_1D = ux; % We look along the entire X-axis
Ex_1D = squeeze(Ex(:, idx_uy_0));
Ey_1D = squeeze(Ey(:, idx_uy_0));
Ez_1D = squeeze(Ez(:, idx_uy_0));

%% --- 2. Calculate Theoretical Critical Angles ---
n_core = 1.977;
n_clad = 1.44;

% Get exact Effective Index (n_eff) dynamically (same method as before)
neff_data_path = 'C:\Users\evyat\Lumerical\pi_shifts_FDTD_results\neff_vs_wl_new\FDE_sweep_results.mat';
try
    neff_file = load(neff_data_path);
    neff_real_values = real(squeeze(neff_file.neff));
    wl_m_sweep = squeeze(neff_file.wl);
    if mean(wl_m_sweep) > 1
        wl_m_sweep = wl_m_sweep * 1e-6;
    end
    [wl_m_sweep, sort_idx] = sort(wl_m_sweep);
    neff_real_values = neff_real_values(sort_idx);
    n_eff_actual = interp1(wl_m_sweep, neff_real_values, lam_m, 'pchip');
catch
    warning('Could not load exact n_eff from file. Defaulting to 1.55.');
    n_eff_actual = 1.55;
end

theta_critical_rad = asin(n_clad / n_eff_actual);
theta_critical_deg = rad2deg(theta_critical_rad);

fprintf('--- Radiation Analysis ---\n');
fprintf('Wavelength: %.3f nm\n', lam_m * 1e9);
fprintf('Monitor Distance: %.2f wavelengths from Core\n', dist_wls);
fprintf('Critical Angle (Theoretical): %.2f deg\n', theta_critical_deg);

%% --- 3. Process Math and Physics ---

% 1. Use the entire unit vector 'ux' directly (this represents kx/k0)
%    |ux| <= 1.0 corresponds to propagating radiation modes.
%    |ux| > 1.0 corresponds to bound evanescent modes.
ux_all = ux_1D;

% To safely calculate physical angles for the propagating part,
% we clip the values going into asin() to [-1, 1] just for the angle vector,
% but we keep the x-axis as ux directly to visualize everything.
theta_rad_safe = asin(min(max(ux_all, -1), 1));
theta_deg_safe = rad2deg(theta_rad_safe);

% 2. Calculate raw intensity |E|^2 for ALL components
I_E2 = abs(Ex_1D).^2 + abs(Ey_1D).^2 + abs(Ez_1D).^2;

% 3. Apply the Jacobian scaling logic required for Angular Projection
%    Power/Angle = Intensity * cos(theta).
%    For evanescent waves (|ux|>1), cos(theta) theoretically becomes imaginary
%    (ikz), representing purely reactive power. We can plot the envelope by
%    just scaling the propagating part and plotting intensity for the rest,
%    or scaling everything by abs(cos(theta)) = abs(sqrt(1 - ux^2)).
cos_theta = sqrt(abs(1 - ux_all.^2)); % abs to handle ux > 1 which would be imaginary
P_all = I_E2 .* cos_theta;

% Normalize to 0 dB maximum for clean visualization
P_theta_norm = P_all / max(P_all(:));
P_theta_dB = 10 * log10(P_theta_norm);

% Apply noise floor to avoid plotting infinite negative noise
noise_floor_dB = -80;
P_theta_dB(P_theta_dB < noise_floor_dB) = noise_floor_dB;

%% --- 4. Plot Results ---
figure('Name', 'Native Lumerical Far-Field Analysis', 'Color', 'w', 'Position', [100, 100, 800, 500]);

% Main angular power line
% We plot against the direction cosine (ux = sin(theta)) to natively support ux > 1
plot(ux_all, P_theta_dB, 'b-', 'LineWidth', 2.0);
hold on;

% Mark the critical angles using direction cosines (ux)
ux_crit = sin(theta_critical_rad);
y_lims = ylim();
xline(ux_crit, '--r', 'LineWidth', 1.5, 'Label', sprintf('+%.1f^\\circ (TIR Limit)', theta_critical_deg), 'LabelOrientation', 'horizontal', 'LabelHorizontalAlignment', 'center', 'LabelVerticalAlignment', 'bottom');
xline(-ux_crit, '--r', 'LineWidth', 1.5, 'Label', sprintf('-%.1f^\\circ (TIR Limit)', theta_critical_deg), 'LabelOrientation', 'horizontal', 'LabelHorizontalAlignment', 'center', 'LabelVerticalAlignment', 'bottom');

% Also mark the boundaries between Radiating and Evanescent zones (|ux| = 1)
xline(1.0, '-k', 'LineWidth', 1.0, 'Label', 'Radiation Limit (+90^\circ)', 'LabelOrientation', 'horizontal', 'LabelHorizontalAlignment', 'center', 'LabelVerticalAlignment', 'top');
xline(-1.0, '-k', 'LineWidth', 1.0, 'Label', 'Radiation Limit (-90^\circ)', 'LabelOrientation', 'horizontal', 'LabelHorizontalAlignment', 'center', 'LabelVerticalAlignment', 'top');

% Embellish the plot
if isnan(dist_wls)
    title_str = sprintf('Far-Field Power vs. Direction Cosine (ux = k_x/k_0)\n\\lambda = %.2f nm', lam_m * 1e9);
else
    title_str = sprintf('Far-Field Power vs. Direction Cosine (ux = k_x/k_0)\n\\lambda = %.2f nm, Monitor Distance = %.2f \\lambda', lam_m * 1e9, dist_wls);
end
title(title_str, 'FontSize', 14);
xlabel('Direction Cosine u_x = sin(\theta) = k_x / k_0', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Normalized Radiation Power density P(u_x) [dB]', 'FontSize', 12, 'FontWeight', 'bold');
% Expand x-axis to show evanescent bounds
xlim([min(-1.5, min(ux_all)) max(1.5, max(ux_all))]);
ylim([-60 5]); % Keep nice bounds for typical phase-shift radiation
grid on;
hold off;

fprintf('Done plotting.\n');
