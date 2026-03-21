% MATLAB Script: calculate_profile_fwhm_comparison.m
addpath(fileparts(fileparts(mfilename('fullpath'))));  % Add project root to MATLAB path

% --- USER CONFIGURATION ---
% Path to your result file
base_dir = "C:\Users\evyat\Lumerical\long_bragg_grating_interconnect\bragg_fdtd_elements_10_apodization_longer\results\";
filename = fullfile(base_dir, 'result_80_periods_10_apodizations_CONST_3D_crop.mat');

% Options
compare_with_2d_slice = true;  % If true, extracts Z=0 slice and compares

% Run Analysis
analyze_field_profile_fwhm(filename, compare_with_2d_slice);


% ---------------------------------------------------------
% MAIN FUNCTION
% ---------------------------------------------------------
function analyze_field_profile_fwhm(filepath, do_2d_compare)

% 1. Load Data
fprintf('Loading Data from: %s\n', filepath);
if ~exist(filepath, 'file'), error('File not found!'); end
data = load(filepath);

% 2. Identify Resonance
if isfield(data, 'T') && isfield(data, 'wl_m')
    [~, idx_peak] = max(data.T);
    wl_res = data.wl_m(idx_peak);
    fprintf('Spectrum Analysis: Resonance detected at %.3f nm\n', wl_res*1e9);
else
    error('Variable "T" or "wl_m" not found inside the .mat file.');
end

% 3. Unpack 3D Data
[x, y, z, lam_3d, E_5D] = unpack_data_robust(data);

% Find the 3D monitor index closest to the resonance
[~, idx_3d] = min(abs(lam_3d - wl_res));
wl_3d_actual = lam_3d(idx_3d);

fprintf('Extracting Field Profile at 3D Monitor Point #%d (%.3f nm)\n', ...
    idx_3d, wl_3d_actual*1e9);

% --- PROCESS A: 3D INTEGRATED (Volumetric) ---
E_slice_3d = squeeze(E_5D(:, :, :, idx_3d, :));
E_sq_3d = sum(abs(E_slice_3d).^2, 4); % Sum components
I_xy_3d = trapz(z, E_sq_3d, 3);       % Integ Z
I_x_3d  = trapz(y, I_xy_3d, 2);       % Integ Y

[I_env_3d, ~, ~] = calculate_envelope(x, I_x_3d);
[fwhm_3d, xL_3d, xR_3d, yHM_3d] = calculate_fwhm_relative(x, I_env_3d);

% --- PROCESS B: 2D SLICE (Z=0 Center Cut) ---
if do_2d_compare
    % Find index closest to Z=0
    [~, idx_z0] = min(abs(z));
    z_val = z(idx_z0);
    fprintf('Extracting 2D Slice at Z = %.3f um\n', z_val*1e6);

    E_slice_2d = squeeze(E_5D(:, :, idx_z0, idx_3d, :)); % [x, y, components]
    E_sq_2d = sum(abs(E_slice_2d).^2, 3);
    I_x_2d  = trapz(y, E_sq_2d, 2); % Integ Y only

    % Normalize 2D to match 3D peak for comparison plotting
    scale_factor = max(I_env_3d) / max(I_x_2d);
    I_x_2d = I_x_2d * scale_factor;

    [I_env_2d, ~, ~] = calculate_envelope(x, I_x_2d);
    [fwhm_2d, xL_2d, xR_2d, yHM_2d] = calculate_fwhm_relative(x, I_env_2d);
end

% --- REPORT RESULTS ---
fprintf('\n--- RESULTS ---\n');
fprintf('3D Integrated FWHM: %.4f mm (%.2f um)\n', fwhm_3d*1e3, fwhm_3d*1e6);
if do_2d_compare
    fprintf('2D Slice (Z=0) FWHM: %.4f mm (%.2f um)\n', fwhm_2d*1e3, fwhm_2d*1e6);
    fprintf('Difference: %.2f%%\n', abs(fwhm_3d - fwhm_2d)/fwhm_3d * 100);
end

% ---------------------------------------------------------
% PLOTTING
% ---------------------------------------------------------
figure('Name', 'Resonance Field Profile Comparison', 'Color', 'w');
hold on;

% --- Plot 3D Data ---
plot(x * 1e6, I_x_3d, 'Color', [0.7 0.7 1], 'LineWidth', 0.5, 'DisplayName', '3D Raw');
plot(x * 1e6, I_env_3d, 'b-', 'LineWidth', 2, 'DisplayName', '3D Envelope');

if fwhm_3d > 0
    % Marker 3D (Blue Dashed)
    plot([xL_3d, xR_3d]*1e6, [yHM_3d, yHM_3d], '--', 'Color', 'b', ...
        'LineWidth', 1.5, 'DisplayName', sprintf('3D FWHM: %.2f um', fwhm_3d*1e6));
end

% --- Plot 2D Data (Optional) ---
if do_2d_compare
    plot(x * 1e6, I_env_2d, 'g--', 'LineWidth', 2, 'DisplayName', '2D Slice Envelope (Z=0)');

    if fwhm_2d > 0
        % Marker 2D (Green Dotted) - Shifted slightly down to visualize
        plot([xL_2d, xR_2d]*1e6, [yHM_2d, yHM_2d]*0.95, ':', 'Color', [0 0.6 0], ...
            'LineWidth', 1.5, 'DisplayName', sprintf('2D FWHM: %.2f um', fwhm_2d*1e6));
    end
end

xlabel('Position X [\mum]');
ylabel('Integrated Energy Density (a.u.)');
title({'Field Profile Comparison', ...
    sprintf('\\lambda_{res} = %.3f nm', wl_3d_actual*1e9)});
legend('Location', 'best');
grid on;
xlim([min(x)*1e6, max(x)*1e6]);
hold off;
end

% ---------------------------------------------------------
% HELPER: ENVELOPE
% ---------------------------------------------------------
function [env, x_pks, y_pks] = calculate_envelope(x, y)
% Physically accurate Envelope calculation using the Hilbert Transform!

% The analytic signal is defined as A(t) = y(t) + j*H(y(t))
analytic_signal = hilbert(double(y));

% The envelope is simply the magnitude of the analytic signal
env = abs(analytic_signal);

% Smooth it slightly to remove any remaining high-frequency carrier jitter
env = smoothdata(env, 'gaussian', 50);

% Findpeaks returned purely for backwards compatibility with plot variables
[y_pks, x_pks] = findpeaks(env, double(x));
end

% ---------------------------------------------------------
% HELPER: FWHM
% ---------------------------------------------------------
function [width, x1, x2, y_target] = calculate_fwhm_relative(x, y)
y_max = max(y); y_min = min(y);
y_target = y_min + 0.5 * (y_max - y_min);

above = y > y_target;
crossings = diff(above);
indices = find(crossings ~= 0);

if length(indices) < 2, width=0; x1=0; x2=0; return; end

x_cross = zeros(length(indices), 1);
for k = 1:length(indices)
    idx = indices(k);
    y1 = y(idx); y2 = y(idx+1);
    x1_pt = x(idx); x2_pt = x(idx+1);
    slope = (y2 - y1) / (x2_pt - x1_pt);
    x_cross(k) = x1_pt + (y_target - y1) / slope;
end

x1 = x_cross(1); x2 = x_cross(end);
width = x2 - x1;
end

% ---------------------------------------------------------
% HELPER: UNPACK
% ---------------------------------------------------------
function [x, y, z, lam, E_5D] = unpack_data_robust(data)
if ~isfield(data, 'field_3d'), error('Missing field_3d'); end
f3d = data.field_3d;
x = double(f3d.x); y = double(f3d.y); z = double(f3d.z);

if isfield(f3d, 'lambda_3d'), lam = double(f3d.lambda_3d);
elseif isfield(f3d, 'lambda'), lam = double(f3d.lambda);
else, lam = 1.55e-6; end
lam = lam(:);

E_raw = f3d.E_res;
dims = size(E_raw);
if ndims(E_raw) == 4
    if dims(4) == 3 && length(lam) == 1
        E_5D = reshape(E_raw, [dims(1), dims(2), dims(3), 1, dims(4)]);
    else, E_5D = E_raw; end
else, E_5D = E_raw; end
end