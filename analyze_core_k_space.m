% analyze_core_k_space.m
% Analyzes the field profile in the core (Z=0, Y=0) of a phase-shift Bragg grating
% 1. Finds resonance inside the bandgap.
% 2. Plots the absolute value squared of the electric field |E|^2 along the propagation axis.
% 3. Converts the field profile to k-space using FFT.
% 4. Highlights radiation region (|kx| < k_clad) vs decaying bound region (|kx| > k_clad).

clear; close all; clc;

%% --- Configuration ---
% Update this to the actual path of your simulation results
result_filepath = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles\results\result_80_periods_CONST.mat";

pitch = 500e-9; % Grating pitch in meters
n_clad = 1.44;  % Approximate cladding refractive index

% --- Defect Isolation Option ---
isolate_defect = false;   % Set to true to ONLY analyze the radiation near x=0
defect_width_um = 10;     % Width of the region to isolate around x=0 (in \mum)

% --- Extraction Position Option ---
% 'core'     : Evaluates K-space at the center of the core (Y = 0)
% 'boundary' : Evaluates K-space at the edge of the simulation domain (far cladding)
extraction_position = 'boundary';

%% --- 1. Load Data & Find Resonance ---
fprintf('Loading data...\n');
if ~exist(result_filepath, 'file')
    error('File not found! Please check result_filepath.');
end
data = load(result_filepath);

if isfield(data, 'T') && isfield(data, 'wl_m')
    T = squeeze(data.T);
    wl = squeeze(data.wl_m);

    % Find resonance inside the bandgap
    stopband_indices = find(T < 0.6);
    if isempty(stopband_indices)
        stopband_indices = find(T < 0.85);
    end

    if isempty(stopband_indices)
        [~, idx_peak] = max(T);
        fprintf('Warning: Stopband not detected. Using global maximum.\n');
    else
        idx_start = stopband_indices(1);
        idx_end = stopband_indices(end);
        [~, local_peak_idx] = max(T(idx_start:idx_end));
        idx_peak = idx_start + local_peak_idx - 1;
    end

    wl_res = wl(idx_peak);
    fprintf('Resonance detected inside bandgap at %.3f nm (T = %.3f)\n', wl_res*1e9, T(idx_peak));
else
    error('Transmission or Wavelength data missing.');
end

%% --- 2. Extract Field at Core Center ---
% Unpack 3D fields
if isfield(data, 'field_3d')
    f3d = data.field_3d;
elseif isfield(data, 'x') && isfield(data, 'E_res')
    f3d = data;
else
    error('Could not find 3D field data.');
end

x = double(f3d.x); y = double(f3d.y); z = double(f3d.z);
if isfield(f3d, 'lambda_3d'); lam_3d = double(f3d.lambda_3d);
elseif isfield(f3d, 'lambda'); lam_3d = double(f3d.lambda);
else; lam_3d = 1.55e-6; end

E_raw = f3d.E_res;
Nx = length(x); Ny = length(y); Nz = length(z); Nlam = length(lam_3d);
E_5D = reshape(E_raw, [Nx, Ny, Nz, Nlam, 3]);

% Nearest wavelength to resonance
[~, idx_lam] = min(abs(lam_3d - wl_res));
wl_3d_actual = lam_3d(idx_lam);

% Core center slice (z=0)
[~, idx_z0] = min(abs(z));

% Determine Y slice index based on option
if strcmp(extraction_position, 'boundary')
    [y_max_m, ~] = max(abs(y));
    boundary_y_m = y_max_m - 0.2e-6; % Step back slightly to avoid PML boundary
    [~, idx_y_extract] = min(abs(y - boundary_y_m));
else
    [~, idx_y_extract] = min(abs(y)); % Default to Y=0 (Core)
end

fprintf('Extracting 1D field profile along propagation axis (X) at Y=%.3fum, Z=%.3fum\n', y(idx_y_extract)*1e6, z(idx_z0)*1e6);

E_core_comp = squeeze(E_5D(:, idx_y_extract, idx_z0, idx_lam, :)); % [Nx, 3] components Ex, Ey, Ez
E_sq_core = sum(abs(E_core_comp).^2, 2);

% --- Plot 1: Real-Space Field Profile ---
figure('Name', sprintf('Real-Space Field (%s)', extraction_position), 'Color', 'w', 'Position', [100 100 700 400]);
plot(x*1e6, E_sq_core, 'b-', 'LineWidth', 1.5);
xlabel('Position along grating (\mum)');
ylabel('|E|^2 [V^2/m^2]');
title(sprintf('Field Profile (|E|^2) at %s (Y=%.2f \\mum), \\lambda = %.3f nm', ...
    extraction_position, y(idx_y_extract)*1e6, wl_3d_actual*1e9));
grid on;

% --- Plot 1b: Top View Field Profile (XY plane at Z=0) ---
E_top_comp = squeeze(E_5D(:, :, idx_z0, idx_lam, :)); % [Nx, Ny, 3] components Ex, Ey, Ez
E_sq_top = sum(abs(E_top_comp).^2, 3); % [Nx, Ny]
E_sq_top_norm = E_sq_top / max(E_sq_top(:));
E_sq_top_dB = 10 * log10(E_sq_top_norm);
E_sq_top_dB(E_sq_top_dB < -60) = -60; % Threshold at -60 dB for visualization

figure('Name', 'Top View Field Profile', 'Color', 'w', 'Position', [150 150 800 400]);
imagesc(x*1e6, y*1e6, E_sq_top_dB');
set(gca, 'YDir', 'normal');
colormap(jet);
cb = colorbar;
ylabel(cb, '10*log_{10}(|E|^2/|E|^2_{max}) [dB]');
xlabel('Position X [\mum]');
ylabel('Position Y [\mum]');
title(sprintf('Top View: Field Profile (XY Plane, Z\\approx 0) at \\lambda = %.3f nm', wl_3d_actual*1e9));


%% --- 3. K-Space Transform (Spatial FFT) ---
% Use a window to reduce finite-domain edge artifacts
if isolate_defect
    fprintf('\nIsolating phase-shift defect at x=0 (Analysis Width: %g \\mum)\n', defect_width_um);
    half_width_m = (defect_width_um / 2) * 1e-6;
    valid_idx = abs(x) <= half_width_m;

    window = zeros(Nx, 1);
    if sum(valid_idx) > 0
        window(valid_idx) = hann(sum(valid_idx));
    else
        warning('Defect width too small. Reverting to full length.');
        window = hann(Nx);
    end
else
    fprintf('\nAnalyzing entire grating length (Full Field).\n');
    window = hann(Nx);
end

E_xw = E_core_comp(:,1) .* window;
E_yw = E_core_comp(:,2) .* window;
E_zw = E_core_comp(:,3) .* window;

dx = mean(diff(x));
zoom_factor = 8;
N_pad = Nx * zoom_factor;
kx = (-N_pad/2 : N_pad/2 - 1)' * (2*pi / (N_pad * dx)); % Wavevector array [rad/m]

fft_Ex = fftshift(fft(E_xw, N_pad));
fft_Ey = fftshift(fft(E_yw, N_pad));
fft_Ez = fftshift(fft(E_zw, N_pad));

% K-space Power Profile
P_kx = abs(fft_Ex).^2 + abs(fft_Ey).^2 + abs(fft_Ez).^2;
P_kx_norm = 10 * log10(P_kx / max(P_kx)); % Normalized dB scale

%% --- 4. Physics Threshold Calculations ---
k0 = 2*pi / wl_3d_actual;
n_eff = wl_3d_actual / (2 * pitch); % Calculated from Bragg condition
k_neff = k0 * n_eff;

% Cladding wavevector determines the boundary between bound vs. radiating fields
% |K_x| < K_clad means radiating (K transverse is real)
% |K_x| > K_clad means decaying/evanescent (K transverse is purely imaginary)
k_clad = k0 * n_clad;

fprintf('\n--- Wavevector Analysis ---\n');
fprintf('Calculated n_eff from Bragg condition: %.4f\n', n_eff);
fprintf('Free space wavevector (k0):  %.2e rad/m\n', k0);
fprintf('Effective bound wavevector (k_neff): %.2e rad/m\n', k_neff);
fprintf('Cladding boundary wavevector (k_clad): %.2e rad/m\n', k_clad);

% --- Plot 2: K-Space Profile ---
figure('Name', 'K-Space Profile', 'Color', 'w', 'Position', [150 150 800 500]);
plot(kx/1e6, P_kx_norm, 'b-', 'LineWidth', 1.5, 'DisplayName', 'FFT Power Spectrum');
hold on;

y_lims = [-60 5];
ylim(y_lims);

% Fill the radiation region for visualization
patch([-k_clad, k_clad, k_clad, -k_clad]/1e6, [y_lims(1) y_lims(1) y_lims(2) y_lims(2)], ...
    'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Radiation Region (|K_x| < k_{clad})');

% Mark Radiation Boundaries
xline(k_clad/1e6, 'r--', 'k_{clad} (Radiation Boundary)', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-k_clad/1e6, 'r--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

% Mark Guided / Bound Peaks
xline(k_neff/1e6, 'k:', 'k_{neff} (Forward Bound Mode)', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-k_neff/1e6, 'k:', 'k_{neff} (Backward Bound Mode)', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');

xlabel('Wavevector K_x (rad / \mum)');
ylabel('Normalized K-Space Power [dB]');
title(sprintf('1D K-Space Field Profile at %s (Y=%.2f \\mum)', extraction_position, y(idx_y_extract)*1e6));
xlim([-k_neff*1.5, k_neff*1.5]/1e6);
legend('Location', 'northeast');
grid on; hold off;

%% --- 5. Radiation Angle Calculation ---
% For components with |K_x| < k_clad (Radiation region):
% Their spatial angle into the cladding is given by \theta = arcsin(K_x / k_clad)
radiating_idx = abs(kx) <= k_clad;
kx_rad = kx(radiating_idx);
P_rad  = P_kx(radiating_idx);

theta_rad = asin(kx_rad / k_clad);
theta_deg = rad2deg(theta_rad);

% Power scaling by Jacobian (cos(theta)) to project dx to dtheta
P_rad_theta = P_rad .* cos(theta_rad);
P_rad_theta_norm = P_rad_theta / max(P_rad_theta);

% --- Plot 3: Extracted Radiation Angles ---
figure('Name', 'Core Field Radiation Spectrum', 'Color', 'w', 'Position', [200 200 700 400]);
plot(theta_deg, P_rad_theta_norm, 'k-', 'LineWidth', 2);
xlabel('Radiation Angle \theta (degrees from normal)');
ylabel('Linear Normalized Power');
title({'Power Radiating vs Angle (Derived directly from core field)', '|\theta| = arcsin(K_x / k_{clad})'});
xlim([-90 90]);
grid on;

fprintf('\nAnalysis Complete. Check figures for real space, k-space, and angle mapping.\n');

%% --- 6. Top View K-Space (2D FFT) ---
fprintf('\n--- Computing 2D K-Space for Top View ---\n');
% Extract the exact top-view field (Z=0 plane)
% Using E_5D which is already extracted in memory [Nx, Ny, Nz, Nlam, 3]
E_top_comp_2d = squeeze(E_5D(:, :, idx_z0, idx_lam, :)); % [Nx, Ny, 3]

% Apply 2D windowing to remove boundary artifacts
% We reuse the X window defined in section 3 (either isolated or full structure)
window_x = window;
window_y = hann(Ny);
[Wm_y, Wm_x] = meshgrid(window_y, window_x); % [Ny, Nx] meshgrid transposed!
% Be careful with meshgrid: meshgrid(y, x) gives size [Nx, Ny]
window_2d = Wm_x .* Wm_y;

E_top_xw = E_top_comp_2d(:,:,1) .* window_2d;
E_top_yw = E_top_comp_2d(:,:,2) .* window_2d;
E_top_zw = E_top_comp_2d(:,:,3) .* window_2d;

% 2D Zero-padding for high resolution in K-space
N_pad_x = Nx * 4;
N_pad_y = Ny * 4;

fft2_Ex = fftshift(fft2(E_top_xw, N_pad_x, N_pad_y));
fft2_Ey = fftshift(fft2(E_top_yw, N_pad_x, N_pad_y));
fft2_Ez = fftshift(fft2(E_top_zw, N_pad_x, N_pad_y));

P_kxy = abs(fft2_Ex).^2 + abs(fft2_Ey).^2 + abs(fft2_Ez).^2;
P_kxy_norm = 10 * log10(P_kxy / max(P_kxy(:)));
P_kxy_norm(P_kxy_norm < -60) = -60;

% Compute 2D wavevectors
dy = mean(diff(y));
kx_2d = (-N_pad_x/2 : N_pad_x/2 - 1) * (2*pi / (N_pad_x * dx));
ky_2d = (-N_pad_y/2 : N_pad_y/2 - 1) * (2*pi / (N_pad_y * dy));

figure('Name', '2D K-Space Top View', 'Color', 'w', 'Position', [250 250 800 600]);
imagesc(kx_2d/1e6, ky_2d/1e6, P_kxy_norm');
set(gca, 'YDir', 'normal');
colormap(jet);
cb = colorbar;
ylabel(cb, 'Normalized 2D K-Space Power [dB]');
xlabel('Wavevector K_x (rad / \mum)');
ylabel('Wavevector K_y (rad / \mum)');
title('2D K-Space Profile (XY Plane, Z\approx0)');

% Overlay physics boundaries
hold on;
% Draw Radiation Circle |K| < K_clad
th = linspace(0, 2*pi, 100);
plot(k_clad*cos(th)/1e6, k_clad*sin(th)/1e6, 'w--', 'LineWidth', 2, 'DisplayName', 'Radiation Boundary (k_{clad})');

% Draw lines for k_neff in x
xline(k_neff/1e6, 'w:', 'k_{neff}', 'LineWidth', 1.5, 'HandleVisibility', 'off', 'LabelVerticalAlignment', 'bottom');
xline(-k_neff/1e6, 'w:', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlim([-k_neff*1.5, k_neff*1.5]/1e6);
ylim([-5, 5]); % As requested by the user, limiting K_y axis manually
legend('Location', 'northeast');
grid on; hold off;
