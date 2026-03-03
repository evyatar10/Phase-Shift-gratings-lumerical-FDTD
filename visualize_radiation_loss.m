% MATLAB Script: visualize_radiation_loss.m
% Visualizes radiation loss in a phase-shift Bragg grating
% by extracting 2D slices (Z=0 and Y=0) from 3D FDTD simulations
% and computing the angular spectrum of the radiated field.

clear; close all; clc;

%% --- USER CONFIGURATION ---
% Path to your result file
% Update this to the actual path of your simulation results
result_filepath = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles\results\result_80_periods_CONST.mat";

% Options
log_scale_db = true;    % true: plot 10*log10(|E|^2), false: linear |E|^2
db_cutoff = -60;        % Minimum dB value to display relative to peak

% Analysis Region Slicing
x_slice_um = [-10, 10];      % X range to plot (around the phase shift at X=0)

%% --- 1. Load Data ---
fprintf('Loading Data from: %s\n', result_filepath);
if ~exist(result_filepath, 'file')
    error('File not found! Please update result_filepath in the script.');
end
data = load(result_filepath);

% Find Resonance Peak
if isfield(data, 'T') && isfield(data, 'wl_m')
    T = squeeze(data.T);
    wl = squeeze(data.wl_m);

    stopband_indices = find(T < 0.6);
    if isempty(stopband_indices)
        stopband_indices = find(T < 0.85);
    end

    if isempty(stopband_indices)
        [~, idx_peak] = max(T);
        fprintf('Warning: No stopband detected. Using global maximum.\n');
    else
        idx_start = stopband_indices(1);
        idx_end = stopband_indices(end);

        [~, local_peak_idx] = max(T(idx_start:idx_end));
        idx_peak = idx_start + local_peak_idx - 1;
    end

    wl_res = wl(idx_peak);
    fprintf('Resonance detected at %.3f nm (T = %.3f)\n', wl_res*1e9, T(idx_peak));
else
    error('Transmission or Wavelength data missing.');
end

%% --- 2. Unpack 3D Data ---
[x, y, z, lam_3d, E_5D] = unpack_data_robust(data);

% Find the 3D monitor index closest to the resonance
[~, idx_3d] = min(abs(lam_3d - wl_res));
wl_3d_actual = lam_3d(idx_3d);

fprintf('Extracting Field Profile at %.3f nm\n', wl_3d_actual*1e9);

% Crop X axis to desired region
x_idx = (x*1e6 >= x_slice_um(1)) & (x*1e6 <= x_slice_um(2));
x_sim = x(x_idx);

E_crop = E_5D(x_idx, :, :, idx_3d, :);

% Calculate |E|^2
E_sq = sum(abs(E_crop).^2, 5); % Sum over Ex, Ey, Ez components
E_sq = reshape(E_sq, [size(E_crop, 1), size(E_crop, 2), size(E_crop, 3)]); % Enforce [Nx, Ny, Nz] shape explicitly

peak_E2 = max(E_sq(:));

if log_scale_db
    E_plot = 10 * log10(E_sq / peak_E2);
    E_plot(E_plot < db_cutoff) = db_cutoff; % Threshold
    c_label = '10*log_{10}(|E|^2/|E|^2_{max}) [dB]';
else
    E_plot = E_sq;
    c_label = '|E|^2 [V^2/m^2]';
end

%% --- 3. Top View: XY Plane (Z = 0) ---
% Automatically use the edge (maximum |y/z|) of the exported monitor data
% We step back slightly (200 nm) from the absolute edge to avoid PML boundary artifacts
[y_max_m, ~] = max(abs(y));
boundary_dist_y_m = y_max_m - 0.2e-6;
[~, idx_y_top] = min(abs(y - boundary_dist_y_m));
boundary_dist_y_um = boundary_dist_y_m * 1e6;

[~, idx_z0] = min(abs(z));
fprintf('Extracting XY slice at Z = %.3f um\n', z(idx_z0)*1e6);

E_xy = squeeze(E_plot(:, :, idx_z0));

figure('Name', 'Radiation Loss - Top View (XY)', 'Color', 'w', 'Position', [100 100 800 400]);
imagesc(x_sim*1e6, y*1e6, E_xy');
set(gca, 'YDir', 'normal');
colormap(jet);
cb = colorbar;
ylabel(cb, c_label);
xlabel('Position X [\mum]');
ylabel('Position Y [\mum]');
title({'Top View: Radiation Loss (XY Plane, Z \approx 0)', sprintf('\\lambda = %.3f nm', wl_3d_actual*1e9)});
% Add line marking the boundary for angular analysis
hold on;
plot([min(x_sim) max(x_sim)]*1e6, [boundary_dist_y_um boundary_dist_y_um], 'k--', 'LineWidth', 1.5);
plot([min(x_sim) max(x_sim)]*1e6, [-boundary_dist_y_um -boundary_dist_y_um], 'k--', 'LineWidth', 1.5);
hold off;


%% --- 4. Side View: XZ Plane (Y = 0) ---
is_3d_data = length(z) > 1;

if is_3d_data
    % Automatically use the edge (maximum |z|) of the exported monitor data
    [boundary_dist_z_m, ~] = max(abs(z));
    boundary_dist_z_um = boundary_dist_z_m * 1e6;

    [~, idx_y0] = min(abs(y));
    fprintf('Extracting XZ slice at Y = %.3f um\n', y(idx_y0)*1e6);

    E_xz = squeeze(E_plot(:, idx_y0, :));

    figure('Name', 'Radiation Loss - Side View (XZ)', 'Color', 'w', 'Position', [150 150 800 400]);
    imagesc(x_sim*1e6, z*1e6, E_xz');
    set(gca, 'YDir', 'normal');
    colormap(jet);
    cb = colorbar;
    ylabel(cb, c_label);
    xlabel('Position X [\mum]');
    ylabel('Position Z [\mum]');
    title({'Side View: Radiation Loss (XZ Plane, Y \approx 0)', sprintf('\\lambda = %.3f nm', wl_3d_actual*1e9)});
    % Add line marking the boundary for angular analysis
    hold on;
    plot([min(x_sim) max(x_sim)]*1e6, [boundary_dist_z_um boundary_dist_z_um], 'k--', 'LineWidth', 1.5);
    plot([min(x_sim) max(x_sim)]*1e6, [-boundary_dist_z_um -boundary_dist_z_um], 'k--', 'LineWidth', 1.5);
    hold off;
else
    fprintf('Data is 2D Z-normal. Skipping Side View (XZ Plane) plot.\n');
end


% 5. Angular Spectrum Analysis
% Physics: To see the angles of radiation, we take the Fourier transform of the
% field strictly OUTSIDE the core, in the cladding region.

% Using the automatically calculated maximum |y|
fprintf('Evaluating Angular Spectrum at Y boundary: %.3f \\mu m\n', boundary_dist_y_um);

[~, idx_z0] = min(abs(z));

% Extract complex field components at the boundary (Z=0, Y=boundary_dist)
E_boundary_comp = squeeze(E_crop(:, idx_y_top, idx_z0, :)); % [x, components]
E_boundary_x = E_boundary_comp(:, 1);
E_boundary_y = E_boundary_comp(:, 2);
E_boundary_z = E_boundary_comp(:, 3);

% Calculate Spatial FFT
dx = mean(diff(x_sim));      % Spatial step in m
Nx = length(x_sim);

% ZERO-PADDING for high angular resolution
% Padding the spatial domain interpolates the k-space domain, heavily increasing resolution
zoom_factor = 8;
N_pad = Nx * zoom_factor;

% Define high-resolution kx vector
kx = (-N_pad/2 : N_pad/2 - 1) * (2*pi / (N_pad * dx)); % rad/m

% Apply a spatial Hann window to eliminate boundary truncation artifacts
% (This stops the massive artificial "horns" at +/- 80 degrees)
spatial_window = hann(Nx);
E_windowed_x = E_boundary_x .* spatial_window;
E_windowed_y = E_boundary_y .* spatial_window;
E_windowed_z = E_boundary_z .* spatial_window;

% Perform Zero-Padded FFT (shift 0 to center)
fft_Ex = fftshift(fft(E_windowed_x, N_pad));
fft_Ey = fftshift(fft(E_windowed_y, N_pad));
fft_Ez = fftshift(fft(E_windowed_z, N_pad));

% Spectral power (Magnitude Squared of the Field Vector)
P_kx = abs(fft_Ex).^2 + abs(fft_Ey).^2 + abs(fft_Ez).^2;

% Convert kx to radiation angle theta
% k_clad = 2*pi / wl_3d_actual * n_clad
n_clad = 1.44; % Assuming SiO2 (Approximate)
k_clad = (2*pi / wl_3d_actual) * n_clad;

% Valid radiation angles exist only for |kx| < k_clad (This IS the critical angle boundary!)
% Light with |kx| > k_clad is completely bound to the core due to Total Internal Reflection
valid_kx_idx = abs(kx) <= k_clad;

kx_rad = kx(valid_kx_idx);
P_rad = P_kx(valid_kx_idx);

if ~isempty(kx_rad)
    % theta = asin(kx / k_clad); angle from broadside (Z-axis)
    theta_rad = asin(kx_rad / k_clad);
    theta_deg = rad2deg(theta_rad);

    % Apply Jacobian physical scaling factor cos(theta)
    % Because P(\theta) d\theta = P(k_x) dk_x, and dk_x = k_clad * cos(\theta) d\theta.
    % This correctly drops power to zero at grazing angles (+/- 90 deg).
    P_rad_theta = P_rad .* cos(theta_rad);
    P_norm = P_rad_theta / max(P_rad_theta);

    % Analytical Calculation (Lorentzian Model)
    compare_analytical = true;
    dn = 0.013;              % Example index contrast
    pitch = 500e-9;          % Grating pitch
    n_eff_approx = wl_res / (2 * pitch); % Approx mode index from Bragg condition
    fprintf('Calculated effective mode index (n_eff) from Bragg condition: %.4f\n', n_eff_approx);

    % Fundamental physics parameters
    kappa = 2 * dn / wl_3d_actual; % Coupling coeff approximation (m^-1)
    k_scatter = (2*pi*n_eff_approx/wl_3d_actual) - (2*pi/pitch); % Primary scattered momentum

    % Phase shift cavity causes a Lorentzian spread in k-space
    % We sum both the forward-scattered and backward-scattered components
    % to make the theoretical curve symmetric (mirror-like) around 0 degrees.
    P_analytical_kx = (kappa^2) ./ ((kx_rad - k_scatter).^2 + kappa^2) + ...
        (kappa^2) ./ ((kx_rad + k_scatter).^2 + kappa^2);

    % Scale analytical by cos(theta) to match the theta domain
    P_analytical_theta = P_analytical_kx .* cos(theta_rad);
    P_analytical_norm = P_analytical_theta / max(P_analytical_theta);

    figure('Name', 'Angular Radiation Spectrum', 'Color', 'w', 'Position', [200 200 700 450]);
    plot(theta_deg, P_norm, 'b-', 'LineWidth', 2, 'DisplayName', 'FDTD Numerical FFT');

    % Calculate True TIR Critical Angle
    % Light inside the core with an angle larger than this cannot escape
    % into the cladding due to Total Internal Reflection (TIR).
    theta_TIR_rad = asin(n_clad / n_eff_approx);
    theta_TIR_deg = rad2deg(theta_TIR_rad);

    if compare_analytical
        hold on;
        plot(theta_deg, P_analytical_norm, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Theory (\\Delta n=%.3f)', dn));
        % Plot TIR Critical Angle boundary markers
        xline(theta_TIR_deg, 'k:', 'LineWidth', 1.5, 'HandleVisibility','off');
        xline(-theta_TIR_deg, 'k:', 'LineWidth', 1.5, 'DisplayName', 'TIR Critical Angle');
        hold off;
    end

    xlabel('Radiation Angle \theta (degrees from normal)');
    ylabel('Normalized Radiated Power');
    title({'Angular Spectrum of Radiation Loss', ...
        sprintf('Evaluated at Y = %.2f \\mum, Z = 0', boundary_dist_y_um)});
    legend('Location', 'best');
    grid on;
    xlim([-90, 90]);
else
    fprintf('Warning: Field grid resolution might not resolve cladding k-vectors well.\n');
end

fprintf('\n--- Analysis Complete ---\n');
fprintf('Calculated TIR Critical Angles (Core to Cladding): +/- %.2f degrees\n', theta_TIR_deg);


%% --- HELPER FUNCTIONS ---
function [x, y, z, lam, E_5D] = unpack_data_robust(data)
if isfield(data, 'field_3d')
    f3d = data.field_3d;
elseif isfield(data, 'x') && isfield(data, 'E_res') % Direct unpacking
    f3d = data;
else
    error('Could not find 3D field data in .mat file.');
end

x = double(f3d.x); y = double(f3d.y); z = double(f3d.z);

if isfield(f3d, 'lambda_3d')
    lam = double(f3d.lambda_3d);
elseif isfield(f3d, 'lambda')
    lam = double(f3d.lambda);
else
    lam = 1.55e-6;
end
lam = lam(:);

E_raw = f3d.E_res;

Nx = length(x);
Ny = length(y);
Nz = length(z);
Nlam = length(lam);

% Robustly reshape the extracted data strictly to the known physical axes
% This flawlessly handles situations where SciPy/Numpy squeezed out singleton dimensions (e.g Nz=1)
E_5D = reshape(E_raw, [Nx, Ny, Nz, Nlam, 3]);

end
