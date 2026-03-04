% analyze_side_monitor.m
% Analyzes the Far-Field radiation pattern and Near-Field spatial FFT
% extracted directly from the 'side_monitor'.

clear; clc; close all;

%% --- 1. Load Data ---
% Specify the result file to analyze.
% Note: Change this to the latest run result path!
data_file = "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles\results\result_80_periods_CONST.mat";

fprintf('Loading data...\n');
if ~exist(data_file, 'file')
    error('File not found: %s', data_file);
end
data = load(data_file);

if isfield(data, 'side_monitor') && ~isempty(fieldnames(data.side_monitor))
    sm_data = data.side_monitor;
else
    error('The loaded file does not contain side_monitor data!');
end

% Wavelength
lam_m = double(squeeze(sm_data.f_monitor));
fprintf('--- Side Monitor Radiation Analysis ---\n');
fprintf('Wavelength: %.3f nm\n', lam_m * 1e9);

%% --- 2. Calculate Numerical TIR Boundary ---
% Define the refractive index of the cladding
n_clad = 1.44; % SiO2

% Assuming resonance condition: lambda_B = 2 * n_eff_approx * pitch
% You can adjust pitch manually if it's different.
pitch = 500e-9;
n_eff_approx = lam_m / (2 * pitch);

% TIR Critical angle
if n_eff_approx > n_clad
    theta_TIR_rad = asin(n_clad / n_eff_approx);
    theta_TIR_deg = rad2deg(theta_TIR_rad);
else
    theta_TIR_deg = 90; % Everything couples if n_eff < n_clad
end
fprintf('TIR Critical Angle: %.2f degrees\n', theta_TIR_deg);

%% --- 3. Process Lumerical Far-Field (Native Projection) ---
ux = double(squeeze(sm_data.ff_ux));
uy = double(squeeze(sm_data.ff_uy));
ux = ux(:);

Ex_ff = double(squeeze(sm_data.ff_Ex));
Ey_ff = double(squeeze(sm_data.ff_Ey));
Ez_ff = double(squeeze(sm_data.ff_Ez));

% Find Z=0 slice (uy ~ 0)
[~, idx_uy_0] = min(abs(uy));
Ex_1D = squeeze(Ex_ff(:, idx_uy_0));
Ey_1D = squeeze(Ey_ff(:, idx_uy_0));
Ez_1D = squeeze(Ez_ff(:, idx_uy_0));

% Convert 'ux' to physical angle
valid_idx_ff = abs(ux) <= 1.0;
ux_valid = ux(valid_idx_ff);
theta_rad_ff = asin(ux_valid);
theta_deg_ff = rad2deg(theta_rad_ff);

I_ff = abs(Ex_1D(valid_idx_ff)).^2 + abs(Ey_1D(valid_idx_ff)).^2 + abs(Ez_1D(valid_idx_ff)).^2;

% Jacobian scaling: P(theta) = I * cos(theta)
P_theta_ff = I_ff .* cos(theta_rad_ff);

% Normalize and convert to DB
P_ff_norm = P_theta_ff / max(P_theta_ff(:));
P_ff_dB = 10 * log10(P_ff_norm);

% Envelope
[pks, locs] = findpeaks(P_ff_dB);
if ~isempty(locs)
    locs_ext = [1; locs; length(P_ff_dB)];
    pks_ext = [P_ff_dB(1); pks; P_ff_dB(end)];
    P_ff_env_dB = interp1(theta_deg_ff(locs_ext), pks_ext, theta_deg_ff, 'pchip');
else
    P_ff_env_dB = P_ff_dB;
end

% Floor
noise_floor_dB = -30;
P_ff_dB(P_ff_dB < noise_floor_dB) = noise_floor_dB;
P_ff_env_dB(P_ff_env_dB < noise_floor_dB) = noise_floor_dB;

%% --- 4. Process Near-Field (Spatial FFT) ---
nf_x = double(squeeze(sm_data.nf_x));
nf_z = double(squeeze(sm_data.nf_z));
nf_E = double(sm_data.nf_E);

Nx = length(nf_x);
Nz = length(nf_z);

% Reshape directly robustly (nf_E is [Nx, Ny=1, Nz, 3] from python)
nf_E_3D = reshape(nf_E, [Nx, Nz, 3]);

% Interpolate slice at Z=0
[~, idx_z0] = min(abs(nf_z));
E_bound_x = squeeze(nf_E_3D(:, idx_z0, 1));
E_bound_y = squeeze(nf_E_3D(:, idx_z0, 2));
E_bound_z = squeeze(nf_E_3D(:, idx_z0, 3));

dx = mean(diff(nf_x));
zoom_factor = 8;
N_pad = Nx * zoom_factor;
kx = (-N_pad/2 : N_pad/2 - 1) * (2*pi / (N_pad * dx));

% Windowing
spatial_window = hann(Nx);
E_win_x = E_bound_x(:) .* spatial_window;
E_win_y = E_bound_y(:) .* spatial_window;
E_win_z = E_bound_z(:) .* spatial_window;

fft_Ex = fftshift(fft(E_win_x, N_pad));
fft_Ey = fftshift(fft(E_win_y, N_pad));
fft_Ez = fftshift(fft(E_win_z, N_pad));

P_kx = abs(fft_Ex).^2 + abs(fft_Ey).^2 + abs(fft_Ez).^2;

k_clad = (2*pi / lam_m) * n_clad;
valid_kx_idx = abs(kx) <= k_clad;
kx_rad = kx(valid_kx_idx);
P_rad = P_kx(valid_kx_idx);

if ~isempty(kx_rad)
    theta_rad_nf = asin(kx_rad / k_clad);
    theta_deg_nf = rad2deg(theta_rad_nf);

    P_rad_theta_nf = P_rad .* cos(theta_rad_nf);
    P_nf_norm = P_rad_theta_nf / max(P_rad_theta_nf(:));
    P_nf_dB = 10 * log10(P_nf_norm);
    P_nf_dB(P_nf_dB < noise_floor_dB) = noise_floor_dB;
else
    theta_deg_nf = [];
    P_nf_dB = [];
end

%% --- 5. Plot Results ---
figure('Name', 'Side Monitor Radiation Analysis', 'Color', 'w', 'Position', [100, 100, 800, 500]);
hold on;

% 1. Native Far Field Output
plot(theta_deg_ff, P_ff_dB, 'b-', 'LineWidth', 1.0, 'DisplayName', 'Native Lumerical FF (Raw)');
plot(theta_deg_ff, P_ff_env_dB, 'r-', 'LineWidth', 2.0, 'DisplayName', 'Native Lumerical FF (Envelope)');

% 2. Spatial FFT
if ~isempty(theta_deg_nf)
    plot(theta_deg_nf, P_nf_dB, 'g--', 'LineWidth', 2.0, 'DisplayName', 'Near-Field Spatial FFT');
end

% 3. Numerical boundaries (TIR)
xline(theta_TIR_deg, 'k:', 'LineWidth', 1.5, 'DisplayName', 'TIR Critical Angle');
xline(-theta_TIR_deg, 'k:', 'LineWidth', 1.5, 'HandleVisibility','off');

title(sprintf('Radiation Power vs. Angle (Side Monitor)\n\\lambda = %.2f nm, n_{eff} = %.3f, TIR_{bound} = \\pm%.1f^\\circ', ...
    lam_m * 1e9, n_eff_approx, theta_TIR_deg), 'FontSize', 12);
xlabel('Radiation Angle \theta (degrees from Broadside)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Normalized Radiation Power [dB]', 'FontSize', 12, 'FontWeight', 'bold');

xlim([-90 90]);
ylim([noise_floor_dB 5]);
grid on;
legend('Location', 'south', 'NumColumns', 2);
hold off;

fprintf('Done plotting.\n');
