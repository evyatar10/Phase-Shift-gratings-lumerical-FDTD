% MATLAB Script: calculate_overlap_physics_based.m

% --- USER CONFIGURATION ---
% Physical Parameters
N_short = 60;       % Periods (each side) for the short device
N_long  = 100;      % Periods (each side) for the long device
pitch   = 500e-9;   % Grating pitch
cav_len = pitch/2;  % Cavity length

% Paths
filename_short = 'C:\Users\evyat\Lumerical\long_bragg_grating_interconnect\bragg_fdtd_elements_v7_3d_profiles\results\result_60_periods_CONST_3D_crop.mat';
filename_long  = 'C:\Users\evyat\Lumerical\long_bragg_grating_interconnect\bragg_fdtd_elements_v7_3d_profiles\results\result_100_periods_CONST_3D_crop.mat';

% Run Analysis
analyze_overlap_physics(filename_long, filename_short, N_short, pitch, cav_len);


% ---------------------------------------------------------
% MAIN FUNCTION
% ---------------------------------------------------------
function analyze_overlap_physics(file_long, file_short, N_short, pitch, cav_len)

    % 1. Load Data
    fprintf('Loading Long Device...\n');
    data_L = load(file_long);
    fprintf('Loading Short Device...\n');
    data_S = load(file_short);

    % 2. Identify Resonance from Transmission Spectrum (T)
    % We use the Short device spectrum as the reference for the "60 period" mode
    if isfield(data_S, 'T') && isfield(data_S, 'wl_m')
        T_spec = data_S.T;
        wl_spec = data_S.wl_m;
        
        % Find Peak Transmission (Resonance)
        [~, idx_peak_global] = max(T_spec);
        wl_resonance_global = wl_spec(idx_peak_global);
        
        fprintf('Spectrum Analysis: True Resonance detected at %.3f nm\n', wl_resonance_global*1e9);
    else
        error('Variable "T" or "wl_m" not found in .mat file. Cannot find resonance.');
    end

    % 3. Unpack 3D Data
    [xL, yL, zL, lamL_3d, EL_All] = unpack_data_robust(data_L, 'Long');
    [xS, yS, zS, lamS_3d, ES_All] = unpack_data_robust(data_S, 'Short');

    num_wls = length(lamS_3d);
    fprintf('3D Monitor contains %d wavelength points.\n', num_wls);

    % 4. Map "True Resonance" to "3D Monitor Points"
    % Find the index in the 3D data closest to the global resonance
    [~, idx_res_3d] = min(abs(lamS_3d - wl_resonance_global));
    
    % Find "Off-Resonance" (Point in 3D data with lowest T)
    % We map T values to the 3D wavelengths
    T_at_3d_points = interp1(wl_spec, T_spec, lamS_3d, 'nearest');
    [~, idx_off_3d] = min(T_at_3d_points);

    fprintf('  -> Selected 3D Slice #%d for Resonance (%.3f nm)\n', idx_res_3d, lamS_3d(idx_res_3d)*1e9);
    fprintf('  -> Selected 3D Slice #%d for Off-Resonance (%.3f nm)\n', idx_off_3d, lamS_3d(idx_off_3d)*1e9);

    % 5. Define Physical Edges
    x_edge_pos = (N_short * pitch) + (cav_len / 2);
    x_target_L = -x_edge_pos;
    x_target_R =  x_edge_pos;

    % 6. Initialize Storage
    factors_left  = zeros(num_wls, 1);
    factors_right = zeros(num_wls, 1);
    
    prof_res = struct('x', [], 'raw', [], 'env', [], 'wl', 0, 'val_L', 0, 'val_R', 0);
    prof_off = struct('x', [], 'raw', [], 'env', [], 'wl', 0, 'val_L', 0, 'val_R', 0);

    % --- LOOP OVER WAVELENGTHS ---
    fprintf('Processing Overlaps...\n');
    for i = 1:num_wls
        wl_current = lamS_3d(i);
        
        % Extract Slices
        E_L_slice = squeeze(EL_All(:, :, :, i, :));
        E_S_slice = squeeze(ES_All(:, :, :, i, :));

        % Calculate Spatial Overlap
        [x_common, overlap_profile] = calculate_spatial_overlap(xL, yL, zL, E_L_slice, xS, yS, zS, E_S_slice);

        % Calculate Envelope (Extrapolated)
        [x_env, y_env] = calculate_envelope_extrapolated(x_common, overlap_profile);

        % Extract Values at Strict Edges
        val_L = interp1(x_env, y_env, x_target_L, 'linear', 'extrap');
        val_R = interp1(x_env, y_env, x_target_R, 'linear', 'extrap');
        
        % Clamp
        val_L = min(max(val_L, 0), 1);
        val_R = min(max(val_R, 0), 1);
        
        factors_left(i)  = val_L;
        factors_right(i) = val_R;

        % Store Profile if it matches Resonance Index
        if i == idx_res_3d
            prof_res.x = x_common;
            prof_res.raw = overlap_profile;
            prof_res.env = interp1(x_env, y_env, x_common, 'pchip', 'extrap');
            prof_res.wl = wl_current;
            prof_res.val_L = val_L;
            prof_res.val_R = val_R;
        end
        
        % Store Profile if it matches Off-Resonance Index
        if i == idx_off_3d
            prof_off.x = x_common;
            prof_off.raw = overlap_profile;
            prof_off.env = interp1(x_env, y_env, x_common, 'pchip', 'extrap');
            prof_off.wl = wl_current;
            prof_off.val_L = val_L;
            prof_off.val_R = val_R;
        end
        
        if mod(i, 5) == 0, fprintf('  Completed %d / %d\n', i, num_wls); end
    end

    % ---------------------------------------------------------
    % PLOTTING
    % ---------------------------------------------------------
    
    % Graph 1: Resonance Profile
    plot_spatial_result(prof_res, x_target_L, x_target_R, 'Resonance Profile');
    
    % Graph 2: Off-Resonance Profile
    plot_spatial_result(prof_off, x_target_L, x_target_R, 'Off-Resonance Profile');

    % Graph 3: Coupling Spectrum
    figure('Name', 'Mismatch Coupling Spectrum', 'Color', 'w');
    plot(lamS_3d * 1e9, factors_left, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'Left Junction (-X)');
    hold on;
    plot(lamS_3d * 1e9, factors_right, 'r--x', 'LineWidth', 1.5, 'DisplayName', 'Right Junction (+X)');
    
    % Mark the Resonance Point on the spectrum
    xline(wl_resonance_global*1e9, 'k--', 'DisplayName', 'True Resonance');
    
    xlabel('Wavelength [nm]');
    ylabel('Coupling Factor \eta');
    title({'Junction Coupling Efficiency', ...
           sprintf('Evaluated at Edges +/- %.2f um', x_target_R*1e6)});
    legend('Location', 'best');
    grid on;
    
    % Print Final Coefficients for User
    fprintf('\n--- FINAL COEFFICIENTS AT RESONANCE (%.3f nm) ---\n', wl_resonance_global*1e9);
    fprintf('Left Edge Coeff:  %.4f\n', prof_res.val_L);
    fprintf('Right Edge Coeff: %.4f\n', prof_res.val_R);
    fprintf('Use these values for your Junction Matrix.\n');
end

% ---------------------------------------------------------
% HELPER: SPATIAL OVERLAP
% ---------------------------------------------------------
function [x_common, overlap_vals] = calculate_spatial_overlap(xL, yL, zL, EL, xS, yS, zS, ES)
    F_Ex = griddedInterpolant({xL, yL, zL}, double(EL(:,:,:,1)), 'linear', 'none');
    F_Ey = griddedInterpolant({xL, yL, zL}, double(EL(:,:,:,2)), 'linear', 'none');
    F_Ez = griddedInterpolant({xL, yL, zL}, double(EL(:,:,:,3)), 'linear', 'none');

    x_min = max(min(xL), min(xS));
    x_max = min(max(xL), max(xS));
    valid_mask = (xS >= x_min) & (xS <= x_max);
    x_common = xS(valid_mask);
    idx_map = find(valid_mask); 

    overlap_vals = zeros(length(x_common), 1);
    [YY, ZZ] = ndgrid(yS, zS);

    for k = 1:length(x_common)
        x_val = x_common(k);
        idx_S = idx_map(k);
        
        E_S_slice = squeeze(ES(idx_S, :, :, :));
        XX_query = repmat(x_val, size(YY));
        Ex_L = F_Ex(XX_query, YY, ZZ); Ex_L(isnan(Ex_L)) = 0;
        Ey_L = F_Ey(XX_query, YY, ZZ); Ey_L(isnan(Ey_L)) = 0;
        Ez_L = F_Ez(XX_query, YY, ZZ); Ez_L(isnan(Ez_L)) = 0;
        E_L_slice = cat(3, Ex_L, Ey_L, Ez_L);

        dot_prod = sum(E_L_slice .* conj(E_S_slice), 3);
        integ_overlap = trapz(yS, trapz(zS, dot_prod, 2), 1);
        norm_L = trapz(yS, trapz(zS, sum(abs(E_L_slice).^2, 3), 2), 1);
        norm_S = trapz(yS, trapz(zS, sum(abs(E_S_slice).^2, 3), 2), 1);

        if (norm_L > 0) && (norm_S > 0)
            overlap_vals(k) = (abs(integ_overlap)^2) / (norm_L * norm_S);
        else
            overlap_vals(k) = 0;
        end
    end
end

% ---------------------------------------------------------
% HELPER: ENVELOPE (EXTRAPOLATED & FIXED CONCAT)
% ---------------------------------------------------------
function [x_env, y_env] = calculate_envelope_extrapolated(x, y)
    [pks, locs] = findpeaks(y, x);
    
    if length(pks) < 3
        x_env = x; y_env = y; return; 
    end
    
    % Force column vectors
    x_peaks = locs(:);
    y_peaks = pks(:);
    
    % Extrapolate Left
    y_start = interp1(x_peaks(1:2), y_peaks(1:2), x(1), 'linear', 'extrap');
    % Extrapolate Right
    y_end   = interp1(x_peaks(end-1:end), y_peaks(end-1:end), x(end), 'linear', 'extrap');
    
    % Assemble
    x_env = [x(1); x_peaks; x(end)];
    y_env = [y_start; y_peaks; y_end];
end

% ---------------------------------------------------------
% HELPER: PLOTTER
% ---------------------------------------------------------
function plot_spatial_result(prof, x_L, x_R, title_str)
    if isempty(prof.x), return; end
    
    figure('Name', title_str, 'Color', 'w');
    hold on;
    plot(prof.x * 1e6, prof.raw, 'Color', [0.7 0.7 1], 'LineWidth', 0.5);
    plot(prof.x * 1e6, prof.env, 'r-', 'LineWidth', 2);
    
    xline(x_L * 1e6, 'k--', 'LineWidth', 1.5);
    plot(x_L * 1e6, prof.val_L, 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
    
    xline(x_R * 1e6, 'k--', 'LineWidth', 1.5);
    plot(x_R * 1e6, prof.val_R, 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 7);

    xlabel('Position X [\mum]');
    ylabel('Overlap Factor');
    title({[title_str ' (\lambda = ' num2str(prof.wl*1e9, '%.2f') ' nm)'], ...
           'Blue=Raw, Red=Envelope'});
    grid on;
    ylim([0, 1.05]);
    xlim([min(prof.x)*1e6, max(prof.x)*1e6]);
    hold off;
end

% ---------------------------------------------------------
% HELPER: UNPACK
% ---------------------------------------------------------
function [x, y, z, lam, E_5D] = unpack_data_robust(data, name)
    if ~isfield(data, 'field_3d'), error('Missing field_3d in %s', name); end
    f3d = data.field_3d;
    x = double(f3d.x); y = double(f3d.y); z = double(f3d.z);
    
    if isfield(f3d, 'lambda_3d'), lam = double(f3d.lambda_3d);
    elseif isfield(f3d, 'lambda'), lam = double(f3d.lambda);
    else, lam = 1.55e-6; end
    lam = lam(:);
    
    E_raw = f3d.E_res;
    dims = size(E_raw);
    if ndims(E_raw) == 4
        E_5D = reshape(E_raw, [dims(1), dims(2), dims(3), 1, dims(4)]);
    else, E_5D = E_raw; end
end