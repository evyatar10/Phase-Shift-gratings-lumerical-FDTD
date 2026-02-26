% MATLAB Script: calculate_overlap_all_strategies.m
% --- USER CONFIGURATION ---
N_short = 80;       % Periods (each side) for the short device
N_long  = 150;      % Periods (each side) for the long device
pitch   = 500e-9;   % Grating pitch
cav_len = pitch/2;  % Cavity length
% Paths
base_dir = "C:\Users\evyat\Lumerical\long_bragg_grating_interconnect\bragg_fdtd_elements_min_80_apod_5\results";
%'C:\Users\evyat\Lumerical\long_bragg_grating_interconnect\bragg_fdtd_elements_v9_3d_profiles_min_80\results';
filename_short = fullfile(base_dir, 'result_80_periods_5_apodizations_CONST_3D_crop.mat');
filename_long  = fullfile(base_dir, 'result_150_periods_5_apodizations_CONST_3D_crop.mat');
% Run Analysis
analyze_overlap_all(filename_long, filename_short, N_short, N_long, pitch, cav_len, base_dir);
% ---------------------------------------------------------
% MAIN FUNCTION
% ---------------------------------------------------------
function analyze_overlap_all(file_long, file_short, N_short, N_long, pitch, cav_len, out_dir)
    % 1. Load Data
    fprintf('Loading Long Device...\n');
    data_L = load(file_long);
    fprintf('Loading Short Device...\n');
    data_S = load(file_short);
    % 2. Identify Resonance
    if isfield(data_S, 'T') && isfield(data_S, 'wl_m')
        T_spec = data_S.T;
        wl_spec = data_S.wl_m;
        [~, idx_peak_global] = max(T_spec);
        wl_resonance_global = wl_spec(idx_peak_global);
        fprintf('Spectrum Analysis: True Resonance detected at %.3f nm\n', wl_resonance_global*1e9);
    else
        error('Variable "T" or "wl_m" not found.');
    end
    % 3. Unpack 3D Data
    [xL, yL, zL, lamL_3d, EL_All] = unpack_data_robust(data_L, 'Long');
    [xS, yS, zS, lamS_3d, ES_All] = unpack_data_robust(data_S, 'Short');
    num_wls = length(lamS_3d);
    fprintf('3D Monitor contains %d wavelengths.\n', num_wls);
    % 4. Find Indices
    [~, idx_res_3d] = min(abs(lamS_3d - wl_resonance_global));
    T_at_3d = interp1(wl_spec, T_spec, lamS_3d, 'nearest');
    [~, idx_off_3d] = min(T_at_3d);
    % 5. Define STRICT Physical Region
    x_edge_pos = (N_short * pitch) + (cav_len / 2);
    x_target_L = -x_edge_pos;
    x_target_R =  x_edge_pos;
    
    fprintf('Physical Edges: +/- %.3f um\n', x_edge_pos*1e6);
    % 6. Initialize Storage
    factors_left   = zeros(num_wls, 1);
    factors_right  = zeros(num_wls, 1);
    factors_global = zeros(num_wls, 1);
    
    prof_res = struct('x', [], 'raw', [], 'env', [], 'wl', 0, 'val_L', 0, 'val_R', 0, 'val_avg', 0);
    prof_off = struct('x', [], 'raw', [], 'env', [], 'wl', 0, 'val_L', 0, 'val_R', 0, 'val_avg', 0);
    % --- LOOP OVER WAVELENGTHS ---
    fprintf('Processing...\n');
    for i = 1:num_wls
        wl_current = lamS_3d(i);
        
        % Extract Slices
        E_L = squeeze(EL_All(:, :, :, i, :));
        E_S = squeeze(ES_All(:, :, :, i, :));
        % Calculate Spatial Overlap Profile
        [x_common, overlap_profile] = calculate_spatial_overlap(xL, yL, zL, E_L, xS, yS, zS, E_S);
        % Calculate Envelope (DENSE Interpolation for averaging)
        [x_env_pts, y_env_pts] = calculate_envelope_points(x_common, overlap_profile);
        env_dense = interp1(x_env_pts, y_env_pts, x_common, 'pchip', 'extrap');
        % --- A. Left/Right Edge Values ---
        val_L = interp1(x_env_pts, y_env_pts, x_target_L, 'linear', 'extrap');
        val_R = interp1(x_env_pts, y_env_pts, x_target_R, 'linear', 'extrap');
        
        % --- B. Global Mean Calculation ---
        mask_device = (x_common >= x_target_L) & (x_common <= x_target_R);
        if any(mask_device)
            val_avg = mean(env_dense(mask_device));
        else
            val_avg = 0; 
        end
        
        % Clamp (Physics check)
        val_L   = min(max(val_L, 0), 1);
        val_R   = min(max(val_R, 0), 1);
        val_avg = min(max(val_avg, 0), 1);
        
        factors_left(i)   = val_L;
        factors_right(i)  = val_R;
        factors_global(i) = val_avg;
        % Store Profiles
        if i == idx_res_3d
            prof_res.x = x_common; prof_res.raw = overlap_profile; prof_res.env = env_dense;
            prof_res.wl = wl_current; prof_res.val_L = val_L; prof_res.val_R = val_R; prof_res.val_avg = val_avg;
        end
        if i == idx_off_3d
            prof_off.x = x_common; prof_off.raw = overlap_profile; prof_off.env = env_dense;
            prof_off.wl = wl_current; prof_off.val_L = val_L; prof_off.val_R = val_R; prof_off.val_avg = val_avg;
        end
        if mod(i, 5) == 0, fprintf('  %d/%d\n', i, num_wls); end
    end
    % ---------------------------------------------------------
    % 7. EXPORT (TXT S-Matrices + MAT FILE)
    % ---------------------------------------------------------
    fprintf('Exporting Data to %s...\n', out_dir);
    
    freq_vec = 2.99792458e8 ./ lamS_3d;
    
    % S21 = sqrt(1 / eta)
    S21_L = sqrt(1 ./ factors_left);
    S21_R = sqrt(1 ./ factors_right);
    S21_G = sqrt(1 ./ factors_global);
    
    % -- Export Text Files --
    name_L = sprintf('junction_N%d_N%d_left.txt', N_short, N_long);
    name_R = sprintf('junction_N%d_N%d_right.txt', N_short, N_long);
    name_G = sprintf('junction_N%d_N%d_global_avg.txt', N_short, N_long);
    
    export_s_matrix(fullfile(out_dir, name_L), freq_vec, S21_L);
    export_s_matrix(fullfile(out_dir, name_R), freq_vec, S21_R);
    export_s_matrix(fullfile(out_dir, name_G), freq_vec, S21_G);
    
    % -- Export MAT File (LEFT Factors Only) --
    mat_filename = sprintf('correction_factors_left_N%d_N%d.mat', N_short, N_long);
    mat_fullpath = fullfile(out_dir, mat_filename);
    
    % Save wavelengths and factors_left ONLY
    wavelengths_m = lamS_3d;
    save(mat_fullpath, 'wavelengths_m', 'factors_left');
    fprintf('  -> Saved MAT file (Left Only): %s\n', mat_fullpath);

    % ---------------------------------------------------------
    % PLOTTING (INCREASED FONT SIZE)
    % ---------------------------------------------------------
    plot_spatial_comparison(prof_res, x_target_L, x_target_R, 'Resonance Profile');
    plot_spatial_comparison(prof_off, x_target_L, x_target_R, 'Off-Resonance Profile');
    
    figure('Name', 'Mismatch Coupling Spectrum', 'Color', 'w');
    hold on;
    % Plot Left/Right (Original Size: LineWidth 1)
    plot(lamS_3d * 1e9, factors_left, 'b--', 'LineWidth', 1, 'DisplayName', 'Left Edge');
    % plot(lamS_3d * 1e9, factors_right, 'r--', 'LineWidth', 1, 'DisplayName', 'Right Edge');
    
    % Plot Global Average (Original Size: LineWidth 2)
    % plot(lamS_3d * 1e9, factors_global, 'g-o', 'LineWidth', 2, 'MarkerFaceColor', 'g', 'DisplayName', 'Global Average');
    
    xline(wl_resonance_global*1e9, 'k-', 'DisplayName', 'Resonance');
    
    % INCREASED FONTS
    xlabel('Wavelength [nm]', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('Overlap Factor \eta', 'FontSize', 16, 'FontWeight', 'bold');
    title('Comparison: Edge vs Distributed Overlap', 'FontSize', 18);
    legend('Location', 'best', 'FontSize', 14);
    
    grid on;
    set(gca, 'FontSize', 14, 'LineWidth', 1.5); % Increases Tick label size
    
    fprintf('\n--- FINAL COEFFICIENTS AT RESONANCE ---\n');
    fprintf('Left Edge:   %.4f\n', prof_res.val_L);
    fprintf('Right Edge:  %.4f\n', prof_res.val_R);
    fprintf('Global Avg:  %.4f\n', prof_res.val_avg);
end
% ---------------------------------------------------------
% HELPER: EXPORT S-MATRIX
% ---------------------------------------------------------
function export_s_matrix(filepath, freq, S21_mag)
    fid = fopen(filepath, 'w');
    for k = 1:length(freq)
        fprintf(fid, '%.6e 0 0 %.6f 0 %.6f 0 0 0\n', freq(k), S21_mag(k), S21_mag(k));
    end
    fclose(fid);
    fprintf('  -> Saved: %s\n', filepath);
end
% ---------------------------------------------------------
% HELPER: SPATIAL OVERLAP
% ---------------------------------------------------------
function [x_common, overlap_vals] = calculate_spatial_overlap(xL, yL, zL, EL, xS, yS, zS, ES)
    F_Ex = griddedInterpolant({xL, yL, zL}, double(EL(:,:,:,1)), 'linear', 'none');
    F_Ey = griddedInterpolant({xL, yL, zL}, double(EL(:,:,:,2)), 'linear', 'none');
    F_Ez = griddedInterpolant({xL, yL, zL}, double(EL(:,:,:,3)), 'linear', 'none');
    x_min = max(min(xL), min(xS)); x_max = min(max(xL), max(xS));
    valid_mask = (xS >= x_min) & (xS <= x_max);
    x_common = xS(valid_mask); idx_map = find(valid_mask); 
    overlap_vals = zeros(length(x_common), 1);
    [YY, ZZ] = ndgrid(yS, zS);
    for k = 1:length(x_common)
        x_val = x_common(k); idx_S = idx_map(k);
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
% HELPER: ENVELOPE (Points Only)
% ---------------------------------------------------------
function [x_peaks, y_peaks] = calculate_envelope_points(x, y)
    [pks, locs] = findpeaks(y, x);
    if length(pks) < 3, x_peaks = x; y_peaks = y; return; end
    x_locs = locs(:); y_pks = pks(:);
    y_start = interp1(x_locs(1:2), y_pks(1:2), x(1), 'linear', 'extrap');
    y_end   = interp1(x_locs(end-1:end), y_pks(end-1:end), x(end), 'linear', 'extrap');
    x_peaks = [x(1); x_locs; x(end)];
    y_peaks = [y_start; y_pks; y_end];
end
% ---------------------------------------------------------
% HELPER: PLOTTER (INCREASED FONT SIZE)
% ---------------------------------------------------------
function plot_spatial_comparison(prof, x_L, x_R, title_str)
    if isempty(prof.x), return; end
    figure('Name', title_str, 'Color', 'w'); hold on;
    
    % Raw & Envelope (Original Sizes)
    plot(prof.x * 1e6, prof.raw, 'Color', [0.8 0.8 1], 'LineWidth', 0.5);
    plot(prof.x * 1e6, prof.env, 'r-', 'LineWidth', 1.5);
    
    % Edge Markers (Original Sizes)
    xline(x_L * 1e6, 'b--', 'LineWidth', 1.5);
    plot(x_L * 1e6, prof.val_L, 'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
    xline(x_R * 1e6, 'r--', 'LineWidth', 1.5);
    plot(x_R * 1e6, prof.val_R, 'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
    
    % Average Line
    yline(prof.val_avg, 'g--', 'LineWidth', 2, 'DisplayName', 'Global Mean');
    
    % INCREASED FONTS
    xlabel('Position X [\mum]', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('Overlap Factor', 'FontSize', 16, 'FontWeight', 'bold');
    
    title_text = {[title_str ' (\lambda = ' num2str(prof.wl*1e9, '%.2f') ' nm)'], ...
                  'Blue Square=Left, Red Square=Right, Green Line=Avg'};
    title(title_text, 'FontSize', 18);
    
    set(gca, 'FontSize', 14, 'LineWidth', 1.5);
    
    grid on; ylim([0, 1.05]); xlim([min(prof.x)*1e6, max(prof.x)*1e6]); hold off;
end
% ---------------------------------------------------------
% HELPER: UNPACK
% ---------------------------------------------------------
function [x, y, z, lam, E_5D] = unpack_data_robust(data, name)
    if ~isfield(data, 'field_3d'), error('Missing field_3d in %s', name); end
    f3d = data.field_3d;
    x = double(f3d.x); y = double(f3d.y); z = double(f3d.z);
    if isfield(f3d, 'lambda_3d'), lam = double(f3d.lambda_3d); else, lam = 1.55e-6; end
    lam = lam(:);
    E_raw = f3d.E_res;
    if ndims(E_raw)==4, E_5D = reshape(E_raw,[size(E_raw,1),size(E_raw,2),size(E_raw,3),1,size(E_raw,4)]);
    else, E_5D = E_raw; end
end