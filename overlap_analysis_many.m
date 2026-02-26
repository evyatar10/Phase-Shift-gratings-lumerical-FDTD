% MATLAB Script: batch_overlap_analysis.m

% --- USER CONFIGURATION ---
% Reference "Short" Device
N_short = 80;
% file_short_name = 'result_60_periods_CONST_3D_crop.mat';
% file_short_name = 'result_80_periods_10_apodizations_CONST_3D_crop.mat';

% List of "Long" Devices to Analyze
% Format: {N_periods, 'filename.mat'}
 % long_devices = {
 %    70,  'result_70_periods_10_apodizations_CONST_3D_crop.mat';
 %    80,  'result_80_periods_10_apodizations_CONST_3D_crop.mat';
 %    90,  'result_90_periods_10_apodizations_CONST_3D_crop.mat';
 %    100, 'result_100_periods_10_apodizations_CONST_3D_crop.mat';
 %    110, 'result_110_periods_10_apodizations_CONST_3D_crop.mat';
 %    120, 'result_120_periods_10_apodizations_CONST_3D_crop.mat';
 % };

long_devices = {
     100, 'result_100_periods_10_apodizations_CONST_3D_crop.mat';
     120, 'result_120_periods_10_apodizations_CONST_3D_crop.mat';
};

% Physical Parameters
pitch   = 500e-9;
cav_len = pitch/2;
base_dir = 'C:\Users\evyat\Lumerical\long_bragg_grating_interconnect\bragg_fdtd_elements_10_apodization_longer\results';

% Run Batch
run_batch_analysis(base_dir, file_short_name, N_short, long_devices, pitch, cav_len);


% ---------------------------------------------------------
% BATCH RUNNER
% ---------------------------------------------------------
function run_batch_analysis(base_dir, file_short_name, N_short, long_list, pitch, cav_len)
    
    % Store summary results for final plot
    num_runs = size(long_list, 1);
    results_summary = zeros(num_runs, 4); % [N, Left, Right, Global]
    
    fprintf('=== STARTING BATCH ANALYSIS ===\n');
    fprintf('Reference Short Device: N=%d (%s)\n', N_short, file_short_name);
    
    path_short = fullfile(base_dir, file_short_name);
    if ~isfile(path_short), error('Reference file not found: %s', path_short); end
    
    for i = 1:num_runs
        N_long = long_list{i, 1};
        file_long_name = long_list{i, 2};
        path_long = fullfile(base_dir, file_long_name);
        
        fprintf('\n--------------------------------------------------\n');
        fprintf('Processing Case %d/%d: N=%d vs N=%d\n', i, num_runs, N_short, N_long);
        
        if ~isfile(path_long)
            warning('File not found: %s. Skipping.', file_long_name);
            results_summary(i, :) = [N_long, NaN, NaN, NaN];
            continue;
        end
        
        % Run the Analysis Function (Returns values at resonance)
        [val_L, val_R, val_G] = analyze_single_pair(path_long, path_short, N_short, N_long, pitch, cav_len, base_dir);
        
        % Store
        results_summary(i, :) = [N_long, val_L, val_R, val_G];
    end
    
    % --- FINAL SUMMARY PLOT ---
    plot_batch_summary(results_summary, N_short);
    
    fprintf('\n=== BATCH COMPLETE ===\n');
end

% ---------------------------------------------------------
% SINGLE PAIR ANALYSIS (Returns scalar values at resonance)
% ---------------------------------------------------------
function [res_L, res_R, res_G] = analyze_single_pair(file_long, file_short, N_short, N_long, pitch, cav_len, out_dir)

    % 1. Load Data
    data_L = load(file_long);
    data_S = load(file_short);

    % 2. Identify Resonance
    if isfield(data_S, 'T') && isfield(data_S, 'wl_m')
        [~, idx_peak] = max(data_S.T);
        wl_res = data_S.wl_m(idx_peak);
        fprintf('  Resonance: %.3f nm\n', wl_res*1e9);
    else
        error('Missing T/wl_m in reference file.');
    end

    % 3. Unpack
    [xL, yL, zL, lamL, EL] = unpack_data_robust(data_L, 'Long');
    [xS, yS, zS, lamS, ES] = unpack_data_robust(data_S, 'Short');
    num_wls = length(lamS);

    % 4. Find Index for Resonance
    [~, idx_res] = min(abs(lamS - wl_res));

    % 5. Physical Edges
    x_edge = (N_short * pitch) + (cav_len / 2);
    x_L_target = -x_edge;
    x_R_target =  x_edge;

    % 6. Loop (Process all wavelengths for export)
    factors_L = zeros(num_wls, 1);
    factors_R = zeros(num_wls, 1);
    factors_G = zeros(num_wls, 1);
    
    res_L = 0; res_R = 0; res_G = 0;

    for k = 1:num_wls
        E_L_slice = squeeze(EL(:, :, :, k, :));
        E_S_slice = squeeze(ES(:, :, :, k, :));

        [x_cm, prof] = calculate_spatial_overlap(xL, yL, zL, E_L_slice, xS, yS, zS, E_S_slice);
        [x_env, y_env] = calculate_envelope_points(x_cm, prof);
        env_dense = interp1(x_env, y_env, x_cm, 'pchip', 'extrap');
        
        % A. Edges
        vL = interp1(x_env, y_env, x_L_target, 'linear', 'extrap');
        vR = interp1(x_env, y_env, x_R_target, 'linear', 'extrap');
        
        % B. Global Average
        mask = (x_cm >= x_L_target) & (x_cm <= x_R_target);
        if any(mask), vG = mean(env_dense(mask)); else, vG = 0; end
        
        % Clamp
        vL = min(max(vL,0),1); vR = min(max(vR,0),1); vG = min(max(vG,0),1);
        
        factors_L(k) = vL; factors_R(k) = vR; factors_G(k) = vG;
        
        if k == idx_res
            res_L = vL; res_R = vR; res_G = vG;
        end
    end
    
    % 7. Export Files (Inverted Sqrt)
    freq_vec = 2.99792458e8 ./ lamS;
    
    name_L = sprintf('junction_N%d_N%d_left.txt', N_short, N_long);
    name_R = sprintf('junction_N%d_N%d_right.txt', N_short, N_long);
    name_G = sprintf('junction_N%d_N%d_global_avg.txt', N_short, N_long);
    
    export_s_matrix(fullfile(out_dir, name_L), freq_vec, sqrt(1./factors_L));
    export_s_matrix(fullfile(out_dir, name_R), freq_vec, sqrt(1./factors_R));
    export_s_matrix(fullfile(out_dir, name_G), freq_vec, sqrt(1./factors_G));
    
    fprintf('  -> Exported S-matrices.\n');
end

% ---------------------------------------------------------
% PLOT SUMMARY (Original Line/Marker Sizes)
% ---------------------------------------------------------
function plot_batch_summary(res, N_ref)
    % Remove NaNs (failed runs)
    valid = ~isnan(res(:,2));
    res = res(valid, :);
    
    N_vals = res(:, 1);
    
    figure('Name', 'Batch Overlap Summary', 'Color', 'w');
    hold on;
    
    % Standard Sizes (1.5 LineWidth, Default MarkerSize)
    plot(N_vals, res(:, 2), 'b--s', 'LineWidth', 1.5, 'MarkerFaceColor', 'b', 'DisplayName', 'Left Edge (-X)');
    plot(N_vals, res(:, 3), 'r--s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r', 'DisplayName', 'Right Edge (+X)');
    plot(N_vals, res(:, 4), 'g-o',  'LineWidth', 2.0, 'MarkerFaceColor', 'g', 'DisplayName', 'Global Average');
    
    yline(1.0, 'k:', 'HandleVisibility', 'off', 'LineWidth', 1.0);
    
    % ONLY THESE LABELS HAVE INCREASED FONT SIZE
    xlabel('Number of Periods (Long Device)', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('Overlap Factor \eta (at Resonance)', 'FontSize', 16, 'FontWeight', 'bold');
    title({sprintf('Coupling Efficiency vs Device Length'), sprintf('(Reference Short Device: N=%d)', N_ref)}, ...
        'FontSize', 18);
    
    legend('Location', 'best');
    
    grid on;
    % Tick numbers (Numbers) stay default size
    ylim([0.90, 1.005]); 
    xticks(N_vals);
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
% HELPER: EXPORT S-MATRIX
% ---------------------------------------------------------
function export_s_matrix(filepath, freq, S21_mag)
    fid = fopen(filepath, 'w');
    for k = 1:length(freq)
        fprintf(fid, '%.6e 0 0 %.6f 0 %.6f 0 0 0\n', freq(k), S21_mag(k), S21_mag(k));
    end
    fclose(fid);
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