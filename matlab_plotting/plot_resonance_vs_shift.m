%% Plot Resonance Peak Wavelength vs. Innermost Teeth Shift
% Loads hardcoded .mat files, reads the stored resonance_wavelength_nm,
% parses the shift from the filename, and plots resonance wavelength
% and peak transmission vs. shift.

clear; clc;
close all;

%% User settings
PLOT_RESONANCE_WL = false; % Also plot resonance wavelength vs. shift

%% Hardcoded file list
RESULTS_DIR = 'C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\leaky_modes_v2\results';

file_names = {
    'result_80_periods_CONST.mat';
    'result_80_periods_CONST_shift_35nm.mat';
    'result_80_periods_CONST_shift_70nm.mat';
    'result_80_periods_CONST_shift_105nm.mat';
    'result_80_periods_CONST_shift_120nm.mat';
    'result_80_periods_CONST_shift_140nm.mat';
    'result_80_periods_CONST_shift_160nm.mat';
    'result_80_periods_CONST_shift_175nm.mat';
};

nFiles = numel(file_names);

%% Preallocate result arrays
shifts_nm   = zeros(1, nFiles);
lambda_res  = nan(1, nFiles);
T_peak      = nan(1, nFiles);

%% Loop: load each file, parse shift, find resonance peak
for k = 1:nFiles
    fname = file_names{k};
    fp    = fullfile(RESULTS_DIR, fname);

    % --- Parse shift from filename ---
    tok = regexp(fname, '_shift_(\d+)nm', 'tokens');
    if ~isempty(tok)
        shifts_nm(k) = str2double(tok{1}{1});
    else
        shifts_nm(k) = 0;
    end

    % --- Load .mat file ---
    fileData = load(fp);

    % Extract wavelength
    if isfield(fileData, 'wl_nm')
        wl_nm = double(fileData.wl_nm);
    elseif isfield(fileData, 'wl_m')
        wl_nm = double(fileData.wl_m) * 1e9;
    elseif isfield(fileData, 'lambda')
        wl_nm = double(fileData.lambda) * 1e9;
    else
        error('No wavelength variable found in %s', fname);
    end

    % Extract T
    if isfield(fileData, 'S21_complex')
        T = abs(fileData.S21_complex).^2;
    elseif isfield(fileData, 'S12_complex')
        T = abs(fileData.S12_complex).^2;
    elseif isfield(fileData, 'S21')
        T = abs(fileData.S21).^2;
    elseif isfield(fileData, 'T')
        T = double(fileData.T);
    else
        error('No transmission variable found in %s', fname);
    end

    % Read resonance wavelength directly from file
    if isfield(fileData, 'resonance_wavelength_nm')
        lambda_res(k) = double(fileData.resonance_wavelength_nm);
    else
        error('resonance_wavelength_nm not found in %s', fname);
    end

    % Read peak transmission at resonance wavelength
    wl_nm_arr = wl_nm(:);
    T_arr     = T(:);
    [~, idx_res] = min(abs(wl_nm_arr - lambda_res(k)));
    T_peak(k) = T_arr(idx_res);
end

%% Sort by shift value
[shifts_sorted, sort_idx] = sort(shifts_nm);
lambda_res_sorted = lambda_res(sort_idx);
T_peak_sorted     = T_peak(sort_idx);

%% Plot 1: Peak Transmission vs. Shift
figure;
plot(shifts_sorted, T_peak_sorted, 's-', 'LineWidth', 1.5, ...
    'MarkerSize', 7, 'MarkerFaceColor', 'auto');
xlabel('Innermost Teeth Shift [nm]');
ylabel('Peak Transmission (T)');
title('Peak Transmission vs. Innermost Teeth Shift');
grid on;
set(gcf, 'Name', 'Peak T vs Shift');
set(gca, 'FontSize', 14);
set(findall(gcf, 'Type', 'text'), 'FontSize', 14);

%% Plot 2: Resonance Wavelength vs. Shift (optional)
if PLOT_RESONANCE_WL
    figure;
    plot(shifts_sorted, lambda_res_sorted, 'o-', 'LineWidth', 1.5, ...
        'MarkerSize', 7, 'MarkerFaceColor', 'auto');
    xlabel('Innermost Teeth Shift [nm]');
    ylabel('Resonance Wavelength [nm]');
    title('Resonance Wavelength vs. Innermost Teeth Shift');
    grid on;
    set(gcf, 'Name', 'Resonance vs Shift');
    set(gca, 'FontSize', 14);
    set(findall(gcf, 'Type', 'text'), 'FontSize', 14);
end
