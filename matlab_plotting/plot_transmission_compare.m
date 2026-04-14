% plot_transmission_compare.m
% Compares transmission vs wavelength for multiple simulation results on one plot.
clear all;
close all;
clc;
addpath(fileparts(fileparts(mfilename('fullpath'))));  % Add project root to MATLAB path

% --- USER CONFIGURATION ---
file_list = {
    "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\leaky_modes_v5\results\result_80_periods_CONST_shift_100.0nm.mat", ...
    "C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\leaky_modes_v5\results\result_80_periods_CONST_shift_100.0nm_fixed_cav.mat", ...
};

labels = {
    '100 nm Shift, Lengthened Cavity', ...
    '100 nm Shift, Fixed Cavity', ...
};

FONT_SIZE        = 13;
LEGEND_FONT_SIZE = 11;

%% --- Plot ---
colors = [
    0.00  0.45  0.74;
    0.85  0.33  0.10;
    0.47  0.67  0.19;
    0.49  0.18  0.56;
    0.93  0.69  0.13;
    0.30  0.75  0.93;
];

figure('Name', 'Transmission Comparison', 'Color', 'w', ...
       'Position', [80, 80, 1100, 450]);

for k = 1:numel(file_list)
    data = load(file_list{k});

    % Extract wavelength
    if isfield(data, 'wl_nm')
        wl_nm = double(data.wl_nm);
    elseif isfield(data, 'wl_m')
        wl_nm = double(data.wl_m) * 1e9;
    else
        error('No wavelength data found in %s', file_list{k});
    end

    % Extract transmission
    if isfield(data, 'S21_complex')
        T = abs(double(data.S21_complex)).^2;
    elseif isfield(data, 'S12_complex')
        T = abs(double(data.S12_complex)).^2;
    elseif isfield(data, 'T')
        T = double(data.T);
    else
        error('No transmission data found in %s', file_list{k});
    end

    T = squeeze(T);
    wl_nm = squeeze(wl_nm);

    c = colors(mod(k-1, size(colors,1)) + 1, :);

    subplot(1, 2, k);
    plot(wl_nm, T, 'Color', c, 'LineWidth', 1.5);
    xlabel('Wavelength [nm]', 'FontSize', FONT_SIZE);
    ylabel('Transmission (T)', 'FontSize', FONT_SIZE);
    title(labels{k}, 'FontSize', FONT_SIZE);
    grid on;
end
