%% Headless TM linear-vs-tanh apodization plot at the TM-corrected pitch (518.3 nm)
% TM resonance re-centered on ~1571 nm (pitch 518.3) so the apodization curves are
% comparable to TE at the same wavelength. Values are the extracted scalars
% (resonance_transmission, fwhm_m) from the Athena runs tm_apod_pitch518 (A2/A5/A10,
% jobs 97304/97336) + tm_apod_518_a20 (A20, job 97355); 0-teeth point reused from
% run_tm/result_N80_TM_avg_tm_P518p3_fields_smp.mat. Hardcoded here so no 550 MB
% field .mat download is needed.
%
% Writes:
%   plot_apod_tm518_transmission.png : peak transmission vs # apodized teeth (lin/tanh)
%   plot_apod_tm518_modewidth.png    : spatial mode width [um] vs teeth, BOTH planes
%
% Mode width is reported in two planes (recomputed identically by
% plane_mode_widths from the 2D field monitors of the same Athena runs):
%   wH = in-plane / "horizontal" width  (|E|^2 integrated over Y, top/yx view)
%   wV = out-of-plane / "vertical" width (|E|^2 integrated over Z, side/zx view)
% Hardcoded here so no 550 MB field .mat download is needed to redraw the figure.
%
% Run: matlab -batch "run('matlab_plotting/plot_apod_tm518_lin_vs_tanh_headless.m')"
clear; clc; close all;
FONT_SIZE = 13;
out_dir = fileparts(mfilename('fullpath'));

teeth = [0 2 5 10 20];

% TM @ pitch 518.3 — linear taper
lin_T  = [0.9584 0.97178 0.97952 0.98359 0.98491];
lin_wH = [18.867 20.537  21.946  24.146  28.778];   % in-plane  (yx)
lin_wV = [17.980 19.702  21.542  23.880  28.354];   % out-of-plane (zx)

% TM @ pitch 518.3 — tanh taper (steepness 2.0); 0-teeth shared with linear
tan_T  = [0.9584 0.96670 0.97173 0.97726 0.98178];
tan_wH = [18.867 19.918  20.399  21.202  22.793];   % in-plane  (yx)
tan_wV = [17.980 18.949  19.535  20.643  22.409];   % out-of-plane (zx)

red = [0.850 0.325 0.098];

% Fig 1: peak transmission
f1 = figure('Name', 'TM518 peak T: linear vs tanh', 'Visible', 'off'); hold on;
plot(teeth, lin_T, 's-',  'Color', red, 'LineWidth', 1.7, 'MarkerSize', 8, ...
    'MarkerFaceColor', 'w',  'DisplayName', 'TM linear');
plot(teeth, tan_T, 's--', 'Color', red, 'LineWidth', 1.7, 'MarkerSize', 8, ...
    'MarkerFaceColor', red,  'DisplayName', 'TM tanh');
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Peak transmission (T)', 'FontSize', FONT_SIZE);
title('TM @ pitch 518.3 nm (\lambda\approx1571 nm): transmission vs apodization', 'FontSize', FONT_SIZE);
legend('Location', 'southeast'); grid on; box on; xticks(teeth);
exportgraphics(f1, fullfile(out_dir, 'plot_apod_tm518_transmission.png'), 'Resolution', 200);

% Fig 2: spatial mode width — two panels, in-plane (yx) and out-of-plane (zx).
f2 = figure('Name', 'TM518 mode width: linear vs tanh (both planes)', ...
            'Visible', 'off', 'Position', [100 100 1180 520]);
allw = [lin_wH lin_wV tan_wH tan_wV];
ylo = floor(min(allw(:))) - 1;  yhi = ceil(max(allw(:))) + 1;

subplot(1, 2, 1); hold on;
plot(teeth, lin_wH, 's-',  'Color', red, 'LineWidth', 1.7, 'MarkerSize', 8, 'MarkerFaceColor', 'w',  'DisplayName', 'TM linear');
plot(teeth, tan_wH, 's--', 'Color', red, 'LineWidth', 1.7, 'MarkerSize', 8, 'MarkerFaceColor', red,  'DisplayName', 'TM tanh');
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Mode width, FWHM [\mum]', 'FontSize', FONT_SIZE);
title('In-plane width (yx, \int|E|^2 dy)', 'FontSize', FONT_SIZE);
legend('Location', 'northwest'); grid on; box on; xticks(teeth); ylim([ylo yhi]);

subplot(1, 2, 2); hold on;
plot(teeth, lin_wV, 's-',  'Color', red, 'LineWidth', 1.7, 'MarkerSize', 8, 'MarkerFaceColor', 'w',  'DisplayName', 'TM linear');
plot(teeth, tan_wV, 's--', 'Color', red, 'LineWidth', 1.7, 'MarkerSize', 8, 'MarkerFaceColor', red,  'DisplayName', 'TM tanh');
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Mode width, FWHM [\mum]', 'FontSize', FONT_SIZE);
title('Out-of-plane width (zx, \int|E|^2 dz)', 'FontSize', FONT_SIZE);
legend('Location', 'northwest'); grid on; box on; xticks(teeth); ylim([ylo yhi]);

sgtitle('TM @ pitch 518.3 nm (\lambda\approx1571 nm): mode width vs apodization', ...
        'FontSize', FONT_SIZE + 1, 'FontWeight', 'bold');
exportgraphics(f2, fullfile(out_dir, 'plot_apod_tm518_modewidth.png'), 'Resolution', 200);

fprintf('Saved:\n  %s\n  %s\n', ...
    fullfile(out_dir, 'plot_apod_tm518_transmission.png'), ...
    fullfile(out_dir, 'plot_apod_tm518_modewidth.png'));
