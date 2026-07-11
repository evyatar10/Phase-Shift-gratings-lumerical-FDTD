%% Headless TM linear-vs-tanh apodization plot at pitch 518.3 nm (~1571 nm)
% TM-only apodization sweep re-run at the TM-corrected pitch (518.3 nm) so the
% resonance sits at ~1571 nm. Overlays the two apodization profiles and writes:
%   plot_tm518_lin_vs_tanh_transmission.png : peak transmission vs # apodized teeth
%   plot_tm518_lin_vs_tanh_modewidth.png    : spatial mode width fwhm_m [um] vs teeth
%
% Two series per figure: TM linear (solid, hollow o) and TM tanh (dashed, filled s).
% Apodized files : results_from_athena/tm_apod_pitch518/results/
%                  result_N80_A{2,5,10,20}_M4_TM_avg.mat        (linear)
%                  result_N80_A{2,5,10,20}_th_M4_TM_avg.mat     (tanh)
% 0-teeth baseline: results_from_athena/run_tm/results/
%                  result_N80_TM_avg_tm_P518p3_fields_smp.mat   (pitch 518.3, no apod)
%
% Run:  matlab -batch "run('matlab_plotting/plot_tm518_linear_vs_tanh_headless.m')"
addpath(fileparts(fileparts(mfilename('fullpath'))));  % project root on path
addpath(fileparts(mfilename('fullpath')));             % matlab_plotting/ (plane_mode_widths)

clear; clc; close all;
FONT_SIZE = 13;

root    = fileparts(fileparts(mfilename('fullpath')));
ap_dir  = fullfile(root, 'results_from_athena', 'tm_apod_pitch518', 'results');
base    = fullfile(root, 'results_from_athena', 'run_tm', 'results', ...
                   'result_N80_TM_avg_tm_P518p3_fields_smp.mat');
out_dir = fileparts(mfilename('fullpath'));

n_teeth = [2 5 10 20];

% Build each series: shared 0-teeth baseline + the apodized points. Mode width
% is reported in both planes: in-plane (yx, wH) and out-of-plane (zx, wV).
[lin_x, lin_T, lin_wH, lin_wV] = series(base, ap_dir, n_teeth, false);   % linear
[tan_x, tan_T, tan_wH, tan_wV] = series(base, ap_dir, n_teeth, true);    % tanh
all_x = [0 n_teeth];

red = [0.850 0.325 0.098];   % TM color (consistent with the TE/TM plots)

% Fig 1: peak transmission
f1 = figure('Name', 'TM518 peak T: linear vs tanh', 'Visible', 'off'); hold on;
plot(lin_x, lin_T, 'o-',  'Color', red, 'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor','w',  'DisplayName','TM linear');
plot(tan_x, tan_T, 's--', 'Color', red, 'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor', red, 'DisplayName','TM tanh');
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Peak transmission (T)', 'FontSize', FONT_SIZE);
title('TM @ pitch 518.3 nm (\lambda\approx1571 nm): transmission vs apodization', 'FontSize', FONT_SIZE);
legend('Location', 'southeast'); grid on; box on; xticks(all_x);
exportgraphics(f1, fullfile(out_dir, 'plot_tm518_lin_vs_tanh_transmission.png'), 'Resolution', 200);

% Fig 2: spatial mode width — two panels, in-plane (yx) and out-of-plane (zx).
f2 = figure('Name', 'TM518 mode width: linear vs tanh (both planes)', ...
            'Visible', 'off', 'Position', [100 100 1180 520]);
allw = [lin_wH lin_wV tan_wH tan_wV];
ylo = floor(min(allw(:), [], 'omitnan')) - 1;
yhi = ceil( max(allw(:), [], 'omitnan')) + 1;

subplot(1, 2, 1); hold on;
plot(lin_x, lin_wH, 'o-',  'Color', red, 'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor','w',  'DisplayName','TM linear');
plot(tan_x, tan_wH, 's--', 'Color', red, 'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor', red, 'DisplayName','TM tanh');
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Mode width, FWHM [\mum]', 'FontSize', FONT_SIZE);
title('In-plane width (yx, \int|E|^2 dy)', 'FontSize', FONT_SIZE);
legend('Location', 'northwest'); grid on; box on; xticks(all_x); ylim([ylo yhi]);

subplot(1, 2, 2); hold on;
plot(lin_x, lin_wV, 'o-',  'Color', red, 'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor','w',  'DisplayName','TM linear');
plot(tan_x, tan_wV, 's--', 'Color', red, 'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor', red, 'DisplayName','TM tanh');
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Mode width, FWHM [\mum]', 'FontSize', FONT_SIZE);
title('Out-of-plane width (zx, \int|E|^2 dz)', 'FontSize', FONT_SIZE);
legend('Location', 'northwest'); grid on; box on; xticks(all_x); ylim([ylo yhi]);

sgtitle('TM @ pitch 518.3 nm (\lambda\approx1571 nm): mode width vs apodization', ...
        'FontSize', FONT_SIZE + 1, 'FontWeight', 'bold');
exportgraphics(f2, fullfile(out_dir, 'plot_tm518_lin_vs_tanh_modewidth.png'), 'Resolution', 200);

% Print the table (wH = in-plane yx, wV = out-of-plane zx).
fprintf('\n teeth |  linear T   wH/wV (um) |   tanh T   wH/wV (um)\n');
for i = 1:numel(all_x)
    fprintf('  %3d  |  %7.4f  %6.2f/%6.2f |  %7.4f  %6.2f/%6.2f\n', ...
        all_x(i), lin_T(i), lin_wH(i), lin_wV(i), tan_T(i), tan_wH(i), tan_wV(i));
end
fprintf('\nSaved:\n  %s\n  %s\n', ...
    fullfile(out_dir, 'plot_tm518_lin_vs_tanh_transmission.png'), ...
    fullfile(out_dir, 'plot_tm518_lin_vs_tanh_modewidth.png'));

%% ── local helpers ───────────────────────────────────────────────────────────
function [x, T, wH, wV] = series(base, ap_dir, n_teeth, is_tanh)
    files = cell(1, numel(n_teeth));
    for i = 1:numel(n_teeth)
        th = ''; if is_tanh, th = '_th'; end
        files{i} = fullfile(ap_dir, sprintf('result_N80_A%d%s_M4_TM_avg.mat', n_teeth(i), th));
    end
    paths = [{base}, files];
    x = [0 n_teeth];
    T = nan(1, numel(paths));  wH = nan(1, numel(paths));  wV = nan(1, numel(paths));
    for k = 1:numel(paths)
        fp = paths{k};
        if ~isfile(fp), warning('Missing file: %s', fp); continue; end
        d = load(fp);
        if isfield(d, 'resonance_transmission')
            T(k) = double(d.resonance_transmission);
        elseif isfield(d,'T') && isfield(d,'wl_nm') && isfield(d,'resonance_wavelength_nm')
            wl = double(d.wl_nm(:)); TT = double(d.T(:));
            [~, ir] = min(abs(wl - double(d.resonance_wavelength_nm)));
            T(k) = TT(ir);
        end
        % In-plane (yx) and out-of-plane (zx) widths from the 2D field monitors;
        % stored scalar fwhm_m is the horizontal fallback if 2D fields are absent.
        [h, v] = plane_mode_widths(fp);
        wH(k) = h;  wV(k) = v;
        if isnan(wH(k)) && isfield(d, 'fwhm_m'), wH(k) = double(d.fwhm_m) * 1e6; end
    end
end
