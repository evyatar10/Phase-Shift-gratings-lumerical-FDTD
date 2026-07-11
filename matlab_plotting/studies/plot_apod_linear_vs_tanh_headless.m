%% Headless combined linear-vs-tanh apodization plot (TE & TM)
% Overlays both apodization profiles for both polarizations on shared axes, and
% writes two PNGs:
%   plot_apod_lin_vs_tanh_transmission.png : peak transmission vs # apodized teeth
%   plot_apod_lin_vs_tanh_modewidth.png    : spatial mode width fwhm_m [um] vs teeth
%
% Four series per figure:
%   TE linear, TM linear, TE tanh, TM tanh   (+ shared 0-teeth baseline point).
% Linear study  : results_from_athena/tm_te_apod/results/result_N80_A*_M4*[_TM]_avg.mat
% Tanh study    : results_from_athena/tm_te_apod_tanh/results/result_N80_A*_th_M4*[_TM]_avg.mat
% 0-teeth base  : results_from_athena/run_tm_vs_te/results/result_N80_avg_te|_TM_avg_tm.mat
%
% Run:  matlab -batch "run('matlab_plotting/plot_apod_linear_vs_tanh_headless.m')"
addpath(fileparts(fileparts(mfilename('fullpath'))));  % project root on path
addpath(fileparts(mfilename('fullpath')));             % matlab_plotting/ (plane_mode_widths)

clear; clc; close all;
FONT_SIZE = 13;

root     = fileparts(fileparts(mfilename('fullpath')));
lin_dir  = fullfile(root, 'results_from_athena', 'tm_te_apod', 'results');
tanh_dir = fullfile(root, 'results_from_athena', 'tm_te_apod_tanh', 'results');
base_dir = fullfile(root, 'results_from_athena', 'run_tm_vs_te', 'results');
out_dir  = fileparts(mfilename('fullpath'));

n_teeth = [2 5 10 20];

% Collect each series: [0-baseline, then n_teeth points]. Mode width is reported
% in both planes: in-plane (yx, suffix wH) and out-of-plane (zx, suffix wV).
[lin_te_x, lin_te_T, lin_te_wH, lin_te_wV] = series(base_dir, lin_dir,  n_teeth, false, false);
[lin_tm_x, lin_tm_T, lin_tm_wH, lin_tm_wV] = series(base_dir, lin_dir,  n_teeth, true,  false);
[tan_te_x, tan_te_T, tan_te_wH, tan_te_wV] = series(base_dir, tanh_dir, n_teeth, false, true);
[tan_tm_x, tan_tm_T, tan_tm_wH, tan_tm_wV] = series(base_dir, tanh_dir, n_teeth, true,  true);

all_x = [0 n_teeth];

% Style: color = polarization (TE blue, TM red); linestyle = profile
% (linear solid, tanh dashed); marker = polarization (o TE, s TM).
blue = [0 0.447 0.741];  red = [0.850 0.325 0.098];

% Fig 1: peak transmission
f1 = figure('Name', 'Peak T: linear vs tanh', 'Visible', 'off'); hold on;
plot(lin_te_x, lin_te_T, 'o-',  'Color', blue, 'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor','w', 'DisplayName','TE linear');
plot(lin_tm_x, lin_tm_T, 's-',  'Color', red,  'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor','w', 'DisplayName','TM linear');
plot(tan_te_x, tan_te_T, 'o--', 'Color', blue, 'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor', blue, 'DisplayName','TE tanh');
plot(tan_tm_x, tan_tm_T, 's--', 'Color', red,  'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor', red,  'DisplayName','TM tanh');
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Peak transmission (T)', 'FontSize', FONT_SIZE);
title('Resonance transmission vs. apodization: linear vs tanh', 'FontSize', FONT_SIZE);
legend('Location', 'southeast'); grid on; box on; xticks(all_x);
exportgraphics(f1, fullfile(out_dir, 'plot_apod_lin_vs_tanh_transmission.png'), 'Resolution', 200);

% Fig 2: spatial mode width — two panels, in-plane (yx) and out-of-plane (zx).
f2 = figure('Name', 'Mode width: linear vs tanh (both planes)', ...
            'Visible', 'off', 'Position', [100 100 1180 520]);
allw = [lin_te_wH lin_te_wV lin_tm_wH lin_tm_wV tan_te_wH tan_te_wV tan_tm_wH tan_tm_wV];
ylo = floor(min(allw(:), [], 'omitnan')) - 1;
yhi = ceil( max(allw(:), [], 'omitnan')) + 1;

subplot(1, 2, 1); hold on;
plot(lin_te_x, lin_te_wH, 'o-',  'Color', blue, 'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor','w', 'DisplayName','TE linear');
plot(lin_tm_x, lin_tm_wH, 's-',  'Color', red,  'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor','w', 'DisplayName','TM linear');
plot(tan_te_x, tan_te_wH, 'o--', 'Color', blue, 'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor', blue, 'DisplayName','TE tanh');
plot(tan_tm_x, tan_tm_wH, 's--', 'Color', red,  'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor', red,  'DisplayName','TM tanh');
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Mode width, FWHM [\mum]', 'FontSize', FONT_SIZE);
title('In-plane width (yx, \int|E|^2 dy)', 'FontSize', FONT_SIZE);
legend('Location', 'northwest'); grid on; box on; xticks(all_x); ylim([ylo yhi]);

subplot(1, 2, 2); hold on;
plot(lin_te_x, lin_te_wV, 'o-',  'Color', blue, 'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor','w', 'DisplayName','TE linear');
plot(lin_tm_x, lin_tm_wV, 's-',  'Color', red,  'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor','w', 'DisplayName','TM linear');
plot(tan_te_x, tan_te_wV, 'o--', 'Color', blue, 'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor', blue, 'DisplayName','TE tanh');
plot(tan_tm_x, tan_tm_wV, 's--', 'Color', red,  'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor', red,  'DisplayName','TM tanh');
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Mode width, FWHM [\mum]', 'FontSize', FONT_SIZE);
title('Out-of-plane width (zx, \int|E|^2 dz)', 'FontSize', FONT_SIZE);
legend('Location', 'northwest'); grid on; box on; xticks(all_x); ylim([ylo yhi]);

sgtitle('Spatial mode width vs. apodization: linear vs tanh', 'FontSize', FONT_SIZE + 1, 'FontWeight', 'bold');
exportgraphics(f2, fullfile(out_dir, 'plot_apod_lin_vs_tanh_modewidth.png'), 'Resolution', 200);

fprintf('\nSaved:\n  %s\n  %s\n', ...
    fullfile(out_dir, 'plot_apod_lin_vs_tanh_transmission.png'), ...
    fullfile(out_dir, 'plot_apod_lin_vs_tanh_modewidth.png'));

%% ── local helpers ───────────────────────────────────────────────────────────
function [x, T, wH, wV] = series(base_dir, study_dir, n_teeth, is_tm, is_tanh)
    % Shared 0-teeth baseline point + the apodized points for one (pol, profile).
    if is_tm
        base = fullfile(base_dir, 'result_N80_TM_avg_tm.mat');
    else
        base = fullfile(base_dir, 'result_N80_avg_te.mat');
    end
    files = cell(1, numel(n_teeth));
    for i = 1:numel(n_teeth)
        th = ''; if is_tanh, th = '_th'; end
        pol = ''; if is_tm, pol = '_TM'; end
        files{i} = fullfile(study_dir, sprintf('result_N80_A%d%s_M4%s_avg.mat', n_teeth(i), th, pol));
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
        % fall back to the stored scalar fwhm_m for the horizontal width when a
        % file has no 2D fields (e.g. the 0-teeth baseline); vertical stays NaN.
        [h, v] = plane_mode_widths(fp);
        wH(k) = h;  wV(k) = v;
        if isnan(wH(k)) && isfield(d, 'fwhm_m'), wH(k) = double(d.fwhm_m) * 1e6; end
    end
end
