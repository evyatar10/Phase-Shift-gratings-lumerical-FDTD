%% Headless apodization comparison: TE @ pitch 500 vs TM @ pitch 518.3
% Both polarizations now resonate at ~1571 nm (TE at pitch 500, TM at the
% TM-corrected pitch 518.3), so this is the apples-to-apples TE-vs-TM apodization
% comparison at a common wavelength. Four series per figure:
%   TE linear, TM linear (solid, hollow markers) ; TE tanh, TM tanh (dashed, filled).
% Writes:
%   plot_te500_tm518_transmission.png : peak transmission vs # apodized teeth
%   plot_te500_tm518_modewidth.png    : spatial mode width fwhm_m [um] vs teeth
%
% TE (pitch 500): tm_te_apod/results/result_N80_A{n}_M4_avg.mat        (linear)
%                 tm_te_apod_tanh/results/result_N80_A{n}_th_M4_avg.mat (tanh)
%                 run_tm_vs_te/results/result_N80_avg_te.mat            (0-teeth)
% TM (pitch 518): tm_apod_pitch518/results/result_N80_A{n}_M4_TM_avg.mat     (linear)
%                 tm_apod_pitch518/results/result_N80_A{n}_th_M4_TM_avg.mat  (tanh)
%                 run_tm/results/result_N80_TM_avg_tm_P518p3_fields_smp.mat  (0-teeth)
%
% Run:  matlab -batch "run('matlab_plotting/plot_te500_vs_tm518_headless.m')"
addpath(fileparts(fileparts(mfilename('fullpath'))));  % project root on path
addpath(fileparts(mfilename('fullpath')));             % matlab_plotting/ (plane_mode_widths)

clear; clc; close all;
FONT_SIZE = 13;

root = fileparts(fileparts(mfilename('fullpath')));
R    = @(varargin) fullfile(root, 'results_from_athena', varargin{:});
out_dir = fileparts(mfilename('fullpath'));

n_teeth = [2 5 10 20];
all_x   = [0 n_teeth];

% Mode width is reported in BOTH planes: the in-plane / "horizontal" width
% (|E|^2 integrated over Y, top/yx view) and the out-of-plane / "vertical"
% width (|E|^2 integrated over Z, side/zx view). Both are recomputed on an
% identical footing by plane_mode_widths(); see that file. Suffixes: _wH / _wV.

% TE @ pitch 500 ---------------------------------------------------------------
te_base = R('run_tm_vs_te','results','result_N80_avg_te.mat');
[te_lin_x, te_lin_T, te_lin_wH, te_lin_wV] = series(te_base, R('tm_te_apod','results'),      n_teeth, '_M4_avg.mat');
[te_tan_x, te_tan_T, te_tan_wH, te_tan_wV] = series(te_base, R('tm_te_apod_tanh','results'), n_teeth, '_th_M4_avg.mat');

% TM @ pitch 518.3 -------------------------------------------------------------
tm_base = R('run_tm','results','result_N80_TM_avg_tm_P518p3_fields_smp.mat');
[tm_lin_x, tm_lin_T, tm_lin_wH, tm_lin_wV] = series(tm_base, R('tm_apod_pitch518','results'), n_teeth, '_M4_TM_avg.mat');
[tm_tan_x, tm_tan_T, tm_tan_wH, tm_tan_wV] = series(tm_base, R('tm_apod_pitch518','results'), n_teeth, '_th_M4_TM_avg.mat');

blue = [0 0.447 0.741];  red = [0.850 0.325 0.098];

% Fig 1: peak transmission
f1 = figure('Name', 'TE500 vs TM518 peak T', 'Visible', 'off'); hold on;
plot(te_lin_x, te_lin_T, 'o-',  'Color', blue, 'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor','w',  'DisplayName','TE linear (p500)');
plot(tm_lin_x, tm_lin_T, 's-',  'Color', red,  'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor','w',  'DisplayName','TM linear (p518)');
plot(te_tan_x, te_tan_T, 'o--', 'Color', blue, 'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor', blue, 'DisplayName','TE tanh (p500)');
plot(tm_tan_x, tm_tan_T, 's--', 'Color', red,  'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor', red,  'DisplayName','TM tanh (p518)');
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Peak transmission (T)', 'FontSize', FONT_SIZE);
title('Apodization @ \lambda\approx1571 nm: TE (pitch 500) vs TM (pitch 518.3)', 'FontSize', FONT_SIZE);
legend('Location', 'southeast'); grid on; box on; xticks(all_x);
exportgraphics(f1, fullfile(out_dir, 'plot_te500_tm518_transmission.png'), 'Resolution', 200);

% Fig 2: spatial mode width — two panels, in-plane (yx) and out-of-plane (zx).
% Both panels share styling/limits so the two planes are read off identically.
f2 = figure('Name', 'TE500 vs TM518 mode width (both planes)', ...
            'Visible', 'off', 'Position', [100 100 1180 520]);

allw = [te_lin_wH te_lin_wV tm_lin_wH tm_lin_wV ...
        te_tan_wH te_tan_wV tm_tan_wH tm_tan_wV];
ylo = floor(min(allw(:), [], 'omitnan')) - 1;
yhi = ceil( max(allw(:), [], 'omitnan')) + 1;

sp1 = subplot(1, 2, 1); hold on;
plot(te_lin_x, te_lin_wH, 'o-',  'Color', blue, 'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor','w',  'DisplayName','TE linear (p500)');
plot(tm_lin_x, tm_lin_wH, 's-',  'Color', red,  'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor','w',  'DisplayName','TM linear (p518)');
plot(te_tan_x, te_tan_wH, 'o--', 'Color', blue, 'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor', blue, 'DisplayName','TE tanh (p500)');
plot(tm_tan_x, tm_tan_wH, 's--', 'Color', red,  'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor', red,  'DisplayName','TM tanh (p518)');
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Mode width, FWHM [\mum]', 'FontSize', FONT_SIZE);
title('In-plane width (yx, \int|E|^2 dy)', 'FontSize', FONT_SIZE);
legend('Location', 'northwest'); grid on; box on; xticks(all_x); ylim([ylo yhi]);

sp2 = subplot(1, 2, 2); hold on;
plot(te_lin_x, te_lin_wV, 'o-',  'Color', blue, 'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor','w',  'DisplayName','TE linear (p500)');
plot(tm_lin_x, tm_lin_wV, 's-',  'Color', red,  'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor','w',  'DisplayName','TM linear (p518)');
plot(te_tan_x, te_tan_wV, 'o--', 'Color', blue, 'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor', blue, 'DisplayName','TE tanh (p500)');
plot(tm_tan_x, tm_tan_wV, 's--', 'Color', red,  'LineWidth', 1.7, 'MarkerSize', 7, 'MarkerFaceColor', red,  'DisplayName','TM tanh (p518)');
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Mode width, FWHM [\mum]', 'FontSize', FONT_SIZE);
title('Out-of-plane width (zx, \int|E|^2 dz)', 'FontSize', FONT_SIZE);
legend('Location', 'northwest'); grid on; box on; xticks(all_x); ylim([ylo yhi]);

sgtitle('Apodization @ \lambda\approx1571 nm: TE (pitch 500) vs TM (pitch 518.3)', ...
        'FontSize', FONT_SIZE + 1, 'FontWeight', 'bold');
exportgraphics(f2, fullfile(out_dir, 'plot_te500_tm518_modewidth.png'), 'Resolution', 200);

% Table — both planes (H = in-plane yx, V = out-of-plane zx).
fprintf('\n teeth |   TE_lin T  wH/wV |   TM_lin T  wH/wV |  TE_tanh T  wH/wV |  TM_tanh T  wH/wV\n');
for i = 1:numel(all_x)
    fprintf('  %3d  | %6.4f %5.2f/%5.2f | %6.4f %5.2f/%5.2f | %6.4f %5.2f/%5.2f | %6.4f %5.2f/%5.2f\n', ...
        all_x(i), te_lin_T(i),te_lin_wH(i),te_lin_wV(i), tm_lin_T(i),tm_lin_wH(i),tm_lin_wV(i), ...
        te_tan_T(i),te_tan_wH(i),te_tan_wV(i), tm_tan_T(i),tm_tan_wH(i),tm_tan_wV(i));
end
fprintf('\nSaved:\n  %s\n  %s\n', ...
    fullfile(out_dir, 'plot_te500_tm518_transmission.png'), ...
    fullfile(out_dir, 'plot_te500_tm518_modewidth.png'));

%% ── local helpers ───────────────────────────────────────────────────────────
function [x, T, wH, wV] = series(base, dir_, n_teeth, suffix)
    files = cell(1, numel(n_teeth));
    for i = 1:numel(n_teeth)
        files{i} = fullfile(dir_, sprintf('result_N80_A%d%s', n_teeth(i), suffix));
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
        % In-plane (yx) and out-of-plane (zx) widths from the 2D field monitors.
        [h, v] = plane_mode_widths(fp);
        wH(k) = h;  wV(k) = v;
        % Fall back to the stored scalar fwhm_m for the horizontal width when a
        % file has no 2D fields (e.g. the 0-teeth TE baseline); vertical stays NaN.
        if isnan(wH(k)) && isfield(d, 'fwhm_m'), wH(k) = double(d.fwhm_m) * 1e6; end
    end
end
