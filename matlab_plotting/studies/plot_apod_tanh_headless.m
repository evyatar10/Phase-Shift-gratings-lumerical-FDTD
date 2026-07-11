%% Headless TE-vs-TM apodization plot for the TANH study
% Non-interactive twin of plot_apodization_vs.m: loads the tanh sweep results
% (result_N80_A*_th_M4*[_TM]_avg.mat) + the reused 0-teeth baseline files, and
% writes two PNGs:
%   plot_apod_tanh_transmission.png : peak transmission vs # apodized teeth (TE & TM)
%   plot_apod_tanh_modewidth.png    : spatial mode width fwhm_m [um] vs teeth (TE & TM)
%
% Same filename parsing as plot_apodization_vs.m:
%   _A<n> -> apodized teeth per side (absent => 0);  _TM -> TM (else TE).
% Run:  matlab -batch "run('matlab_plotting/plot_apod_tanh_headless.m')"
addpath(fileparts(fileparts(mfilename('fullpath'))));  % project root on path
addpath(fileparts(mfilename('fullpath')));             % matlab_plotting/ (plane_mode_widths)

clear; clc; close all;
FONT_SIZE = 13;

root      = fileparts(fileparts(mfilename('fullpath')));
tanh_dir  = fullfile(root, 'results_from_athena', 'tm_te_apod_tanh', 'results');
base_dir  = fullfile(root, 'results_from_athena', 'run_tm_vs_te', 'results');
out_dir   = fileparts(mfilename('fullpath'));

% Tanh apodized results (8) + reused 0-teeth baseline (2)
paths = [ ...
    cellstr(fullfile(tanh_dir, {...
        'result_N80_A2_th_M4_avg.mat',  'result_N80_A2_th_M4_TM_avg.mat', ...
        'result_N80_A5_th_M4_avg.mat',  'result_N80_A5_th_M4_TM_avg.mat', ...
        'result_N80_A10_th_M4_avg.mat', 'result_N80_A10_th_M4_TM_avg.mat', ...
        'result_N80_A20_th_M4_avg.mat', 'result_N80_A20_th_M4_TM_avg.mat'})), ...
    cellstr(fullfile(base_dir, {...
        'result_N80_avg_te.mat', 'result_N80_TM_avg_tm.mat'})) ];

nFiles   = numel(paths);
n_apod   = zeros(1, nFiles);
is_tm    = false(1, nFiles);
T_peak   = nan(1, nFiles);
width_H  = nan(1, nFiles);   % in-plane  (yx)
width_V  = nan(1, nFiles);   % out-of-plane (zx)

for k = 1:nFiles
    fp = paths{k};
    if ~isfile(fp), warning('Missing file: %s', fp); continue; end
    [~, base, ext] = fileparts(fp);
    fname = [base ext];

    tok = regexp(fname, '_A(\d+)', 'tokens', 'once');   % apodized teeth/side
    if isempty(tok), n_apod(k) = 0; else, n_apod(k) = str2double(tok{1}); end
    is_tm(k) = contains(fname, '_TM');                  % polarization

    d = load(fp);
    if isfield(d, 'resonance_transmission')
        T_peak(k) = double(d.resonance_transmission);
    elseif isfield(d, 'T') && isfield(d, 'wl_nm') && isfield(d, 'resonance_wavelength_nm')
        wl = double(d.wl_nm(:)); T = double(d.T(:));
        [~, ir] = min(abs(wl - double(d.resonance_wavelength_nm)));
        T_peak(k) = T(ir);
    else
        warning('No transmission info in %s', fname);
    end
    % In-plane (yx) / out-of-plane (zx) widths from the 2D field monitors; fall
    % back to the stored scalar fwhm_m for the horizontal width when 2D fields
    % are absent (e.g. the 0-teeth baseline); vertical stays NaN there.
    [width_H(k), width_V(k)] = plane_mode_widths(fp);
    if isnan(width_H(k)) && isfield(d, 'fwhm_m'), width_H(k) = double(d.fwhm_m) * 1e6; end

    fprintf('%-40s  pol=%s  n_apod=%2d  T=%.4f  wH=%.3f wV=%.3f um\n', ...
        fname, ternary(is_tm(k),'TM','TE'), n_apod(k), T_peak(k), width_H(k), width_V(k));
end

[te_x, te_T, te_wH, te_wV] = collect(~is_tm, n_apod, T_peak, width_H, width_V);
[tm_x, tm_T, tm_wH, tm_wV] = collect( is_tm, n_apod, T_peak, width_H, width_V);
all_x = unique(n_apod);

% Fig 1: peak transmission vs apodization
f1 = figure('Name', 'Peak T vs apodization (tanh)', 'Visible', 'off'); hold on;
if ~isempty(te_x), plot(te_x, te_T, 'o-', 'LineWidth', 1.6, 'MarkerSize', 7, ...
        'MarkerFaceColor', 'w', 'DisplayName', 'TE'); end
if ~isempty(tm_x), plot(tm_x, tm_T, 's-', 'LineWidth', 1.6, 'MarkerSize', 7, ...
        'MarkerFaceColor', 'w', 'DisplayName', 'TM'); end
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Peak transmission (T)', 'FontSize', FONT_SIZE);
title('Resonance transmission vs. apodization (tanh)', 'FontSize', FONT_SIZE);
legend('Location', 'best'); grid on; box on;
if ~isempty(all_x), xticks(all_x); end
exportgraphics(f1, fullfile(out_dir, 'plot_apod_tanh_transmission.png'), 'Resolution', 200);

% Fig 2: spatial mode width vs apodization — in-plane (yx) and out-of-plane (zx).
f2 = figure('Name', 'Mode width vs apodization (tanh, both planes)', ...
            'Visible', 'off', 'Position', [100 100 1180 520]);
allw = [te_wH te_wV tm_wH tm_wV];
ylo = floor(min(allw(:), [], 'omitnan')) - 1;
yhi = ceil( max(allw(:), [], 'omitnan')) + 1;

subplot(1, 2, 1); hold on;
if ~isempty(te_x), plot(te_x, te_wH, 'o-', 'LineWidth', 1.6, 'MarkerSize', 7, ...
        'MarkerFaceColor', 'w', 'DisplayName', 'TE'); end
if ~isempty(tm_x), plot(tm_x, tm_wH, 's-', 'LineWidth', 1.6, 'MarkerSize', 7, ...
        'MarkerFaceColor', 'w', 'DisplayName', 'TM'); end
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Mode width, FWHM [\mum]', 'FontSize', FONT_SIZE);
title('In-plane width (yx, \int|E|^2 dy)', 'FontSize', FONT_SIZE);
legend('Location', 'best'); grid on; box on;
if ~isempty(all_x), xticks(all_x); end
if isfinite(ylo) && isfinite(yhi), ylim([ylo yhi]); end

subplot(1, 2, 2); hold on;
if ~isempty(te_x), plot(te_x, te_wV, 'o-', 'LineWidth', 1.6, 'MarkerSize', 7, ...
        'MarkerFaceColor', 'w', 'DisplayName', 'TE'); end
if ~isempty(tm_x), plot(tm_x, tm_wV, 's-', 'LineWidth', 1.6, 'MarkerSize', 7, ...
        'MarkerFaceColor', 'w', 'DisplayName', 'TM'); end
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Mode width, FWHM [\mum]', 'FontSize', FONT_SIZE);
title('Out-of-plane width (zx, \int|E|^2 dz)', 'FontSize', FONT_SIZE);
legend('Location', 'best'); grid on; box on;
if ~isempty(all_x), xticks(all_x); end
if isfinite(ylo) && isfinite(yhi), ylim([ylo yhi]); end

sgtitle('Spatial mode width vs. apodization (tanh)', 'FontSize', FONT_SIZE + 1, 'FontWeight', 'bold');
exportgraphics(f2, fullfile(out_dir, 'plot_apod_tanh_modewidth.png'), 'Resolution', 200);

fprintf('\nSaved:\n  %s\n  %s\n', ...
    fullfile(out_dir, 'plot_apod_tanh_transmission.png'), ...
    fullfile(out_dir, 'plot_apod_tanh_modewidth.png'));

%% ── local helpers ───────────────────────────────────────────────────────────
function [x, y1, yH, yV] = collect(mask, n_apod, T_peak, width_H, width_V)
    x = n_apod(mask); y1 = T_peak(mask); yH = width_H(mask); yV = width_V(mask);
    [x, si] = sort(x); y1 = y1(si); yH = yH(si); yV = yV(si);
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
