% plot_compare_8_devices.m
% Grouped bar chart comparing 8 pi-shift Bragg devices (TE & TM x
% {baseline, linear-10, tanh-10, optimized}) on two metrics:
%   (1) peak transmission, (2) spatial mode-width FWHM (um).
%
% Reads result_*_cmp_<label>.mat written by runners/single/compare_8_devices.py.
% All bars are standard TE/TM colors; the TE-optimized VALUE LABELS (best
% transmission/mode-width tradeoff) are highlighted.
%
% Headless run:
%   matlab -batch "cd('matlab_plotting'); plot_compare_8_devices"

clear; close all;

resultsDir = fullfile('..', 'results_from_athena', 'compare_8_devices', 'results');
if ~isfolder(resultsDir)
    resultsDir = fullfile('results_from_athena', 'compare_8_devices', 'results');
end

cats   = {'Baseline', 'Linear-10', 'Tanh-10', 'Optimized (2 teeth)'};
labels = { ...
    {'TE_baseline',  'TM_baseline'}, ...
    {'TE_linear10',  'TM_linear10'}, ...
    {'TE_tanh10',    'TM_tanh10'}, ...
    {'TE_optimized', 'TM_optimized'} };

nCat = numel(cats);
T_peak   = nan(nCat, 2);   % cols: [TE TM]
width_um = nan(nCat, 2);

for i = 1:nCat
    for p = 1:2
        lab = labels{i}{p};
        f = dir(fullfile(resultsDir, ['result_*_cmp_' lab '.mat']));
        if isempty(f); warning('Missing result for %s', lab); continue; end
        d = load(fullfile(f(1).folder, f(1).name));
        if isfield(d, 'resonance_transmission'); T_peak(i, p) = double(d.resonance_transmission); end
        if isfield(d, 'fwhm_m'); width_um(i, p) = double(d.fwhm_m) * 1e6; end
        fprintf('%-14s  T=%.4f  width=%.3f um\n', lab, T_peak(i, p), width_um(i, p));
    end
end

% Override the TM "Optimized" bar (row 4, col 2 = TM) with the 5-parameter
% WITH-SHIFT optimum (PSO job 97635), accurate-mesh verified. Replaces the
% earlier 3-param (no-shift) TM optimized values per request.
T_peak(4, 2)   = 0.9618;    % accurate-mesh true_peak_T (was 0.9558, 3-param)
width_um(4, 2) = 20.870;    % accurate-mesh spatial FWHM (was 20.091, 3-param)
fprintf('TM_optimized   T=%.4f  width=%.3f um  [5-param/with-shift override]\n', ...
    T_peak(4, 2), width_um(4, 2));

%% Figure: two stacked panels (transmission, mode width)
fig = figure('Visible', 'off', 'Position', [100 100 1100 860], 'Color', 'w');
cTE = [0.20 0.45 0.80];   % blue
cTM = [0.85 0.33 0.20];   % orange-red
HI  = 0;                  % no value-label highlight (all labels plain black)

% --- Panel 1: peak transmission ---
ax1 = subplot(2, 1, 1);
b1 = bar(T_peak, 'grouped');
b1(1).FaceColor = cTE;  b1(2).FaceColor = cTM;
set(ax1, 'XTickLabel', cats, 'FontSize', 13);
ylabel('Peak transmission', 'FontSize', 14, 'FontWeight', 'bold');
title('Peak transmission by device', 'FontSize', 15);
ylim([0.78, 1.04]);
grid on; bar_value_labels(b1, T_peak, '%.3f', HI, 1);
legend([b1(1) b1(2)], {'TE', 'TM'}, 'Location', 'southoutside', ...
    'Orientation', 'horizontal', 'FontSize', 13);

% --- Panel 2: spatial mode-width FWHM ---
ax2 = subplot(2, 1, 2);
b2 = bar(width_um, 'grouped');
b2(1).FaceColor = cTE;  b2(2).FaceColor = cTM;
set(ax2, 'XTickLabel', cats, 'FontSize', 13);
ylabel('Mode width FWHM (\mum)', 'FontSize', 14, 'FontWeight', 'bold');
title('Cavity mode width (spatial FWHM) by device', 'FontSize', 15);
ylim([0, 28]);
grid on; bar_value_labels(b2, width_um, '%.1f', HI, 1);
legend([b2(1) b2(2)], {'TE', 'TM'}, 'Location', 'southoutside', ...
    'Orientation', 'horizontal', 'FontSize', 13);

sgtitle({'\pi-shift Bragg grating: TE vs TM device comparison (accurate mesh)', ...
    '\lambda_{TE} \approx 1556-1569 nm    \lambda_{TM} \approx 1569 nm'}, ...
    'FontSize', 14, 'FontWeight', 'bold');

outPng = 'plot_compare_8_devices.png';
exportgraphics(fig, outPng, 'Resolution', 200);
set(fig, 'Visible', 'on');                     % so the saved .fig opens visibly
savefig(fig, 'plot_compare_8_devices.fig');   % editable MATLAB figure
fprintf('Saved %s and plot_compare_8_devices.fig\n', outPng);

% --- helper: horizontal bold value labels; highlight one (hi_g,hi_j) ---
function bar_value_labels(bars, Y, fmt, hi_g, hi_j)
    if nargin < 4, hi_g = 0; end
    if nargin < 5, hi_j = 0; end
    [ngroups, nbars] = size(Y);
    yl = ylim;  off = 0.012 * (yl(2) - yl(1));
    for j = 1:nbars
        x = bars(j).XEndPoints;
        for g = 1:ngroups
            if isnan(Y(g, j)); continue; end
            if g == hi_g && j == hi_j
                text(x(g), Y(g, j) + off, sprintf(fmt, Y(g, j)), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                    'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.80 0.50 0]);
            else
                text(x(g), Y(g, j) + off, sprintf(fmt, Y(g, j)), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                    'FontSize', 12, 'FontWeight', 'bold');
            end
        end
    end
end
