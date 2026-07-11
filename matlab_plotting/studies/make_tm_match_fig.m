function make_tm_match_fig()
% Build a MATLAB .fig of the combined transmission: TE@80 vs period-matched TM.
% Loads the bisection result_*.mat files, plots T(lambda) for both polarizations
% with Q in the legend, and saves an editable .fig (+ a .png) you can open and
% edit interactively in MATLAB (Property Editor, Plot Tools, drag annotations).

resdir = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'results_from_athena', 'tm_match_bisect', 'results');

te = load(fullfile(resdir, 'result_N80_avg_te_smp.mat'));
tm = load(fullfile(resdir, 'result_N132_TM_avg_tm_P518p3_smp.mat'));

[te_wl, te_T, te_Q] = curve(te);
[tm_wl, tm_T, tm_Q] = curve(tm);

f = figure('Color', 'w', 'Name', 'TE@80 vs period-matched TM', ...
           'Position', [100 100 980 600]);
ax = axes(f); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
plot(ax, te_wl, te_T, '-', 'LineWidth', 2.0, 'Color', [0.85 0.33 0.10], ...
     'DisplayName', sprintf('TE  N=80  (Q \\approx %.0f)', te_Q));
plot(ax, tm_wl, tm_T, '-', 'LineWidth', 2.0, 'Color', [0.00 0.45 0.74], ...
     'DisplayName', sprintf('TM  N=132  (Q \\approx %.0f)', tm_Q));
xlabel(ax, 'Wavelength [nm]', 'FontSize', 13);
ylabel(ax, 'Transmission, T', 'FontSize', 13);
title(ax, 'Combined transmission: TE@80 vs period-matched TM', 'FontSize', 14);
legend(ax, 'Location', 'northeast', 'FontSize', 12);
set(ax, 'FontSize', 12);

out_fig = fullfile(resdir, 'combined_transmission_TE80_vs_TMmatched.fig');
out_png = fullfile(resdir, 'combined_transmission_TE80_vs_TMmatched_matlab.png');
savefig(f, out_fig);
exportgraphics(ax, out_png, 'Resolution', 150);
fprintf('WROTE_FIG: %s\n', out_fig);
fprintf('WROTE_PNG: %s\n', out_png);
end

function [wl, T, Q] = curve(s)
% Sorted (wl, T) spectrum and spectral Q = lambda / |spectral_fwhm_nm|.
wl = double(s.wl_nm(:));
T  = double(s.T(:));
[wl, idx] = sort(wl); T = T(idx);
lam  = double(s.resonance_wavelength_nm);
fwhm = abs(double(s.spectral_fwhm_nm));   % stored sign-flipped (descending lambda axis)
Q = lam / fwhm;
end
