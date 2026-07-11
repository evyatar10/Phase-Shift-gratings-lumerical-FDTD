function plot_tm_wide_mode_H200_1590()
% Transmission spectrum + spatial mode envelope for the 200 nm-height TM
% wide-mode pi-shift grating RETARGETED to ~1590 nm (pitch 545.959 nm, the
% corrugation that gives an 80 um spatial mode FWHM at 1590 nm). Two-panel:
%   LEFT : T(lambda), with resonance wavelength, peak transmission and Q
%   RIGHT: normalized spatial energy envelope, with the mode FWHM
% Data-agnostic: globs the study dir and picks the device whose spatial FWHM is
% closest to 80 um, so it works whatever exact corrugation the secant landed on.

here = fileparts(mfilename('fullpath'));
ddir = fullfile(here, '..', 'results_from_athena', 'tm_wide_mode_H200_P546', 'results');

d = pick_80um_device(ddir);

% --- spectra / envelope ---
wl    = double(d.wl_nm(:));
T     = double(d.T(:));
x_um  = double(d.field_x(:)) * 1e6;
env   = double(d.field_envelope_1D(:));
env_n = (env - min(env)) / (max(env) - min(env));   % floor-relative, 0..1

% --- key figures of merit ---
lam_res = double(d.resonance_wavelength_nm);
T_res   = double(d.resonance_transmission);
Q       = lam_res / abs(double(d.spectral_fwhm_nm));   % spectral_fwhm stored signed
fwhm_um = double(d.fwhm_m) * 1e6;

% --- device dimensions ---
h_nm     = double(d.core_height_m)           * 1e9;
w_nm     = double(d.avg_corrugation_width_m) * 1e9;
pitch_nm = double(d.pitch_m)                 * 1e9;
corr_nm  = double(d.corrugation_depth_m)     * 1e9;
N        = double(d.n_periods_each_side);

fig = figure('Color', 'w', 'Position', [100 100 1250 480]);

% ===== LEFT: transmission vs wavelength =====
subplot(1, 2, 1);
plot(wl, T, 'b-', 'LineWidth', 1.4); hold on;
xline(lam_res, 'r--', 'LineWidth', 1.1);
plot(lam_res, T_res, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
grid on; box on;
xlabel('Wavelength [nm]');
ylabel('Transmission');
xlim([min(wl) max(wl)]); ylim([0 1]);
title(sprintf('Transmission:  \\lambda_{res} = %.2f nm,  T_{peak} = %.3f,  Q = %.0f', ...
      lam_res, T_res, Q), 'Interpreter', 'tex');

% ===== RIGHT: spatial mode envelope =====
subplot(1, 2, 2);
plot(x_um, env_n, 'b-', 'LineWidth', 1.4); hold on;
yline(0.5, 'k:', 'LineWidth', 1.0);
xline(-fwhm_um/2, 'r--', 'LineWidth', 1.0);
xline( fwhm_um/2, 'r--', 'LineWidth', 1.0);
grid on; box on;
xlabel('x [\mum]');
ylabel('Normalized energy envelope');
xlim([min(x_um) max(x_um)]); ylim([0 1.05]);
title(sprintf('Spatial mode:  FWHM = %.1f \\mum', fwhm_um), 'Interpreter', 'tex');

% ===== overall title: device dimensions =====
title_l1 = sprintf(['\\pi-shift TM grating @ ~1590 nm   |   height %.0f nm,  ' ...
    'width %.0f nm,  pitch %.2f nm'], h_nm, w_nm, pitch_nm);
title_l2 = sprintf(['corrugation %.1f nm,  N = %.0f/side   ' ...
    '(n_{core} = 1.97 / n_{clad} = 1.444)'], corr_nm, N);
sgtitle({title_l1, title_l2}, 'Interpreter', 'tex', 'FontWeight', 'bold', 'FontSize', 12);

outpng = fullfile(ddir, 'plot_tm_wide_mode_H200_1590.png');
outfig = fullfile(ddir, 'plot_tm_wide_mode_H200_1590.fig');
exportgraphics(fig, outpng, 'Resolution', 150);
savefig(fig, outfig);
fprintf('saved %s\n', outpng);
fprintf('saved %s\n', outfig);
end


function d = pick_80um_device(ddir)
% Return the loaded result struct whose spatial FWHM is closest to 80 um.
files = dir(fullfile(ddir, 'result_corrmatch_tm_C*.mat'));
assert(~isempty(files), 'no result_corrmatch_tm_C*.mat in %s', ddir);
best = []; bestErr = inf;
for k = 1:numel(files)
    dk = load(fullfile(ddir, files(k).name));
    if ~isfield(dk, 'fwhm_m') || ~isfield(dk, 'field_envelope_1D'); continue; end
    err = abs(double(dk.fwhm_m) * 1e6 - 80.0);
    if err < bestErr; bestErr = err; best = dk; bestName = files(k).name; end
end
assert(~isempty(best), 'no usable result with fwhm_m + envelope in %s', ddir);
fprintf('selected %s  (FWHM %.2f um)\n', bestName, double(best.fwhm_m) * 1e6);
d = best;
end
