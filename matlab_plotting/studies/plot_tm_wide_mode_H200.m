function plot_tm_wide_mode_H200()
% Transmission spectrum + spatial mode envelope for the 200 nm-height TM
% wide-mode pi-shift grating (the corrugation = 227.7 nm device that gives an
% 80 um spatial mode FWHM). Two-panel figure:
%   LEFT : T(lambda), with resonance wavelength, peak transmission and Q in title
%   RIGHT: normalized spatial energy envelope, with the mode FWHM in title
% Saves a .png and a .fig next to this script.

here    = fileparts(mfilename('fullpath'));
matpath = fullfile(here, '..', 'results_from_athena', 'tm_wide_mode_H200', ...
                   'results', 'result_corrmatch_tm_C227685.mat');
d = load(matpath);

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
h_nm     = double(d.core_height_m)       * 1e9;
w_nm     = double(d.avg_corrugation_width_m) * 1e9;
pitch_nm = double(d.pitch_m)             * 1e9;
corr_nm  = double(d.corrugation_depth_m) * 1e9;
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
title_l1 = sprintf(['\\pi-shift TM grating   |   height %.0f nm,  width %.0f nm,  ' ...
    'pitch %.1f nm'], h_nm, w_nm, pitch_nm);
title_l2 = sprintf(['corrugation %.1f nm,  N = %.0f/side   ' ...
    '(n_{core} = 1.97 / n_{clad} = 1.444)'], corr_nm, N);
sgtitle({title_l1, title_l2}, 'Interpreter', 'tex', 'FontWeight', 'bold', 'FontSize', 12);

% ===== save (outputs live with the data in results_from_athena, NOT here) =====
outdir = fullfile(here, '..', 'results_from_athena', 'tm_wide_mode_H200', 'results');
if ~isfolder(outdir), mkdir(outdir); end
outpng = fullfile(outdir, 'plot_tm_wide_mode_H200.png');
outfig = fullfile(outdir, 'plot_tm_wide_mode_H200.fig');
exportgraphics(fig, outpng, 'Resolution', 150);
savefig(fig, outfig);
fprintf('saved %s\n', outpng);
fprintf('saved %s\n', outfig);
end
