function plot_tm_H200_corr_compare()
% Overlay transmission spectra T(lambda) for the three corrugation depths the
% wide-mode search actually evaluated on the 200 nm-height device
% (N=300, pitch 531.5 nm, SPAN_MULT=4 -> physical T). Lets you compare how the
% corrugation changes the bandgap, the defect peak, and the blue-side loss.

here = fileparts(mfilename('fullpath'));
ddir = fullfile(here, '..', 'results_from_athena', 'tm_wide_mode_H200', 'results');

files = {'result_corrmatch_tm_C180000.mat', ...
         'result_corrmatch_tm_C227685.mat', ...
         'result_corrmatch_tm_C260000.mat'};
cols  = {[0 0.45 0.74], [0.85 0.33 0.10], [0.47 0.67 0.19]};

fig = figure('Color', 'w', 'Position', [100 100 950 560]);
hold on; leg = cell(1, numel(files));
for k = 1:numel(files)
    d = load(fullfile(ddir, files{k}));
    wl = double(d.wl_nm(:));  T = double(d.T(:));
    corr = double(d.corrugation_depth_m) * 1e9;
    fwhm = double(d.fwhm_m) * 1e6;
    Tres = double(d.resonance_transmission);
    Q    = double(d.resonance_wavelength_nm) / abs(double(d.spectral_fwhm_nm));
    plot(wl, T, '-', 'LineWidth', 1.6, 'Color', cols{k});
    leg{k} = sprintf('corr %.0f nm:  FWHM %.0f \\mum,  T_{res} %.2f,  Q %.0f', ...
                     corr, fwhm, Tres, Q);
end
% mark the common radiation threshold (n_eff+n_clad)*Lambda
lam_rad = (1.4585 + 1.444) * 531.5e-9 * 1e9;
xline(lam_rad, 'k:', 'LineWidth', 1.2);
text(lam_rad - 0.5, 0.05, sprintf('radiation onset %.1f nm', lam_rad), ...
     'HorizontalAlignment', 'right', 'Rotation', 90, 'FontSize', 9);

grid on; box on;
xlabel('Wavelength [nm]'); ylabel('Transmission');
ylim([0 1]);
legend(leg, 'Location', 'south', 'Interpreter', 'tex');
title({'TM 200 nm-height device: transmission vs corrugation', ...
       'pitch 531.5 nm,  N = 300/side,  SPAN\_MULT = 4  (physical T)'}, ...
      'Interpreter', 'tex', 'FontWeight', 'bold');

outdir = ddir;
if ~isfolder(outdir), mkdir(outdir); end
outpng = fullfile(outdir, 'plot_tm_H200_corr_compare.png');
exportgraphics(fig, outpng, 'Resolution', 150);
fprintf('saved %s\n', outpng);
end
