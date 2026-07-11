function plot_tm_H200_TRL()
% Energy budget T(lambda), R(lambda), loss(lambda) for the FINAL 80 um device
% (200 nm height, pitch 531.5 nm, corrugation 227.7 nm, N=300, SPAN_MULT=4).
% T + R + loss = 1 everywhere; "loss" here is the radiation into the cladding.

here = fileparts(mfilename('fullpath'));
ddir = fullfile(here, '..', 'results_from_athena', 'tm_wide_mode_H200', 'results');
d = load(fullfile(ddir, 'result_corrmatch_tm_C227685.mat'));

wl   = double(d.wl_nm(:));
T    = double(d.T(:));
R    = double(d.R(:));
L    = double(d.loss(:));
tot  = T + R + L;

lam_res = double(d.resonance_wavelength_nm);
Tres    = double(d.resonance_transmission);
Q       = lam_res / abs(double(d.spectral_fwhm_nm));
fwhm    = double(d.fwhm_m) * 1e6;
lam_rad = (1.4585 + 1.444) * 531.5e-9 * 1e9;   % radiation onset

fig = figure('Color', 'w', 'Position', [100 100 1000 580]);
hold on;
plot(wl, T,   '-', 'LineWidth', 1.7, 'Color', [0 0.45 0.74]);
plot(wl, R,   '-', 'LineWidth', 1.7, 'Color', [0.85 0.33 0.10]);
plot(wl, L,   '-', 'LineWidth', 1.7, 'Color', [0.47 0.67 0.19]);
plot(wl, tot, '--','LineWidth', 1.0, 'Color', [0.4 0.4 0.4]);

xline(lam_res, 'k:',  'LineWidth', 1.0);
xline(lam_rad, 'k-.', 'LineWidth', 1.0);
text(lam_rad - 0.4, 0.92, sprintf('radiation onset %.1f nm', lam_rad), ...
     'Rotation', 90, 'HorizontalAlignment', 'right', 'FontSize', 9);

grid on; box on;
xlabel('Wavelength [nm]'); ylabel('Fraction of power');
xlim([min(wl) max(wl)]); ylim([0 1.05]);
legend({'T (transmission)', 'R (reflection)', 'loss (radiation)', ...
        'T+R+loss = 1 (check)'}, 'Location', 'east', 'Interpreter', 'tex');
title({sprintf(['Energy budget of the final 80 \\mum device  ' ...
        '(\\lambda_{res}=%.2f nm, T_{res}=%.2f, Q=%.0f, FWHM=%.0f \\mum)'], ...
        lam_res, Tres, Q, fwhm), ...
       ['200 nm height, width 800 nm, pitch 531.5 nm, corrugation 227.7 nm, ' ...
        'N=300/side, SPAN\_MULT=4']}, ...
      'Interpreter', 'tex', 'FontWeight', 'bold');

outpng = fullfile(ddir, 'plot_tm_H200_TRL.png');
exportgraphics(fig, outpng, 'Resolution', 150);
fprintf('saved %s\n', outpng);
end
