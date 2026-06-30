function plot_tm_H200_1590_TRL()
% Energy budget T(lambda), R(lambda), loss(lambda) for the 1590 nm 80 um device
% (200 nm height, pitch 545.959 nm, N=300, SPAN_MULT=4). T + R + loss = 1
% everywhere; "loss" is the radiation into the cladding. Data-agnostic: picks the
% device whose spatial FWHM is closest to 80 um from the study dir.

here = fileparts(mfilename('fullpath'));
ddir = fullfile(here, '..', 'results_from_athena', 'tm_wide_mode_H200_P546', 'results');

d = pick_80um_device(ddir);

wl   = double(d.wl_nm(:));
T    = double(d.T(:));
R    = double(d.R(:));
L    = double(d.loss(:));
tot  = T + R + L;

lam_res  = double(d.resonance_wavelength_nm);
Tres     = double(d.resonance_transmission);
Q        = lam_res / abs(double(d.spectral_fwhm_nm));
fwhm     = double(d.fwhm_m) * 1e6;
pitch_nm = double(d.pitch_m) * 1e9;
lam_rad  = (1.4585 + 1.444) * pitch_nm;   % radiation onset (n_eff+n_clad)*Lambda

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
title({sprintf(['Energy budget of the 1590 nm 80 \\mum device  ' ...
        '(\\lambda_{res}=%.2f nm, T_{res}=%.2f, Q=%.0f, FWHM=%.0f \\mum)'], ...
        lam_res, Tres, Q, fwhm), ...
       sprintf(['200 nm height, pitch %.2f nm, corrugation %.1f nm, ' ...
        'N=300/side, SPAN\\_MULT=4'], pitch_nm, double(d.corrugation_depth_m)*1e9)}, ...
      'Interpreter', 'tex', 'FontWeight', 'bold');

outpng = fullfile(ddir, 'plot_tm_H200_1590_TRL.png');
exportgraphics(fig, outpng, 'Resolution', 150);
fprintf('saved %s\n', outpng);
end


function d = pick_80um_device(ddir)
files = dir(fullfile(ddir, 'result_corrmatch_tm_C*.mat'));
assert(~isempty(files), 'no result_corrmatch_tm_C*.mat in %s', ddir);
best = []; bestErr = inf;
for k = 1:numel(files)
    dk = load(fullfile(ddir, files(k).name));
    if ~isfield(dk, 'fwhm_m') || ~isfield(dk, 'loss'); continue; end
    err = abs(double(dk.fwhm_m) * 1e6 - 80.0);
    if err < bestErr; bestErr = err; best = dk; bestName = files(k).name; end
end
assert(~isempty(best), 'no usable result with fwhm_m + loss in %s', ddir);
fprintf('selected %s  (FWHM %.2f um)\n', bestName, double(best.fwhm_m) * 1e6);
d = best;
end
