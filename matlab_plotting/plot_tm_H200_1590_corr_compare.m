function plot_tm_H200_1590_corr_compare()
% Overlay transmission spectra T(lambda) for every corrugation depth the
% wide-mode secant evaluated on the 1590 nm 200 nm-height device
% (N=300, pitch 545.959 nm, SPAN_MULT=4 -> physical T). Shows how the
% corrugation changes the bandgap, the defect peak and the blue-side loss.
% Data-agnostic: globs every result_corrmatch_tm_C*.mat in the study dir.

here = fileparts(mfilename('fullpath'));
ddir = fullfile(here, '..', 'results_from_athena', 'tm_wide_mode_H200_P546', 'results');

files = dir(fullfile(ddir, 'result_corrmatch_tm_C*.mat'));
assert(~isempty(files), 'no result_corrmatch_tm_C*.mat in %s', ddir);

% sort by corrugation depth so legend/colours are ordered
recs = struct('name', {}, 'corr', {});
for k = 1:numel(files)
    dk = load(fullfile(ddir, files(k).name));
    if ~isfield(dk, 'T') || ~isfield(dk, 'wl_nm'); continue; end
    recs(end+1) = struct('name', files(k).name, ...
                         'corr', double(dk.corrugation_depth_m) * 1e9); %#ok<AGROW>
end
[~, ord] = sort([recs.corr]);
recs = recs(ord);
cols = lines(numel(recs));

fig = figure('Color', 'w', 'Position', [100 100 950 560]);
hold on; leg = cell(1, numel(recs)); pitch_nm = NaN;
for k = 1:numel(recs)
    d = load(fullfile(ddir, recs(k).name));
    wl = double(d.wl_nm(:));  T = double(d.T(:));
    corr = double(d.corrugation_depth_m) * 1e9;
    fwhm = double(d.fwhm_m) * 1e6;
    Tres = double(d.resonance_transmission);
    Q    = double(d.resonance_wavelength_nm) / abs(double(d.spectral_fwhm_nm));
    pitch_nm = double(d.pitch_m) * 1e9;
    plot(wl, T, '-', 'LineWidth', 1.6, 'Color', cols(k, :));
    leg{k} = sprintf('corr %.0f nm:  FWHM %.0f \\mum,  T_{res} %.2f,  Q %.0f', ...
                     corr, fwhm, Tres, Q);
end

lam_rad = (1.4585 + 1.444) * pitch_nm;   % radiation onset (n_eff+n_clad)*Lambda
xline(lam_rad, 'k:', 'LineWidth', 1.2);
text(lam_rad - 0.5, 0.05, sprintf('radiation onset %.1f nm', lam_rad), ...
     'HorizontalAlignment', 'right', 'Rotation', 90, 'FontSize', 9);

grid on; box on;
xlabel('Wavelength [nm]'); ylabel('Transmission');
ylim([0 1]);
legend(leg, 'Location', 'south', 'Interpreter', 'tex');
title({'TM 1590 nm 200 nm-height device: transmission vs corrugation', ...
       sprintf('pitch %.2f nm,  N = 300/side,  SPAN\\_MULT = 4  (physical T)', ...
       pitch_nm)}, 'Interpreter', 'tex', 'FontWeight', 'bold');

outpng = fullfile(ddir, 'plot_tm_H200_1590_corr_compare.png');
exportgraphics(fig, outpng, 'Resolution', 150);
fprintf('saved %s\n', outpng);
end
