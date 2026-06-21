function make_te80_tm80_field_fig()
% Overlaid longitudinal cavity field envelopes for TE@80 and TM@80 (both 80
% periods/side), with the spatial mode FWHM (fwhm_m) in each legend label.
% Saves an editable .fig (+ a .png).

resdir = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'results_from_athena', 'tm_match_bisect', 'results');

te = load(fullfile(resdir, 'result_N80_avg_te_smp.mat'));
tm = load(fullfile(resdir, 'result_N80_TM_avg_tm_P518p3_smp.mat'));

f = figure('Color', 'w', 'Name', 'Cavity field profiles: TE@80 vs TM@80', ...
           'Position', [100 100 980 600]);
ax = axes(f); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');

envelope(ax, te, 'TE  N=80', [0.85 0.33 0.10]);
envelope(ax, tm, 'TM  N=80', [0.00 0.45 0.74]);

xlabel(ax, 'Position x [\mum]', 'FontSize', 13);
ylabel(ax, 'Integrated energy density (a.u.)', 'FontSize', 13);
title(ax, 'Longitudinal cavity field envelope: TE@80 vs TM@80', 'FontSize', 14);
legend(ax, 'Location', 'northeast', 'FontSize', 12);
set(ax, 'FontSize', 12);

out_fig = fullfile(resdir, 'field_profiles_TE80_vs_TM80.fig');
out_png = fullfile(resdir, 'field_profiles_TE80_vs_TM80_matlab.png');
savefig(f, out_fig);
exportgraphics(ax, out_png, 'Resolution', 150);
fprintf('WROTE_FIG: %s\n', out_fig);
fprintf('WROTE_PNG: %s\n', out_png);
end

function envelope(ax, s, label, col)
x    = double(s.field_x(:)) * 1e6;                 % um
env  = double(s.field_envelope_1D(:));
fwhm = double(s.fwhm_m) * 1e6;                      % um
plot(ax, x, env, '-', 'Color', col, 'LineWidth', 2.0, ...
     'DisplayName', sprintf('%s   (FWHM = %.2f \\mum)', label, fwhm));
end
