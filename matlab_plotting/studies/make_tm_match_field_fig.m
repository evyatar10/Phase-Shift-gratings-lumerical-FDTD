function make_tm_match_field_fig()
% Build a MATLAB .fig of the 1D longitudinal cavity field profiles for the
% TE@80 and period-matched TM@132 designs (one panel each), with the spatial
% mode FWHM (fwhm_m) shown in each legend. Saves an editable .fig (+ a .png).

resdir = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'results_from_athena', 'tm_match_bisect', 'results');

te = load(fullfile(resdir, 'result_N80_avg_te_smp.mat'));
tm = load(fullfile(resdir, 'result_N132_TM_avg_tm_P518p3_smp.mat'));

f = figure('Color', 'w', 'Name', 'Cavity field profiles: TE@80 vs TM@132', ...
           'Position', [100 100 980 600]);
ax = axes(f); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');

envelope(ax, te, 'TE  N=80',  [0.85 0.33 0.10]);
envelope(ax, tm, 'TM  N=132', [0.00 0.45 0.74]);

xlabel(ax, 'Position x [\mum]', 'FontSize', 13);
ylabel(ax, 'Integrated energy density (a.u.)', 'FontSize', 13);
title(ax, 'Longitudinal cavity field envelope: TE@80 vs TM@132', 'FontSize', 14);
legend(ax, 'Location', 'northeast', 'FontSize', 12);
set(ax, 'FontSize', 12);

out_fig = fullfile(resdir, 'field_profiles_TE80_vs_TM132.fig');
out_png = fullfile(resdir, 'field_profiles_TE80_vs_TM132_matlab.png');
savefig(f, out_fig);
exportgraphics(f, out_png, 'Resolution', 150);
fprintf('WROTE_FIG: %s\n', out_fig);
fprintf('WROTE_PNG: %s\n', out_png);
end

function envelope(ax, s, label, col)
% Plot just the peak field envelope vs x, with the spatial FWHM (fwhm_m) in the
% legend label.
x    = double(s.field_x(:)) * 1e6;                 % um
env  = double(s.field_envelope_1D(:));
fwhm = double(s.fwhm_m) * 1e6;                      % um
plot(ax, x, env, '-', 'Color', col, 'LineWidth', 2.0, ...
     'DisplayName', sprintf('%s   (FWHM = %.2f \\mum)', label, fwhm));
end
