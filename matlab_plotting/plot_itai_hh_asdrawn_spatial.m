% Spatial mode profile of Itai's HH device AS DRAWN (full-depth profile, his
% pitch 514 nm, N=98/side). Study: runners/sweeps/itai_hh_asdrawn.py | IGUM 63441
% envelope = peak envelope of the standing-wave pattern (the fwhm_m convention).
clear; close all;
d = fullfile(fileparts(mfilename('fullpath')), '..', 'results_from_igum');
te = load(fullfile(d, 'hh_asdrawn_spatial_TE.mat'));
tm = load(fullfile(d, 'hh_asdrawn_spatial_TM.mat'));

fig = figure('Position', [100 100 950 430], 'Color', 'w');
for k = 1:2
    s = {te, tm}; s = s{k}; c = {[0 0.35 0.85], [0.85 0.25 0]}; c = c{k};
    subplot(1,2,k);
    plot(s.x_um, s.energy_density / max(s.energy_density), '-', 'Color', [0.7 0.7 0.7]); hold on;
    plot(s.x_um, s.envelope / max(s.envelope), '-', 'Color', c, 'LineWidth', 1.8);
    yline(0.5, 'k:', 'LineWidth', 1.0);
    xline(-s.fwhm_um/2, 'k--'); xline(s.fwhm_um/2, 'k--');
    grid on; xlim([-40 40]); ylim([0 1.05]);
    xlabel('x (\mum)'); ylabel('normalised energy');
    legend({'|E|^2', 'envelope', 'half max'}, 'Location', 'northeast', 'FontSize', 9);
    title([s.polarization '   FWHM = ' sprintf('%.2f', s.fwhm_um) ' \mum']);
end
sgtitle('Itai HH apodization, AS DRAWN (N=98/side, pitch 514 nm) - spatial mode profile');
savefig(fig, fullfile(d, 'itai_hh_asdrawn_spatial.fig'));
exportgraphics(fig, fullfile(d, 'itai_hh_asdrawn_spatial.png'), 'Resolution', 150);
fprintf('TE fwhm %.3f um | TM fwhm %.3f um\n', te.fwhm_um, tm.fwhm_um);
