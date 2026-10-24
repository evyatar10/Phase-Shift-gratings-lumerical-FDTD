% plot_postarray_spectrum_mode.m -- T(lambda) and the 1D mode profile (energy vs x) of the best post-array device vs its control.
% Study dir: runners/sweeps/farfield_sph_15um_postarray_overshoot | Jobs: Athena 269589 (r2pack_x30) vs 267605_0/269582 (control) | 2026-10-24
% Short runs (~2 decay times): the spectrum is shown for shape only (CLAUDE.md 2: no Q from a short-run spectrum); the corrected peak T
% (T x gamma^2) and Q_i come from python_tools/radcancel/solve_v4.py and are quoted in the legend.
R = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results_from_athena');
F = {fullfile(R, 'budget_overshoot', 'dy29_box6p8.mat'), fullfile(R, 'comb_overshoot', 'r2pack_x30_dy29.mat')};
lab = {'control (no posts): T 0.671, Q_i 5.67e5', 'r2pack (211 posts/side, rows y 1.2/1.5 um): T 0.778, Q_i 9.38e5 (x1.65)'};
col = {[0.2 0.2 0.2], [0.85 0.3 0.15]};
f = figure('Color', 'w', 'Position', [60 60 1200 820]); tl = tiledlayout(f, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
ax1 = nexttile(tl); hold(ax1, 'on'); ax2 = nexttile(tl); hold(ax2, 'on');
for k = 1:2
    m = load(F{k}, 'wl_m', 'T', 'resonance_wavelength_nm', 'field_x', 'field_energy_density_1D', 'fwhm_m');
    lam = m.wl_m(:) * 1e9; lr = m.resonance_wavelength_nm;
    plot(ax1, lam - lr, m.T(:), '-', 'Color', col{k}, 'LineWidth', 1.6, 'DisplayName', sprintf('%s; lambda_0 %.3f nm', lab{k}, lr));
    u = m.field_energy_density_1D(:); u = u / max(u);
    plot(ax2, m.field_x(:) * 1e6, u, '-', 'Color', col{k}, 'LineWidth', 1.6, 'DisplayName', sprintf('%s; FWHM %.2f um', strtok(lab{k}, ':'), m.fwhm_m * 1e6));
end
xlim(ax1, [-0.12 0.12]); xlabel(ax1, '\lambda - \lambda_0 (nm)'); ylabel(ax1, 'T (short-run, uncorrected)'); grid(ax1, 'on');
legend(ax1, 'Location', 'northeast', 'Interpreter', 'none', 'FontSize', 9); set(ax1, 'FontSize', 10);
xlim(ax2, [-35 35]); xlabel(ax2, 'x along the waveguide (\mum); \pi-shift at x = 0'); ylabel(ax2, 'energy density (norm.)'); grid(ax2, 'on');
legend(ax2, 'Location', 'northeast', 'Interpreter', 'none', 'FontSize', 9); set(ax2, 'FontSize', 10);
title(tl, 'TE \pi-shift overshoot Nt80x (SiN 350 nm, pitch 514 nm, 96 periods/side), 1619.4 nm: control vs r2pack posts', 'FontSize', 12);
exportgraphics(f, fullfile(R, 'comb_overshoot', 'r2pack_spectrum_mode.png'), 'Resolution', 150); savefig(f, fullfile(R, 'comb_overshoot', 'r2pack_spectrum_mode.fig'));
