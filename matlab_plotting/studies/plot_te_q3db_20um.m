% plot_te_q3db_20um - TE no-trench 20 um mode @ -3 dB: T(N)/Q(N), T(lambda) dB,
% and the mode envelope. Study: runners/sweeps/te_q3db_20um.py
% Jobs: Athena 128580/128581/128593/128730/128733 | 2026-08-05
% Trench arm absent by design: trench is a MEASURED TE null (trench_te_apod).

res_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), '..', ...
    'results_from_athena', 'te_q3db_20um', 'results');
files = dir(fullfile(res_dir, 'result_*C250*.mat'));
assert(~isempty(files), 'no corr-250 results in %s', res_dir);

N = []; T = []; Q = []; M = {};
for k = 1:numel(files)
    m = load(fullfile(res_dir, files(k).name));
    N(end+1) = double(m.n_periods_each_side);           %#ok<*SAGROW>
    T(end+1) = m.resonance_transmission;
    Q(end+1) = m.resonance_wavelength_nm / abs(m.spectral_fwhm_nm);
    M{end+1} = m;
end
[~, best] = min(abs(T - 0.5));
[Ns, si] = sort(N);

% ── Fig 1: T(N) and Q(N), crossing starred ──────────────────────────────────
fig1 = figure('Position', [100 100 720 600]);
ax1 = subplot(2,1,1); hold on; grid on;
ax2 = subplot(2,1,2); hold on; grid on;
c0 = [0.00 0.45 0.74];
plot(ax1, Ns, T(si), 'o-', 'Color', c0, 'MarkerFaceColor', c0);
yline(ax1, 0.5, 'k--', 'T = 0.5 (-3 dB)');
plot(ax1, N(best), T(best), 'kp', 'MarkerSize', 13);
plot(ax2, Ns, Q(si), 'o-', 'Color', c0, 'MarkerFaceColor', c0);
plot(ax2, N(best), Q(best), 'kp', 'MarkerSize', 13);
text(ax2, N(best), Q(best), sprintf('  N=%d, Q=%.0f', N(best), Q(best)), ...
    'VerticalAlignment', 'top');
ylabel(ax1, 'Peak transmission'); set(ax1, 'XTickLabel', []);
ylabel(ax2, 'Loaded Q'); xlabel(ax2, 'N periods per side');
title(ax1, {'\pi-shift TE, 20 \mum mode @ -3 dB, no trench', ...
    'corrugation 250 nm, height 350 nm, pitch 500 nm'});
out = fullfile(res_dir, '..', 'te_q3db_20um_T_Q');
savefig(fig1, [out '.fig']); exportgraphics(fig1, [out '.png'], 'Resolution', 150);
fprintf('saved %s.png\n', out);

% ── Fig 2: T(lambda) in dB, final device ────────────────────────────────────
mb = M{best};
qb = mb.resonance_wavelength_nm / abs(mb.spectral_fwhm_nm);
fig2 = figure('Position', [100 100 820 480]); hold on; grid on;
plot(mb.wl_nm, 10*log10(mb.T), '-', 'Color', c0, 'DisplayName', ...
    sprintf(['TE no trench, N=%d:  {\\bf Q = %.0f}   (\\lambda_{res} %.2f nm, ' ...
    'peak %.2f dB, FWHM %.2f \\mum)'], N(best), qb, ...
    mb.resonance_wavelength_nm, 10*log10(mb.resonance_transmission), ...
    mb.fwhm_m * 1e6));
yline(-3, 'k--', '-3 dB', 'HandleVisibility', 'off');
xlabel('Wavelength [nm]'); ylabel('Transmission [dB]');
xlim([min(mb.wl_nm) max(mb.wl_nm)]); ylim([-45 2]);
legend('Location', 'south');
title({'\pi-shift TE, 20 \mum mode @ -3 dB: final device', ...
    'corrugation 250 nm, height 350 nm, pitch 500 nm'});
out2 = fullfile(res_dir, '..', 'te_q3db_20um_final_T_dB');
savefig(fig2, [out2 '.fig']); exportgraphics(fig2, [out2 '.png'], 'Resolution', 150);
fprintf('saved %s.png\n', out2);

% ── Fig 3: mode envelope ────────────────────────────────────────────────────
fig3 = figure('Position', [100 100 820 480]); hold on; grid on;
env = mb.field_envelope_1D(:) / max(mb.field_envelope_1D);
plot(mb.field_x(:) * 1e6, env, '-', 'Color', c0, 'LineWidth', 1.5, ...
    'DisplayName', sprintf('TE no trench, N=%d  (FWHM %.2f \\mum)', ...
    N(best), mb.fwhm_m * 1e6));
yline(0.5, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
xlabel('x [\mum]'); ylabel('Normalized energy envelope');
legend('Location', 'northeast');
title({'\pi-shift TE, 20 \mum mode @ -3 dB: cavity-mode envelope', ...
    'corrugation 250 nm, height 350 nm, pitch 500 nm'});
out3 = fullfile(res_dir, '..', 'te_q3db_20um_final_envelope');
savefig(fig3, [out3 '.fig']); exportgraphics(fig3, [out3 '.png'], 'Resolution', 150);
fprintf('saved %s.png\n', out3);
