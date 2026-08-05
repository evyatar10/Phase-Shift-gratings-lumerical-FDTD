% plot_trench_q3db_20um - T(N) and Q(N) at corr 325 nm: trench vs no-trench.
% Study: runners/metal_mirror/trench_q3db_20um.py | IGUM 47910/48458/48711/48973
% Date: 2026-08-03 | Purpose: max loaded Q at -3 dB peak T for a 20 um TM mode.
% Reads results_from_igum/trench_q3db_20um/results/result_*C325*.mat.

res_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), '..', ...
    'results_from_igum', 'trench_q3db_20um', 'results');
files = dir(fullfile(res_dir, 'result_*C325*.mat'));
assert(~isempty(files), 'no C325 results found in %s', res_dir);

N = []; T = []; Q = []; tr = logical([]);
for k = 1:numel(files)
    m = load(fullfile(res_dir, files(k).name));
    N(end+1)  = double(m.n_periods_each_side);          %#ok<*SAGROW>
    T(end+1)  = m.resonance_transmission;
    Q(end+1)  = m.resonance_wavelength_nm / abs(m.spectral_fwhm_nm);
    tr(end+1) = contains(files(k).name, 'scRECT');
    % Q needs >=10 sample points across the linewidth (5 pm spacing) to be
    % trusted; under-resolved points stay in the T panel only.
    if abs(m.spectral_fwhm_nm) < 0.050, Q(end) = NaN; end
end

% Operating points (closest integer-N device to T = 0.5 per arm)
best = struct();
for arm = [false true]
    i = find(tr == arm); [~, j] = min(abs(T(i) - 0.5)); b = i(j);
    if arm, best.tr = b; else, best.ct = b; end
end

fig = figure('Position', [100 100 760 640]);
ax1 = subplot(2, 1, 1); hold(ax1, 'on'); grid(ax1, 'on');
ax2 = subplot(2, 1, 2); hold(ax2, 'on'); grid(ax2, 'on');
cols = {[0.00 0.45 0.74], [0.85 0.33 0.10]};   % ctrl, trench
for arm = [false true]
    i = find(tr == arm); [~, s] = sort(N(i)); i = i(s);
    c = cols{arm + 1};
    nm = ternary(arm, 'air trench (full-z)', 'no trench');
    plot(ax1, N(i), T(i), 'o-', 'Color', c, 'MarkerFaceColor', c, ...
        'DisplayName', nm);
    plot(ax2, N(i), Q(i), 'o-', 'Color', c, 'MarkerFaceColor', c, ...
        'DisplayName', nm);
end
yline(ax1, 0.5, 'k--', 'T = 0.5 (-3 dB)', 'HandleVisibility', 'off');
for b = [best.ct best.tr]
    plot(ax1, N(b), T(b), 'kp', 'MarkerSize', 13, 'HandleVisibility', 'off');
    plot(ax2, N(b), Q(b), 'kp', 'MarkerSize', 13, 'HandleVisibility', 'off');
    text(ax2, N(b), Q(b), sprintf('  N=%d, Q=%.0f', N(b), Q(b)), ...
        'VerticalAlignment', 'top');
end
ylabel(ax1, 'Peak transmission');
ylabel(ax2, 'Loaded Q'); xlabel(ax2, 'N periods per side');
set(ax1, 'XTickLabel', []);
legend(ax1, 'Location', 'northeast');
title(ax1, sprintf(['\\pi-shift TM, 20 \\mum mode (corr 325 nm, h350, ' ...
    'p516.83) @ -3 dB: Q %.0f vs %.0f'], Q(best.ct), Q(best.tr)));

out = fullfile(res_dir, '..', 'trench_q3db_20um_T_Q');
savefig(fig, [out '.fig']);
exportgraphics(fig, [out '.png'], 'Resolution', 150);
fprintf('saved %s.fig/.png\n', out);

% ── Final two devices: T(lambda) in dB + field-envelope overlays ─────────────
ct = load(fullfile(res_dir, 'result_N165_TM_avg_C325_Ybox8p0_Zbox8p8.mat'));
tn = load(fullfile(res_dir, ['result_N170_TM_avg_C325_Ybox8p0_Zbox8p8_' ...
    'scRECT_L178000xW800_X0_Y1800_pair_hole_H12000.mat']));
q_ct = ct.resonance_wavelength_nm / abs(ct.spectral_fwhm_nm);
q_tn = tn.resonance_wavelength_nm / abs(tn.spectral_fwhm_nm);
lab = {sprintf(['no trench, N=165:  {\\bf Q = %.0f}   (\\lambda_{res} ' ...
           '%.2f nm, peak %.2f dB, FWHM %.2f \\mum)'], q_ct, ...
           ct.resonance_wavelength_nm, 10*log10(ct.resonance_transmission), ...
           ct.fwhm_m * 1e6), ...
       sprintf(['air trench, N=170:  {\\bf Q = %.0f}   (\\lambda_{res} ' ...
           '%.2f nm, peak %.2f dB, FWHM %.2f \\mum)'], q_tn, ...
           tn.resonance_wavelength_nm, 10*log10(tn.resonance_transmission), ...
           tn.fwhm_m * 1e6)};

fig2 = figure('Position', [100 100 820 480]); hold on; grid on;
plot(ct.wl_nm, 10*log10(ct.T), '-', 'Color', cols{1}, 'DisplayName', lab{1});
plot(tn.wl_nm, 10*log10(tn.T), '-', 'Color', cols{2}, 'DisplayName', lab{2});
yline(-3, 'k--', '-3 dB', 'HandleVisibility', 'off');
xlabel('Wavelength [nm]'); ylabel('Transmission [dB]');
xlim([min(ct.wl_nm) max(ct.wl_nm)]); ylim([-45 2]);
legend('Location', 'south');
title({'\pi-shift TM, 20 \mum mode @ -3 dB', ...
    'corrugation 325 nm, height 350 nm, pitch 516.83 nm'});
out2 = fullfile(res_dir, '..', 'trench_q3db_20um_final_T_dB');
savefig(fig2, [out2 '.fig']);
exportgraphics(fig2, [out2 '.png'], 'Resolution', 150);
fprintf('saved %s.fig/.png\n', out2);

fig3 = figure('Position', [100 100 820 480]); hold on; grid on;
for s = {{ct, cols{1}, 'no trench, N=165'}, {tn, cols{2}, 'air trench, N=170'}}
    d = s{1}{1};
    env = d.field_envelope_1D(:) / max(d.field_envelope_1D);
    plot(d.field_x(:) * 1e6, env, '-', 'Color', s{1}{2}, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('%s  (FWHM %.2f \\mum)', s{1}{3}, d.fwhm_m * 1e6));
end
yline(0.5, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
xlabel('x [\mum]'); ylabel('Normalized energy envelope');
legend('Location', 'northeast');
title({'\pi-shift TM, 20 \mum mode @ -3 dB: mode envelopes', ...
    'corrugation 325 nm, height 350 nm, pitch 516.83 nm'});
out3 = fullfile(res_dir, '..', 'trench_q3db_20um_final_envelopes');
savefig(fig3, [out3 '.fig']);
exportgraphics(fig3, [out3 '.png'], 'Resolution', 150);
fprintf('saved %s.fig/.png\n', out3);

function y = ternary(c, a, b)
if c, y = a; else, y = b; end
end
