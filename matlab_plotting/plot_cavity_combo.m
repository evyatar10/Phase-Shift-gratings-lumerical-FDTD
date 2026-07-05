% plot_cavity_combo.m — shape-on-top-of-the-optimal-cavity test (job 117553).
% Question: does any cavity SHAPE (barrel / tri7 / tilt) add anything on top of
% the optimally-sized 1050 nm rectangle? Answer: no — the optimum is scalar.
% Headless-safe.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
d_cmb = fullfile(proj, 'results_from_athena', 'cavity_combo_study', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'cavity_combo_study');

    function m = met(fp)
        d = load(fp);
        [~, i] = min(abs(d.wl_nm - d.resonance_wavelength_nm));
        m = struct('loss', 1 - d.resonance_transmission - d.R(i), ...
                   'T', d.resonance_transmission, 'fw', d.fwhm_m * 1e6, ...
                   'lam', d.resonance_wavelength_nm);
    end

files = {'result_N80_TM_avg_Ybox6p8_Zbox8p8.mat', ...
         'result_N80_TM_W1050_Ybox6p8_Zbox8p8.mat', ...
         'result_N80_TM_W1050_cavtilt150_Ybox6p8_Zbox8p8.mat', ...
         'result_N80_TM_W1050_cavtilt300_Ybox6p8_Zbox8p8.mat', ...
         'result_N80_TM_W1050_cavbarr75_Ybox6p8_Zbox8p8.mat', ...
         'result_N80_TM_W1050_cavbarr150_Ybox6p8_Zbox8p8.mat', ...
         'result_N80_TM_W1050_cavtri7191_Ybox6p8_Zbox8p8.mat'};
lb = {'control (rect 800)', 'rect 1050', '+ tilt 150', '+ tilt 300', ...
      '+ barrel 75', '+ barrel 150', '+ tri7 191'};
r = cellfun(@(f) met(fullfile(d_cmb, f)), files);

jitter = 0.002;   % dx=50 nm coarse-mesh jitter floor (measured 2026-07-02)

fig = figure('Visible', 'off', 'Position', [80 80 900 470]);
hold on; grid on;
xb = 1:numel(r);
fill([1.5 7.5 7.5 1.5], r(2).loss + jitter * [-1 -1 1 1], [0.85 0.85 0.85], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'rect-1050 \pm jitter floor');
b = bar(xb, [r.loss], 'FaceColor', 'flat', 'HandleVisibility', 'off');
b.CData = [0.35 0.35 0.35; 0.19 0.45 0.72; repmat([0.62 0.62 0.75], 2, 1); ...
           repmat([0.72 0.27 0.19], 3, 1)];
yline(r(1).loss, 'k--', 'HandleVisibility', 'off');
for k = 1:numel(r)
    text(k, r(k).loss + 0.003, sprintf('%.4f\nfwhm %+.1f%%', r(k).loss, ...
        (r(k).fw / r(1).fw - 1) * 100), 'HorizontalAlignment', 'center', 'FontSize', 8.5);
end
ylim([0 0.125]);
set(gca, 'XTick', xb, 'XTickLabel', lb, 'XTickLabelRotation', 18, ...
    'TickLabelInterpreter', 'none');
ylabel('resonant loss 1 - T - R');
legend('Location', 'southeast');
title(sprintf(['TM \\pi-shift — shapes ON TOP of the optimal 1050 nm cavity: optimum is SCALAR\n' ...
    'job 117553, converged box, \\lambda_{res} %.1f nm, control T = %.3f; ' ...
    'extra area hurts, tilt within noise'], r(1).lam, r(1).T));

exportgraphics(fig, fullfile(out_dir, 'cavity_combo_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'cavity_combo_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'cavity_combo_summary.png'));
