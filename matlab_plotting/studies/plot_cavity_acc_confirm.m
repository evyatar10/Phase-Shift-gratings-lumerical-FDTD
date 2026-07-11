% plot_cavity_acc_confirm.m — accurate-mesh (dx~35 nm) final confirm (job 117784).
% Confirms the in-scope champion (rect cavity 1050 nm, -30% loss) and adjudicates
% the coarse-mesh tilt candidate: tilt survives together at -0.0007 (floor ~1e-4)
% -> real but tiny (-0.9% vs rect 1050), saturated by depth 150.
% Headless-safe.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
d_acc = fullfile(proj, 'results_from_athena', 'cavity_acc_confirm', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'cavity_acc_confirm');

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
         'result_N80_TM_W1050_cavtilt300_Ybox6p8_Zbox8p8.mat'};
lb = {'control (rect 800)', 'rect 1050', '+ tilt 150', '+ tilt 300'};
r = cellfun(@(f) met(fullfile(d_acc, f)), files);

floor_acc = 1e-4;   % accurate-mesh jitter floor (pillar precedent 2026-07-02)

fig = figure('Visible', 'off', 'Position', [80 80 720 470]);
hold on; grid on;
xb = 1:numel(r);
fill([1.5 4.5 4.5 1.5], r(2).loss + floor_acc * [-1 -1 1 1], [0.85 0.85 0.85], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.8, 'DisplayName', 'rect-1050 \pm accurate floor');
b = bar(xb, [r.loss], 'FaceColor', 'flat', 'HandleVisibility', 'off');
b.CData = [0.35 0.35 0.35; 0.19 0.45 0.72; repmat([0.30 0.55 0.35], 2, 1)];
yline(r(1).loss, 'k--', 'HandleVisibility', 'off');
for k = 1:numel(r)
    text(k, r(k).loss + 0.004, sprintf('%.4f\nfwhm %+.2f%%', r(k).loss, ...
        (r(k).fw / r(1).fw - 1) * 100), 'HorizontalAlignment', 'center', 'FontSize', 9);
end
ylim([0 0.135]);
set(gca, 'XTick', xb, 'XTickLabel', lb, 'TickLabelInterpreter', 'none');
ylabel('resonant loss 1 - T - R');
legend('Location', 'southeast');
title(sprintf(['TM \\pi-shift — ACCURATE-mesh confirm (dx\\approx35 nm), job 117784\n' ...
    'rect 1050: -29.9%% loss; tilt tiny (-0.9%%); \\lambda_{res} %.1f nm, ' ...
    'control T = %.3f'], r(1).lam, r(1).T));

exportgraphics(fig, fullfile(out_dir, 'cavity_acc_confirm_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'cavity_acc_confirm_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'cavity_acc_confirm_summary.png'));
