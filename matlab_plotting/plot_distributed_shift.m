% plot_distributed_shift.m — distributed pi-shift theory test (job 117530).
% The pi phase slip is spread over N inner gaps (uniform / Gaussian density)
% with total round-trip phase conserved; theory predicted radiated loss falls
% with spread length. Result: FALSIFIED — every distributed variant is worse.
% Headless-safe.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
d_dps = fullfile(proj, 'results_from_athena', 'distributed_shift_study', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'distributed_shift_study');

    function m = met(fp)
        d = load(fp);
        [~, i] = min(abs(d.wl_nm - d.resonance_wavelength_nm));
        m = struct('loss', 1 - d.resonance_transmission - d.R(i), ...
                   'T', d.resonance_transmission, 'fw', d.fwhm_m * 1e6, ...
                   'lam', d.resonance_wavelength_nm);
    end

files = {'result_N80_TM_avg_Ybox6p8_Zbox8p8.mat', ...
         'result_N80_TM_avg_dsh2Sm65sm32_Ybox6p8_Zbox8p8.mat', ...
         'result_N80_TM_avg_dsh2Sm119sm60_Ybox6p8_Zbox8p8.mat', ...
         'result_N80_TM_avg_dsh4Sm119sm53_Ybox6p8_Zbox8p8.mat', ...
         'result_N80_TM_avg_dsh8Sm119sm33_Ybox6p8_Zbox8p8.mat', ...
         'result_N80_TM_avg_dsh8Sm119sm15_Ybox6p8_Zbox8p8.mat'};
lb = {'lumped (control)', 'N=2 half-spread', 'N=2 full', ...
      'N=4 Gaussian', 'N=8 Gaussian', 'N=8 uniform'};
r = cellfun(@(f) met(fullfile(d_dps, f)), files);

fig = figure('Visible', 'off', 'Position', [80 80 1150 460]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile; hold on; grid on;
b = bar([r.loss], 'FaceColor', 'flat');
b.CData = [0.35 0.35 0.35; repmat([0.72 0.27 0.19], 5, 1)];
yline(r(1).loss, 'k--', 'HandleVisibility', 'off');
for k = 1:numel(r)
    text(k, r(k).loss + 0.004, sprintf('%.3f', r(k).loss), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end
ylim([0 0.175]);
set(gca, 'XTick', 1:numel(r), 'XTickLabel', lb, 'XTickLabelRotation', 20, ...
    'TickLabelInterpreter', 'none');
ylabel('resonant loss 1 - T - R');
title('Spreading the \pi slip RAISES loss (+21% .. +39%)');

nexttile; hold on; grid on;
dl = ([r(2:end).loss] / r(1).loss - 1) * 100;
df = ([r(2:end).fw] / r(1).fw - 1) * 100;
cols = lines(5);
for k = 1:5
    plot(df(k), dl(k), 'o', 'MarkerSize', 9, 'MarkerFaceColor', cols(k,:), ...
        'Color', cols(k,:), 'DisplayName', lb{k+1});
end
yline(0, 'k--', 'HandleVisibility', 'off'); xline(0, 'k--', 'HandleVisibility', 'off');
xlabel('spatial mode width change (%)'); ylabel('loss change vs control (%)');
title('No trade-off either: mode also widens');
legend('Location', 'southeast', 'FontSize', 8);

sgtitle(sprintf(['TM \\pi-shift — DISTRIBUTED phase-slip test (theory falsified)\n' ...
    'job 117530, converged box, \\lambda_{res} %.1f nm, control T = %.3f'], ...
    r(1).lam, r(1).T));

exportgraphics(fig, fullfile(out_dir, 'distributed_shift_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'distributed_shift_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'distributed_shift_summary.png'));
