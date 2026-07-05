% plot_cavity_width_ladder.m — rect-cavity width optimum (job 117486 + round-2
% points from job 117434). Left: loss vs cavity width for the anchored device
% (optimum ~1050 nm) and the wide device (optimum ~1250). Right: the new best
% all-rectangle design waterfall. Headless-safe.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
d_lad = fullfile(proj, 'results_from_athena', 'cavity_width_ladder', 'results');
d_cd  = fullfile(proj, 'results_from_athena', 'cavity_design_study', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'cavity_width_ladder');

    function m = met(fp)
        d = load(fp);
        [~, i] = min(abs(d.wl_nm - d.resonance_wavelength_nm));
        m = struct('loss', 1 - d.resonance_transmission - d.R(i), ...
                   'T', d.resonance_transmission, 'fw', d.fwhm_m * 1e6);
    end

% anchored device: control + round-2 rects + ladder rects
anc_w = [800 895.5 950 1050 1150 1250 1400];
anc_f = {fullfile(d_lad, 'result_N80_TM_avg_Ybox6p8_Zbox8p8.mat'), ...
         fullfile(d_cd,  'result_N80_TM_W896_Ybox6p8_Zbox8p8.mat'), ...
         fullfile(d_cd,  'result_N80_TM_W950_Ybox6p8_Zbox8p8.mat'), ...
         fullfile(d_lad, 'result_N80_TM_W1050_Ybox6p8_Zbox8p8.mat'), ...
         fullfile(d_lad, 'result_N80_TM_W1150_Ybox6p8_Zbox8p8.mat'), ...
         fullfile(d_lad, 'result_N80_TM_W1250_Ybox6p8_Zbox8p8.mat'), ...
         fullfile(d_lad, 'result_N80_TM_W1400_Ybox6p8_Zbox8p8.mat')};
anc = cellfun(@met, anc_f);

wide_w = [1000 1250];
wide_f = {fullfile(d_lad, 'result_N80_TM_avg_Wavg1000_C500_Ybox6p8_Zbox8p9.mat'), ...
          fullfile(d_lad, 'result_N80_TM_W1250_Wavg1000_C500_Ybox6p8_Zbox8p9.mat')};
wide = cellfun(@met, wide_f);

fig = figure('Visible', 'off', 'Position', [80 80 1200 470]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile; hold on; grid on;
plot(anc_w, [anc.loss], 'o-', 'Color', [0.19 0.45 0.72], 'LineWidth', 1.5, ...
    'DisplayName', 'anchored device (W800/corr400)');
plot(wide_w, [wide.loss], 's-', 'Color', [0.13 0.55 0.33], 'LineWidth', 1.5, ...
    'DisplayName', 'wide device (W1000/corr500)');
yline(anc(1).loss, 'k--', 'HandleVisibility', 'off');
xlabel('rect cavity width (nm)'); ylabel('resonant loss 1 - T - R');
title('Cavity-width optimum: \sim1.3\times the core width, then it reverses');
legend('Location', 'northwest');

nexttile; hold on; grid on;
wf = [anc(1).loss, anc(4).loss, wide(1).loss, wide(2).loss];
fw = [anc(1).fw,   anc(4).fw,   wide(1).fw,   wide(2).fw];
lb = {'anchored control', 'cavity 1050', 'W1000/corr500', 'W1000 + cavity 1250'};
b = bar(wf, 'FaceColor', 'flat');
b.CData = [0.35 0.35 0.35; 0.19 0.45 0.72; 0.19 0.45 0.72; 0.13 0.55 0.33];
for k = 1:4
    text(k, wf(k) + 0.004, sprintf('%.3f\nfwhm %.1f \\mum', wf(k), fw(k)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end
ylim([0 0.135]);
set(gca, 'XTick', 1:4, 'XTickLabel', lb, 'XTickLabelRotation', 15, 'TickLabelInterpreter', 'none');
ylabel('resonant loss 1 - T - R');
title('All-rectangle best design: loss 0.110 \rightarrow 0.044 (-60%) at fwhm +6%');

sgtitle(sprintf(['TM \\pi-shift — plain RECTANGULAR cavity widening (no curved shapes)\n' ...
    'jobs 117434 + 117486, converged box, all \\Delta\\lambda_{res} < 0.35 nm']));

exportgraphics(fig, fullfile(out_dir, 'cavity_width_ladder_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'cavity_width_ladder_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'cavity_width_ladder_summary.png'));
