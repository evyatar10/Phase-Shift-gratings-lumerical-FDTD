% plot_barrel_followup.m — barrel-cavity confirmation study (job 117054).
% Left: loss vs barrel depth (optimization + accurate mesh series).
% Right: the lever "waterfall" — control / barrel / width / width+barrel,
% all optimization mesh, same converged box, fwhm annotated. Headless-safe.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
d_bf = fullfile(proj, 'results_from_athena', 'barrel_followup',   'results');
d_is = fullfile(proj, 'results_from_athena', 'inner_shape_study', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'barrel_followup');

    function m = met(fp)
        d = load(fp);
        [~, i] = min(abs(d.wl_nm - d.resonance_wavelength_nm));
        m = struct('loss', 1 - d.resonance_transmission - d.R(i), ...
                   'T', d.resonance_transmission, 'fw', d.fwhm_m * 1e6, ...
                   'lam', d.resonance_wavelength_nm);
    end

% Depth ladders
opt_d = [0 75 150 225 400];
opt_f = {fullfile(d_is, 'result_N80_TM_avg_Ybox6p8_Zbox8p8.mat'), ...
         fullfile(d_is, 'result_N80_TM_avg_cavbarr75_Ybox6p8_Zbox8p8.mat'), ...
         fullfile(d_is, 'result_N80_TM_avg_cavbarr150_Ybox6p8_Zbox8p8.mat'), ...
         fullfile(d_bf, 'result_N80_TM_avg_cavbarr225_Ybox6p8_Zbox8p8.mat'), ...
         fullfile(d_bf, 'result_N80_TM_avg_cavbarr400_Ybox6p8_Zbox8p8.mat')};
acc_d = [0 150 167.5 300];
acc_f = {fullfile(d_bf, 'result_N80_TM_avg_Ybox6p8_Zbox8p8.mat'), ...
         fullfile(d_bf, 'result_N80_TM_avg_cavbarr150_Ybox6p8_Zbox8p8.mat'), ...
         fullfile(d_bf, 'result_N80_TM_avg_cavbarr168_Ybox6p8_Zbox8p8.mat'), ...
         fullfile(d_bf, 'result_N80_TM_avg_cavbarr300_Ybox6p8_Zbox8p8.mat')};
opt_loss = cellfun(@(f) getfield(met(f), 'loss'), opt_f); %#ok<GFLD>
acc_loss = cellfun(@(f) getfield(met(f), 'loss'), acc_f); %#ok<GFLD>

fig = figure('Visible', 'off', 'Position', [80 80 1200 470]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile; hold on; grid on;
plot(opt_d, opt_loss, 'o-', 'Color', [0.19 0.45 0.72], 'LineWidth', 1.4, ...
    'DisplayName', 'dx = 50 nm (optimization)');
plot(acc_d, acc_loss, 's-', 'Color', [0.13 0.55 0.33], 'LineWidth', 1.4, ...
    'DisplayName', 'dx \approx 35 nm (accurate)');
xlabel('barrel bulge depth (nm)'); ylabel('resonant loss 1 - T - R');
title('Loss vs cavity-bulge depth (W 800 / corr 400)');
legend('Location', 'northeast');

% Waterfall (all optimization mesh)
wf_lab = {'anchored control', 'barrel 400', 'W1000/corr500', 'W1000 + barrel 190'};
wf_f = {opt_f{1}, opt_f{5}, ...
        fullfile(d_bf, 'result_N80_TM_avg_Wavg1000_C500_Ybox6p8_Zbox8p9.mat'), ...
        fullfile(d_bf, 'result_N80_TM_avg_Wavg1000_C500_cavbarr190_Ybox6p8_Zbox8p9.mat')};
wf = cellfun(@met, wf_f);

nexttile; hold on; grid on;
b = bar([wf.loss], 'FaceColor', 'flat');
b.CData = [0.35 0.35 0.35; 0.19 0.45 0.72; 0.19 0.45 0.72; 0.13 0.55 0.33];
for k = 1:4
    text(k, wf(k).loss + 0.004, sprintf('%.3f\nfwhm %.1f \\mum', wf(k).loss, wf(k).fw), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end
ylim([0, 0.135]);
set(gca, 'XTick', 1:4, 'XTickLabel', wf_lab, 'XTickLabelRotation', 15, ...
    'TickLabelInterpreter', 'none');
ylabel('resonant loss 1 - T - R');
title('Levers stack: loss 0.110 \rightarrow 0.059 (-46%) at fwhm +6%');

sgtitle(sprintf(['TM \\pi-shift (h 350 nm, N=80) — barrel-cavity confirmation + stacking\n' ...
    'accurate-mesh confirm: barrel 150 \\DeltaT=+0.022; depth trend mesh-consistent; \\Delta\\lambda < 0.2 nm']));

exportgraphics(fig, fullfile(out_dir, 'barrel_followup_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'barrel_followup_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'barrel_followup_summary.png'));
