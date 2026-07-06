% plot_shift_frontier.m — job 118214 (20 rows, accurate mesh).
% The measured loss-vs-mode-width frontier of the inner gap-shift family:
% ~ -1e-3 loss per +0.1% fwhm, common slope for singles/pairs/triples/length.
% Best in-bound: W1050 + pair[+20,+20] + see-saw(1040,980) -> 0.0545, T 0.9449.
% Headless-safe.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
d_res = fullfile(proj, 'results_from_athena', 'tm_shift_frontier', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'tm_shift_frontier');

    function m = met(fp)
        d = load(fp);
        [~, i] = min(abs(d.wl_nm - d.resonance_wavelength_nm));
        m = struct('loss', 1 - d.resonance_transmission - d.R(i), ...
                   'T', d.resonance_transmission, 'fw', d.fwhm_m * 1e6);
    end

F = @(s) fullfile(d_res, sprintf('result_N80_%s_Ybox6p8_Zbox8p8.mat', s));
ctrl = met(F('TM_avg'));
blue = [0.19 0.45 0.72]; red = [0.85 0.33 0.10]; grn = [0.30 0.55 0.35];
prp = [0.49 0.18 0.56];

% groups: {label, marker, color, {keys}}
G = {
 'single (tooth-1)', 'o', blue, {'S35_TM_W1050_dsh1S35s35', 'S40_TM_W1050_dsh1S40s40', 'S50_TM_W1050_dsh1S50s50'};
 'pair (2 teeth)', 's', grn, {'TM_W1050_dsh2S30s15', 'TM_W1050_dsh2S40s20', 'TM_W1050_dsh2S50s25', 'TM_W1050_dsh2S50s20', 'TM_W1050_dsh2S50s30', 'TM_W1050_dsh2S60s30'};
 'triple (3 teeth)', '^', prp, {'TM_W1050_dsh3S45s15', 'TM_W1050_dsh3S50s20', 'TM_W1050_dsh3S60s20'};
 'pair + length det', 'v', red, {'D20p00_TM_W1050_dsh2S40s20', 'D-20p00_TM_W1050_dsh2S40s20'};
 'pair + see-saw', 'd', [0.93 0.69 0.13], {'TM_W1050_dsh2S40s20_ptw2W1020to980', 'TM_W1050_dsh2S40s20_ptw2W1040to980'};
};

fig = figure('Visible', 'off', 'Position', [60 60 860 640], 'Color', 'w');
hold on; grid on;
fill([-1 1 1 -1], [0.03 0.03 0.13 0.13], [0.92 0.96 0.92], 'EdgeColor', 'none', ...
    'DisplayName', 'in-bound (|\Delta fwhm| \leq 1%)');
for g = 1:size(G, 1)
    xs = []; ys = [];
    for k = 1:numel(G{g, 4})
        r = met(F(G{g, 4}{k}));
        xs(end+1) = (r.fw / ctrl.fw - 1) * 100; %#ok<*SAGROW>
        ys(end+1) = r.loss;
    end
    plot(xs, ys, G{g, 2}, 'Color', G{g, 3}, 'MarkerFaceColor', G{g, 3}, ...
        'MarkerSize', 7, 'DisplayName', G{g, 1});
end
b = met(F('TM_W1050_dsh2S40s20_ptw2W1040to980'));
plot((b.fw / ctrl.fw - 1) * 100, b.loss, 'p', 'MarkerSize', 17, ...
    'MarkerFaceColor', grn, 'MarkerEdgeColor', 'k', ...
    'DisplayName', sprintf('BEST in-bound: 0.0545 (T=%.3f)', b.T));
t3 = met(F('TM_W1050_dsh3S60s20'));
text((t3.fw / ctrl.fw - 1) * 100, t3.loss - 0.003, ...
    sprintf('  triple 20\\times3: %.4f (T=%.3f)', t3.loss, t3.T), 'FontSize', 9);
plot(0, ctrl.loss, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', [0.4 0.4 0.4], ...
    'DisplayName', 'W800 baseline');
r1050 = 0.0823;   % rect-1050 anchor (117814/117927)
plot(0.62, r1050, 'o', 'MarkerSize', 10, 'MarkerFaceColor', blue, ...
    'MarkerEdgeColor', 'k', 'DisplayName', 'rect-1050');
xline(1.0, ':', 'Color', red, 'HandleVisibility', 'off');
xlabel('\Delta fwhm_m vs W800 control (%)');
ylabel('resonant loss 1 - T - R');
legend('Location', 'northeast', 'FontSize', 9);
title(sprintf(['TM \\pi-shift W800/corr400/pitch516.83/h350, N=80 — gap-shift frontier ' ...
    '(job 118214, accurate)\ncommon tradeoff \\approx -1\\times10^{-3} loss per +0.1%% fwhm; ' ...
    '\\lambda_{res} 1556.6 nm']));
ylim([0.035 0.125]);

exportgraphics(fig, fullfile(out_dir, 'shift_frontier_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'shift_frontier_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'shift_frontier_summary.png'));
