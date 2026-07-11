% plot_pareto_stack_vs_apod.m — job 118293 (12 rows, accurate) + frontier
% anchors (job 118214): the loss-vs-mode-width Pareto comparison.
% Verdict: the defect-local (shift/stack) family dominates apodization up to
% ~+3% mode width; apodization wins at large widths; under an apodized
% envelope the local modules INVERT (same physical resource: interface
% matching). Headless-safe.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
d_par = fullfile(proj, 'results_from_athena', 'tm_pareto_stack_vs_apod', 'results');
d_frn = fullfile(proj, 'results_from_athena', 'tm_shift_frontier', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'tm_pareto_stack_vs_apod');

    function m = met(fp)
        d = load(fp);
        [~, i] = min(abs(d.wl_nm - d.resonance_wavelength_nm));
        m = struct('loss', 1 - d.resonance_transmission - d.R(i), ...
                   'T', d.resonance_transmission, 'fw', d.fwhm_m * 1e6);
    end

P = @(s) fullfile(d_par, sprintf('result_N80_%s_Ybox6p8_Zbox8p8.mat', s));
Fr = @(s) fullfile(d_frn, sprintf('result_N80_%s_Ybox6p8_Zbox8p8.mat', s));
ctrl = met(P('TM_avg'));
X = @(r) (r.fw / ctrl.fw - 1) * 100;
blue = [0.19 0.45 0.72]; red = [0.85 0.33 0.10]; grn = [0.30 0.55 0.35];
prp = [0.49 0.18 0.56];

fig = figure('Visible', 'off', 'Position', [60 60 900 660], 'Color', 'w');
hold on; grid on;
fill([-1 1 1 -1], [0.01 0.01 0.13 0.13], [0.92 0.96 0.92], 'EdgeColor', 'none', ...
    'DisplayName', 'user bound |\Delta fwhm| \leq 1%');

% apodization ladder
ap = arrayfun(@(n) met(P(sprintf('A%d_TM_avg', n))), [5 10 20]);
apx = [0 arrayfun(X, ap)]; apy = [ctrl.loss ap.loss];
plot(apx, apy, 'o-', 'Color', red, 'MarkerFaceColor', red, 'LineWidth', 1.4, ...
    'DisplayName', 'apodization ladder (n = 0, 5, 10, 20)');

% defect-local (shift family) frontier: stack + frontier points
fr_keys = {'TM_W1050_dsh2S40s20_ptw2W1040to980', 'TM_W1050_dsh2S50s20', ...
           'TM_W1050_dsh3S50s20', 'TM_W1050_dsh2S60s30', 'TM_W1050_dsh3S60s20'};
frx = []; fry = [];
st = met(P('TM_W1050_dsh2S40s20_ptw2W1040to980'));
frx(1) = X(st); fry(1) = st.loss;
for k = 2:numel(fr_keys)
    r = met(Fr(fr_keys{k})); frx(k) = X(r); fry(k) = r.loss;
end
r1050 = met(Fr('S35_TM_W1050_dsh1S35s35'));   % low-dose anchor
plot([0.62 X(r1050) frx], [0.0823 r1050.loss fry], 's-', 'Color', grn, ...
    'MarkerFaceColor', grn, 'LineWidth', 1.4, ...
    'DisplayName', 'defect-local family (rect-1050 \rightarrow shift frontier)');
plot(X(st), st.loss, 'p', 'MarkerSize', 17, 'MarkerFaceColor', grn, ...
    'MarkerEdgeColor', 'k', 'DisplayName', ...
    sprintf('stack (in-bound): loss %.4f, T %.3f', st.loss, st.T));

% combinations under apod-10
combo = {met(P('A10_TM_W1050_dsh2S40s20_ptw2W1040to980')), ...
         met(P('A10_TM_W1050_dsh2S40s20')), met(P('A10_TM_W1050'))};
cx = cellfun(X, combo); cy = cellfun(@(r) r.loss, combo);
plot(cx, cy, 'd', 'Color', prp, 'MarkerFaceColor', prp, 'MarkerSize', 8, ...
    'DisplayName', 'apod-10 + local modules (transfer INVERTS)');
a10 = met(P('A10_TM_avg'));
plot(X(a10), a10.loss, 'o', 'Color', red, 'MarkerFaceColor', 'w', ...
    'HandleVisibility', 'off');
text(X(a10) + 0.5, a10.loss, ' apod-10 alone', 'FontSize', 8);
text(cx(1) + 0.5, cy(1), ' apod-10 + full stack (beats pure apod at this width)', 'FontSize', 8);

plot(0, ctrl.loss, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', [0.4 0.4 0.4], ...
    'DisplayName', 'W800 baseline');
set(gca, 'YScale', 'log');
xlabel('\Delta fwhm_m vs W800 baseline (%)');
ylabel('resonant loss 1 - T - R  (log)');
legend('Location', 'northeast', 'FontSize', 8);
xlim([-2 52]); ylim([0.012 0.13]);
title(sprintf(['TM \\pi-shift W800/corr400/pitch516.83/h350, N=80 — loss vs mode width ' ...
    '(jobs 118293 + 118214, accurate)\ndefect-local family dominates to ~+3%% width; ' ...
    'apodization wins only at large widths; \\lambda_{res} ~1556.6 nm']));

exportgraphics(fig, fullfile(out_dir, 'pareto_stack_vs_apod.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'pareto_stack_vs_apod.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'pareto_stack_vs_apod.png'));
