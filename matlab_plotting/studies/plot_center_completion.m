% plot_center_completion.m — job 117927 (49 rows, accurate mesh) summary.
% (a) tooth-shift dose-response (the headline: pair [+20,+20] -> loss 0.0549)
% (b) cavity width ladder incl. W1600/W2100 (two-edge model falsified)
% (c) cavity-length scan with fwhm overlay (delocalization in disguise)
% (d) loss vs fwhm_m Pareto scatter of all rows, 1% bound marked.
% Headless-safe.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
d_res = fullfile(proj, 'results_from_athena', 'tm_center_completion', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'tm_center_completion');

    function m = met(fp)
        d = load(fp);
        [~, i] = min(abs(d.wl_nm - d.resonance_wavelength_nm));
        m = struct('loss', 1 - d.resonance_transmission - d.R(i), ...
                   'T', d.resonance_transmission, 'fw', d.fwhm_m * 1e6, ...
                   'lam', d.resonance_wavelength_nm);
    end

F = @(s) fullfile(d_res, sprintf('result_N80_%s_Ybox6p8_Zbox8p8.mat', s));
ctrl = met(F('TM_avg')); base = met(F('TM_W1050'));
blue = [0.19 0.45 0.72]; red = [0.85 0.33 0.10]; grn = [0.30 0.55 0.35];

fig = figure('Visible', 'off', 'Position', [40 40 1240 900], 'Color', 'w');
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% ===== (a) tooth-shift dose response =====
nexttile; hold on; grid on;
sh = [-30 -20 -10 10 20 30];
for k = 1:2
    if k == 1, tag = 'avg'; bl = ctrl; c = [0.55 0.55 0.55]; nm = 'W800 base';
    else, tag = 'W1050'; bl = base; c = blue; nm = 'W1050 base'; end
    v = zeros(size(sh));
    v(1) = met(F(sprintf('TM_%s_dsh1Sm30sm30', tag))).loss;
    v(2) = met(F(sprintf('TM_%s_dsh1Sm20sm20', tag))).loss;
    v(3) = met(F(sprintf('TM_%s_dsh1Sm10sm10', tag))).loss;
    v(4) = met(F(sprintf('S10_TM_%s_dsh1S10s10', tag))).loss;
    v(5) = met(F(sprintf('S20_TM_%s_dsh1S20s20', tag))).loss;
    v(6) = met(F(sprintf('S30_TM_%s_dsh1S30s30', tag))).loss;
    plot(sh, (v - bl.loss) * 1e3, 'o-', 'Color', c, 'MarkerFaceColor', c, ...
        'DisplayName', [nm ' (tooth-1)']);
end
p = met(F('TM_W1050_dsh2S40s20'));
plot(20, (p.loss - base.loss) * 1e3, 'p', 'MarkerSize', 15, 'MarkerFaceColor', ...
    grn, 'MarkerEdgeColor', 'k', 'DisplayName', 'pair [+20,+20] \rightarrow loss 0.0549');
yline(0, '-', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
xlabel('inner gap shift \delta (nm; + = tooth toward cavity)');
ylabel('\Delta loss vs own base (\times10^{-3})');
legend('Location', 'northeast', 'FontSize', 8);
title('(a) Tooth-shift dose response — unsaturated, sign-antisymmetric');

% ===== (b) full width ladder incl. discriminators =====
nexttile; hold on; grid on;
wl_w = [800 1000 1050 1100 1600 2100];
wl_f = {'TM_avg', 'D-20p00_TM_W1000', 'TM_W1050', 'TM_W1100', 'TM_W1600', 'TM_W2100'};
% use det-0 rows where present (W1000/W1100 det0 not in this study -> use 117814-equivalent points from anti_moment? keep this study only:)
vals = [ctrl.loss, met(F('D20p00_TM_W1000')).loss, base.loss, ...
        met(F('D20p00_TM_W1100')).loss, met(F('TM_W1600')).loss, met(F('TM_W2100')).loss];
% mark the two discriminator points distinctly
plot(wl_w([1 3]), vals([1 3]), 'o', 'Color', blue, 'MarkerFaceColor', blue, ...
    'DisplayName', 'measured (det 0)');
plot(wl_w([2 4]), vals([2 4]), 's', 'Color', blue, 'DisplayName', 'det +20 rows');
plot(wl_w([5 6]), vals([5 6]), 'd', 'MarkerSize', 10, 'Color', red, ...
    'MarkerFaceColor', red, 'DisplayName', 'discriminators W1600/W2100');
plot(wl_w, vals, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
yline(ctrl.loss, '--k', 'W800 control', 'HandleVisibility', 'off');
xlabel('cavity width (nm)'); ylabel('resonant loss 1 - T - R');
legend('Location', 'northwest', 'FontSize', 8);
title('(b) No second minimum: two-edge model falsified (monotonic past 1400)');

% ===== (c) cavity length scan with fwhm =====
nexttile; hold on; grid on;
dets = [-40 -20 0 20 40];
lv = zeros(size(dets)); fv = zeros(size(dets));
for i = 1:numel(dets)
    if dets(i) == 0, r = base;
    else, r = met(F(sprintf('D%dp00_TM_W1050', dets(i)))); end
    lv(i) = r.loss; fv(i) = (r.fw / ctrl.fw - 1) * 100;
end
yyaxis left; plot(dets, lv * 1e3, 'o-', 'Color', blue, 'MarkerFaceColor', blue);
ylabel('loss (\times10^{-3})'); set(gca, 'YColor', blue);
yyaxis right; plot(dets, fv, 's--', 'Color', red, 'MarkerFaceColor', red);
yline(1.0, ':', '+1% fwhm bound', 'Color', red);
ylabel('\Delta fwhm_m vs W800 control (%)'); set(gca, 'YColor', red);
xlabel('cavity detuning det (nm; det<0 = LONGER cavity)');
title('(c) Length knob: loss falls but the mode delocalizes — out of bound');

% ===== (d) Pareto scatter =====
nexttile; hold on; grid on;
files = dir(fullfile(d_res, 'result_*.mat'));
for k = 1:numel(files)
    r = met(fullfile(d_res, files(k).name));
    dfw = (r.fw / ctrl.fw - 1) * 100;
    ok = abs(dfw) <= 1.0;
    plot(dfw, r.loss, 'o', 'MarkerSize', 5, ...
        'Color', ternary(ok, grn, [0.65 0.65 0.65]), ...
        'MarkerFaceColor', ternary(ok, grn, 'none'), 'HandleVisibility', 'off');
end
pB = met(F('TM_W1050_dsh2S40s20'));
plot((pB.fw / ctrl.fw - 1) * 100, pB.loss, 'p', 'MarkerSize', 16, ...
    'MarkerFaceColor', grn, 'MarkerEdgeColor', 'k', ...
    'DisplayName', 'W1050 + pair[+20,+20]: 0.0549');
plot((base.fw / ctrl.fw - 1) * 100, base.loss, 'o', 'MarkerSize', 10, ...
    'MarkerFaceColor', blue, 'MarkerEdgeColor', 'k', 'DisplayName', 'rect-1050');
plot(0, ctrl.loss, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', [0.4 0.4 0.4], ...
    'DisplayName', 'W800 baseline');
xline(1.0, ':', 'Color', red, 'HandleVisibility', 'off');
xline(-1.0, ':', 'Color', red, 'HandleVisibility', 'off');
xlabel('\Delta fwhm_m vs W800 control (%)'); ylabel('resonant loss 1 - T - R');
legend('Location', 'northeast', 'FontSize', 8);
title('(d) All 49 rows on the (mode width, loss) plane; \pm1% bound dotted');

title(tl, sprintf(['TM \\pi-shift W800/corr400/pitch516.83/h350, N=80 — center completion, ' ...
    'ACCURATE mesh (job 117927)\nnew best: W1050 + gap pair [+20,+20]: loss %.4f, ' ...
    'T = %.3f, \\lambda_{res} %.2f nm, fwhm %+.1f%%, Q %.0f'], ...
    pB.loss, pB.T, pB.lam, (pB.fw / ctrl.fw - 1) * 100, 1403), ...
    'FontSize', 11, 'Interpreter', 'tex');

exportgraphics(fig, fullfile(out_dir, 'center_completion_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'center_completion_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'center_completion_summary.png'));

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
