% plot_anti_moment_cavity.m — anti-moment families, accurate mesh (job 117814).
% Family B: fine cavity-width ladder around the 1050 champion (+1052 jitter partner).
% Family A: zero-net-area inner-tooth see-saw on the 1050 base (wide tooth +-1 =
% 1000+delta, tooth +-2 = 1000-delta). OUTCOME: B flat (1050 optimal); A shows a
% real signed, saturating trend — +delta helps (loss 0.0823 -> 0.0810), interference
% cancellation of the residual cavity-local radiating moment. Headless-safe.
% Four panels: (a) geometry schematic, (b) width ladder, (c) see-saw dose-response,
% (d) resonance spectra.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
d_res = fullfile(proj, 'results_from_athena', 'anti_moment_cavity', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'anti_moment_cavity');

    function m = met(fp)
        d = load(fp);
        [~, i] = min(abs(d.wl_nm - d.resonance_wavelength_nm));
        m = struct('loss', 1 - d.resonance_transmission - d.R(i), ...
                   'T', d.resonance_transmission, 'fw', d.fwhm_m * 1e6, ...
                   'lam', d.resonance_wavelength_nm, 'wl', d.wl_nm(:), 'Ts', d.T(:));
    end

f_ctrl = met(fullfile(d_res, 'result_N80_TM_avg_Ybox6p8_Zbox8p8.mat'));

% Family B: cavity-width ladder (1052 = sub-mesh jitter partner of 1050)
wB = [1000 1025 1050 1052 1075 1100];
rB = arrayfun(@(w) met(fullfile(d_res, sprintf( ...
    'result_N80_TM_W%d_Ybox6p8_Zbox8p8.mat', w))), wB);
champ = rB(wB == 1050);

% Family A: see-saw pair on W1050 — delta>0 => tooth1=1000+d, tooth2=1000-d
dA = [-30 -20 -10 10 20 30];
fA = @(d) fullfile(d_res, sprintf( ...
    'result_N80_TM_W1050_ptw2W%dto%d_Ybox6p8_Zbox8p8.mat', 1000 + d, 1000 - d));
rA = arrayfun(@(d) met(fA(d)), dA);
best = met(fA(20));

floor_acc = abs(champ.loss - rB(wB == 1052).loss);   % in-study jitter = 2e-4

fig = figure('Visible', 'off', 'Position', [40 40 1240 880], 'Color', 'w');
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% ===== (a) geometry schematic (top view; x horizontal) =====
nexttile; hold on; axis equal;
pitch = 0.51683; hp = pitch / 2;
Wn = 0.600; Ww = 1.000; Wc = 1.050;
gray = [0.78 0.82 0.87]; hi = [0.85 0.33 0.10]; lo = [0.00 0.45 0.74];
draw = @(x0, x1, w, c) fill([x0 x1 x1 x0], [-w -w w w] / 2, c, ...
    'EdgeColor', [0.25 0.25 0.25], 'LineWidth', 0.4);
% cavity block (centered), length ~ 2*hp for the schematic
draw(-hp, hp, Wc, gray);
% right arm: narrow, tooth1(+30), narrow, tooth2(-30), narrow, tooth3(nom)
xr = hp;
draw(xr, xr + hp, Wn, gray); xr = xr + hp;
draw(xr, xr + hp, Ww + 0.030, hi); xr = xr + hp;      % tooth 1 wider
draw(xr, xr + hp, Wn, gray); xr = xr + hp;
draw(xr, xr + hp, Ww - 0.030, lo); xr = xr + hp;      % tooth 2 narrower
draw(xr, xr + hp, Wn, gray); xr = xr + hp;
draw(xr, xr + hp, Ww, gray);
% left arm mirror
xl = -hp;
draw(xl - hp, xl, Wn, gray); xl = xl - hp;
draw(xl - hp, xl, Ww + 0.030, hi); xl = xl - hp;
draw(xl - hp, xl, Wn, gray); xl = xl - hp;
draw(xl - hp, xl, Ww - 0.030, lo); xl = xl - hp;
draw(xl - hp, xl, Wn, gray); xl = xl - hp;
draw(xl - hp, xl, Ww, gray);
plot([0 0], [-Wc/2 Wc/2], 'k-', 'LineWidth', 1.5);
text(0, 0.72, '\pi-shift cavity 1050', 'HorizontalAlignment', 'center', 'FontSize', 9);
text(hp + hp/2, (Ww + 0.030) / 2 + 0.14, 'tooth 1: +\delta', 'Color', hi, ...
    'FontSize', 9, 'HorizontalAlignment', 'center');
text(3 * hp + hp/2, (Ww - 0.030) / 2 + 0.30, 'tooth 2: -\delta', 'Color', lo, ...
    'FontSize', 9, 'HorizontalAlignment', 'center');
xlim([-1.7 1.7]); ylim([-0.85 0.95]);
xlabel('x (\mum) — propagation'); ylabel('y (\mum)');
title('(a) The see-saw: inner tooth 1 wider, tooth 2 narrower (zero net area)');

% ===== (b) width ladder =====
nexttile; hold on; grid on;
fill([995 1105 1105 995], champ.loss + floor_acc * [-1 -1 1 1], [0.85 0.85 0.88], ...
    'EdgeColor', 'none', 'DisplayName', 'champion \pm floor');
plot(wB, [rB.loss], 'o-', 'Color', lo, 'MarkerFaceColor', lo, ...
    'DisplayName', 'rect cavity ladder');
plot(1050, champ.loss, 'p', 'MarkerSize', 13, 'MarkerFaceColor', hi, ...
    'MarkerEdgeColor', 'k', 'DisplayName', 'champion 1050');
xlabel('cavity width (nm)'); ylabel('resonant loss  1 - T - R');
legend('Location', 'north', 'FontSize', 8); xlim([990 1110]);
title('(b) Width ladder: flat plateau — 1050 already optimal');

% ===== (c) see-saw dose-response =====
nexttile; hold on; grid on;
yline(0, '-', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
fill([-33 33 33 -33], 1e3 * floor_acc * [-1 -1 1 1], [0.85 0.85 0.88], ...
    'EdgeColor', 'none', 'DisplayName', '\pm numerical floor');
dl = ([rA.loss] - champ.loss) * 1e3;
plot(dA, dl, 'o-', 'Color', hi, 'LineWidth', 1.5, 'MarkerFaceColor', 'w', ...
    'DisplayName', 'see-saw 1000\pm\delta');
plot(20, ([best.loss] - champ.loss) * 1e3, 'p', 'MarkerSize', 13, ...
    'MarkerFaceColor', [0.30 0.69 0.29], 'MarkerEdgeColor', 'k', 'DisplayName', 'best \delta=+20');
text(18, -1.6, '+\delta: interference', 'Color', [0.2 0.5 0.2], 'FontSize', 9, ...
    'HorizontalAlignment', 'center');
text(-18, 3.2, 'wrong sign hurts', 'Color', [0.6 0 0], 'FontSize', 9, ...
    'HorizontalAlignment', 'center');
xlabel('see-saw \delta (nm): tooth 1 = 1000+\delta,  tooth 2 = 1000-\delta');
ylabel('\Deltaloss vs rect-1050  (\times10^{-3})');
legend('Location', 'northwest', 'FontSize', 8); xlim([-33 33]);
title('(c) See-saw: antisymmetric & saturating = interference cancellation');

% ===== (d) resonance spectra =====
nexttile; hold on; grid on;
plot(f_ctrl.wl, f_ctrl.Ts, 'LineWidth', 1.0, 'Color', [0.5 0.5 0.5]);
plot(champ.wl, champ.Ts, 'LineWidth', 1.3, 'Color', lo);
plot(best.wl, best.Ts, 'LineWidth', 1.3, 'Color', hi);
xlim([1553 1559]);
xlabel('wavelength (nm)'); ylabel('transmission T');
legend({sprintf('control W800  (T=%.3f, loss %.3f)', f_ctrl.T, f_ctrl.loss), ...
        sprintf('rect-1050  (T=%.3f, loss %.3f)', champ.T, champ.loss), ...
        sprintf('+see-saw \\delta=20  (T=%.3f, loss %.3f)', best.T, best.loss)}, ...
       'Location', 'northeast', 'FontSize', 8);
title('(d) Resonance peak T: 0.878 \rightarrow 0.917 \rightarrow 0.918');

title(tl, sprintf(['TM \\pi-shift, accurate mesh.   Champion = rect-1050 + see-saw \\delta+20:  ' ...
    'loss %.4f\\rightarrow%.4f (-31%% vs control), T %.3f\\rightarrow%.3f, fwhm +%.1f%%'], ...
    f_ctrl.loss, best.loss, f_ctrl.T, best.T, (best.fw / f_ctrl.fw - 1) * 100), ...
    'FontSize', 11, 'Interpreter', 'tex');

exportgraphics(fig, fullfile(out_dir, 'anti_moment_cavity_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'anti_moment_cavity_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'anti_moment_cavity_summary.png'));
