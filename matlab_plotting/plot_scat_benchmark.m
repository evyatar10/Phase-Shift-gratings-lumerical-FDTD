% Scatterer-program presentation set: 3 figures from MEASURED N=80 results.
% Study dir: results_from_athena/scat_e_validate/figures/  |  2026-07-21
% Jobs: stage E ladder 121239..121816, round 7 widths 121830, stage G apod 122367.
% Data: scat_benchmark_data.mat (extracted from the result .mat files by
% scratchpad extract_scat_benchmark.py; each var = [lam_nm T loss Q fwhm_um]).
%   Fig 1 benchmark: T + mode width, 3 device groups x (plain / +scatterer pair)
%   Fig 2 ladder:    T vs scatterer standoff -> converges to plain wide-cavity T
%   Fig 3 pair:      T bars, same scatterer pair on the three devices (sign flip)

res_dir = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'results_from_athena', 'scat_e_validate', 'figures');
d = load(fullfile(res_dir, 'scat_benchmark_data.mat'));

C_PLAIN = [0.55 0.55 0.55];          % gray = no scatterers
C_PAIR  = [0.20 0.45 0.85];          % blue = pair helps
C_HURT  = [0.80 0.25 0.25];          % red  = pair hurts
FS = 14;

T = @(v) v(2);

groups = {sprintf('regular\\newlinecavity 800 nm'), ...
          sprintf('wide cavity\\newline1050 nm'), ...
          sprintf('apodized\\newline10 periods')};
plain = [d.w800_ctrl; d.w1050; d.apod10_ctrl];
pair  = [d.pair700;   d.w1050_pair; d.apod10_pair];

%% ---- Fig 1: benchmark (T linear + spatial mode width; Q written as text) ----
fig = figure('Position', [80 80 1250 520], 'Color', 'w');
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, sprintf(['TM pi-shift grating, N=80/side, \\lambda\\approx1559 nm — ' ...
    'effect of the r=80 nm scatterer pair [0, 270]']), 'FontSize', FS+3);

nexttile; hold on
b = bar([plain(:,2) pair(:,2)], 'grouped');
b(1).FaceColor = C_PLAIN; b(2).FaceColor = C_PAIR;
for i = 1:3
    % T values inside the bars (horizontal, white, just below each bar top)
    text(b(1).XEndPoints(i), plain(i,2)-0.014, sprintf('%.3f', plain(i,2)), ...
        'HorizontalAlignment', 'center', 'FontSize', FS-2, ...
        'FontWeight', 'bold', 'Color', 'w');
    text(b(2).XEndPoints(i), pair(i,2)-0.014, sprintf('%.3f', pair(i,2)), ...
        'HorizontalAlignment', 'center', 'FontSize', FS-2, ...
        'FontWeight', 'bold', 'Color', 'w');
    dT = pair(i,2) - plain(i,2);
    if dT > 0, c_dT = C_PAIR*0.8; else, c_dT = C_HURT*0.85; end
    text(b(2).XEndPoints(i), max(plain(i,2), pair(i,2))+0.010, sprintf('%+.4f', dT), ...
        'HorizontalAlignment', 'center', 'FontSize', FS, 'FontWeight', 'bold', ...
        'Color', c_dT);
    text(i, 0.617, sprintf('Q %d / %d', round(plain(i,4)), round(pair(i,4))), ...
        'HorizontalAlignment', 'center', 'FontSize', FS-2, 'Color', [0.25 0.25 0.25], ...
        'BackgroundColor', 'w', 'Margin', 1);
end
set(gca, 'XTick', 1:3, 'XTickLabel', groups, 'FontSize', FS);
ylabel('resonance transmission T', 'FontSize', FS);
ylim([0.6 1.02]); grid on; box on
legend({'no scatterers', 'with scatterer pair'}, 'Location', 'northwest', 'FontSize', FS-1);

nexttile; hold on
b = bar([plain(:,5) pair(:,5)], 'grouped');
b(1).FaceColor = C_PLAIN; b(2).FaceColor = C_PAIR;
for i = 1:3
    text(b(1).XEndPoints(i), plain(i,5)-0.8, sprintf('%.1f', plain(i,5)), ...
        'HorizontalAlignment', 'center', 'FontSize', FS-2, ...
        'FontWeight', 'bold', 'Color', 'w');
    text(b(2).XEndPoints(i), pair(i,5)-0.8, sprintf('%.1f', pair(i,5)), ...
        'HorizontalAlignment', 'center', 'FontSize', FS-2, ...
        'FontWeight', 'bold', 'Color', 'w');
end
set(gca, 'XTick', 1:3, 'XTickLabel', groups, 'FontSize', FS);
ylabel('spatial mode FWHM (\mum)', 'FontSize', FS);
ylim([0 23]); grid on; box on

exportgraphics(fig, fullfile(res_dir, 'scat_benchmark_tm.png'), 'Resolution', 200);
set(fig, 'Visible', 'on'); savefig(fig, fullfile(res_dir, 'scat_benchmark_tm.fig'));

%% ---- Fig 2: standoff ladder — scatterers converge to the plain-widening value ----
y_nm = [700 650 600 580 500 480];
Tlad = [T(d.pair700) T(d.pair650) T(d.pair600) T(d.pair580) T(d.pair500) T(d.pair480)];

fig = figure('Position', [100 100 900 560], 'Color', 'w');
hold on
plot(y_nm, Tlad, '-o', 'Color', C_PAIR, 'MarkerFaceColor', C_PAIR, ...
    'LineWidth', 2.2, 'MarkerSize', 8);
text(497, 0.9268, 'scatterer touches cavity', 'FontSize', FS-1, ...
    'Color', C_PAIR*0.8, 'HorizontalAlignment', 'left');
set(gca, 'FontSize', FS, 'XTick', 500:50:700);
xlabel('scatterer pair y position (nm)', 'FontSize', FS);
ylabel('resonance transmission T', 'FontSize', FS);
title('TM N=80: transmission vs scatterer pair y position', 'FontSize', FS+1);
xlim([460 720]); ylim([0.88 0.932]); grid on; box on
% straight arrow from the label to the touching point (normalized fig coords)
ax = gca;
xn = @(x) ax.Position(1) + ax.Position(3)*(x - ax.XLim(1))/diff(ax.XLim);
yn = @(y) ax.Position(2) + ax.Position(4)*(y - ax.YLim(1))/diff(ax.YLim);
annotation('arrow', [xn(494) xn(482)], [yn(0.9264) yn(0.9246)], ...
    'Color', C_PAIR*0.8, 'LineWidth', 1.6, 'HeadLength', 9, 'HeadWidth', 9);

exportgraphics(fig, fullfile(res_dir, 'scat_ladder_tm.png'), 'Resolution', 200);
set(fig, 'Visible', 'on'); savefig(fig, fullfile(res_dir, 'scat_ladder_tm.fig'));

%% ---- Fig 3: same scatterer pair on three devices — T bars, sign-coded colors ----
fig = figure('Position', [120 120 900 560], 'Color', 'w');
hold on
b = bar([plain(:,2) pair(:,2)], 'grouped');
b(1).FaceColor = C_PLAIN;
b(2).FaceColor = 'flat';
b(2).CData = [C_PAIR; C_HURT; C_HURT];
for i = 1:3
    text(b(1).XEndPoints(i), plain(i,2)-0.006, sprintf('%.3f', plain(i,2)), ...
        'HorizontalAlignment', 'center', 'FontSize', FS-1, ...
        'FontWeight', 'bold', 'Color', 'w');
    text(b(2).XEndPoints(i), pair(i,2)-0.006, sprintf('%.3f', pair(i,2)), ...
        'HorizontalAlignment', 'center', 'FontSize', FS-1, ...
        'FontWeight', 'bold', 'Color', 'w');
    dT = pair(i,2) - plain(i,2);
    if dT > 0, c_dT = C_PAIR*0.8; else, c_dT = C_HURT*0.85; end
    text(b(2).XEndPoints(i), max(plain(i,2), pair(i,2))+0.005, sprintf('%+.4f', dT), ...
        'HorizontalAlignment', 'center', 'FontSize', FS+1, 'FontWeight', 'bold', ...
        'Color', c_dT);
end
set(gca, 'XTick', 1:3, 'XTickLabel', groups, 'FontSize', FS);
ylabel('resonance transmission T', 'FontSize', FS);
title(sprintf(['TM N=80: the same scatterer pair [0, 270] helps only the ' ...
    'regular cavity']), 'FontSize', FS+1);
ylim([0.85 1.005]); grid on; box on
% legend via dummy patches so gain/loss colors are both explained
hp = [patch(NaN, NaN, C_PLAIN), patch(NaN, NaN, C_PAIR), patch(NaN, NaN, C_HURT)];
legend(hp, {'no scatterers', 'scatterer pair T gain', 'scatterer pair T loss'}, ...
    'Location', 'northwest', 'FontSize', FS-1);

exportgraphics(fig, fullfile(res_dir, 'scat_signflip_tm.png'), 'Resolution', 200);
set(fig, 'Visible', 'on'); savefig(fig, fullfile(res_dir, 'scat_signflip_tm.fig'));

%% ---- Fig 4: far field (side), regular vs wide — grazing needles at ±11.5° ----
% scat_farfield_data.mat: P(ux) = side-monitor E2 integrated over uy, box 6.8,
% at resonance (stage-E ff files). Needle MEASURED at ux = ±0.980 both devices.
ffd = load(fullfile(res_dir, 'scat_farfield_data.mat'));
Pmax = max(ffd.w800_P);

fig = figure('Position', [100 100 950 540], 'Color', 'w');
hold on
for u = [-0.980 0.980]   % dashed markers at the needle directions
    xline(u, '--', 'Color', [0.85 0.55 0.10], 'LineWidth', 1.8);
end
% beyond |ux| = 1 (kx > n_clad*k0) the field is evanescent in the cladding
% (TIR / bound to the guide) — shaded gray
for s = [-1 1]
    fill(s*[1 1.05 1.05 1], [-25 -25 3 3], [0.90 0.90 0.90], 'EdgeColor', 'none');
end
h1 = plot(ffd.w800_ux, 10*log10(ffd.w800_P / Pmax), '-', ...
    'Color', [0.25 0.25 0.25], 'LineWidth', 1.8);
h2 = plot(ffd.w1050_ux, 10*log10(ffd.w1050_P / Pmax), '-', 'Color', C_PAIR, ...
    'LineWidth', 1.8);
text(-0.955, 1.5, '\theta \approx -11.5\circ', 'HorizontalAlignment', 'left', ...
    'FontSize', FS, 'FontWeight', 'bold', 'Color', [0.75 0.45 0.05]);
text(0.955, 1.5, '\theta \approx +11.5\circ', 'HorizontalAlignment', 'right', ...
    'FontSize', FS, 'FontWeight', 'bold', 'Color', [0.75 0.45 0.05]);
set(gca, 'FontSize', FS);
xlabel('direction cosine u_x', 'FontSize', FS);
ylabel('radiated power (dB, norm.)', 'FontSize', FS);
title('TM N=80 side far field, \lambda\approx1558.5 nm — grazing needles', ...
    'FontSize', FS+1);
xlim([-1.05 1.05]); ylim([-25 3]); grid on; box on
legend([h1 h2], {'regular cavity 800 nm', 'wide cavity 1050 nm'}, ...
    'Location', 'southeast', 'FontSize', FS-1);

exportgraphics(fig, fullfile(res_dir, 'scat_farfield_tm.png'), 'Resolution', 200);
set(fig, 'Visible', 'on'); savefig(fig, fullfile(res_dir, 'scat_farfield_tm.fig'));

%% ---- Fig 5: needle-matched comb (Λ = 551 nm) — transmission unchanged ----
% scat_retro_data.mat (stage H, job 123563): each var = [lam_nm T], box 16 control.
rd = load(fullfile(res_dir, 'scat_retro_data.mat'));
Tr = [rd.ctrl(2) rd.d30(2) rd.d51(2) rd.rows2(2)];
labels5 = {'regular (no comb)', 'comb d = 3 \mum', 'comb d = 5.1 \mum', ...
           sprintf('comb, 2 rows\\newlined = 3, 5.68 \\mum')};

fig = figure('Position', [120 120 950 540], 'Color', 'w');
hold on
for i = 1:4
    if i == 1, c = C_PLAIN; else, c = [0.55 0.65 0.85]; end
    bar(i, Tr(i), 0.6, 'FaceColor', c);
    text(i, Tr(i)-0.006, sprintf('%.4f', Tr(i)), 'HorizontalAlignment', 'center', ...
        'FontSize', FS-1, 'FontWeight', 'bold', 'Color', 'w');
    if i > 1
        text(i, Tr(i)+0.005, sprintf('%+.4f', Tr(i)-Tr(1)), ...
            'HorizontalAlignment', 'center', 'FontSize', FS-2, 'Color', [0.35 0.35 0.35]);
    end
end
set(gca, 'XTick', 1:4, 'XTickLabel', labels5, 'FontSize', FS-1, ...
    'XTickLabelRotation', 0);
ylabel('resonance transmission T', 'FontSize', FS);
title('TM N=80: scatterer comb, \Lambda = 551 nm', 'FontSize', FS+1);
xlim([0.4 4.6]); ylim([0.85 1.005]); grid on; box on

exportgraphics(fig, fullfile(res_dir, 'scat_retrocomb_tm.png'), 'Resolution', 200);
set(fig, 'Visible', 'on'); savefig(fig, fullfile(res_dir, 'scat_retrocomb_tm.fig'));

disp('DONE: 5 figures in results_from_athena/scat_e_validate/figures/');
