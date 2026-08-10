% plot_comb_difference.m
% Study: scat_w_dscan (best comb d=1.5, job 130179) vs stored ctrl (123563)
% Date: 2026-08-10
% Purpose: make the comb's far-field action VISIBLE: (A) control map,
%   (B) the comb's own emitted wave |E_comb - E_ctrl|^2 (coherent difference =
%   the anti-beam), (C) ratio map comb/ctrl in dB (what got deleted),
%   (D) 1D ratio cut. Side monitor (where the lobes live), Ez (95% of power).

ROOT = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results_from_athena');
ctrl = load(fullfile(ROOT, 'scat_h_retrocomb', 'results', ...
    'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat'));
comb = load(fullfile(ROOT, 'scat_w_dscan', 'results', ...
    ['result_N80_TM_avg_Ybox16p0_Zbox8p8_scR82_arr31_' ...
     'X-7582to8348_Y1500to1500_C400_pair_ff.mat']));

ux = ctrl.farfield_side.ux;
uy = ctrl.farfield_side.uy;
Ec = ctrl.farfield_side.Ez_c;              % rows = ux (verified convention)
Eb = comb.farfield_side.Ez_c;
ref = max(abs(Ec(:)).^2);

fig = figure('Visible', 'off', 'Position', [60 60 1000 760]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(tl);
imagesc(ax, ux, uy, 10*log10(max(abs(Ec').^2/ref, 1e-6)));
axis(ax, 'xy', 'square'); clim(ax, [-50 0]);
xlabel(ax, 'u_x'); ylabel(ax, 'u_y');
title(ax, 'A — control: the device''s own radiation');

ax = nexttile(tl);
imagesc(ax, ux, uy, 10*log10(max(abs((Eb - Ec)').^2/ref, 1e-6)));
axis(ax, 'xy', 'square'); clim(ax, [-50 0]);
xlabel(ax, 'u_x'); ylabel(ax, 'u_y');
title(ax, 'B — what the pillars EMIT: |E_{comb} - E_{ctrl}|^2 (the anti-beam)');

ax = nexttile(tl);
imagesc(ax, ux, uy, 10*log10(max(abs(Eb').^2, 1e-30) ./ max(abs(Ec').^2, 1e-30)));
axis(ax, 'xy', 'square'); clim(ax, [-10 10]); colormap(ax, 'turbo');
xlabel(ax, 'u_x'); ylabel(ax, 'u_y');
cb = colorbar(ax); cb.Label.String = 'comb / ctrl  [dB]';
title(ax, 'C — the CHANGE: ratio map (blue = deleted radiation)');

ax = nexttile(tl);
band = abs(uy) <= 0.35;
pc = sum(abs(Ec(:, band)).^2, 2);
pb = sum(abs(Eb(:, band)).^2, 2);
plot(ax, ux, 10*log10(pb ./ pc), 'LineWidth', 1.4);
yline(ax, 0, 'k:');
grid(ax, 'on'); ylim(ax, [-12 6]);
xlabel(ax, 'u_x'); ylabel(ax, 'comb / ctrl  [dB]');
title(ax, 'D — 1D ratio (|u_y|\leq0.35 band): the lobe deletion');

title(tl, {'Where the pillars act — comb d=1.5 \mum, r=82 (T 0.897) vs control (T 0.885)'; ...
    'TM corr 400, W800, N=80, side monitor, E_z component'});

set(fig, 'Visible', 'on');
OUT = fullfile(ROOT, 'scat_w_dscan');
savefig(fig, fullfile(OUT, 'comb_difference_maps.fig'));
set(fig, 'Visible', 'off');
exportgraphics(fig, fullfile(OUT, 'comb_difference_maps.png'), 'Resolution', 150);
disp(fullfile(OUT, 'comb_difference_maps.png'));
