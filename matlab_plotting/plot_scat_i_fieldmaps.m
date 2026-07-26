% plot_scat_i_fieldmaps.m — stage-I |E|^2 field-map images (scatterer program).
% Study dir: results_from_athena/scat_i_fieldmaps | Job 123991 | 2026-07-18
% Purpose: real-space comparison of (0) W800 control, (1) pillar pair [0,270]
% @y=0.7um r=80 (stage-E winner, +0.0227), (2) retro comb d=3.0um Lambda=551nm
% (stage-H null — comb autopsy). Data = *_planes.mat (converted from the
% server-extracted *_planes.npz; resonance-nearest recorded lambda).
% Views per project convention: XY monitor = "Side view", XZ = "Top view".

res_dir = fullfile(fileparts(mfilename('fullpath')), '..', ...
                   'results_from_athena', 'scat_i_fieldmaps', 'results');
out_dir = fullfile(res_dir, '..');

runs = { ...
  'result_N80_TM_avg_Ybox16p0_Zbox8p8_planes.mat', ...
      'W800 control  (T 0.885)'; ...
  'result_N80_TM_avg_Ybox16p0_Zbox8p8_scR80_arr2_X0to270_Y700to700_C400_pair_planes.mat', ...
      'pillar pair r=80  [0,270] nm @ y=\pm0.7 \mum  (T 0.909)'; ...
  'result_N80_TM_avg_Ybox16p0_Zbox8p8_scR110_arr151_X-41325to41325_Y3000to3000_C400_pair_planes.mat', ...
      'retro comb r=110  \Lambda=551 nm @ d=\pm3.0 \mum  (T 0.885)'};

% Shared log scale: normalize every panel to the control's global max so the
% three devices are directly comparable.
d0 = load(fullfile(res_dir, runs{1,1}));
norm_max = max(d0.field_xy_E2(:));
clim_lo = -6;                                 % 60 dB dynamic range

fig = figure('Visible', 'on', 'Position', [50 50 1500 900]);
tl = tiledlayout(fig, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:3
    d = load(fullfile(res_dir, runs{k,1}));

    % — XY plane (z=0): "Side view" (project convention); pillars/comb live here
    nexttile(tl);
    imagesc(d.field_xy_x*1e6, d.field_xy_y*1e6, ...
            log10(max(double(d.field_xy_E2'), 1e-30)/norm_max));
    axis xy; clim([clim_lo 0]); colormap(turbo);
    ylabel('y (\mum)');
    if k == 3, xlabel('x (\mum)'); end
    title(sprintf('%s — Side view (XY)', runs{k,2}));
    if k == 2
        hold on; plot([0 0.27 0 0.27], [0.7 0.7 -0.7 -0.7], 'wo', ...
                      'MarkerSize', 4, 'LineWidth', 0.8); hold off;
    elseif k == 3
        hold on; yline( 3.0, 'w--', 'LineWidth', 0.5);
                 yline(-3.0, 'w--', 'LineWidth', 0.5); hold off;
    end

    % — XZ plane (y=0): "Top view" (project convention); vertical channel
    nexttile(tl);
    imagesc(d.field_xz_side_x*1e6, d.field_xz_side_z*1e6, ...
            log10(max(double(d.field_xz_side_E2'), 1e-30)/norm_max));
    axis xy; clim([clim_lo 0]); colormap(turbo);
    ylabel('z (\mum)');
    if k == 3, xlabel('x (\mum)'); end
    title('Top view (XZ)');
    cb = colorbar; cb.Label.String = 'log_{10} |E|^2 (norm.)';
end

title(tl, sprintf(['TM corr-400 W800, N=80/side — |E|^2 at \\lambda=%.3f nm ' ...
                   '(box y=16 \\mum, opt mesh)'], d0.field_xy_lambda_used_nm));

png_path = fullfile(out_dir, 'scat_i_fieldmaps.png');
fig_path = fullfile(out_dir, 'scat_i_fieldmaps.fig');
exportgraphics(fig, png_path, 'Resolution', 200);
savefig(fig, fig_path);
fprintf('saved: %s\nsaved: %s\n', png_path, fig_path);
