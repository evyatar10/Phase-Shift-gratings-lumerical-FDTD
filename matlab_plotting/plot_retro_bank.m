% plot_retro_bank.m — retro-reflector bank verdict figure (stage K).
% Study dir: results_from_athena/retro_bank | Job 124310 | 2026-07-19
% Purpose: dT per retro variant vs the 0.0018 jitter floor. The Al bar is an
% INSTRUMENT result (staircase-mesh metal absorption at dx=50 nm), not physics.

labels = {'flat PEC d=2.80', 'flat Al d=3.00 (mesh artifact)', ...
          'PEC comb r110', 'PEC comb r150', 'PEC comb 2-row', ...
          'corner wall d=3.000', 'corner wall d=3.054'};
dT = [0.0004 -0.0274 -0.0019 0.0009 -0.0014 -0.0026 -0.0023];   % MEASURED 124310
T_ctrl = 0.8851; lam_res = 1558.612;

fig = figure('Visible', 'on', 'Position', [80 80 950 520]);
hold on;
fill([0.4 7.6 7.6 0.4], 1e3*[-0.0018 -0.0018 0.0018 0.0018], ...
     [0.92 0.92 0.92], 'EdgeColor', 'none');
b = bar(1:7, 1e3*dT, 0.55, 'FaceColor', [0 0.45 0.74]);
b.FaceColor = 'flat';
b.CData(2,:) = [0.6 0.6 0.6];              % Al = instrument bar, grayed
yline(0, 'k-');
yline(1e3*0.0019, ':', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.1, ...
      'Label', 'flat-wall best (d-scan, +0.0019)');
hold off;
grid on; box on;
set(gca, 'XTick', 1:7, 'XTickLabel', labels, 'XTickLabelRotation', 25);
ylabel('\DeltaT vs control (\times10^{-3})');
ylim(1e3*[-0.0285 0.004]);
title(sprintf(['Retro bank (350 nm layer) — TM corr-400 W800, ' ...
               '\\lambda_{res}=%.3f nm, T_{ctrl}=%.4f  |  band = \\pm0.0018 floor'], ...
              lam_res, T_ctrl));

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', ...
                   'results_from_athena', 'retro_bank');
exportgraphics(fig, fullfile(out_dir, 'retro_bank_verdict.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'retro_bank_verdict.fig'));
fprintf('saved: %s\nsaved: %s\n', fullfile(out_dir, 'retro_bank_verdict.png'), ...
        fullfile(out_dir, 'retro_bank_verdict.fig'));
