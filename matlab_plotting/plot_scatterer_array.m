% plot_scatterer_array.m — final array-study comparison (job 116940)
% Bar chart of dT and dloss vs the same-run control for the 6 multi-scatterer
% geometries, accurate mesh, converged box (y=6.8 um, z=8.8 um).
% Headless-safe: no dialogs; saves .fig + .png next to the data.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
res_dir = fullfile(proj, 'results_from_athena', 'tm_scatterer_array', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'tm_scatterer_array');

% (file token, display label, N)
rows = { ...
  'arr1_X810to810_Y1000',        'single pair @0.81',        1; ...
  'arr4_X810to6075_Y1000',       'winners N=4',              4; ...
  'arr3_X4050to5097_Y1000to1259','lobe-ray N=3 (diag B)',    3; ...
  'arr3_X4050to3895_Y1000to1500','same-arc N=3 (diag A)',    3; ...
  'arr3_X810to2145_Y1000',       'rho-comb N=3 fixed-y',     3; ...
  'arr6_X810to3858_Y1000',       'rho-comb N=6 fixed-y',     6};

ctrl = load(fullfile(res_dir, 'result_N80_TM_avg_Ybox6p8_Zbox8p8.mat'));
[T0, loss0, lam0] = row_metrics(ctrl);

n = size(rows, 1);
dT = zeros(1, n); dloss = zeros(1, n);
for k = 1:n
    f = dir(fullfile(res_dir, ['result_*' rows{k,1} '*.mat']));
    assert(numel(f) == 1, 'expected exactly one file for %s', rows{k,1});
    d = load(fullfile(f.folder, f.name));
    [T, lossk, ~] = row_metrics(d);
    dT(k) = T - T0; dloss(k) = lossk - loss0;
end

fig = figure('Visible', 'off', 'Position', [100 100 900 420]);
b = bar([dT; dloss]', 'grouped');
b(1).FaceColor = [0.19 0.45 0.72]; b(2).FaceColor = [0.85 0.42 0.20];
yline(0, 'k-');
set(gca, 'XTickLabel', rows(:,2), 'XTickLabelRotation', 20, 'TickLabelInterpreter', 'none');
ylabel('\Delta vs in-study control');
legend({'\DeltaT at resonance', '\Deltaloss (1-T-R)'}, 'Location', 'southwest');
grid on;
title(sprintf(['\\pi-shift TM corr 400 nm, r=100 nm pillar pairs, accurate mesh, box 6.8/8.8 \\mum\n' ...
    'control: \\lambda_{res}=%.2f nm, T=%.4f, loss=%.4f'], lam0, T0, loss0));

exportgraphics(fig, fullfile(out_dir, 'array_study_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'array_study_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'array_study_summary.png'));

function [T, loss, lam] = row_metrics(d)
    lam = d.resonance_wavelength_nm;
    T = d.resonance_transmission;
    [~, i] = min(abs(d.wl_nm - lam));
    loss = 1 - T - d.R(i);
end
