% plot_scatterer_program_summary.m — one figure, the WHOLE scatterer program.
% Panel A: pillar pair T vs axial position (3 radii, 187-task scan)
% Panel B: in-core SiO2 hole T vs position (98-task scan)
% Panel C: radius ladder at winning positions (accurate mesh, converged box)
% Panel D: multi-scatterer geometries (accurate mesh, converged box)
% Headless-safe; saves .fig + .png next to the data.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
d_scan  = fullfile(proj, 'results_from_athena', 'tm_scatterer_scan',   'results');
d_hole  = fullfile(proj, 'results_from_athena', 'tm_hole_scan',        'results');
d_rad   = fullfile(proj, 'results_from_athena', 'tm_scatterer_radius', 'results');
d_arr   = fullfile(proj, 'results_from_athena', 'tm_scatterer_array',  'results');
out_dir = fullfile(proj, 'results_from_athena', 'tm_scatterer_scan');

fig = figure('Visible', 'off', 'Position', [50 50 1400 900]);
tl = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% ── Panel A: pillar position scan ─────────────────────────────────────────
nexttile; hold on; grid on;
[xs, Ts, rs] = load_scan(d_scan);
T0A = Ts(rs == 0);                                   % scan's own control
cols = containers.Map({100, 150, 200}, {[0.19 0.45 0.72], [0.20 0.62 0.36], [0.80 0.33 0.20]});
for r = [100 150 200]
    m = (rs == r);
    [xr, is] = sort(xs(m)); Tr = Ts(m); Tr = Tr(is);
    plot(xr * 1e6, Tr, '.-', 'Color', cols(r), 'DisplayName', sprintf('r = %d nm', r));
end
yline(T0A, 'k--', 'DisplayName', 'no scatterer');
xlabel('pillar-pair position x (\mum)'); ylabel('peak T');
title('A: SiN pillar pair (x, \pm1 \mum) — position scan (dx=50 nm, 4.8 \mum box)');
legend('Location', 'southeast', 'NumColumns', 2);

% ── Panel B: in-core hole scan ────────────────────────────────────────────
nexttile; hold on; grid on;
[xh, Th, rh] = load_scan(d_hole);
T0B = Th(rh == 0);
m = rh > 0; [xhs, is] = sort(xh(m)); Ths = Th(m); Ths = Ths(is);
plot(xhs * 1e6, Ths, '.-', 'Color', [0.55 0.35 0.65], 'DisplayName', 'r=100 nm SiO_2 hole');
yline(T0B, 'k--', 'DisplayName', 'no hole');
xlabel('hole position x (\mum)'); ylabel('peak T');
title('B: SiO_2 hole ON-axis in the core — parasitic-to-neutral, never a gain');
legend('Location', 'southeast');

% ── Panel C: radius ladder (accurate mesh, converged box) ─────────────────
nexttile; hold on; grid on;
[xr_, Tr_, rr_] = load_scan(d_rad);
T0C = Tr_(rr_ == 0);
Rs = [80 100 125]; Xs = [540 810 4050];
M = nan(numel(Rs), numel(Xs));
for i = 1:numel(Rs)
    for j = 1:numel(Xs)
        k = find(rr_ == Rs(i) & abs(xr_ * 1e9 - Xs(j)) < 1, 1);
        if ~isempty(k), M(i, j) = (Tr_(k) - T0C) * 1e3; end   % in 1e-3 units (avoids exponent label)
    end
end
b = bar(M', 'grouped');
for i = 1:numel(Rs), b(i).DisplayName = sprintf('r = %d nm', Rs(i)); end
yline(0, 'k-', 'HandleVisibility', 'off');
set(gca, 'XTick', 1:numel(Xs), 'XTickLabel', arrayfun(@(x) sprintf('x = %g nm', x), Xs, 'UniformOutput', false));
ylabel('\DeltaT vs control (\times10^{-3})');
title('C: radius ladder at winners (dx\approx35 nm, converged 6.8/8.8 \mum box) — optimum \approx100 nm');
legend('Location', 'northeast');

% ── Panel D: multi-scatterer geometries ───────────────────────────────────
nexttile; hold on; grid on;
rows = { ...
  'arr1_X810to810_Y1000',        'single @810';
  'arr4_X810to6075_Y1000',       'winners N=4';
  'arr3_X4050to5097_Y1000to1259','lobe-ray N=3';
  'arr3_X4050to3895_Y1000to1500','same-arc N=3';
  'arr3_X810to2145_Y1000',       'comb N=3';
  'arr6_X810to3858_Y1000',       'comb N=6'};
c0 = load(fullfile(d_arr, 'result_N80_TM_avg_Ybox6p8_Zbox8p8.mat'));
T0D = c0.resonance_transmission;
dT = nan(1, size(rows, 1));
for k = 1:size(rows, 1)
    f = dir(fullfile(d_arr, ['result_*' rows{k,1} '*.mat']));
    d = load(fullfile(f(1).folder, f(1).name));
    dT(k) = d.resonance_transmission - T0D;
end
bar(dT, 'FaceColor', [0.19 0.45 0.72]);
yline(0, 'k-');
set(gca, 'XTick', 1:size(rows,1), 'XTickLabel', rows(:,2), 'XTickLabelRotation', 20, ...
    'TickLabelInterpreter', 'none');
ylabel('\DeltaT vs control');
title('D: arrays (r=100 nm) — gains do NOT stack; close combs destructive');

title(tl, sprintf(['TM \\pi-shift grating (pitch 516.83 nm, corr 400 nm, h 350 nm) — scatterer recycling program\n' ...
    'best single: \\DeltaT=+0.0026 @ (r=100 nm, x=810 nm) \\cdot best overall: +0.0034 (N=4) \\cdot budget: loss 0.11']), ...
    'FontSize', 12);

exportgraphics(fig, fullfile(out_dir, 'scatterer_program_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'scatterer_program_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'scatterer_program_summary.png'));

function [xs, Ts, rs] = load_scan(dd)
    fl = dir(fullfile(dd, 'result_*.mat'));
    n = numel(fl); xs = nan(1, n); Ts = nan(1, n); rs = nan(1, n);
    for k = 1:n
        d = load(fullfile(fl(k).folder, fl(k).name), ...
                 'resonance_transmission', 'scatterer_r_m', 'scatterer_x_m');
        Ts(k) = d.resonance_transmission;
        rs(k) = round(d.scatterer_r_m * 1e9);
        xs(k) = d.scatterer_x_m;
    end
    % keep a single control value (first r==0 file)
    if numel(find(rs == 0)) > 1
        keep = true(1, n); extra = find(rs == 0); keep(extra(2:end)) = false;
        xs = xs(keep); Ts = Ts(keep); rs = rs(keep);
    end
end
