% plot_comb_period_scan.m
% Study: comb handoff (runners/scatterers/COMB_HANDOFF.md)  |  Date: 2026-08-27
% Purpose: peak T against the CIRCLE PERIOD (Lambda), at two offsets: the
%   cancelling phase (270 deg) and the reference phase (0 deg). The period sets
%   WHERE the circles re-radiate (the aim); the phase sets whether that beam adds
%   or cancels. At 0 deg no period helps -- every point sits below the plain
%   device. At 270 deg the aim curve peaks near Lambda = 530 nm.
% Rows scanned: N=80/side, TM, corr 400, W800, h350, box y=16/z=8.8, 20 nm/1501,
%   optimization mesh, circles r 110 nm, 31 per row, d = 1.8 um, core height.
%   Sources: scat_p_antineedle (0 deg series), scat_s_refine + scat_r_aim536 +
%   scat_q_r80phase (270 deg series). Control = the shared _ff control row.
% Output: results_from_athena/q20um_3db_benchmark/comb_period_scan.{png,fig}

ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % studies/ -> repo root
RA   = fullfile(ROOT, 'results_from_athena');
OUT  = fullfile(RA, 'q20um_3db_benchmark');
if ~isfolder(OUT); mkdir(OUT); end

MAX_LAMBDA = 545;        % last period measured on BOTH branches — plot stops there

CTRL = load(fullfile(RA, 'air_trench_dscan', 'results', ...
                     'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat'));
T0 = CTRL.resonance_transmission;

% scan BOTH clusters: the aim-extension rows (scat_aim_extend) ran on IGUM.
RI    = fullfile(ROOT, 'results_from_igum');
files = [dir(fullfile(RA, 'scat_*', 'results', 'result_N80_TM_avg_Ybox16p0*scR110_arr31*.mat'));
         dir(fullfile(RI, 'scat_*', 'results', 'result_N80_TM_avg_Ybox16p0*scR110_arr31*.mat'))];
lam = []; phi = []; T = [];
for k = 1:numel(files)
    nm = files(k).name;
    tok = regexp(nm, 'arr(\d+)_X(-?\d+)to(-?\d+)_Y(\d+)to(\d+)_C(\d+)', 'tokens', 'once');
    if isempty(tok); continue; end
    if contains(nm, 'Zminm'); continue; end               % flush (deep-etch) variant
    n = str2double(tok{1}); x0 = str2double(tok{2}); x1 = str2double(tok{3});
    d = str2double(tok{4}); corr = str2double(tok{6});
    if d ~= 1800 || corr ~= 400; continue; end            % one standoff, one device
    if (x1 - x0) / (n - 1) > MAX_LAMBDA; continue; end     % both branches end together
    m = load(fullfile(files(k).folder, nm));
    if m.resonance_wavelength_nm < 1558 || m.resonance_wavelength_nm > 1559.5
        continue                                          % accurate-mesh twins
    end
    L  = (x1 - x0) / (n - 1);
    ph = mod((x0 + x1) / 2 / L * 360, 360);
    lam(end+1) = L;   phi(end+1) = ph;   T(end+1) = m.resonance_transmission;   %#ok<SAGROW>
end

is270 = abs(phi - 270) < 12;                              % the cancelling phase
is000 = (phi < 12) | (phi > 348);                         % the reference phase
% Lambda repeats exist on purpose (mesh-registration twins: same phase, the comb
% moved one whole period). Keep every point visible; draw the line through the mean.
[L270u, ~, g] = unique(lam(is270));  t270all = T(is270);
m270 = accumarray(g(:), t270all(:), [], @mean).';
[L000u, ~, g0] = unique(lam(is000)); t000all = T(is000);
m000 = accumarray(g0(:), t000all(:), [], @mean).';
lam270 = lam(is270);  lam000 = lam(is000);

c270 = [0.00 0.45 0.74];  c000 = [0.85 0.33 0.10];  ink = [0.13 0.13 0.13];
fig = figure('Visible', 'off', 'Position', [60 60 940 580]);
ax = axes(fig); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');

yline(ax, T0, 'k--', 'LineWidth', 1.4, 'DisplayName', 'no circles');
xq270 = linspace(min(L270u), max(L270u), 300);
plot(ax, xq270, interp1(L270u, m270, xq270, 'pchip'), '-', 'Color', c270, ...
     'LineWidth', 1.8, 'HandleVisibility', 'off');
plot(ax, lam270, t270all, 'o', 'Color', c270, 'MarkerFaceColor', c270, ...
     'MarkerSize', 7, 'DisplayName', '270\circ (cancelling)');
xq000 = linspace(min(L000u), max(L000u), 300);
plot(ax, xq000, interp1(L000u, m000, xq000, 'pchip'), '-', 'Color', c000, ...
     'LineWidth', 1.8, 'HandleVisibility', 'off');
plot(ax, lam000, t000all, 's', 'Color', c000, 'MarkerFaceColor', c000, ...
     'MarkerSize', 7, 'DisplayName', '0\circ (reference)');

allL = [L270u(:); L000u(:)];
xlim(ax, [min(allL) - 2.5, max(allL) + 2.5]);
xlabel(ax, 'circle period  \Lambda  [nm]');
ylabel(ax, 'peak transmission');
legend(ax, 'Location', 'southoutside', 'Orientation', 'horizontal');
title(ax, 'Transmission vs circle period', ...
      'FontSize', 15, 'FontWeight', 'bold', 'Color', ink);

exportgraphics(fig, fullfile(OUT, 'comb_period_scan.png'), 'Resolution', 200);
savefig(fig, fullfile(OUT, 'comb_period_scan.fig'));
close(fig);

fprintf('uniform: T %.5f\n', T0);
fprintf('270 deg series (mean per Lambda):\n');
for k = 1:numel(L270u); fprintf('  Lambda %5.1f   T %.5f (%+.4f)\n', L270u(k), m270(k), m270(k) - T0); end
fprintf('0 deg series:\n');
for k = 1:numel(L000u); fprintf('  Lambda %5.1f   T %.5f (%+.4f)\n', L000u(k), m000(k), m000(k) - T0); end
