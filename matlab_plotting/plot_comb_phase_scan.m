% plot_comb_phase_scan.m
% Study: cross-study benchmark  |  Date: 2026-08-27
% Purpose: the comb PHASE circle -- peak T against the post offset dx,
%   at the fixed comb pitch, with the plain uniform device as the reference
%   line. This is the oscillation: the same circles lose 0.045 in T at 90 deg
%   and gain 0.008 at 270 deg, which is why the design carries dx = 0.75*Lambda.
% Rows: N=80/side, TM, corr 400, W800, h350, box y=16/z=8.8, optimization mesh,
%   comb Lambda 536 nm, r 110 nm, 31 posts per row, d = 1.8 um, core height.
%   0/90/180/270 deg from scat_r_aim536; 250 and 290 deg from scat_s_refine's
%   refinement of the peak. Uniform control = the shared _ff control at these
%   numerics (identical in five studies).
% Output: results_from_athena/q20um_3db_benchmark/comb_phase_scan.{png,fig}

ROOT = fileparts(fileparts(mfilename('fullpath')));
RA   = fullfile(ROOT, 'results_from_athena');
OUT  = fullfile(RA, 'q20um_3db_benchmark');
if ~isfolder(OUT); mkdir(OUT); end

LAMBDA = 536;                                 % the circle period this scan sits at

CTRL = load(fullfile(RA, 'air_trench_dscan', 'results', ...
                     'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat'));
T0 = CTRL.resonance_transmission;
Q0 = CTRL.resonance_wavelength_nm / abs(CTRL.spectral_fwhm_nm);

files = dir(fullfile(RA, 'scat_*', 'results', 'result_N80_TM_avg_Ybox16p0*scR*.mat'));
ph = []; T = []; Q = [];
for k = 1:numel(files)
    nm = files(k).name;
    if contains(nm, 'RECT') || contains(nm, 'Zminm3975'); continue; end
    tok = regexp(nm, 'scR(\d+)_arr(\d+)_X(-?\d+)to(-?\d+)_Y(\d+)', 'tokens', 'once');
    if isempty(tok); continue; end
    r = str2double(tok{1}); n = str2double(tok{2});
    x0 = str2double(tok{3}); x1 = str2double(tok{4}); d = str2double(tok{5});
    if r ~= 110 || n ~= 31 || d ~= 1800; continue; end
    if round((x1 - x0) / (n - 1)) ~= LAMBDA; continue; end
    m = load(fullfile(files(k).folder, nm));
    if m.resonance_wavelength_nm < 1558 || m.resonance_wavelength_nm > 1559.5
        continue                                      % accurate-mesh twin
    end
    ph(end+1) = (x0 + x1) / 2 / LAMBDA * 360;                            %#ok<SAGROW>
    T(end+1)  = m.resonance_transmission;                                %#ok<SAGROW>
    Q(end+1)  = m.resonance_wavelength_nm / abs(m.spectral_fwhm_nm);     %#ok<SAGROW>
end
[ph, o] = sort(ph); T = T(o); Q = Q(o);

phw = ph; Tw = T;

c1 = [0.00 0.45 0.74]; ink = [0.13 0.13 0.13];
fig = figure('Visible','off', 'Position', [60 60 940 580]);
ax  = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

yline(ax, T0, 'k--', 'LineWidth', 1.4, 'DisplayName', 'no circles');
% smooth curve THROUGH the measured points (shape-preserving interpolation,
% not a fit): it crosses every point exactly, and it stops at the last one.
phq = linspace(min(ph), max(ph), 400);
plot(ax, phq, interp1(ph, T, phq, 'pchip'), '-', 'Color', c1, ...
     'LineWidth', 1.8, 'HandleVisibility', 'off');
plot(ax, ph, T, 'o', 'Color', c1, 'MarkerFaceColor', c1, 'MarkerSize', 7, ...
     'DisplayName', sprintf('SiN circles, \\Lambda = %d nm', LAMBDA));
xlabel(ax, 'circle offset  \deltax  from the cavity centre  [deg of \Lambda]'); ylabel(ax, 'peak transmission');
xlim(ax, [min(ph) - 25, max(ph) + 25]); xticks(ax, 0:90:270);
legend(ax, 'Location', 'southoutside', 'Orientation', 'horizontal');

title(ax, 'Transmission vs circle offset', ...
      'FontSize', 15, 'FontWeight', 'bold', 'Color', ink);

exportgraphics(fig, fullfile(OUT, 'comb_phase_scan.png'), 'Resolution', 200);
savefig(fig, fullfile(OUT, 'comb_phase_scan.fig')); close(fig);

fprintf('uniform: T %.5f  Q %.0f\n', T0, Q0);
for k = 1:numel(ph)
    fprintf('  dx %3.0f deg   T %.5f (%+.4f)   Q %6.0f (%+.0f)\n', ...
            ph(k), T(k), T(k) - T0, Q(k), Q(k) - Q0);
end
