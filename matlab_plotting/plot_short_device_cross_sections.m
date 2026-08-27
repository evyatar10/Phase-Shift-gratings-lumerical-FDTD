% plot_short_device_cross_sections.m
% Study: cross-study benchmark  |  Date: 2026-08-27
% Purpose: the three decorations of the SHORT device side by side as fab-stack
%   cross sections (same slide style as plot_trench_stack_cross_sections), each
%   annotated with its measured peak T, Q and spatial mode width, and the
%   relative change against the plain uniform device.
% Device: TM, N=80/side, corr 400 nm, pitch 516.83, W800, h350, box y=16/z=8.8,
%   z-symmetry OFF, optimization mesh, window 1548.5-1568.5 nm.
% All three rows are ONE study, scat_u_flushcomb: same job batch, same numerics,
%   its own z-asymmetric control (z-symmetry is OFF there -- section-2 BCs).
%   uniform          that study's control row
%   SiN circles      31 posts/row, r 70 nm, d = 1.8 um
%   air half trench  W 800 nm slot, d = 1.8 um, floor at the BOX interface
% ! The posts are DRAWN at the core height by user request (2026-08-27). In this
%   study they were actually etched down to the BOX like the trench, so the
%   middle panel's drawing and its number describe different post depths. The
%   core-height measurement of the same idea is scat_y_polish r89/47 posts,
%   T 0.90012 / Q 1338 / mode 15.469 um -- swap MATS{2} to use it.
% Mode is ~15.5 um here: this is the corr-400 device, NOT the 20 um-mode
%   corr-325 family.
% Output: results_from_athena/q20um_3db_benchmark/short_device_cross_sections.{png,fig}

ROOT = fileparts(fileparts(mfilename('fullpath')));
RA   = fullfile(ROOT, 'results_from_athena');
OUT  = fullfile(RA, 'q20um_3db_benchmark');
if ~isfolder(OUT); mkdir(OUT); end

PANELS = {'Uniform', 'SiN circles', 'Air half trench'};
SRC  = fullfile(RA, 'scat_u_flushcomb', 'results');
MATS = { ...
  fullfile(SRC,'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat'); ...
  fullfile(SRC,'result_N80_TM_avg_Ybox16p0_Zbox8p8_scR70_arr31_X-7567to8363_Y1800to1800_C400_pair_Zminm3975_ff.mat'); ...
  fullfile(SRC,'result_N80_TM_avg_Ybox16p0_Zbox8p8_scRECT_L84000xW800_arr1_X0to0_Y1800to1800_C400_pair_hole_Zminm3975_ff.mat')};

% ── geometry, um (drawn to scale) ─────────────────────────────────────────
CORE_H = 0.35; BOX_T = 3.8; TOP_CLAD = 2.2; CORE_W = 0.8; SI_T = 1.3;
X_HALF = 3.2;
POST_Y = 1.8;  POST_R = 0.07;           % 31 SiN posts per row, r 70 nm
TRENCH_Y = [1.4 2.2];                   % air rect W800, flush slot BOX -> SiN top

zc  = CORE_H/2;
zsi = -(zc + BOX_T);                    % -3.975 = Si top face
ztp = zc + TOP_CLAD;
zbt = zsi - SI_T;

C_SI = [0.470 0.470 0.470]; C_OX = [0.847 0.871 0.925]; C_SIN = [0.165 0.640 0.610];
INK  = [0.13 0.13 0.13];
EDGE = {'EdgeColor', INK, 'LineWidth', 1.1};

T = nan(1,3); Q = nan(1,3); W = nan(1,3);
for k = 1:3
    d = load(MATS{k});
    assert(double(d.n_periods_each_side) == 80, 'panel %d is not the N=80 device', k);
    T(k) = d.resonance_transmission;
    Q(k) = d.resonance_wavelength_nm / abs(d.spectral_fwhm_nm);
    W(k) = d.fwhm_m * 1e6;
end

fig = figure('Visible','off', 'Position', [40 40 1320 700], 'Color', 'w');
tl  = tiledlayout(fig, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

for k = 1:3
    ax = nexttile(tl); hold(ax, 'on');
    rectangle(ax, 'Position', [-X_HALF zbt 2*X_HALF zsi-zbt], 'FaceColor', C_SI,  EDGE{:});
    rectangle(ax, 'Position', [-X_HALF zsi 2*X_HALF ztp-zsi], 'FaceColor', C_OX,  EDGE{:});
    rectangle(ax, 'Position', [-CORE_W/2 -zc CORE_W CORE_H],  'FaceColor', C_SIN, EDGE{:});

    if k == 2                            % SiN posts: the core layer only
        for s = [-1 1]
            rectangle(ax, 'Position', [s*POST_Y - POST_R, -zc, 2*POST_R, CORE_H], ...
                      'FaceColor', C_SIN, EDGE{:});
        end
    elseif k == 3                        % air slot, BOX -> SiN top
        for s = [-1 1]
            rectangle(ax, 'Position', [min(s*TRENCH_Y) zsi diff(TRENCH_Y) zc-zsi], ...
                      'FaceColor', 'w', EDGE{:});
            text(ax, s*mean(TRENCH_Y), -1.9, 'Air', 'Rotation', 90, ...
                 'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', INK);
        end
    end

    text(ax, 0, (zsi+zbt)/2, 'Si', 'HorizontalAlignment','center', 'FontSize', 14, 'Color', 'w');
    text(ax, 0, -2.7, 'SiO_2', 'HorizontalAlignment','center', 'FontSize', 13, 'Color', INK);
    text(ax, 0,  1.45, 'SiN',  'HorizontalAlignment','center', 'FontSize', 12, 'Color', INK);
    plot(ax, [0 0], [1.15 zc+0.06], '-', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.7);

    title(ax, PANELS{k}, 'FontSize', 15, 'FontWeight', 'bold', 'Color', INK);
    text(ax, 0, zbt - 1.05, ...
        {sprintf('T = %.4f%s',        T(k), rel(T(k), T(1))), ...
         sprintf('Q = %s%s',          commas(Q(k)), rel(Q(k), Q(1))), ...
         sprintf('mode = %.2f %sm%s', W(k), '\mu', rel(W(k), W(1)))}, ...
        'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold', 'Color', INK);

    axis(ax, 'equal'); axis(ax, 'off');
    xlim(ax, [-X_HALF-0.5, X_HALF+0.5]); ylim(ax, [zbt-2.3, ztp+0.9]);
end

title(tl, {'\pi-shift Bragg grating, 80 periods per side, TM, corrugation 400 nm', ...
           '\lambda_{res} \approx 1558.6 nm'}, 'FontSize', 15, 'Color', INK);
annotation(fig, 'rectangle', [0.002 0.002 0.996 0.996], 'Color', 'w');

exportgraphics(fig, fullfile(OUT, 'short_device_cross_sections.png'), 'Resolution', 220);
savefig(fig, fullfile(OUT, 'short_device_cross_sections.fig')); close(fig);

for k = 1:3
    fprintf('%-14s T %.5f  Q %6.0f  mode %.3f um\n', PANELS{k}, T(k), Q(k), W(k));
end

% ── helpers ───────────────────────────────────────────────────────────────
function s = rel(v, ref)
% "(+1.7%)" against the uniform row -- the plain signed change, so a wider mode
% reads positive rather than being re-signed into a fake improvement
if v == ref; s = ''; return; end
s = sprintf('   (%+.1f%%)', 100 * (v - ref) / ref);
end

function s = commas(v)
s = regexprep(sprintf('%d', round(v)), '(\d)(?=(\d{3})+$)', '$1,');
end
