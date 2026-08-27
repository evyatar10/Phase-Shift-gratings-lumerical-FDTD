% plot_tooth_profiles.m
% Study: cross-study benchmark  |  Date: 2026-08-27
% Purpose: the per-tooth design of the two apodized devices -- corrugation on
%   top, our tooth shift underneath -- over the innermost 65 teeth.
% Layout note: the two panels are STACKED and share one tooth axis (1..65) on
%   purpose. The shift only exists for the 25 free inner teeth, so if the two
%   panels had different x ranges they would look like the same span; sharing
%   the axis (plus the dotted marker at tooth 25 and the shaded tail) makes the
%   different extents impossible to misread.
% Data (hard-coded here as data, same convention as itai_hh_nt60w20.py):
%   TM  BEST_T9636, runners/lumopt2_design/best_designs.py -- 25 free inner
%       teeth, then the frozen outer corrugation 325 nm. Shifts likewise stop
%       at tooth 25; the outer section is a regular grating.
%   TE  Itai's Nt60 overshoot apodization, runners/sweeps/itai_hh_nt60w20.py.
%       His tables are cavity-first, so the cavity entry is dropped and teeth
%       1..60 are the apodized ones; tooth 61 onward is the bulk, 494.2 nm.
% Output: results_from_athena/q20um_3db_benchmark/tooth_profiles.{png,fig}

ROOT = fileparts(fileparts(mfilename('fullpath')));
OUT  = fullfile(ROOT, 'results_from_athena', 'q20um_3db_benchmark');
if ~isfolder(OUT); mkdir(OUT); end

N_SHOW = 65;                 % teeth drawn; both devices are far longer
N_FREE = 25;                 % TM: free inner teeth = where the shift lives

TM_CORR = [324.61 331.14 345.55 353.68 355.16 358.44 360.78 361.66 360.74 ...
           362.10 364.26 363.68 361.75 363.98 364.14 361.77 363.61 363.67 ...
           361.09 362.08 362.27 360.03 360.61 361.52 360.51];
TM_OUTER = 325;              % frozen outer corrugation
TM_SHIFT = [3.165 2.814 4.200 5.351 5.876 6.426 6.118 5.344 4.893 3.985 ...
            3.044 2.305 1.466 0.911 0.898 0.890 0.890 0.897 0.909 0.925 ...
            0.947 0.969 0.995 1.026 1.055];

TE_CORR = [ 56.2  75.9 108.5 145.8 182.9 218.8 253.8 288.3 322.2 355.5 ...
           388.0 419.6 449.8 478.7 505.6 530.5 553.7 575.4 594.2 609.7 ...
           621.6 629.7 634.0 634.3 630.5 622.5 610.4 596.5 580.5 562.7 ...
           543.5 523.7 503.8 484.5 466.2 448.9 432.7 417.7 405.8 397.2 ...
           392.3 390.8 392.2 396.1 402.0 408.3 414.7 421.4 428.5 435.9 ...
           443.3 450.7 457.8 464.5 471.1 477.7 484.1 489.4 493.0 494.2];
TE_BULK = 494.2;

tm = [TM_CORR, repmat(TM_OUTER, 1, N_SHOW - numel(TM_CORR))];
te = [TE_CORR, repmat(TE_BULK,  1, N_SHOW - numel(TE_CORR))];
x  = 1:N_SHOW;

cTM = [0.20 0.50 0.15]; cTE = [0.64 0.08 0.18]; ink = [0.13 0.13 0.13];
fig = figure('Visible','off', 'Position', [60 60 1000 700]);
tl  = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% ── corrugation ───────────────────────────────────────────────────────────
ax1 = nexttile(tl, 1, [2 1]); hold(ax1,'on'); grid(ax1,'on'); box(ax1,'on');
plot(ax1, x, te, '-o', 'Color', cTE, 'MarkerFaceColor', cTE, 'MarkerSize', 3.5, ...
     'LineWidth', 1.5, 'DisplayName', 'overshoot apodization, TE  (60 teeth)');
plot(ax1, x, tm, '-o', 'Color', cTM, 'MarkerFaceColor', cTM, 'MarkerSize', 3.5, ...
     'LineWidth', 1.5, 'DisplayName', 'inverse design, TM  (25 teeth, {\bfnot fully optimized})');
xline(ax1, N_FREE, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.2, 'HandleVisibility','off');
ylabel(ax1, 'corrugation [nm]');
set(ax1, 'XTickLabel', []);
ylim(ax1, [0 700]);
legend(ax1, 'Location', 'southeast', 'FontSize', 10);

% ── tooth shift (our TM device only, and only where it exists) ────────────
ax2 = nexttile(tl, 3); hold(ax2,'on'); grid(ax2,'on'); box(ax2,'on');
% the outer section is a regular grating, so its shift IS zero -- drawn as zero
% across the full axis rather than stopped at 25, which also removes any doubt
% about the two panels covering the same tooth range
plot(ax2, x, [TM_SHIFT, zeros(1, N_SHOW - N_FREE)], '-o', 'Color', cTM, ...
     'MarkerFaceColor', cTM, 'MarkerSize', 3.5, 'LineWidth', 1.5);
xline(ax2, N_FREE, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.2);
ylabel(ax2, 'tooth shift [nm]'); xlabel(ax2, 'tooth number (1 = innermost)');
ylim(ax2, [0 7.5]);

linkaxes([ax1 ax2], 'x'); xlim(ax1, [0.5 N_SHOW+0.5]);
xticks(ax1, [1 N_FREE 40 60 N_SHOW]); xticks(ax2, [1 N_FREE 40 60 N_SHOW]);

title(tl, sprintf(['Per-tooth design, innermost %d teeth only ' ...
      '(full devices: TM 220, TE 130 per side)'], N_SHOW), ...
      'FontSize', 13, 'FontWeight', 'bold', 'Color', ink);

exportgraphics(fig, fullfile(OUT, 'tooth_profiles.png'), 'Resolution', 200);
savefig(fig, fullfile(OUT, 'tooth_profiles.fig')); close(fig);

fprintf('TM corr %.1f-%.1f nm (25 free) then %d nm | shift %.2f-%.2f nm\n', ...
        min(TM_CORR), max(TM_CORR), TM_OUTER, min(TM_SHIFT), max(TM_SHIFT));
fprintf('TE corr %.1f-%.1f nm (60 teeth) then %.1f nm\n', ...
        min(TE_CORR), max(TE_CORR), TE_BULK);
