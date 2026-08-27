% plot_q20um_3db_benchmark.m
% Study: cross-study benchmark  |  Date: 2026-08-26
% Purpose: one chart of the Q factor every device family reaches at the common
%   operating point -- 20 um spatial FWHM, peak T = -3 dB.
% The four locked devices are read from their stored .mat. The inverse-design
%   and apodization rows come from the numbers the user supplied (2026-08-26);
%   the two that are still projections carry a "~" and nothing else.
% Output: results_from_athena/q20um_3db_benchmark/q20um_3db_benchmark.{png,fig}

ROOT = fileparts(fileparts(mfilename('fullpath')));
RA   = fullfile(ROOT, 'results_from_athena');
RI   = fullfile(ROOT, 'results_from_igum');
OUT  = fullfile(RA, 'q20um_3db_benchmark');
if ~isfolder(OUT); mkdir(OUT); end

% ── the four locked devices, ascending Q ───────────────────────────────────
meas = { ...
  'uniform',          'TE', fullfile(RA,'te_q3db_20um','results','result_N166_avg_C250.mat'); ...
  'uniform',          'TM', fullfile(RI,'trench_q3db_20um','results','result_N165_TM_avg_C325_Ybox8p0_Zbox8p8.mat'); ...
  'circles',          'TM', fullfile(RA,'comb_q3db','results','result_N169_TM_avg_C325_Ybox8p0_Zbox8p8_scR80_arr57_X-14467to15269_Y1900to1900_C325_pair.mat'); ...
  'air half trench',  'TM', fullfile(RA,'trench_flush_q3db','results','result_N168_TM_avg_C325_Ybox8p0_Zbox8p8_scRECT_L176000xW800_X0_Y1800_pair_hole_Zminm3975.mat')};

nm = {}; sub = {}; pol = {}; Q = []; approx = [];
for k = 1:size(meas,1)
    d = load(meas{k,3});
    nm{end+1}  = meas{k,1}; sub{end+1} = ''; pol{end+1} = meas{k,2};     %#ok<SAGROW>
    Q(end+1)   = d.resonance_wavelength_nm / abs(d.spectral_fwhm_nm);    %#ok<SAGROW>
    approx(end+1) = false;                                               %#ok<SAGROW>
end

% ── user-supplied rows (2026-08-26). The + circles value is the measured
% crossing from the companion session; the bare-grating one is scaled off it
% by the measured circles gain, so it keeps the "~".
est = {'inverse design 25 teeth',           'not fully optimized', 'TM', 7.6e4, true; ...
       'inverse design 25 teeth + circles', 'not fully optimized', 'TM', 88868, false; ...
       'overshoot apodization 60 teeth',    '',                    'TE', 5.6e5, true};
for k = 1:size(est,1)
    nm{end+1} = est{k,1}; sub{end+1} = est{k,2}; pol{end+1} = est{k,3};  %#ok<SAGROW>
    Q(end+1)  = est{k,4}; approx(end+1) = est{k,5};                      %#ok<SAGROW>
end

col = [0.55 0.55 0.55; 0.28 0.28 0.28; 0.00 0.45 0.74; 0.93 0.69 0.13; ...
       0.47 0.67 0.19; 0.20 0.50 0.15; 0.64 0.08 0.18];
n = numel(Q);

lab  = cell(1, n);      % value text
tick = cell(1, n);      % axis label (one line; the note is drawn separately)
for k = 1:n
    if Q(k) >= 1e5
        e = floor(log10(Q(k)));
        lab{k} = sprintf('%.1f%s10^{%d}', Q(k)/10^e, '\times', e);
    else
        lab{k} = regexprep(sprintf('%d', round(Q(k))), '(\d)(?=(\d{3})+$)', '$1,');
    end
    if approx(k); lab{k} = ['~' lab{k}]; end
    tick{k} = sprintf('%s  (%s)', nm{k}, pol{k});
end

fig = figure('Visible','off','Position',[80 80 1120 640]);
ax = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
for k = 1:n
    barh(ax, k, Q(k), 0.62, 'FaceColor', col(k,:), 'EdgeColor', 'k');
    text(ax, Q(k)*1.12, k, lab{k}, 'FontWeight','bold', 'FontSize', 11, ...
         'VerticalAlignment','middle');
end
set(ax, 'XScale','log', 'YTick', 1:n, 'YTickLabel', repmat({''}, 1, n), ...
    'YDir','reverse', 'FontSize', 11, 'Position', [0.295 0.13 0.66 0.755]);
ylim(ax, [0.4 n+0.6]); xlim(ax, [8e3 2.2e6]);

% Row labels are drawn by hand, in NORMALIZED units, so the note can be
% centred under its own device name: as a second line of a tick label it
% inherits the label's left edge instead, and the labels differ in length.
yn = @(v) (n + 0.6 - v) / (n + 0.2);
mid = nan(1, n);
for k = 1:n
    t = text(ax, -0.012, yn(k), tick{k}, 'Units','normalized', 'FontSize', 11, ...
             'HorizontalAlignment','right', 'VerticalAlignment','middle');
    if isempty(sub{k}); continue; end
    e = get(t, 'Extent');                       % [x y w h], normalized
    mid(k) = e(1) + e(3)/2;
end
% one shared centre for every note, so they line up with each other (the names
% above them differ in length, so per-name centring would stagger them)
xc = mean(mid(~isnan(mid)));
for k = find(~isnan(mid))
    text(ax, xc, yn(k + 0.30), sub{k}, 'Units','normalized', ...
         'FontSize', 11, 'FontWeight','bold', ...
         'HorizontalAlignment','center', 'VerticalAlignment','middle');
end
xlabel(ax, 'Q factor at peak T = -3 dB');
title(ax, {'Q factor at a 20 \mum spatial FWHM and -3 dB insertion loss', ...
           'h 350 nm, SiN 1.97 / SiO_2 1.444, \lambda \approx 1560 nm'});

exportgraphics(fig, fullfile(OUT,'q20um_3db_benchmark.png'), 'Resolution', 200);
savefig(fig, fullfile(OUT,'q20um_3db_benchmark.fig')); close(fig);

for k = 1:n
    fprintf('%-34s %-3s  Q = %9.0f  %s\n', nm{k}, pol{k}, Q(k), lab{k});
end
