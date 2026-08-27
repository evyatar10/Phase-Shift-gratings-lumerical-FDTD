% plot_q20um_short_benchmark.m
% Study: cross-study benchmark  |  Date: 2026-08-26
% Purpose: the SHORT-DEVICE companion to plot_q20um_3db_benchmark. Same 20 um
%   spatial-FWHM spec, but every device held at ~100 periods per side instead
%   of lengthened to -3 dB, so transmission and Q are read off one common
%   device length. Only three of the seven families have a measurement at that
%   length; the rest are only measured near their own -3 dB point and are
%   deliberately absent rather than interpolated.
% Sources (all MEASURED):
%   uniform TM N=100                   results_from_igum/tm_nladder_c325
%   inverse design + circles N=100     results_from_igum/invdesign_q3db_20um
%   overshoot apodization TE N=98      results_from_igum/itai_hh_nt60w20_summary.csv (IGUM 63722)
% Output: results_from_athena/q20um_3db_benchmark/q20um_short_benchmark.{png,fig}

ROOT = fileparts(fileparts(mfilename('fullpath')));
RA   = fullfile(ROOT, 'results_from_athena');
RI   = fullfile(ROOT, 'results_from_igum');
OUT  = fullfile(RA, 'q20um_3db_benchmark');
if ~isfolder(OUT); mkdir(OUT); end

files = { ...
  'uniform',                           'TM', fullfile(RI,'tm_nladder_c325','results','result_N100_TM_avg_C325_Ybox8p0_Zbox8p8.mat'); ...
  'inverse design 25 teeth + circles', 'TM', fullfile(RI,'invdesign_q3db_20um','results','result_N100_TM_W961_C325_dsh25S66s3_ptw25W964to981_ptn25W640to620_Ybox8p0_Zbox8p8_scR80_arr57_X-14467to15269_Y1900to1900_C325_pair.mat')};

nm = {}; pol = {}; N = []; T = []; Q = []; W = []; Lhalf = [];
for k = 1:size(files,1)
    d = load(files{k,3});
    nm{end+1} = files{k,1}; pol{end+1} = files{k,2};                     %#ok<SAGROW>
    N(end+1)  = double(d.n_periods_each_side);                           %#ok<SAGROW>
    T(end+1)  = d.resonance_transmission;                                %#ok<SAGROW>
    Q(end+1)  = d.resonance_wavelength_nm / abs(d.spectral_fwhm_nm);     %#ok<SAGROW>
    W(end+1)  = d.fwhm_m * 1e6;                                          %#ok<SAGROW>
    Lhalf(end+1) = N(end) * d.pitch_m * 1e6;                             %#ok<SAGROW>
end

% the apodized TE device: its .mat lives on IGUM, the stored summary row here
s = readtable(fullfile(RI, 'itai_hh_nt60w20_summary.csv'));
r = s(s.N == 98, :);
nm{end+1} = 'overshoot apodization 60 teeth'; pol{end+1} = 'TE';
N(end+1) = 98; T(end+1) = r.Tres; Q(end+1) = r.QL; W(end+1) = r.fwhm_um;
Lhalf(end+1) = 98 * 0.49106;                     % its own pitch, 491.06 nm

col = [0.28 0.28 0.28; 0.20 0.50 0.15; 0.64 0.08 0.18];
n = numel(Q);
lab = arrayfun(@(k) sprintf('%s  (%s)  -  %d periods', nm{k}, pol{k}, N(k)), ...
               1:n, 'UniformOutput', false);

fig = figure('Visible','off','Position',[80 80 1240 480]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');

axT = nexttile(tl); hold(axT,'on'); grid(axT,'on'); box(axT,'on');
for k = 1:n
    barh(axT, k, T(k), 0.6, 'FaceColor', col(k,:), 'EdgeColor', 'k');
    text(axT, T(k) + 0.004, k, sprintf('%.4f  (%.2f dB)', T(k), 10*log10(T(k))), ...
         'FontWeight','bold', 'FontSize', 10, 'VerticalAlignment','middle');
end
set(axT, 'YTick', 1:n, 'YTickLabel', lab, 'YDir','reverse', 'FontSize', 10);
ylim(axT, [0.4 n+0.6]); xlim(axT, [0.85 1.06]);
xlabel(axT, 'peak transmission');
title(axT, 'Transmission');

axQ = nexttile(tl); hold(axQ,'on'); grid(axQ,'on'); box(axQ,'on');
for k = 1:n
    barh(axQ, k, Q(k), 0.6, 'FaceColor', col(k,:), 'EdgeColor', 'k');
    text(axQ, Q(k) + 150, k, regexprep(sprintf('%d', round(Q(k))), ...
         '(\d)(?=(\d{3})+$)', '$1,'), 'FontWeight','bold', 'FontSize', 10, ...
         'VerticalAlignment','middle');
end
set(axQ, 'YTick', 1:n, 'YTickLabel', [], 'YDir','reverse', 'FontSize', 10);
ylim(axQ, [0.4 n+0.6]); xlim(axQ, [0 9500]);
xlabel(axQ, 'Q factor');
title(axQ, 'Q factor');

title(tl, {'Short device: the same 20 \mum-mode designs held at ~100 periods per side', ...
           'h 350 nm, SiN 1.97 / SiO_2 1.444  |  not lengthened to -3 dB'}, ...
      'FontWeight','bold');

exportgraphics(fig, fullfile(OUT,'q20um_short_benchmark.png'), 'Resolution', 200);
savefig(fig, fullfile(OUT,'q20um_short_benchmark.fig')); close(fig);

for k = 1:n
    fprintf('%-34s %-3s N=%3d  half %5.1f um  T %.5f (%+.3f dB)  Q %6.0f  mode %.2f um\n', ...
            nm{k}, pol{k}, N(k), Lhalf(k), T(k), 10*log10(T(k)), Q(k), W(k));
end
