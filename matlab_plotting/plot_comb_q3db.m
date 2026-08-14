% plot_comb_q3db.m
% Study: results_from_athena/comb_q3db (jobs 130458 wave-1 + 130548 wave-2)
% Date: 2026-08-11
% Purpose: benchmark the comb q3db lock against the stored corr-325 q3db
%   families — ctrl (IGUM trench_q3db_20um), full-z trench (IGUM), flush
%   trench (Athena) — T(dB) and Q vs N with the -3 dB line. Cross-cluster
%   mixing is legitimate: Athena ctrl N165 reproduced the IGUM anchor exactly
%   (Q 13930, job 130458).
%   Also plots the LOCKED device itself (N=169, -3.04 dB): T(lambda) and the
%   mode-width profile, with the measured numbers written on the figures.
% Output: results_from_athena/comb_q3db/comb_q3db_benchmark.{png,fig}
%         results_from_athena/comb_q3db/comb_q3db_N169_transmission.{png,fig}
%         results_from_athena/comb_q3db/comb_q3db_N169_mode_width.{png,fig}

ROOT  = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results_from_athena');
ROOTI = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results_from_igum');
OUT   = fullfile(ROOT, 'comb_q3db');

tq = @(n) fullfile(ROOTI, 'trench_q3db_20um', 'results', ...
    sprintf('result_N%d_TM_avg_C325_Ybox8p0_Zbox8p8.mat', n));
tz = @(n, L) fullfile(ROOTI, 'trench_q3db_20um', 'results', ...
    sprintf('result_N%d_TM_avg_C325_Ybox8p0_Zbox8p8_scRECT_L%dxW800_X0_Y1800_pair_hole_H12000.mat', n, L));
fl = @(n, L) fullfile(ROOT, 'trench_flush_q3db', 'results', ...
    sprintf('result_N%d_TM_avg_C325_Ybox8p0_Zbox8p8_scRECT_L%dxW800_X0_Y1800_pair_hole_Zminm3975.mat', n, L));
cb = @(n) fullfile(ROOT, 'comb_q3db', 'results', ...
    sprintf('result_N%d_TM_avg_C325_Ybox8p0_Zbox8p8_scR80_arr57_X-14467to15269_Y1900to1900_C325_pair.mat', n));

fam = struct( ...
    'name',  {'no decoration (ctrl)', 'full-z trench', 'flush trench', 'comb \Lambda531/270\circ'}, ...
    'col',   {[0.3 0.3 0.3], [0.85 0.33 0.1], [0.93 0.69 0.13], [0 0.45 0.74]}, ...
    'files', { ...
      {tq(150), tq(165), tq(180), tq(195)}, ...
      {tz(165,173000), tz(169,177000), tz(170,178000), tz(185,194000), tz(205,214000), tz(225,235000)}, ...
      {fl(168,176000), fl(169,177000), fl(170,178000)}, ...
      {cb(165), cb(167), cb(168), cb(169)}});

fig = figure('Visible', 'off', 'Position', [80 80 860 720]);
tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
ax1 = nexttile(tl); hold(ax1, 'on'); grid(ax1, 'on');
ax2 = nexttile(tl); hold(ax2, 'on'); grid(ax2, 'on');

for k = 1:numel(fam)
    Ns = []; dB = []; Q = [];
    for j = 1:numel(fam(k).files)
        d = load(fam(k).files{j});
        tok = regexp(fam(k).files{j}, 'result_N(\d+)_', 'tokens', 'once');
        Ns(end+1) = str2double(tok{1});                                    %#ok<SAGROW>
        dB(end+1) = 10*log10(d.resonance_transmission);                    %#ok<SAGROW>
        Q(end+1)  = d.resonance_wavelength_nm / abs(d.spectral_fwhm_nm);   %#ok<SAGROW>
    end
    [Ns, o] = sort(Ns); dB = dB(o); Q = Q(o);
    plot(ax1, Ns, dB, 'o-', 'Color', fam(k).col, 'LineWidth', 1.3, 'MarkerFaceColor', fam(k).col, 'MarkerSize', 5);
    plot(ax2, Ns, Q,  'o-', 'Color', fam(k).col, 'LineWidth', 1.3, 'MarkerFaceColor', fam(k).col, 'MarkerSize', 5);
end
yline(ax1, -3, 'k--', '-3 dB', 'LineWidth', 1.0, 'LabelHorizontalAlignment', 'left');
xlabel(ax1, 'N periods each side'); ylabel(ax1, 'peak T (dB)');
xlabel(ax2, 'N periods each side'); ylabel(ax2, 'loaded Q');
legend(ax1, {fam.name}, 'Location', 'southwest', 'FontSize', 8);
title(ax1, 'Peak transmission vs N (each family locks where it crosses -3 dB)', 'FontSize', 10);
title(ax2, 'Loaded Q vs N', 'FontSize', 10);
title(tl, {'TM q3db benchmark — corr 325, W800, h350, 20 \mum mode'; ...
    'comb: 57 posts r=80, \Lambda=531, 270\circ, d=1.9 \mum (jobs 130458/130548)'}, 'FontSize', 11);

set(fig, 'Visible', 'on');
savefig(fig, fullfile(OUT, 'comb_q3db_benchmark.fig'));
set(fig, 'Visible', 'off');
exportgraphics(fig, fullfile(OUT, 'comb_q3db_benchmark.png'), 'Resolution', 160);
disp(fullfile(OUT, 'comb_q3db_benchmark.png'));

%% The locked device (N=169, -3.04 dB): T(lambda) and mode width
d    = load(cb(169));
lam  = d.resonance_wavelength_nm;
Tpk  = d.resonance_transmission;
fw   = abs(d.spectral_fwhm_nm);          % spectral FWHM is stored negative
Qf   = lam / fw;
wx   = d.fwhm_m * 1e6;

[wl, o] = sort(d.wl_nm(:));              % stored descending
T = d.T(o); Rr = d.R(o);
[~, ires] = min(abs(wl - lam));

res_txt = { sprintf('N = %d each side  (L = %.1f \\mum)', d.n_periods_each_side, d.L_device*1e6), ...
            sprintf('\\lambda_{res} = %.3f nm', lam), ...
            sprintf('T_{peak} = %.4f   (%.2f dB)', Tpk, 10*log10(Tpk)), ...
            sprintf('R = %.4f,  loss = %.4f', Rr(ires), 1 - Tpk - Rr(ires)), ...
            sprintf('linewidth = %.1f pm', fw*1e3), ...
            sprintf('\\bfQ = %.0f\\rm', Qf), ...
            sprintf('mode width = %.2f \\mum', wx)};
dev_title = {'TM \pi-shift Bragg grating + anti-needle comb'; ...
    sprintf('corr 325 nm, W 800 nm, h 350 nm, pitch 516.83 nm  |  comb r=80, \\Lambda=531 nm, 270\\circ, d=1.9 \\mum, 57 posts'); ...
    sprintf('N = %d  |  \\lambda_{res} = %.3f nm  |  T_{peak} = %.4f (%.2f dB)  |  Q = %.0f', ...
        d.n_periods_each_side, lam, Tpk, 10*log10(Tpk), Qf)};

% --- transmission spectrum (full recorded window + resonance line inset) ---
f2 = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 940 640]);
axm = axes(f2); hold(axm, 'on'); grid(axm, 'on');
plot(axm, wl, T, 'Color', [0 0.45 0.74], 'LineWidth', 1.4);
plot(axm, lam, Tpk, 'v', 'Color', [0.85 0.33 0.1], 'MarkerFaceColor', [0.85 0.33 0.1], 'MarkerSize', 7);
xlabel(axm, 'Wavelength [nm]'); ylabel(axm, 'Transmission T');
xlim(axm, [min(wl) max(wl)]); ylim(axm, [0 1.05]);
title(axm, dev_title, 'FontSize', 11);
text(axm, 0.03, 0.60, res_txt, 'Units', 'normalized', 'FontSize', 10, ...
    'VerticalAlignment', 'top', 'BackgroundColor', [1 1 1], 'EdgeColor', [0.6 0.6 0.6]);

axi = axes(f2, 'Position', [0.60 0.42 0.28 0.30]); hold(axi, 'on'); grid(axi, 'on');
m = abs(wl - lam) <= 0.5;
plot(axi, wl(m), T(m), 'Color', [0 0.45 0.74], 'LineWidth', 1.4);
plot(axi, [lam-fw/2 lam+fw/2], [Tpk/2 Tpk/2], 'k--', 'LineWidth', 1.2);
text(axi, lam, Tpk/2, sprintf(' %.1f pm', fw*1e3), 'FontSize', 9, 'VerticalAlignment', 'bottom');
xlabel(axi, '\lambda [nm]'); ylabel(axi, 'T'); xlim(axi, lam + [-0.5 0.5]);
title(axi, 'resonance line', 'FontSize', 9);

set(f2, 'Visible', 'on');
savefig(f2, fullfile(OUT, 'comb_q3db_N169_transmission.fig'));
set(f2, 'Visible', 'off');
exportgraphics(f2, fullfile(OUT, 'comb_q3db_N169_transmission.png'), 'Resolution', 160);
disp(fullfile(OUT, 'comb_q3db_N169_transmission.png'));

% --- mode width (energy density along x at resonance) ---
x   = d.field_x(:) * 1e6;
I   = d.field_energy_density_1D(:);
env = d.field_envelope_1D(:);
sc  = 10^(-floor(log10(max(env))));
yHM = min(env) + 0.5*(max(env) - min(env));
ic  = find(diff(env > yHM) ~= 0);        % half-max crossings, linearly interpolated
xc  = arrayfun(@(k) x(k) + (yHM - env(k)) * (x(k+1)-x(k)) / (env(k+1)-env(k)), ic);
fprintf('mode FWHM: stored %.3f um | from envelope %.3f um\n', wx, xc(end)-xc(1));

f3 = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 940 640]);
ax3 = axes(f3); hold(ax3, 'on'); grid(ax3, 'on');
area(ax3, x, I*sc, 'FaceColor', [0.75 0.75 1], 'EdgeColor', [0.55 0.55 0.85], 'FaceAlpha', 0.55);
plot(ax3, x, env*sc, 'r-', 'LineWidth', 2);
plot(ax3, [xc(1) xc(end)], [yHM yHM]*sc, 'k--', 'LineWidth', 1.5);
text(ax3, mean([xc(1) xc(end)]), yHM*sc*1.06, sprintf('FWHM = %.2f \\mum', wx), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
legend(ax3, {'|E|^2 (integrated over y,z)', 'envelope', 'half maximum'}, 'Location', 'northeast', 'FontSize', 9);
xlabel(ax3, 'Position x [\mum]'); ylabel(ax3, 'Integrated energy density (a.u.)');
xlim(ax3, [min(x) max(x)]);
title(ax3, [dev_title; {sprintf('mode width (spatial FWHM) = %.2f \\mum   (spec 20 \\mum)', wx)}], 'FontSize', 11);
text(ax3, 0.03, 0.95, res_txt, 'Units', 'normalized', 'FontSize', 10, ...
    'VerticalAlignment', 'top', 'BackgroundColor', [1 1 1], 'EdgeColor', [0.6 0.6 0.6]);

set(f3, 'Visible', 'on');
savefig(f3, fullfile(OUT, 'comb_q3db_N169_mode_width.fig'));
set(f3, 'Visible', 'off');
exportgraphics(f3, fullfile(OUT, 'comb_q3db_N169_mode_width.png'), 'Resolution', 160);
disp(fullfile(OUT, 'comb_q3db_N169_mode_width.png'));
