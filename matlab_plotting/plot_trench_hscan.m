% plot_trench_hscan.m — trench-height scan verdict figure (N=150 W800, IGUM).
% Study dir: results_from_igum/trench_n150_hscan | Jobs 41767/42317/42331 | 2026-07-26
% Purpose: peak T (dB, left panel) and Q (right panel) vs air-trench height for
% the full-size TM device; 15 points ctrl..full-z, all measured on IGUM at
% identical numerics (box y=8 um, opt mesh, window 15 nm / 2001 pts).

res_dir = fullfile(fileparts(mfilename('fullpath')), '..', ...
                   'results_from_igum', 'trench_n150_hscan', 'results');
files = dir(fullfile(res_dir, 'result_*.mat'));
assert(~isempty(files), 'no result_*.mat under %s', res_dir);

h_nm = []; T = []; Q = []; lam = [];
for k = 1:numel(files)
    d = load(fullfile(res_dir, files(k).name));
    tok = regexp(files(k).name, '_H(\d+)\.mat$', 'tokens', 'once');
    if ~isempty(tok)
        h = str2double(tok{1});                      % explicit height tag
    elseif contains(files(k).name, 'scRECT')
        h = 350;                                     % core-height trench (no _H tag)
    else
        h = 0;                                       % control, no trench
    end
    h_nm(end+1)  = h;                                            %#ok<*SAGROW>
    T(end+1)     = d.resonance_transmission;
    lam(end+1)   = d.resonance_wavelength_nm;
    Q(end+1)     = d.resonance_wavelength_nm / abs(d.spectral_fwhm_nm);
end
[h_nm, order] = sort(h_nm); T = T(order); Q = Q(order); lam = lam(order);
% User 2026-07-26: plot up to one point before the end — drop the full-z
% (12 um, through-PML) point; the physical curve ends at h = 6.9 um.
keep = h_nm < 12000;
h_nm = h_nm(keep); T = T(keep); Q = Q(keep); lam = lam(keep);
h_um = h_nm / 1e3;

blue = [0 0.45 0.74];
fig = figure('Visible', 'on', 'Position', [60 60 1100 460]);
tl = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
plot(h_um, 10*log10(T), 'o-', 'Color', blue, 'MarkerFaceColor', blue, ...
     'MarkerSize', 5.5, 'LineWidth', 1.2);
grid on; box on;
xlabel('trench height (\mum)');
ylabel('peak transmission (dB)');

nexttile;
plot(h_um, Q, 's-', 'Color', [0.85 0.33 0.10], ...
     'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerSize', 5.5, 'LineWidth', 1.2);
grid on; box on;
xlabel('trench height (\mum)');
ylabel('Q factor');

title(tl, sprintf(['\\pi-shift Bragg TM, W800, 150 periods, air trench ' ...
    'd = 1.8 \\mum, \\lambda_{res} %.1f to %.1f nm'], ...
    min(lam), max(lam)), 'FontSize', 11);

out_dir = fullfile(res_dir, '..');
exportgraphics(fig, fullfile(out_dir, 'trench_hscan_T_Q.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'trench_hscan_T_Q.fig'));
fprintf('saved: %s\nsaved: %s\n', fullfile(out_dir, 'trench_hscan_T_Q.png'), ...
        fullfile(out_dir, 'trench_hscan_T_Q.fig'));
