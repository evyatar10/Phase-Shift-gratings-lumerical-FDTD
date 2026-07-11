%% Scatterer x-scan, peak transmission vs position — r = 80/100/150/200 nm overlay
% Single-panel MATLAB remake of the peak-T-vs-x figure, with the new r = 80 nm
% curve stacked on the original r = 100/150/200 nm scan (tm_scatterer_scan).
%
% Data (identical numerics/domain across both studies — control 0.79946 shared):
%   r=100/150/200 : results_from_athena/tm_scatterer_scan/results/
%   r=80          : results_from_athena/tm_scatterer_r80_xscan/results/  (first 5 x)
% Control (no scatterer, r=0) = result_N80_TM_avg.mat in the old-scan folder.
%
% Device: TM pi-shift Bragg grating, pitch 516.83 nm, corr 400 nm, h 350 nm, N=80,
% n 1.97/1.444; SiN cylinder PAIR at (x, +/-1.0 um). Deliverables: editable .fig + PNG.
% Headless-safe. Run: plot_scatterer_scan_r80

addpath(fileparts(fileparts(mfilename('fullpath'))));
proj = fileparts(fileparts(mfilename('fullpath')));

old_dir = fullfile(proj, 'results_from_athena', 'tm_scatterer_scan', 'results');
r80_dir = fullfile(proj, 'results_from_athena', 'tm_scatterer_r80_xscan', 'results');

% ── control T (r=0) from the old-scan no-scatterer file ──────────────────────
ctrl_file = fullfile(old_dir, 'result_N80_TM_avg.mat');
assert(isfile(ctrl_file), 'control file not found: %s', ctrl_file);
cd0 = load(ctrl_file);
T_ctrl = double(cd0.resonance_transmission);
lam0   = double(cd0.resonance_wavelength_nm);

% ── collect (r_nm, x_um, T) from both folders ────────────────────────────────
[r_old, x_old, T_old] = read_points(old_dir);
[r_80,  x_80,  T_80 ] = read_points(r80_dir);
r_nm = [r_old, r_80];  x_um = [x_old, x_80];  Tpk = [T_old, T_80];

% ── plot ─────────────────────────────────────────────────────────────────────
FS = 13;
radii = [80 100 150 200];
cols  = [0.47 0.25 0.80;   ...  % r=80  purple (new, stands out)
         0.00 0.45 0.74;   ...  % r=100 blue
         0.85 0.33 0.10;   ...  % r=150 orange
         0.20 0.60 0.20];       % r=200 green

fig = figure('Color', 'w', 'Position', [100 100 980 560]);
ax = axes(fig); hold(ax, 'on'); box(ax, 'on');

yline(ax, T_ctrl, '--', sprintf('control T = %.3f (no scatterer)', T_ctrl), ...
      'Color', [0.4 0.4 0.4], 'LineWidth', 1.2, 'FontSize', FS-2, ...
      'LabelHorizontalAlignment', 'center', 'LabelVerticalAlignment', 'top', ...
      'Interpreter', 'none');

h = gobjects(0); labels = {};
for i = 1:numel(radii)
    m = (r_nm == radii(i));
    if ~any(m), continue; end
    [xs, ord] = sort(x_um(m));  ts = Tpk(m);  ts = ts(ord);
    lw = 1.6 + 1.4*(radii(i) == 80);           % emphasize the new r=80 curve
    ms = 6   + 2*(radii(i) == 80);
    h(end+1) = plot(ax, xs, ts, '-o', 'Color', cols(i,:), 'LineWidth', lw, ...
                    'MarkerSize', ms, 'MarkerFaceColor', 'w'); %#ok<AGROW>
    labels{end+1} = sprintf('r = %d nm', radii(i)); %#ok<AGROW>
end

xlabel(ax, 'scatterer x-position  (\mum)', 'FontSize', FS);
ylabel(ax, 'peak T', 'FontSize', FS);
title(ax, {'Scatterer x-scan \bullet peak T vs position   (SiN pillar pair at \pmy=1 \mum)', ...
      sprintf('TM \\pi-shift, \\Lambda 516.83 / corr 400 / h350 / N80, \\lambda_{res} %.1f nm', lam0)}, ...
      'FontSize', FS-1);
legend(ax, h, labels, 'Location', 'southeast', 'FontSize', FS-1, 'Box', 'off');
xlim(ax, [-0.1 5]);
set(ax, 'FontSize', FS-1); grid(ax, 'on');

% ── save editable .fig + PNG ─────────────────────────────────────────────────
out = fullfile(old_dir, '..', 'scatterer_scan_r80_overlay');
savefig(fig, [out '.fig']);
exportgraphics(fig, [out '.png'], 'Resolution', 200);
fprintf('Saved:\n  %s.fig\n  %s.png\n', out, out);


% ═══════════════════════════════════════════════════════════════════════════════
function [r_nm, x_um, T] = read_points(dirpath)
    % radius/x come from the file-name token (_scR{r}_X{x}_Y{y}); the .mat does
    % not store radius as a field. 'm' prefix = negative nm. T from the .mat.
    r_nm = []; x_um = []; T = [];
    if ~isfolder(dirpath), return; end
    files = dir(fullfile(dirpath, 'result_*_pair.mat'));   % scatterer rows only
    for k = 1:numel(files)
        tok = regexp(files(k).name, '_scR(\d+)_X(m?\d+)_', 'tokens', 'once');
        if isempty(tok), continue; end
        d = load(fullfile(files(k).folder, files(k).name));
        xs = tok{2};  neg = startsWith(xs, 'm');
        xn = str2double(erase(xs, 'm')) * (1 - 2*neg);
        r_nm(end+1) = str2double(tok{1});           %#ok<AGROW>
        x_um(end+1) = xn / 1000;                     %#ok<AGROW>
        T(end+1)    = double(d.resonance_transmission); %#ok<AGROW>
    end
end
