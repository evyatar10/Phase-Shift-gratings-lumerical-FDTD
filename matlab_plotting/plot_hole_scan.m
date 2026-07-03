%% In-core hole scan (TM, flipped material): lambda_res / T / Q / loss vs hole position
% Study: runners/sweeps/tm_hole_scan.py -> results_from_athena/tm_hole_scan/
%
% A single SiO2 cylinder (r = 100 nm, n = 1.444, full core height) punched into the
% SiN core ON the guide axis, stepped from the cavity center outward at pitch/8 —
% the scan resolves the defect-mode standing wave (hole at a NODE ~ benign, at an
% ANTINODE ~ maximally scattering; expect a BLUE lambda shift near the cavity).
%
% Panels (shared x = hole x-position, um): lambda_res, peak T, Q, resonant loss.
% Dotted vertical lines every pitch/2 (the half-period lattice of the grating).
% Deliverables: editable .fig + PNG. Headless-safe (no dialogs); override with
%   data_dir = '...\results_from_athena\tm_hole_scan\results'; plot_hole_scan

addpath(fileparts(fileparts(mfilename('fullpath'))));  % project root on path

FONT_SIZE = 12;
PITCH_UM  = 0.51683;

if ~exist('data_dir', 'var') || isempty(data_dir)
    proj = fileparts(fileparts(mfilename('fullpath')));
    data_dir = fullfile(proj, 'results_from_athena', 'tm_hole_scan', 'results');
end
files = dir(fullfile(data_dir, 'result_*.mat'));
assert(~isempty(files), 'No result_*.mat in %s', data_dir);

n = numel(files);
r_nm  = nan(1, n);  x_um = nan(1, n);
lam   = nan(1, n);  Tpk  = nan(1, n);  Q = nan(1, n);  Lres = nan(1, n);

for k = 1:n
    d = load(fullfile(files(k).folder, files(k).name));
    r_nm(k) = round(double(d.scatterer_r_m) * 1e9);          % 0 = control
    x_um(k) = round(double(d.scatterer_x_m) * 1e9) / 1000;
    lam(k)  = double(d.resonance_wavelength_nm);
    Tpk(k)  = double(d.resonance_transmission);
    Q(k)    = lam(k) / abs(double(d.spectral_fwhm_nm));      % TM fwhm stored NEGATIVE
    wl = double(d.wl_nm(:));  Lv = double(d.loss(:));
    [~, ir] = min(abs(wl - lam(k)));
    Lres(k) = Lv(ir);
end

% sanity: in-window resonance; T floor. A hole CAN legitimately spoil the cavity,
% so out-of-family points are reported, plotted, but flagged.
bad = isnan(lam) | lam < 1543.5 | lam > 1573.5 | Tpk < 0.02;
if any(bad)
    fprintf('NOTE: %d task(s) out of family (off-window or T<0.02):\n', nnz(bad));
    fprintf('  %s\n', files(bad).name);
end

ctrl = (r_nm == 0) & ~bad;
assert(nnz(ctrl) >= 1, 'No valid control (r=0) task found.');
lam0 = mean(lam(ctrl));  T0 = mean(Tpk(ctrl));  Q0 = mean(Q(ctrl));  L0 = mean(Lres(ctrl));

sel = (r_nm > 0) & ~bad;
[xs, io] = sort(x_um(sel));

fig = figure('Position', [80 60 950 980], 'Color', 'w');
tl = tiledlayout(4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

panels = { ...
    struct('vals', lam,  'base', lam0, 'lab', '\lambda_{res}  [nm]'), ...
    struct('vals', Tpk,  'base', T0,   'lab', 'peak T'), ...
    struct('vals', Q,    'base', Q0,   'lab', 'Q = \lambda/|FWHM|'), ...
    struct('vals', Lres, 'base', L0,   'lab', 'loss 1-R-T @ res')};

ax = gobjects(1, 4);
for p = 1:4
    ax(p) = nexttile;  hold(ax(p), 'on');  grid(ax(p), 'on');
    P = panels{p};
    yline(ax(p), P.base, 'k--', 'DisplayName', 'no-hole control');
    for m = 1:ceil(max(x_um) / (PITCH_UM / 2))
        xline(ax(p), m * PITCH_UM / 2, ':', 'Color', [0.4 0.4 0.4], ...
              'HandleVisibility', 'off');
    end
    vv = P.vals(sel);  vv = vv(io);
    plot(ax(p), xs, vv, '.-', 'Color', [0.49 0.18 0.56], 'MarkerSize', 9, ...
         'DisplayName', 'SiO_2 hole r = 100 nm');
    ylabel(ax(p), P.lab, 'FontSize', FONT_SIZE);
    set(ax(p), 'FontSize', FONT_SIZE - 1);
    if p < 4, set(ax(p), 'XTickLabel', []); end
end
xlabel(ax(4), 'hole x-position  [\mum]', 'FontSize', FONT_SIZE);
legend(ax(1), 'Location', 'best', 'FontSize', FONT_SIZE - 2);
linkaxes(ax, 'x');  xlim(ax(1), [-0.1, max(x_um) + 0.15]);

title(tl, {'\pi-shift Bragg TM, pitch 516.83 nm, corr 400 nm, h 350 nm, N80', ...
    sprintf('in-core SiO_2 hole @ y = 0 — \\lambda_{res}(0) = %.2f nm, T_0 = %.3f', ...
    lam0, T0)}, 'FontSize', FONT_SIZE);

out_dir = fileparts(data_dir);
png_path = fullfile(out_dir, 'hole_scan_summary.png');
fig_path = fullfile(out_dir, 'hole_scan_summary.fig');
exportgraphics(fig, png_path, 'Resolution', 200);
savefig(fig, fig_path);

fprintf('\nControl: lambda_res = %.3f nm, T = %.4f, Q = %.0f, loss = %.4f\n', lam0, T0, Q0, L0);
fprintf('Saved: %s\n', png_path);
fprintf('Saved: %s\n', fig_path);
