%% Scatterer-position scan (TM, radiation recycling): lambda_res / T / Q / loss vs position
% Study: runners/sweeps/tm_scatterer_scan.py -> results_from_athena/tm_scatterer_scan/
%
% One SiN cylinder PAIR at (x_s, +/-1.0 um) beside the anchored TM pi-shift grating
% (pitch 516.83 nm, corr 400 nm, h 350 nm, N=80, n 1.97/1.444). Per-task metrics are
% read from the stored fields (scatterer_r_m == 0 marks the in-study control); the
% file-name token _scR{r}_X{x}_Y{y} is only a fallback for older files.
%
% Panels (shared x = scatterer x-position, um):
%   1: resonance wavelength [nm]     2: peak transmission T
%   3: Q = lambda_res/|spectral_fwhm_nm|  (NOT plot_transmission.m's Q — known bug)
%   4: resonant loss 1-R-T at the resonance sample
% Vertical dotted lines: radial recoupling condition rho = m*lambda0/(2*n_clad),
% i.e. x_m = sqrt((m*d)^2 - y0^2). Shaded band: control +/- mesh-jitter noise floor
% (from the +25 nm half-cell offset tasks). Deliverables: editable .fig + PNG.
%
% Headless-safe (no dialogs). Override the data folder before running if needed:
%   data_dir = 'c:\...\results_from_athena\tm_scatterer_scan\results'; plot_scatterer_scan

addpath(fileparts(fileparts(mfilename('fullpath'))));  % project root on path

FONT_SIZE = 12;
N_CLAD = 1.444;
Y0_UM  = 1.0;                     % fixed lateral offset of the pair (um)

if ~exist('data_dir', 'var') || isempty(data_dir)
    proj = fileparts(fileparts(mfilename('fullpath')));
    data_dir = fullfile(proj, 'results_from_athena', 'tm_scatterer_scan', 'results');
end
files = dir(fullfile(data_dir, 'result_*.mat'));
assert(~isempty(files), 'No result_*.mat in %s', data_dir);

n = numel(files);
r_nm  = nan(1, n);  x_um = nan(1, n);
lam   = nan(1, n);  Tpk  = nan(1, n);  Q = nan(1, n);  Lres = nan(1, n);

for k = 1:n
    fp = fullfile(files(k).folder, files(k).name);
    d = load(fp);

    % position/radius: stored fields first, filename token as fallback
    if isfield(d, 'scatterer_r_m')
        % round to integer nm: positions/radii are integer nm by construction,
        % and float round-trip (150e-9*1e9 ~= 150) breaks exact == matching
        r_nm(k) = round(double(d.scatterer_r_m) * 1e9);
        x_um(k) = round(double(d.scatterer_x_m) * 1e9) / 1000;
    else
        tok = regexp(files(k).name, '_scR(\d+)_X(m?\d+)_Y(m?\d+)', 'tokens', 'once');
        if isempty(tok)
            r_nm(k) = 0;  x_um(k) = 0;               % control (no token)
        else
            r_nm(k) = str2double(tok{1});
            xs = tok{2};  sgn = 1;
            if xs(1) == 'm', sgn = -1; xs = xs(2:end); end
            x_um(k) = sgn * str2double(xs) / 1000;
        end
    end

    lam(k)  = double(d.resonance_wavelength_nm);
    Tpk(k)  = double(d.resonance_transmission);
    Q(k)    = lam(k) / abs(double(d.spectral_fwhm_nm));   % TM fwhm stored NEGATIVE
    wl = double(d.wl_nm(:));  Lv = double(d.loss(:));
    [~, ir] = min(abs(wl - lam(k)));
    Lres(k) = Lv(ir);
end

% sanity: resonance must sit inside the scan window (dead/off-window guard)
bad = isnan(lam) | lam < 1543.5 | lam > 1573.5 | Tpk < 0.05;
if any(bad)
    fprintf('WARNING: %d task(s) failed the sanity check (off-window/dead):\n', nnz(bad));
    fprintf('  %s\n', files(bad).name);
end

ctrl = (r_nm == 0) & ~bad;
assert(nnz(ctrl) >= 1, 'No valid control (r=0) task found.');
lam0 = mean(lam(ctrl));  T0 = mean(Tpk(ctrl));  Q0 = mean(Q(ctrl));  L0 = mean(Lres(ctrl));

% mesh-jitter noise floor: +25 nm half-cell tasks vs their main-line partners (r=150)
jit_x = [2.725, 5.425, 8.125];                     % um
noise = struct('lam', 0, 'T', 0, 'Q', 0, 'L', 0);
for jx = jit_x
    kj = find(abs(x_um - jx)     < 1e-6 & r_nm == 150 & ~bad, 1);
    km = find(abs(x_um - jx+0.025) < 1e-6 & r_nm == 150 & ~bad, 1);
    if ~isempty(kj) && ~isempty(km)
        noise.lam = max(noise.lam, abs(lam(kj)  - lam(km)));
        noise.T   = max(noise.T,   abs(Tpk(kj)  - Tpk(km)));
        noise.Q   = max(noise.Q,   abs(Q(kj)    - Q(km)));
        noise.L   = max(noise.L,   abs(Lres(kj) - Lres(km)));
    end
end

% recoupling-phase gridlines: rho = m*d, d = lambda0/(2 n_clad); x = sqrt(rho^2-y0^2)
d_um = lam0 / (2 * N_CLAD) / 1000;
m_max = ceil(sqrt(max(x_um)^2 + Y0_UM^2) / d_um);
x_grid = [];
for m = 1:m_max
    rho = m * d_um;
    if rho >= Y0_UM
        x_grid(end+1) = sqrt(rho^2 - Y0_UM^2); %#ok<SAGROW>
    end
end

%% Figure: 4 stacked panels, one line per radius
radii  = [100 150 200];
colors = [0.00 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19];
fig = figure('Position', [80 60 950 980], 'Color', 'w');
tl = tiledlayout(4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

panels = { ...
    struct('vals', lam,  'base', lam0, 'nz', noise.lam, 'lab', '\lambda_{res}  [nm]'), ...
    struct('vals', Tpk,  'base', T0,   'nz', noise.T,   'lab', 'peak T'), ...
    struct('vals', Q,    'base', Q0,   'nz', noise.Q,   'lab', 'Q = \lambda/|FWHM|'), ...
    struct('vals', Lres, 'base', L0,   'nz', noise.L,   'lab', 'loss 1-R-T @ res')};

ax = gobjects(1, 4);
for p = 1:4
    ax(p) = nexttile;  hold(ax(p), 'on');  grid(ax(p), 'on');
    P = panels{p};
    % noise band around the control value
    xb = [min(x_um) max(x_um)];
    fill(ax(p), [xb fliplr(xb)], ...
         [P.base-P.nz P.base-P.nz P.base+P.nz P.base+P.nz], ...
         [0.5 0.5 0.5], 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
         'DisplayName', 'control \pm jitter');
    yline(ax(p), P.base, 'k--', 'HandleVisibility', 'off');
    for xg = x_grid
        xline(ax(p), xg, ':', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
    end
    for ri = 1:numel(radii)
        sel = (r_nm == radii(ri)) & ~bad;
        if ~any(sel), continue; end
        [xs, io] = sort(x_um(sel));  vv = P.vals(sel);  vv = vv(io);
        plot(ax(p), xs, vv, '.-', 'Color', colors(ri, :), 'MarkerSize', 9, ...
             'DisplayName', sprintf('r = %d nm', radii(ri)));
    end
    ylabel(ax(p), P.lab, 'FontSize', FONT_SIZE);
    set(ax(p), 'FontSize', FONT_SIZE - 1);
    if p < 4, set(ax(p), 'XTickLabel', []); end
end
xlabel(ax(4), 'scatterer x-position  [\mum]', 'FontSize', FONT_SIZE);
legend(ax(1), 'Location', 'best', 'FontSize', FONT_SIZE - 2);
linkaxes(ax, 'x');  xlim(ax(1), [min(x_um) - 0.2, max(x_um) + 0.2]);

% r = 80 nm was NOT position-swept here (only r = 100/150/200). A finer radius
% ladder (tm_scatterer_radius, converged box y6.8/z8.8, accurate mesh) sampled
% r = 80/100/125 nm at the optimal site x = 810 nm; quoted as a note because its
% numerics differ from this old-box scan (control T there = 0.878, not 0.799).
annotation(fig, 'textbox', [0.135 0.605 0.46 0.045], ...
    'String', ['Radius ladder @ x = 810 nm (converged box):  ' ...
               '\DeltaT_{peak} = +0.0020 / +0.0026 / +0.0018 for r = 80 / 100 / 125 nm  ' ...
               '\rightarrow optimum \approx 100 nm'], ...
    'FontSize', FONT_SIZE - 3, 'EdgeColor', [0.6 0.6 0.6], ...
    'BackgroundColor', [1 1 1], 'FaceAlpha', 0.85, 'FitBoxToText', 'on', ...
    'Interpreter', 'tex');

title(tl, {'\pi-shift Bragg TM, pitch 516.83 nm, corr 400 nm, h 350 nm, N80', ...
    sprintf('SiN pair @ \\pm%.1f \\mum — \\lambda_{res}(0) = %.2f nm, T_0 = %.3f', ...
    Y0_UM, lam0, T0)}, 'FontSize', FONT_SIZE);

%% Save deliverables (editable .fig + PNG) next to the data
out_dir = fileparts(data_dir);
png_path = fullfile(out_dir, 'scatterer_scan_summary.png');
fig_path = fullfile(out_dir, 'scatterer_scan_summary.fig');
exportgraphics(fig, png_path, 'Resolution', 200);
savefig(fig, fig_path);

fprintf('\nControl: lambda_res = %.3f nm, T = %.4f, Q = %.0f, loss = %.4f\n', lam0, T0, Q0, L0);
fprintf('Jitter noise floor: dlam = %.4f nm, dT = %.4f, dQ = %.1f, dloss = %.4f\n', ...
        noise.lam, noise.T, noise.Q, noise.L);
fprintf('Recoupling period d = lambda0/(2 n_clad) = %.4f um\n', d_um);
fprintf('Saved: %s\n', png_path);
fprintf('Saved: %s\n', fig_path);
