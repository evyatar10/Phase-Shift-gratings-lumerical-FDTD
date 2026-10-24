% plot_nt80x_phase_devices.m -- top views of the devices in the off-lattice tab (phase) test, from the runner's exact geometry.
% Study dir: runners/sweeps/ (farfield_sph_15um_phase, _phase_plain) | Jobs: Athena 265799 (plain), 265802 (Nt80x) | 2026-10-21
% Purpose (user): show which devices are being tried. Input: results_from_athena/farfield_sph_15um_phase/phase_devices.json
% (written from the runner modules: teeth per period + tab x positions + protrusion). Upper sidewall only (the lower one is the mirror
% image); tabs drawn x10 in protrusion so the 5-nm features are visible. Output: phase_devices.{fig,png} in the same folder.
out = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results_from_athena', 'farfield_sph_15um_phase');
D = jsondecode(fileread(fullfile(out, 'phase_devices.json')));
EX = 10; XL = [-5 5];
N = D.NT80X; Pl = D.PLAIN;
panels = {
  {N, N.nar0, N.wid0, [],          'Nt80x lab device (Itai overshoot, N = 96): control  --  radiation 1.000, Q_i 5.35e5'}
  {N, N.nar,  N.wid,  [],          'parent: contrast +20 nm on the first two periods  --  MEASURED radiation 0.684, Q_i +40 %, lambda/W held'}
  {N, N.nar,  N.wid,  N.rows.r0_d5, 'parent + off-lattice tabs, registration r0, 5 nm  --  running (265802)'}
  {N, N.nar,  N.wid,  N.rows.rh_d5, 'parent + the same tabs shifted by half a period (+257 nm), 5 nm  --  running (265802)'}
  {Pl, Pl.nar, Pl.wid, Pl.rows.r0_d10, 'plain pi-shift device (corr 316, pitch 500) + tabs r0, 10 nm  --  MEASURED radiation 1.010 (r0+half: 0.991): no overlap'}
};
f = figure('Color', 'w', 'Position', [60 60 1500 1050]);
for k = 1:numel(panels)
    p = panels{k}; G = p{1}; nar = p{2}(:)' * 1e-3; wid = p{3}(:)' * 1e-3; P = G.pitch * 1e-3; wcav = G.cav * 1e-3;
    subplot(numel(panels), 1, k); hold on;
    [xe, ye] = edge_line(nar, wid, wcav, P, XL);
    fill([xe fliplr(xe)], [ye zeros(size(ye))], [0.75 0.82 0.95], 'EdgeColor', 'none');
    plot(xe, ye, 'Color', [0.2 0.35 0.7], 'LineWidth', 1.2);
    if k == 2                                                % show what the contrast knob changed
        [xs, ys] = edge_line(N.nar0(:)' * 1e-3, N.wid0(:)' * 1e-3, wcav, P, XL);
        plot(xs, ys, 'k--', 'LineWidth', 0.8);
    end
    if ~isempty(p{4})
        T = p{4}; xt = T.x(:)' * 1e-3; dd = T.d * 1e-3 * EX;
        for j = 1:numel(xt)
            [~, ii] = min(abs(xe - xt(j))); yw = ye(ii);
            rectangle('Position', [xt(j) - 0.05, yw, 0.10, dd], 'FaceColor', [0.85 0.3 0.15], 'EdgeColor', 'none');
        end
    end
    xline(0, ':', 'Color', [0.4 0.4 0.4]);
    xlim(XL); ylim([0.25 0.70]); set(gca, 'FontSize', 10); ylabel('y (\mum)');
    title(p{5}, 'Interpreter', 'none', 'FontSize', 11);
    if k == numel(panels), xlabel('x along the waveguide (\mum); \pi-shift segment at x = 0'); end
end
sgtitle(sprintf('Top view, upper sidewall: devices in the off-lattice tab (phase) test, 15.5 \\mum mode, \\lambda 1619.5 nm (tabs drawn x%d in height; dashed = control edge)', EX), 'FontSize', 12);
exportgraphics(f, fullfile(out, 'phase_devices.png'), 'Resolution', 150); savefig(f, fullfile(out, 'phase_devices.fig'));

% ---- figure 2 (2026-10-21, user: "images of the device and what happens"): every family on one page, zoomed on the centre,
% from devices_overview.json (written from the runner modules). Tabs orange, notches = cut-outs with a black outline, x10 in height.
V = jsondecode(fileread(fullfile(out, 'devices_overview.json'))); XZ = [-2.6 2.6]; Pz = V.pitch * 1e-3; wc = V.cav * 1e-3;
f2 = figure('Color', 'w', 'Position', [60 60 1500 1150]);
for k = 1:numel(V.panels)
    if iscell(V.panels), p = V.panels{k}; else, p = V.panels(k); end
    subplot(numel(V.panels), 1, k); hold on;
    [xe, ye] = edge_line(p.nar(:)' * 1e-3, p.wid(:)' * 1e-3, wc, Pz, XZ);
    fill([xe fliplr(xe)], [ye zeros(size(ye))], [0.75 0.82 0.95], 'EdgeColor', 'none');
    plot(xe, ye, 'Color', [0.2 0.35 0.7], 'LineWidth', 1.2);
    if isfield(p, 'dashed') && p.dashed
        [xs, ys] = edge_line(V.nar0(:)' * 1e-3, V.wid0(:)' * 1e-3, wc, Pz, XZ); plot(xs, ys, 'k--', 'LineWidth', 0.9);
    end
    if isfield(p, 'plane'), xline(p.plane, ':', 'Color', [0.6 0.1 0.1], 'LineWidth', 1.2); end
    F = p.feats;
    for j = 1:numel(F)
        if iscell(F), q = F{j}; else, q = F(j); end
        xt = q.x * 1e-3; [~, ii] = min(abs(xe - xt)); yw = ye(ii); dd = q.d * 1e-3 * EX;
        if strcmp(q.kind, 'tab')
            rectangle('Position', [xt - 0.05, yw, 0.10, dd], 'FaceColor', [0.85 0.3 0.15], 'EdgeColor', 'none');
        else
            rectangle('Position', [xt - 0.05, yw - dd, 0.10, dd], 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 1);
        end
    end
    patch([-Pz/4 Pz/4 Pz/4 -Pz/4], [0.26 0.26 0.27 0.27], [0.3 0.3 0.3], 'EdgeColor', 'none');   % cavity segment marker
    xlim(XZ); ylim([0.25 0.62]); set(gca, 'FontSize', 10); ylabel('y (\mum)');
    title(p.title, 'Interpreter', 'none', 'FontSize', 10.5);
    if k == numel(V.panels), xlabel('x along the waveguide (\mum); grey bar = the 257-nm \pi-shift cavity segment'); end
end
sgtitle(['Nt80x lab device, upper sidewall near the \pi shift (15.5 \mum mode, \lambda 1619.5 nm): what we tried (features x' num2str(EX) ' in height)'], 'FontSize', 12);
exportgraphics(f2, fullfile(out, 'devices_overview.png'), 'Resolution', 150); savefig(f2, fullfile(out, 'devices_overview.fig'));

% ---- figures 3-4 (2026-10-21, user): working devices from panel JSON files (plain-TE junction shoulders; Nt80x notch device)
panel_figure(fullfile(fileparts(out), 'farfield_sph_15um_junction_plain', 'plain_devices.json'), ', 15.3 \mum mode, \lambda 1558.5 nm');
panel_figure(fullfile(fileparts(out), 'farfield_sph_15um_cavity', 'nt80x_notch_device.json'), ', 15.5 \mum mode, \lambda 1619.5 nm');

function panel_figure(jf, sub)
% One figure from a panel JSON: per panel teeth widths, optional dashed control edge / cavity width, tabs (orange) and notches (white).
Vp = jsondecode(fileread(jf)); outp = fileparts(jf); Pp = Vp.pitch * 1e-3; XP = [-Vp.zoom Vp.zoom]; YL = [0.25 0.55];
if isfield(Vp, 'ylim'), YL = Vp.ylim(:)'; end
f = figure('Color', 'w', 'Position', [60 60 1400 330 * numel(Vp.panels)]);
for k = 1:numel(Vp.panels)
    if iscell(Vp.panels), p = Vp.panels{k}; else, p = Vp.panels(k); end
    subplot(numel(Vp.panels), 1, k); hold on;
    wcp = Vp.cav * 1e-3; if isfield(p, 'cavw'), wcp = p.cavw * 1e-3; end
    [xe, ye] = edge_line(p.nar(:)' * 1e-3, p.wid(:)' * 1e-3, wcp, Pp, XP);
    fill([xe fliplr(xe)], [ye zeros(size(ye))], [0.75 0.82 0.95], 'EdgeColor', 'none'); plot(xe, ye, 'Color', [0.2 0.35 0.7], 'LineWidth', 1.2);
    if isfield(p, 'dashed') && p.dashed
        [xs, ys] = edge_line(Vp.nar0(:)' * 1e-3, Vp.wid0(:)' * 1e-3, Vp.cav * 1e-3, Pp, XP); plot(xs, ys, 'k--', 'LineWidth', 0.9);
    end
    F = p.feats;
    for j = 1:numel(F)
        if iscell(F), q = F{j}; else, q = F(j); end
        xt = q.x * 1e-3; [~, ii] = min(abs(xe - xt)); dd = q.d * 1e-3 * Vp.ex;
        if strcmp(q.kind, 'tab'), rectangle('Position', [xt - 0.05, ye(ii), 0.10, dd], 'FaceColor', [0.85 0.3 0.15], 'EdgeColor', 'none');
        else, rectangle('Position', [xt - 0.05, ye(ii) - dd, 0.10, dd], 'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 1); end
    end
    patch([-Pp/4 Pp/4 Pp/4 -Pp/4], YL(1) + [0.01 0.01 0.02 0.02], [0.3 0.3 0.3], 'EdgeColor', 'none');
    xlim(XP); ylim(YL); set(gca, 'FontSize', 10); ylabel('y (\mum)'); title(p.title, 'Interpreter', 'none', 'FontSize', 11);
    if k == numel(Vp.panels), xlabel('x along the waveguide (\mum); grey bar = cavity segment'); end
end
sgtitle([Vp.sup sub], 'FontSize', 12);
exportgraphics(f, fullfile(outp, [Vp.out '.png']), 'Resolution', 150); savefig(f, fullfile(outp, [Vp.out '.fig']));
end

function [xe, ye] = edge_line(nar, wid, wcav, P, XL)
% Upper edge sampled every 1 nm: cavity |x| < P/4; right arm narrow then wide in each period, left arm wide (cavity side) then narrow
% (same rule as bragg_device / the runner's half_width_nm).
xe = XL(1):0.001:XL(2); ye = zeros(size(xe));
for i = 1:numel(xe)
    s = abs(xe(i)) - P/4;
    if s < 0, ye(i) = wcav / 2; continue; end
    j = floor(s / P) + 1; fr = (s - (j - 1) * P) / P;
    if xe(i) > 0, w = nar(j) * (fr < 0.5) + wid(j) * (fr >= 0.5); else, w = wid(j) * (fr < 0.5) + nar(j) * (fr >= 0.5); end
    ye(i) = w / 2;
end
end
