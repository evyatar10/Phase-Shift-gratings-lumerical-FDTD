% plot_best_T9609_device.m — layout + profiles of the program-best device.
% Study: runners/lumopt2_design (lumopt2 corr-325 campaign) | Athena job 133530,
% eval 3 | 2026-08-17 | Draws the MEASURED geometry stored in
% runners/lumopt2_design/best_designs.py::BEST_T9609 (T 0.9609, Q_i 103,149).
% Geometry CSVs are emitted by the python side; this script only draws.

SCR  = 'C:\Users\evyat\AppData\Local\Temp\claude\c--Users-evyat-Lumerical-phase-shift-grating-FTDT-codes\24f4c06e-f9a7-4594-af02-45c87057065a\scratchpad\';
OUT  = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\lumopt2_c325_logs\';
R    = readmatrix([SCR 'dev_rects.csv']);      % xc_nm, xspan_nm, yspan_nm
C    = readmatrix([SCR 'dev_comb.csv']);       % x_nm, r_nm, y_nm
PITCH = 516.83; NFREE = 25;
CORE = [0.20 0.45 0.75]; POST = [0.85 0.45 0.15]; CAV = [0.75 0.25 0.25];

f = figure('Visible','off','Position',[60 60 1180 860],'Color','w');
tl = tiledlayout(f,3,1,'TileSpacing','compact','Padding','compact');

% ── layout, inner region (cavity ± 6 periods) ────────────────────────────────
ax1 = nexttile; hold(ax1,'on');
for k = 1:size(R,1)
    xc = R(k,1); xs = R(k,2); ys = R(k,3);
    isCav = (k == (size(R,1)+1)/2);
    if isCav, col = CAV; else, col = CORE; end
    rectangle(ax1,'Position',[xc-xs/2, -ys/2, xs, ys],'FaceColor',col, ...
              'EdgeColor',[0.15 0.15 0.15],'LineWidth',0.4);
end
th = linspace(0,2*pi,40);
for k = 1:size(C,1)
    for sgn = [1 -1]
        patch(ax1, C(k,1)+C(k,2)*cos(th), sgn*C(k,3)+C(k,2)*sin(th), POST, ...
              'EdgeColor',[0.35 0.2 0.05],'LineWidth',0.3);
    end
end
xlim(ax1,[-6.5*PITCH 6.5*PITCH]); ylim(ax1,[-2350 2350]);
xlabel(ax1,'x  (nm)   — propagation'); ylabel(ax1,'y  (nm)');
title(ax1,'Layout near the \pi-shift: SiN core (blue), cavity (red), comb posts (orange)');
grid(ax1,'on'); box(ax1,'on');

% ── corrugation profile ──────────────────────────────────────────────────────
corr = [271.070 287.152 306.579 314.416 314.663 317.163 319.541 320.599 ...
        319.664 320.933 322.966 322.469 320.712 322.808 323.014 320.849 ...
        322.522 322.611 320.222 321.083 321.294 319.181 319.703 320.638 319.641];
shft = [3.224 2.843 4.228 5.345 5.814 6.292 5.948 5.183 4.734 3.868 2.969 ...
        2.259 1.446 0.902 0.889 0.881 0.880 0.887 0.897 0.913 0.933 0.954 ...
        0.979 1.008 1.035];
d = 1:NFREE;
ax2 = nexttile; hold(ax2,'on');
plot(ax2,d,corr,'o-','Color',CORE,'MarkerFaceColor',CORE,'LineWidth',1.5,'MarkerSize',5);
yline(ax2,325,'--','Color',[0.5 0.5 0.5],'LineWidth',1.2,'Label','seed 325 nm', ...
      'LabelHorizontalAlignment','right','FontSize',9);
xlabel(ax2,'period index d   (1 = innermost, next to the cavity)');
ylabel(ax2,'corrugation (nm)');
title(ax2,'Corrugation: inner dip to 271 nm  (\rho = 0.974, outer 75 periods/side frozen at 325)');
grid(ax2,'on'); box(ax2,'on'); xlim(ax2,[0.5 NFREE+0.5]);

% ── tooth shift ──────────────────────────────────────────────────────────────
ax3 = nexttile; hold(ax3,'on');
bar(ax3,d,shft,0.6,'FaceColor',POST,'EdgeColor',[0.35 0.2 0.05]);
xlabel(ax3,'period index d'); ylabel(ax3,'tooth shift s (nm)');
title(ax3,'Tooth shift: 2\Sigmas = 130.6 nm total, absorbed by the cavity (device length constant)');
grid(ax3,'on'); box(ax3,'on'); xlim(ax3,[0.5 NFREE+0.5]);

title(tl, {['\pi-shift Bragg grating, TM, h350 / pitch 516.83 / 100 periods per side / ' ...
            'comb 57 posts r80 d1.9 \mum'], ...
           ['\lambda_{res} = 1566.16 nm,  T = 0.9609,  Q_i = 103,149,  mode width 17.79 \mum ' ...
            '(cavity width 993 nm)']}, 'FontWeight','bold');

savefig(f,[OUT 'best_T9609_device.fig']);
exportgraphics(f,[OUT 'best_T9609_device.png'],'Resolution',180);
disp('done');
