% plot_best_T9635_retrim_device.m — layout + profiles of the RETRIMMED best.
% Study: runners/lumopt2_design (v2 campaign seed) | jobs 136051 (retrim) /
% 136141 (campaign eval) | 2026-08-23 | Draws BEST_T9635 + 52.5 nm uniform
% corr (the width-legal best: T 0.960, lam 1566.06 nm, mode FWHM 17.755 um,
% Q_i ~1e5). Geometry CSVs emitted from best_designs.py (same dir as OUT).
% Comb is UNIFORM: r 80.0+-0.2 nm, spacing 531.0+-0.2 nm, rows at y +-1.9 um.

OUT = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\lumopt2_c325_logs\';
R = readmatrix([OUT 'dev_rects_T9635r.csv']);   % xc, xspan, yspan, iscav
C = readmatrix([OUT 'dev_comb_T9635r.csv']);    % x, r, y
PITCH = 516.83; NFREE = 25;
CORE = [0.20 0.45 0.75]; POST = [0.85 0.45 0.15]; CAV = [0.75 0.25 0.25];

corr = [335.12 341.68 356.11 364.21 365.68 368.95 371.29 372.18 371.25 ...
        372.61 374.78 374.20 372.26 374.49 374.66 372.28 374.13 374.19 ...
        371.60 372.60 372.79 370.55 371.13 372.04 371.05];
shft = [3.224 2.843 4.228 5.345 5.814 6.292 5.948 5.183 4.734 3.868 2.969 ...
        2.259 1.446 0.902 0.889 0.881 0.880 0.887 0.897 0.913 0.933 0.954 ...
        0.979 1.008 1.035];

f = figure('Visible','off','Position',[60 60 1180 880],'Color','w');
tl = tiledlayout(f,3,1,'TileSpacing','compact','Padding','compact');
title(tl, ['\pi-shift Bragg grating, retrimmed best  —  T 0.960,  ' ...
           '\lambda_{res} 1566.06 nm,  mode FWHM 17.76 \mum  (N=100 surrogate)']);

% -- layout near the pi-shift (cavity +- ~6 periods) --------------------------
ax1 = nexttile; hold(ax1,'on');
for k = 1:size(R,1)
    if R(k,4) == 1, col = CAV; else, col = CORE; end
    rectangle(ax1,'Position',[R(k,1)-R(k,2)/2, -R(k,3)/2, R(k,2), R(k,3)], ...
              'FaceColor',col,'EdgeColor',[0.15 0.15 0.15],'LineWidth',0.4);
end
th = linspace(0,2*pi,40);
for k = 1:size(C,1)
    if abs(C(k,1)) > 6.5*PITCH, continue; end
    for sgn = [1 -1]
        patch(ax1, C(k,1)+C(k,2)*cos(th), sgn*C(k,3)+C(k,2)*sin(th), POST, ...
              'EdgeColor',[0.35 0.2 0.05],'LineWidth',0.3);
    end
end
xlim(ax1,[-6.5*PITCH 6.5*PITCH]); ylim(ax1,[-2350 2350]);
xlabel(ax1,'x  (nm)   — propagation'); ylabel(ax1,'y  (nm)');
title(ax1,['Layout: SiN core (blue), cavity 961\times389 nm (red), ' ...
           'UNIFORM comb r=80 nm / 531 nm spacing / y=\pm1.9 \mum (orange)']);
grid(ax1,'on'); box(ax1,'on');

% -- corrugation profile ------------------------------------------------------
ax2 = nexttile; hold(ax2,'on');
d = 1:NFREE;
plot(ax2,d,corr,'o-','Color',CORE,'MarkerFaceColor',CORE,'LineWidth',1.5,'MarkerSize',5);
plot(ax2,NFREE+(1:5),325*ones(1,5),'s--','Color',[0.5 0.5 0.5],'LineWidth',1.2, ...
     'MarkerFaceColor',[0.5 0.5 0.5],'MarkerSize',4);
plot(ax2,[NFREE NFREE+1],[corr(end) 325],':','Color',[0.7 0.3 0.3],'LineWidth',1.2);
text(ax2,NFREE+0.6,349,'46 nm step','FontSize',9,'Color',[0.7 0.3 0.3]);
xlabel(ax2,'period index d   (1 = innermost, next to the cavity)');
ylabel(ax2,'corrugation (nm)');
title(ax2,'Corrugation: taper 335\rightarrow372 nm over ~6 teeth, plateau ~372; outer 75/side frozen at 325');
grid(ax2,'on'); box(ax2,'on'); xlim(ax2,[0.5 NFREE+5.5]); ylim(ax2,[315 385]);

% -- tooth shift --------------------------------------------------------------
ax3 = nexttile; hold(ax3,'on');
bar(ax3,d,shft,0.6,'FaceColor',POST,'EdgeColor',[0.35 0.2 0.05]);
xlabel(ax3,'period index d'); ylabel(ax3,'tooth shift s (nm)');
title(ax3,'Tooth shift: bump peaking 6.3 nm at d=6; 2\Sigmas = 130.6 nm cavity elongation');
grid(ax3,'on'); box(ax3,'on'); xlim(ax3,[0.5 NFREE+0.5]);

exportgraphics(f,[OUT 'best_T9635_retrim_device.png'],'Resolution',180);
savefig(f,[OUT 'best_T9635_retrim_device.fig']);
close(f); disp('written');
