% plot_comb_and_shift_studies.m — the two closed decoration/geometry studies.
% Study: runners/lumopt2_design | jobs 133718 (comb basin, 9 tasks), 133793
% (comb count, 2), 134033 (shift ladder, 3) | 2026-08-18 | All MEASURED at the
% campaign numerics; controls are stored rows, never re-run. Jitter floor 0.002.

FLOOR = 0.002;                       % repeat-measurement jitter floor in T
CORE = [0.20 0.45 0.75]; POST = [0.85 0.45 0.15]; CAV = [0.75 0.25 0.25];
GREY = [0.45 0.45 0.45];

f = figure('Visible','off','Position',[60 60 1180 820],'Color','w');
tl = tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');

% ── 1. comb phase circle ─────────────────────────────────────────────────────
ax = nexttile; hold(ax,'on');
ph = [0 90 180 270 360];  Tph = [0.94629 0.93958 0.93333 0.94006 0.94629];
yline(ax,0.94147,'--','Color',CAV,'LineWidth',1.4,'Label','no comb at all', ...
      'LabelHorizontalAlignment','left','FontSize',9);
plot(ax,ph,Tph,'o-','Color',POST,'MarkerFaceColor',POST,'LineWidth',1.8,'MarkerSize',7);
plot(ax,0,0.94629,'p','MarkerSize',15,'MarkerFaceColor',[0.9 0.75 0.1],'MarkerEdgeColor','k');
xlim(ax,[-15 375]); xticks(ax,0:90:360); grid(ax,'on'); box(ax,'on');
xlabel(ax,'comb phase offset (deg)'); ylabel(ax,'peak transmission T');
title(ax,'Comb phase: built value is the global maximum');

% ── 2. comb pitch vs the light line ──────────────────────────────────────────
ax = nexttile; hold(ax,'on');
pit = [516.83 524 531 540];  Tpit = [0.94144 0.94374 0.94629 0.94165];
yline(ax,0.94147,'--','Color',CAV,'LineWidth',1.4,'Label','no comb','FontSize',9);
xline(ax,529.22,':','Color',GREY,'LineWidth',1.6,'Label','light line 529.2 nm', ...
      'LabelVerticalAlignment','bottom','FontSize',9);
plot(ax,pit,Tpit,'o-','Color',CORE,'MarkerFaceColor',CORE,'LineWidth',1.8,'MarkerSize',7);
plot(ax,531,0.94629,'p','MarkerSize',15,'MarkerFaceColor',[0.9 0.75 0.1],'MarkerEdgeColor','k');
grid(ax,'on'); box(ax,'on'); xlim(ax,[513 543]);
xlabel(ax,'comb pitch \Lambda (nm)'); ylabel(ax,'peak transmission T');
title(ax,'Comb pitch: below the light line the comb is inert');

% ── 3. comb post count ───────────────────────────────────────────────────────
ax = nexttile; hold(ax,'on');
nn = [29 57 113];  Tn = [0.96104 0.9609 0.96167];
ctrl = 0.9609;
fill(ax,[20 130 130 20],[ctrl-FLOOR ctrl-FLOOR ctrl+FLOOR ctrl+FLOOR], ...
     GREY,'FaceAlpha',0.13,'EdgeColor','none');
plot(ax,nn,Tn,'s-','Color',POST,'MarkerFaceColor',POST,'LineWidth',1.8,'MarkerSize',8);
set(ax,'XScale','log'); xticks(ax,nn); xticklabels(ax,{'29','57','113'});
xlim(ax,[24 135]); grid(ax,'on'); box(ax,'on');
xlabel(ax,'comb posts per side'); ylabel(ax,'peak transmission T');
title(ax,'Comb count: flat over 4\times in length (band = \pm jitter floor)');

% ── 4. tooth-shift ladder, T and width together ──────────────────────────────
ax = nexttile; hold(ax,'on');
s2 = [0 65.3 130.6 195.9];  Tl = [0.93613 0.95222 0.9635 0.96747];
rat = [1.0001 1.0055 1.0173 1.0325];
yyaxis(ax,'left');
plot(ax,s2,Tl,'o-','Color',CORE,'MarkerFaceColor',CORE,'LineWidth',1.8,'MarkerSize',7);
plot(ax,130.6,0.9635,'p','MarkerSize',15,'MarkerFaceColor',[0.9 0.75 0.1],'MarkerEdgeColor','k');
ylabel(ax,'peak transmission T'); ax.YColor = CORE;
yyaxis(ax,'right');
plot(ax,s2,rat,'^--','Color',CAV,'MarkerFaceColor',CAV,'LineWidth',1.5,'MarkerSize',6);
yline(ax,1.02,'-','Color',CAV,'LineWidth',1.6,'Label','mode-width limit +2%', ...
      'LabelHorizontalAlignment','left','FontSize',9);
ylabel(ax,'mode width \sigma / \sigma_0'); ax.YColor = CAV;
grid(ax,'on'); box(ax,'on'); xlabel(ax,'total tooth shift 2\Sigmas (nm)');
title(ax,'Tooth shift: no interior optimum - the width spec is the stop');

title(tl, {'\pi-shift Bragg grating corr-325: decoration and shift studies (2026-08-18)', ...
           ['built design T = 0.9635, Q_i = 110,874, \sigma = 17.795 \mum | ' ...
            'stars mark the as-built value | jitter floor 0.002 in T']}, ...
      'FontWeight','bold');

OUT = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\lumopt2_c325_logs\';
savefig(f,[OUT 'comb_and_shift_studies.fig']);
exportgraphics(f,[OUT 'comb_and_shift_studies.png'],'Resolution',180);
disp('done');
