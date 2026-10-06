% plot_te_s1_v3_design.m — TE seed S1 under the v3 optimizer: what the design became.
% Study: runners/lumopt2_design (TE lane) | jobs 169105 (v3 toy), 170253 (campaign) |
% 2026-10-06 | Reads the campaign eval log (its first 4 rows are the toy, copied in as
% the warm start) and draws corrugation / tooth shift per free tooth plus T, width
% and Q_i per evaluation. Re-run after fetching a newer lumopt2_te_s1_v3_evals.jsonl.
% Every logged eval is drawn as accepted (true through campaign eval 3); once a trial is
% rejected, filter the rows with lumopt2_te_s1_v3_proj.jsonl.

DIR = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\campaign_te_s1\';
NFREE = 60;  CORR0 = 250;  N_TOY = 4;           % rows 1..4 = toy evals 0..3
SPEC_W = 19.121 * [0.98 1.02];                   % ±2 % width spec (um)

L = splitlines(strtrim(fileread([DIR 'lumopt2_te_s1_v3_evals.jsonl'])));
n = numel(L);
P = zeros(n, 3*NFREE);  T = zeros(n,1);  W = T;  LAM = T;  QI = T;  CAV = T;
for k = 1:n
    r = jsondecode(L{k});
    P(k,:) = r.params(1:3*NFREE);  CAV(k) = r.params(end);
    T(k) = r.t_pk;  W(k) = r.fwhm_env_um;  LAM(k) = r.lam_pk_nm;  QI(k) = r.q_i;
end
corr = P(:, 1:NFREE);  shft = P(:, 2*NFREE+1:3*NFREE);
d = 1:NFREE;
SEED = [0.55 0.55 0.55];  TOY = [0.30 0.55 0.85];  NOW = [0.80 0.20 0.15];

f = figure('Visible','off','Position',[60 60 1250 860],'Color','w');
tl = tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
title(tl, sprintf(['TE \\pi-shift Bragg grating S1 (pitch 500 nm, 98 periods/side, ' ...
    '60 free)  —  \\lambda_{res} %.2f nm,  T %.3f'], LAM(end), T(end)));

ax = nexttile(tl); hold(ax,'on');
plot(ax, d, corr(1,:), '--', 'Color', SEED, 'LineWidth', 1.2);
plot(ax, d, corr(N_TOY,:), 'o-', 'Color', TOY, 'MarkerSize', 3, 'LineWidth', 1.0);
plot(ax, d, corr(end,:), 'o-', 'Color', NOW, 'MarkerFaceColor', NOW, 'MarkerSize', 4, 'LineWidth', 1.5);
xlabel(ax, 'tooth index  (1 = next to the cavity)');  ylabel(ax, 'corrugation (nm)');
legend(ax, {'seed (uniform)', 'end of toy', sprintf('now (campaign eval %d)', n - N_TOY - 1)}, ...
       'Location','southeast');
title(ax, 'Corrugation: the 3 teeth next to the cavity taper down');
grid(ax,'on'); box(ax,'on'); xlim(ax,[0.5 NFREE+0.5]);

ax = nexttile(tl); hold(ax,'on');
bar(ax, d, corr(end,:) - CORR0, 'FaceColor', NOW, 'EdgeColor', 'none');
xlabel(ax, 'tooth index');  ylabel(ax, '\Delta corrugation vs seed (nm)');
title(ax, sprintf('Change vs seed: %+.0f / %+.0f / %+.0f nm inner, ~%+.0f nm outer', ...
      corr(end,1)-CORR0, corr(end,2)-CORR0, corr(end,3)-CORR0, median(corr(end,20:end))-CORR0));
grid(ax,'on'); box(ax,'on'); xlim(ax,[0.5 NFREE+0.5]);

ax = nexttile(tl); hold(ax,'on');
plot(ax, d, shft(N_TOY,:), 'o-', 'Color', TOY, 'MarkerSize', 3);
plot(ax, d, shft(end,:), 'o-', 'Color', NOW, 'MarkerFaceColor', NOW, 'MarkerSize', 4, 'LineWidth', 1.5);
xlabel(ax, 'tooth index');  ylabel(ax, 'tooth shift (nm)');
title(ax, sprintf('Tooth shifts stay tiny (max %.2f nm); cavity width %.0f \\rightarrow %.0f nm', ...
      max(shft(end,:)), CAV(1), CAV(end)));
legend(ax, {'end of toy', 'now'}, 'Location','northeast');
grid(ax,'on'); box(ax,'on'); xlim(ax,[0.5 15.5]);

ax = nexttile(tl); hold(ax,'on');
ev = [0:N_TOY-1, N_TOY-1 + (1:n-N_TOY-1)];          % campaign eval 0 repeats toy eval 3
keep = [1:N_TOY, N_TOY+2:n];
yyaxis(ax,'left');
plot(ax, ev, T(keep), 'o-', 'LineWidth', 1.5, 'MarkerFaceColor', 'auto');  ylabel(ax, 'T at resonance');
yyaxis(ax,'right');
plot(ax, ev, W(keep), 's-', 'LineWidth', 1.2);  ylabel(ax, 'mode width FWHM (\mum)');
yline(ax, SPEC_W(2), ':', '+2 % width spec', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2, ...
      'LabelHorizontalAlignment', 'left');
xline(ax, N_TOY-1, '--', 'campaign start', 'LabelVerticalAlignment', 'bottom');
xlabel(ax, 'accepted evaluation');
title(ax, sprintf('T %.3f \\rightarrow %.3f,  Q_i %.0fk \\rightarrow %.0fk at fixed width', ...
      T(1), T(end), QI(1)/1e3, QI(end)/1e3));
grid(ax,'on'); box(ax,'on');

savefig(f, [DIR 'te_s1_v3_design.fig']);
exportgraphics(f, [DIR 'te_s1_v3_design.png'], 'Resolution', 150);
