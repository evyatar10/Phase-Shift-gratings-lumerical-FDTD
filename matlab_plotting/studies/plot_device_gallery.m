% plot_device_gallery.m — accurate top-view geometry of every device family
% tested this round, each drawn to scale (inner region: cavity +- ~4 periods,
% where all modifications live) with its MEASURED accurate-mesh result read
% from the result .mat. Modified segments highlighted. Headless-safe.
%
% Builder-faithful walk (bragg_device.py): left arm ends with a WIDE tooth
% adjacent to the cavity; right arm starts with a NARROW gap adjacent (the
% pi-shift). Positive gap shift shortens the narrow gap and the cavity absorbs
% 2*sum. x = propagation (horizontal), y = lateral. Colors: base SiN vs the
% highlighted change per device.

clear; close all;
proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
R = fullfile(proj, 'results_from_athena');
out_dir = fullfile(R, 'device_gallery'); if ~exist(out_dir, 'dir'), mkdir(out_dir); end

HP = 0.51683 / 2;                 % half pitch (um)
SIN = [0.82 0.60 0.28]; HL = [0.20 0.55 0.85]; HL2 = [0.35 0.70 0.35];
OX = [0.93 0.96 0.99]; STK = [0.85 0.33 0.10];

% ---- device specs: name, file, cav(um), shifts(nm), wideTeeth(nm from inner),
%      narrowTeeth(nm), apodN, strips[d_um w_um], hlColor, note ----
D = {};
D{end+1} = struct('nm','(1) W800 baseline','f',fullfile(R,'tm_center_completion','results','result_N80_TM_avg_Ybox6p8_Zbox8p8.mat'),'cav',0.800,'sh',[],'wt',[],'nt',[],'apod',0,'st',[],'hc',SIN,'note','plain uniform grating, cavity = avg width');
D{end+1} = struct('nm','(2) rect-1050','f',fullfile(R,'tm_center_completion','results','result_N80_TM_W1050_Ybox6p8_Zbox8p8.mat'),'cav',1.050,'sh',[],'wt',[],'nt',[],'apod',0,'st',[],'hc',HL,'note','cavity widened 800\rightarrow1050 nm (a wider rectangle)');
D{end+1} = struct('nm','(3) see-saw','f',fullfile(R,'tm_center_completion','results','result_N80_TM_W1050_ptw2W1040to980_Ybox6p8_Zbox8p8.mat'),'cav',1.050,'sh',[],'wt',[1.040 0.980],'nt',[],'apod',0,'st',[],'hc',HL2,'note','tooth1 1040 / tooth2 980 (width see-saw)');
D{end+1} = struct('nm','(4) gap-shift pair','f',fullfile(R,'tm_center_completion','results','result_N80_TM_W1050_dsh2S40s20_Ybox6p8_Zbox8p8.mat'),'cav',1.050,'sh',[20 20],'wt',[],'nt',[],'apod',0,'st',[],'hc',HL2,'note','2 inner teeth pulled +20 nm toward cavity');
D{end+1} = struct('nm','(5) THE STACK  (best)','f',fullfile(R,'tm_shift_frontier','results','result_N80_TM_W1050_dsh2S40s20_ptw2W1040to980_Ybox6p8_Zbox8p8.mat'),'cav',1.050,'sh',[20 20],'wt',[1.040 0.980],'nt',[],'apod',0,'st',[],'hc',STK,'note','1050 + pair[+20,+20] + see-saw(1040,980)');
D{end+1} = struct('nm','(6) apodization n=10','f',fullfile(R,'tm_pareto_stack_vs_apod','results','result_N80_A10_TM_avg_Ybox6p8_Zbox8p8.mat'),'cav',0.800,'sh',[],'wt',[],'nt',[],'apod',10,'st',[],'hc',[0.6 0.4 0.7],'note','corrugation tapered over 10 inner teeth');
D{end+1} = struct('nm','(7) strip reflector','f',fullfile(R,'tm_strip_reflector','results','result_N80_TM_W1050_dsh2S40s20_ptw2W1040to980_Ybox7p6_Zbox8p8_scRECT_L84000xW198_X0_Y1200_pair.mat'),'cav',1.050,'sh',[20 20],'wt',[1.040 0.980],'nt',[],'apod',0,'st',[1.20 0.198],'hc',STK,'note','stack + SiN strips at \pmd (drain, worse)');
D{end+1} = struct('nm','(8) derived profile','f',fullfile(R,'tm_derived_profile','results','result_N80_TM_W1031_dsh2S40s20_ptw3W1028to1010_ptn3W619to588_Ybox6p8_Zbox8p8.mat'),'cav',1.0312,'sh',[20 20],'wt',[1.0277 0.9978 1.0098],'nt',[0.6187 0.5870 0.5877],'apod',0,'st',[],'hc',[0.55 0.35 0.35],'note','math-derived shape (worse, both signs)');

% control fwhm for percent
mc = load(D{1}.f); [~,ic] = min(abs(mc.wl_nm - mc.resonance_wavelength_nm));
FW0 = mc.fwhm_m * 1e6;

fig = figure('Visible','off','Position',[20 20 1500 1180],'Color','w');
tl = tiledlayout(fig, 3, 3, 'TileSpacing','compact','Padding','compact');

for q = 1:numel(D)
    s = D{q}; nexttile; hold on;
    m = load(s.f); [~,i] = min(abs(m.wl_nm - m.resonance_wavelength_nm));
    loss = 1 - m.resonance_transmission - m.R(i); T = m.resonance_transmission;
    fw = m.fwhm_m * 1e6; lam = m.resonance_wavelength_nm;

    ND = 4;                                  % periods each side to draw
    shf = @(d) local_pick(s.sh, d, 0) / 1000;      % gap shift of tooth d (um)
    cav_extra = 2 * sum(s.sh) / 1000;

    % apod mod-depth envelope (linear taper of corrugation over apod teeth)
    if s.apod > 0
        modd = @(d) (0.100 + (0.400-0.100) * min(max(d/s.apod,0),1));
    else
        modd = @(d) 0.400;
    end
    wideW = @(d) local_pick(s.wt, d, 0.800 + modd(d)/2);
    narrW = @(d) local_pick(s.nt, d, 0.800 - modd(d)/2);

    % ---- build segment list (x0,x1,w,ishl) left arm -> cavity -> right arm ----
    segs = [];                                % rows: x0 x1 w flag
    x = 0;
    for d = ND:-1:1
        gl = HP - shf(d);
        segs(end+1,:) = [x x+gl narrW(d) (d<=numel(s.sh))]; x = x+gl;
        segs(end+1,:) = [x x+HP wideW(d) (d<=max(numel(s.wt),numel(s.sh)))]; x = x+HP;
    end
    cavx0 = x; cavlen = HP + cav_extra;
    segs(end+1,:) = [x x+cavlen s.cav 2]; x = x + cavlen;   % flag 2 = cavity
    for d = 1:ND
        if d >= 2, sp = shf(d-1); else, sp = 0; end
        gl = HP - sp;
        segs(end+1,:) = [x x+gl narrW(d) (d<=numel(s.sh))]; x = x+gl;
        segs(end+1,:) = [x x+HP wideW(d) (d<=max(numel(s.wt),numel(s.sh)))]; x = x+HP;
    end
    xc = cavx0 + cavlen/2;                    % center on cavity
    segs(:,1:2) = segs(:,1:2) - xc;

    ylimv = 0.78; if ~isempty(s.st), ylimv = 1.55; end
    rectangle('Position',[-2.4 -ylimv 4.8 2*ylimv],'FaceColor',OX,'EdgeColor','none');

    for r = 1:size(segs,1)
        x0=segs(r,1); x1=segs(r,2); w=segs(r,3); fl=segs(r,4);
        c = SIN;
        if fl == 2, c = s.hc; elseif fl == 1, c = s.hc; end
        fill([x0 x1 x1 x0],[-w -w w w]/2, c, 'EdgeColor',[0.25 0.25 0.25],'LineWidth',0.4);
    end
    plot([0 0],[-s.cav s.cav]/2,'k-','LineWidth',1.2);
    text(0, 0, '\pi','HorizontalAlignment','center','FontSize',11,'FontWeight','bold','Color','w');

    if ~isempty(s.st)                          % strips
        d = s.st(1); ws = s.st(2);
        for sgn = [1 -1]
            fill([-2.4 2.4 2.4 -2.4], sgn*d + [-ws -ws ws ws]/2, [0.55 0.40 0.20], ...
                'EdgeColor',[0.3 0.2 0.1],'LineWidth',0.4);
        end
        text(1.9, d, 'SiN strip','FontSize',7,'Color',[0.3 0.2 0.1],'HorizontalAlignment','right');
    end

    xlim([-2.4 2.4]); ylim([-ylimv ylimv]);
    set(gca,'XTick',[],'YTick',[]); box on;
    dfw = (fw/FW0 - 1)*100;
    ttl = sprintf('%s\nloss %.4f   T %.4f   fwhm %+.1f%%', s.nm, loss, T, dfw);
    title(ttl, 'FontSize', 9, 'Interpreter','tex');
    text(0, -ylimv*0.82, s.note, 'HorizontalAlignment','center','FontSize',7.2, ...
        'Color',[0.35 0.35 0.35], 'Interpreter','tex');
end

% ---- panel 9: results summary (loss vs mode width, all devices) ----
nexttile; hold on; grid on;
fill([-1 1 1 -1],[0.03 0.03 0.13 0.13],[0.92 0.96 0.92],'EdgeColor','none','HandleVisibility','off');
for q = 1:numel(D)
    s = D{q}; m = load(s.f); [~,i] = min(abs(m.wl_nm - m.resonance_wavelength_nm));
    loss = 1 - m.resonance_transmission - m.R(i); fw = m.fwhm_m*1e6;
    dfw = (fw/FW0-1)*100;
    mk = 'o'; ms = 8; if q==5, mk='p'; ms=17; end
    plot(dfw, loss, mk, 'MarkerSize', ms, 'MarkerFaceColor', s.hc, ...
        'MarkerEdgeColor','k', 'DisplayName', s.nm);
end
xline(1,':','Color',STK,'HandleVisibility','off');
xlabel('\Delta mode width vs baseline (%)'); ylabel('resonant loss 1-T-R');
set(gca,'YScale','log'); ylim([0.02 0.14]); xlim([-2 30]);
legend('Location','eastoutside','FontSize',6.5);
title('(9) results — loss vs mode width','FontSize',9);

title(tl, sprintf(['TM \\pi-shift Bragg cavity — devices tested (accurate mesh) drawn to scale, ' ...
    'inner region \\pm4 periods.   \\lambda_{res}\\approx1556 nm.   ' ...
    'BEST = the stack: loss 0.0545, T 0.9449, mode width +0.9%%']), ...
    'FontSize', 12, 'Interpreter','tex');

exportgraphics(fig, fullfile(out_dir,'device_gallery.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir,'device_gallery.fig'));
fprintf('saved: %s\n', fullfile(out_dir,'device_gallery.png'));

function w = local_pick(arr, d, def)
if d >= 1 && numel(arr) >= d, w = arr(d); else, w = def; end
end
