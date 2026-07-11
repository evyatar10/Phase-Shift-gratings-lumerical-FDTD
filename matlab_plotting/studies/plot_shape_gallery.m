% plot_shape_gallery.m — the NON-rectangular teeth & cavity shapes we tested,
% each drawn from the builder's actual polygon math (bragg_device.py:
% add_shaped_tooth + the cavity_shape profiles) with its MEASURED result.
% All rows are OPTIMIZATION mesh (dx=50), avg-800 rect cavity baseline = 0.1098
% -> compare WITHIN this set; the rectangular winners (rect-1050, the stack)
% are marked as reference lines. Headless-safe.

clear; close all;
proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
R = fullfile(proj,'results_from_athena');
out_dir = fullfile(R,'device_gallery'); if ~exist(out_dir,'dir'), mkdir(out_dir); end
HP = 0.51683/2; SIN=[0.82 0.60 0.28]; OX=[0.93 0.96 0.99];
GOOD=[0.30 0.55 0.35]; BAD=[0.80 0.30 0.20]; NEU=[0.55 0.55 0.60];

BASE   = met(fullfile(R,'inner_shape_study','results','result_N80_TM_avg_Ybox6p8_Zbox8p8.mat'));
RC1050 = met(fullfile(R,'cavity_width_ladder','results','result_N80_TM_W1050_Ybox6p8_Zbox8p8.mat'));

% device list: name, file, kind('tooth'|'cav'|'rectwide'|'none'), shape, depth(um for cav), note
% Trimmed set: baseline, the plain wider rect winner, and the two transmission-
% improving bulges (Hann, Gaussian). Barrel is shown as a depth SCAN panel below.
S = {};
S{end+1}=struct('nm','baseline: rect teeth + rect cavity','d','inner_shape_study','f','result_N80_TM_avg_Ybox6p8_Zbox8p8.mat','k','none','sh','','dp',0,'note','all rectangles (reference)');
S{end+1}=struct('nm','rect-1050: plain wider cavity','d','cavity_width_ladder','f','result_N80_TM_W1050_Ybox6p8_Zbox8p8.mat','k','rectwide','sh','','dp',0,'note','plain rectangle, wider (1050 nm)','Wcav',1.050);
S{end+1}=struct('nm','CAVITY: Hann','d','cavity_design_study','f','result_N80_TM_avg_cavhann300_Ybox6p8_Zbox8p8.mat','k','cav','sh','hann','dp',0.300,'note','raised-cosine bulge, +area');
S{end+1}=struct('nm','CAVITY: Gaussian','d','cavity_design_study','f','result_N80_TM_avg_cavgaus300_Ybox6p8_Zbox8p8.mat','k','cav','sh','gauss','dp',0.300,'note','Gaussian bulge, +area');

fig=figure('Visible','off','Position',[20 20 1500 900],'Color','w');
tl=tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');

for q=1:numel(S)
    s=S{q}; nexttile; hold on;
    m=met(fullfile(R,s.d,'results',s.f));
    dT=(m.T-BASE.T)*1e3;
    if dT > 2, col=GOOD; elseif dT < -2, col=BAD; else, col=NEU; end

    yl=0.82; rectangle('Position',[-1.5 -yl 3 2*yl],'FaceColor',OX,'EdgeColor','none');
    % draw cavity +- 3 periods (rect teeth), replacing cavity or innermost tooth
    Wn=0.600; Ww=1.000;
    if isfield(s,'Wcav'), Wc=s.Wcav; else, Wc=0.800; end
    % segments as [x0 x1 w]; build symmetric walk (approx: cavity centered)
    segL={}; x=0;
    for d=3:-1:1
        segL{end+1}=[x x+HP Wn]; x=x+HP;               % narrow
        segL{end+1}=[x x+HP Ww d];  x=x+HP;            % wide (tag inner=1)
    end
    cav0=x; cavlen=HP; x=x+cavlen;
    segR={};
    for d=1:3
        segR{end+1}=[x x+HP Wn]; x=x+HP;
        segR{end+1}=[x x+HP Ww d]; x=x+HP;
    end
    xc=cav0+cavlen/2;
    drawseg=@(v,shaded) fill([v(1) v(2) v(2) v(1)]-xc,[-v(3) -v(3) v(3) v(3)]/2, ...
        shaded,'EdgeColor',[0.25 0.25 0.25],'LineWidth',0.4);

    % left teeth (last wide = innermost, adjacent to cavity)
    for r=1:numel(segL)
        v=segL{r}; isInner=(numel(v)>=4 && v(4)==1 && r==numel(segL));
        if isInner && strcmp(s.k,'tooth')
            drawShapedTooth(v(1)-xc, v(2)-xc, Wn, Ww, s.sh, 'L', col);
        else
            drawseg(v(1:3), SIN);
        end
    end
    % cavity
    if strcmp(s.k,'cav')
        drawShapedCavity(cav0-xc, cavlen, Wc, s.sh, s.dp, col);
    elseif strcmp(s.k,'rectwide')
        drawseg([cav0 cav0+cavlen Wc], col);   % plain wider rect, colored
    else
        drawseg([cav0 cav0+cavlen Wc], SIN);
    end
    % right teeth (first wide after gap = innermost on right)
    for r=1:numel(segR)
        v=segR{r}; isInner=(numel(v)>=4 && v(4)==1 && r==2);
        if isInner && strcmp(s.k,'tooth')
            drawShapedTooth(v(1)-xc, v(2)-xc, Wn, Ww, s.sh, 'R', col);
        else
            drawseg(v(1:3), SIN);
        end
    end
    plot([0 0],[-Wc Wc]/2,'k-','LineWidth',1);
    xlim([-1.5 1.5]); ylim([-yl yl]); set(gca,'XTick',[],'YTick',[]); box on;
    title(sprintf('%s\nT = %.4f  (%+.1f\\times10^{-3} vs rect)', s.nm, m.T, dT), ...
        'FontSize',8.5,'Interpreter','tex');
    text(0,-yl*0.86,s.note,'HorizontalAlignment','center','FontSize',7,'Color',[0.35 0.35 0.35]);
end

% ---- panel 5: barrel bulge DEPTH scan (T vs depth) ----
nexttile; hold on; grid on;
bd=[75 150 168 225 300 400];
bdir={'inner_shape_study','barrel_followup','barrel_followup','barrel_followup','barrel_followup','barrel_followup'};
bT=zeros(size(bd));
for j=1:numel(bd)
    mm=met(fullfile(R,bdir{j},'results',sprintf('result_N80_TM_avg_cavbarr%d_Ybox6p8_Zbox8p8.mat',bd(j))));
    bT(j)=mm.T;
end
yline(RC1050.T,'-','Color',[0.19 0.45 0.72],'LineWidth',1.4,'Label','rect-1050','LabelHorizontalAlignment','left','FontSize',7);
yline(BASE.T,'--','Color',NEU,'LineWidth',1.2,'Label','baseline 800','LabelHorizontalAlignment','left','FontSize',7);
plot(bd,bT,'-o','Color',GOOD,'MarkerFaceColor',GOOD,'LineWidth',1.6,'MarkerSize',5);
xlabel('barrel bulge depth (nm)'); ylabel('transmission T'); xlim([50 420]); ylim([0.882 0.926]);
title('barrel bulge depth scan','FontSize',8.5);

% ---- panel 6: results bars vs the stack winner ----
nexttile; hold on; grid on;
names={}; vals=[]; cols=[];
for q=1:numel(S)
    s=S{q}; m=met(fullfile(R,s.d,'results',s.f));
    short=regexprep(s.nm,'^(TEETH|CAVITY): ','');
    if q==1, short='baseline'; elseif strcmp(s.k,'rectwide'), short='rect-1050'; end
    names{end+1}=short; vals(end+1)=m.T;
    dT=(m.T-BASE.T)*1e3;
    if dT>2, cols(end+1,:)=GOOD; elseif dT<-2, cols(end+1,:)=BAD; else, cols(end+1,:)=NEU; end
end
[vs,ord]=sort(vals,'descend');   % highest T at bottom
b=barh(vs,'FaceColor','flat'); b.CData=cols(ord,:);
set(gca,'YTick',1:numel(vs),'YTickLabel',names(ord),'FontSize',8);
xline(0.9449,'-','Color',[0.85 0.33 0.10],'LineWidth',1.5,'Label','the stack','FontSize',7);
xlabel('resonant transmission T  (opt mesh)'); xlim([0.85 0.955]);
title('shapes vs the stack winner','FontSize',8.5);

title(tl,{'TM \pi-shift Bragg cavity — transmission-improving cavity bulges vs a plain wider rectangle (opt mesh, to scale)', ...
    'green = higher transmission vs the rect-800 baseline; every bulge just adds area, and the plain rect-1050 matches or beats them'}, ...
    'FontSize',9,'Interpreter','tex');

exportgraphics(fig,fullfile(out_dir,'shape_gallery.png'),'Resolution',200);
savefig(fig,fullfile(out_dir,'shape_gallery.fig'));
fprintf('saved: %s\n', fullfile(out_dir,'shape_gallery.png'));

% ===================== shape drawing helpers =====================
function m = met(fp)
    d=load(fp); [~,i]=min(abs(d.wl_nm-d.resonance_wavelength_nm));
    m=struct('loss',1-d.resonance_transmission-d.R(i),'T',d.resonance_transmission,'fw',d.fwhm_m*1e6);
end

function drawShapedTooth(x1,x2,wn,ww,shape,arm,col)
    base=0.5*wn; h=0.5*(ww-wn);
    % base narrow segment
    fill([x1 x2 x2 x1],[-base -base base base],[0.82 0.60 0.28],'EdgeColor',[0.25 0.25 0.25],'LineWidth',0.4);
    switch shape
        case 'ellipse'
            u=linspace(-1,1,25); xc=0.5*(x1+x2); a=0.5*(x2-x1);
            px=xc+a*u; py=base+h*sqrt(max(0,1-u.^2));
            vx=[x1 px x2]; vy=[base py base];
        case 'tri'
            vx=[x1 0.5*(x1+x2) x2]; vy=[base base+h base];
        case 'wedge_cav'
            if arm=='L', vx=[x1 x2 x2]; vy=[base base base+h];
            else,        vx=[x1 x1 x2]; vy=[base base+h base]; end
        otherwise
            vx=[x1 x2 x2 x1]; vy=[base base base base];
    end
    for sgn=[1 -1]
        fill(vx, sgn*vy, col,'EdgeColor',[0.25 0.25 0.25],'LineWidth',0.4);
    end
end

function drawShapedCavity(x0,L,Wc,shape,dp,col)
    u=linspace(0,1,60);
    switch shape
        case 'barrel', w=Wc+dp*sin(pi*u);
        case 'hour',   w=Wc-dp*sin(pi*u);
        case 'hann',   w=Wc+dp*sin(pi*u).^2;
        case 'gauss',  w=Wc+dp*exp(-((u-0.5)/0.15).^2);
        case 'tri3',   p=0.3; w=Wc+dp*min(u/p,(1-u)/(1-p));
        case 'tri5',   p=0.5; w=Wc+dp*min(u/p,(1-u)/(1-p));
        case 'dbl2',   u2=mod(u,0.5)/0.5; w=Wc+dp*(1-abs(u2-0.5)/0.5);
        case 'slup',   w=0.6+(1.0-0.6)*u;      % narrow->wide ramp (zero-area)
        case 'sldn',   w=1.0+(0.6-1.0)*u;
        otherwise,     w=Wc*ones(size(u));
    end
    xs=x0+u*L;
    fill([xs fliplr(xs)],[w/2 -fliplr(w)/2],col,'EdgeColor',[0.25 0.25 0.25],'LineWidth',0.5);
end
