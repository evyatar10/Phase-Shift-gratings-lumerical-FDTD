% plot_barrel_geometry.m — TOP-VIEW schematic: what the "barrel cavity" IS.
% Draws the central region (cavity + 3 periods each side) of the anchored TM
% device: rectangular control vs the barrel-bulged pi-shift segment.
% Faithful to bragg_device.py construction. Headless-safe; .fig + PNG.

clear; close all;

p   = 516.83;            % pitch (nm)
hp  = p / 2;             % half-pitch
Wc  = 800;              % cavity nominal width (avg)
hwN = 300;              % narrow half-width (W_narrow/2 = 600/2)
hwW = 500;              % wide   half-width (W_wide/2  = 1000/2)
Lc  = hp;               % cavity length = pitch/2
xc  = Lc / 2;           % cavity spans [-xc, +xc]
depth = 150;            % barrel bulge (nm) -> mid width 950

xg = linspace(-3.4*hp - xc, 3.4*hp + xc, 4000);

    function hw = halfwidth(x, xc, hp, p, hwN, hwW, Wc, depth, isBarrel)
        hw = zeros(size(x));
        for k = 1:numel(x)
            xi = x(k);
            if abs(xi) <= xc
                if isBarrel
                    u = (xi + xc) / (2 * xc);            % 0..1 across cavity
                    hw(k) = (Wc + depth * sin(pi * u)) / 2;
                else
                    hw(k) = Wc / 2;
                end
            elseif xi < -xc
                s = (-xi) - xc;  pos = mod(s, p);         % left arm: adjacent = WIDE
                hw(k) = (pos < hp) * hwW + (pos >= hp) * hwN;
            else
                s = xi - xc;     pos = mod(s, p);         % right arm: adjacent = NARROW
                hw(k) = (pos < hp) * hwN + (pos >= hp) * hwW;
            end
        end
    end

fig = figure('Visible', 'off', 'Position', [80 80 1200 560]);
tl = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

for panel = 1:2
    isBarrel = (panel == 2);
    hw = halfwidth(xg, xc, hp, p, hwN, hwW, Wc, depth, isBarrel);
    nexttile; hold on;
    fill([xg, fliplr(xg)] / 1000, [hw, -fliplr(hw)] / 1000, ...
        [0.55 0.78 0.92], 'EdgeColor', [0.1 0.25 0.45], 'LineWidth', 1.0);
    % mark the cavity segment
    xline(-xc/1000, 'k:'); xline(xc/1000, 'k:');
    if isBarrel
        % dashed outline of where the rect wall WOULD be
        plot([-xc xc]/1000, [Wc/2 Wc/2]/1000, '--', 'Color', [0.7 0.2 0.2], 'LineWidth', 1.1);
        plot([-xc xc]/1000, -[Wc/2 Wc/2]/1000, '--', 'Color', [0.7 0.2 0.2], 'LineWidth', 1.1);
        text(0, 0, sprintf('barrel: 800\\rightarrow950 nm bulge'), 'HorizontalAlignment', 'center', ...
            'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.5 0.1 0.1]);
        title('(b) BARREL cavity — sidewalls bowed outward (half-sine, +150 nm at middle)');
    else
        text(0, 0, 'cavity: straight 800 nm', 'HorizontalAlignment', 'center', 'FontSize', 10);
        title('(a) Rectangular cavity (the standard device)');
    end
    axis equal; grid on;
    xlim([xg(1) xg(end)]/1000); ylim([-0.62 0.62]);
    xlabel('x along guide (\mum)'); ylabel('y (\mum)');
end

title(tl, sprintf(['What the "barrel cavity" is — top view, TM \\pi-shift (pitch 516.83 nm, corr 400 nm)\n' ...
    'ONLY the central \\pi-shift segment changes; all 80 mirror periods each side stay rectangular']), ...
    'FontSize', 12, 'FontWeight', 'bold');

out = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\inner_shape_study';
exportgraphics(fig, fullfile(out, 'barrel_geometry_schematic.png'), 'Resolution', 200);
savefig(fig, fullfile(out, 'barrel_geometry_schematic.fig'));
fprintf('saved: %s\n', fullfile(out, 'barrel_geometry_schematic.png'));
