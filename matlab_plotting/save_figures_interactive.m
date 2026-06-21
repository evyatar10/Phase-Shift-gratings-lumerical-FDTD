function save_figures_interactive()
% SAVE_FIGURES_INTERACTIVE  Interactively choose figures to save, names, formats, and folder.
addpath(fileparts(fileparts(mfilename('fullpath'))));  % Add project root to MATLAB path
%   Use after running a script that creates figures. Prompts for:
%   1) Which figures to save (multi-select list)
%   2) Output folder
%   3) Format for all: FIG only, PNG only, or Both (applies to every selected figure)
%   4) For each selected figure: base filename (no extension) only
%   Saves .fig (MATLAB figure) and/or .png as requested.

    figs = findobj('Type', 'figure');
    if isempty(figs)
        msgbox('No figures are open.', 'No figures', 'warn');
        return;
    end

    % Sort by figure number for stable ordering
    [~, idx] = sort([figs.Number]);
    figs = figs(idx);

    % Build list strings: "Figure N" or "Figure N: Name" if Name is set
    list = arrayfun(@(f) figListItem(f), figs, 'UniformOutput', false);

    % 1) Select which figures to save
    [sel, ok] = listdlg('ListString', list, ...
        'Name', 'Save figures', ...
        'PromptString', 'Select figures to save (Ctrl/Shift for multiple):', ...
        'ListSize', [400 200], ...
        'SelectionMode', 'multiple');
    if ~ok || isempty(sel)
        return;
    end
    figsToSave = figs(sel);

    % 2) Select output folder
    outDir = uigetdir(pwd, 'Select folder to save figures');
    if isequal(outDir, 0)
        return;
    end

    % 3) Choose format for all figures (one choice, applied to every selected figure)
    [fmtIdx, ok] = listdlg('ListString', {'FIG only (.fig)', 'PNG only (.png)', 'Both (.fig and .png)'}, ...
        'Name', 'Format for all', ...
        'PromptString', 'Save format (same for all selected figures):', ...
        'ListSize', [280 80], ...
        'SelectionMode', 'single');
    if ~ok || isempty(fmtIdx)
        return;
    end
    saveFIG = (fmtIdx == 1 || fmtIdx == 3);
    savePNG = (fmtIdx == 2 || fmtIdx == 3);

    % 4) For each selected figure: get filename only (format already chosen)
    n = numel(figsToSave);
    for i = 1:n
        f = figsToSave(i);
        defaultName = figDefaultName(f);
        ans_ = inputdlg('Filename (no extension):', sprintf('Figure %d of %d', i, n), [1 60], {defaultName});
        if isempty(ans_)
            continue; % skip this figure
        end
        baseName = strtrim(ans_{1});
        if isempty(baseName)
            baseName = defaultName;
        end
        [~, baseName, ~] = fileparts(baseName);
        if isempty(baseName)
            baseName = defaultName;
        end

        basePath = fullfile(outDir, baseName);
        if saveFIG
            savefig(f, [basePath '.fig']);
        end
        if savePNG
            print(f, [basePath '.png'], '-dpng', '-r150');
        end
    end

    msgbox(sprintf('Saved %d figure(s) to:\n%s', n, outDir), 'Done', 'help');
end

function s = figListItem(f)
    name = get(f, 'Name');
    if isempty(name)
        s = sprintf('Figure %d', f.Number);
    else
        s = sprintf('Figure %d: %s', f.Number, name);
    end
end

function name = figDefaultName(f)
% Default filename = the text shown on the figure, sanitized.
% Prefers the figure window Name (e.g. "XZ Zoomed Side View"); falls back
% to the axes/sgtitle title text, then to "figure_N".
    raw = strtrim(get(f, 'Name'));
    if isempty(raw)
        raw = figTitleText(f);
    end
    name = sanitizeFilename(raw);
    if isempty(name)
        name = sprintf('figure_%d', f.Number);
    end
end

function t = figTitleText(f)
% Best-effort extraction of a visible title from the figure.
    t = '';
    % sgtitle (super title) if present
    sg = findobj(f, 'Type', 'subplottext', 'Tag', 'subplottitle');
    if ~isempty(sg)
        t = strtrim(charText(get(sg(1), 'String')));
    end
    if isempty(t)
        ax = findobj(f, 'Type', 'axes');
        for k = 1:numel(ax)
            ttl = get(ax(k), 'Title');
            t = strtrim(charText(get(ttl, 'String')));
            if ~isempty(t)
                break;
            end
        end
    end
end

function s = charText(str)
% Flatten a title String (char, cellstr, or string array) to one line.
    if iscell(str) || (isstring(str) && numel(str) > 1)
        s = strjoin(cellstr(str), ' ');
    else
        s = char(str);
    end
    s = strtrim(s);
end

function s = sanitizeFilename(s)
% Turn arbitrary title text into a safe base filename.
    s = char(s);
    s = regexprep(s, '\s+', '_');        % whitespace -> underscore
    s = regexprep(s, '[\\/:*?"<>|]', ''); % strip filesystem-illegal chars
    s = regexprep(s, '_+', '_');          % collapse repeated underscores
    s = regexprep(s, '^_+|_+$', '');      % trim leading/trailing underscores
end
