% Check ALL TM result files

% Find all .mat files with "TM" or "tm" in the name within results directories
d = dir('results_from_athena/**/result_*tm*.mat');
tm_files = {};
for i = 1:length(d)
    tm_files{i} = fullfile(d(i).folder, d(i).name);
end

fprintf('Found %d TM result files to check\n\n', length(tm_files));

both = {};
only_one = {};
none = {};

for i = 1:length(tm_files)
    fpath = tm_files{i};
    
    try
        data = load(fpath);
        has_xy = isfield(data, 'field_xy');
        has_xz = isfield(data, 'field_xz_side');
        has_yz = isfield(data, 'field_yz_cross');
        
        [folder, fname, ~] = fileparts(fpath);
        relpath = strrep(fpath, pwd, '');
        if startsWith(relpath, '/') || startsWith(relpath, '\')
            relpath = relpath(2:end);
        end
        
        if has_xy && has_xz
            both{end+1} = {fname, relpath};
        elseif has_xy || has_xz || has_yz
            only_one{end+1} = {fname, relpath, has_xy, has_xz, has_yz};
        else
            none{end+1} = {fname, relpath};
        end
    catch ME
        fprintf('ERROR %s: %s\n', fpath, ME.message);
    end
end

fprintf('==================== RESULTS ====================\n\n');

fprintf('FILES WITH BOTH XY AND XZ FIELD PROFILES: %d\n', length(both));
fprintf('─────────────────────────────────────────────\n');
for i=1:length(both)
    fprintf('%s\n', both{i}{2});
end

fprintf('\n\nFILES WITH INCOMPLETE FIELD PROFILES: %d\n', length(only_one));
fprintf('─────────────────────────────────────────────\n');
for i=1:length(only_one)
    entry = only_one{i};
    planes = '';
    if entry{3}, planes = [planes 'XY']; end
    if entry{4}, if ~isempty(planes), planes = [planes '+']; end; planes = [planes 'XZ']; end
    if entry{5}, if ~isempty(planes), planes = [planes '+']; end; planes = [planes 'YZ']; end
    fprintf('[%s] %s\n', planes, entry{2});
end

fprintf('\n\nFILES WITH NO FIELD PROFILES: %d\n', length(none));
fprintf('─────────────────────────────────────────────\n');
for i=1:length(none)
    fprintf('%s\n', none{i}{2});
end

fprintf('\n\n==================== SUMMARY ====================\n');
fprintf('Total files analyzed:       %d\n', length(tm_files));
fprintf('With BOTH XY and XZ:        %d\n', length(both));
fprintf('With only one plane:        %d\n', length(only_one));
fprintf('With no field profiles:     %d\n', length(none));
