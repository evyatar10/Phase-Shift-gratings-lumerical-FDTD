% Quick check of TM field profiles

tm_files = {
    "results_from_athena/run_tm/results/result_N80_TM_avg_ff_tm_P518p3_fields_smp.mat",
    "results_from_athena/run_tm/results/result_N80_TM_avg_ff_tm_P518p3_smp.mat",
    "results_from_athena/run_tm/results/result_N80_TM_avg_tm_P518.mat",
    "results_from_athena/run_tm/results/result_N80_TM_avg_tm_P518p3.mat",
    "results_from_athena/run_tm/results/result_N80_TM_avg_tm_P518p3_M1p0_smp.mat",
    "results_from_athena/run_tm/results/result_N80_TM_avg_tm_P518p3_M1p5_smp.mat",
    "results_from_athena/run_tm/results/result_N80_TM_avg_tm_P518p3_M1p8_smp.mat",
    "results_from_athena/run_tm/results/result_N80_TM_avg_tm_P518p3_M2p7_fields_smp.mat",
    "results_from_athena/run_tm/results/result_N80_TM_avg_tm_P518p3_M2p7_smp.mat",
    "results_from_athena/run_tm/results/result_N80_TM_avg_tm_P518p3_acc.mat",
    "results_from_athena/run_tm/results/result_N80_TM_avg_tm_P518p3_fields_smp.mat",
    "results_from_athena/run_tm/results/result_N80_TM_avg_tm_P519.mat",
    "results_from_athena/run_tm_vs_te/results/result_N80_TM_avg_tm.mat",
    "results_from_athena/run_tm_vs_te/results/result_N80_TM_avg_tm_obj.mat",
    "results_from_athena/run_tm_vs_te/results/result_N80_TM_avg_tm_smp.mat",
    "results_from_athena/tm_match_bisect/results/result_N120_TM_avg_tm_P518p3_smp.mat",
    "results_from_athena/tm_match_bisect/results/result_N130_TM_avg_tm_P518p3_smp.mat",
    "results_from_athena/tm_match_bisect/results/result_N131_TM_avg_tm_P518p3_smp.mat",
    "results_from_athena/tm_match_bisect/results/result_N132_TM_avg_tm_P518p3_smp.mat",
    "results_from_athena/tm_match_bisect/results/result_N135_TM_avg_tm_P518p3_smp.mat",
    "results_from_athena/tm_match_bisect/results/result_N140_TM_avg_tm_P518p3_smp.mat",
    "results_from_athena/tm_match_bisect/results/result_N160_TM_avg_tm_P518p3_smp.mat",
    "results_from_athena/tm_match_bisect/results/result_N80_TM_avg_tm_P518p3_smp.mat",
    "results_from_athena/tm_te/results/result_N80_TM_avg_ff_tm_P518p3_fields_smp.mat",
    "results_from_athena/tm_te_pitch_matched/result_N80_TM_avg_tm.mat",
    "results_from_athena/tm_te_pitch_matched/result_N80_TM_avg_tm_P518p3.mat",
    "results_from_athena/tm_te_pitch_matched/result_N80_TM_avg_tm_P518p3_acc.mat",
    "results_from_athena/tm_te_shift/results/result_N80_TM_avg_tm.mat",
};

both = {};
only_one = {};
none = {};

for i = 1:length(tm_files)
    fpath = tm_files{i};
    
    if ~isfile(fpath)
        continue;
    end
    
    try
        data = load(fpath);
        has_xy = isfield(data, 'field_xy');
        has_xz = isfield(data, 'field_xz_side');
        has_yz = isfield(data, 'field_yz_cross');
        
        [~, fname, ~] = fileparts(fpath);
        
        if has_xy && has_xz
            both{end+1} = {fname, fpath};
        elseif has_xy || has_xz || has_yz
            only_one{end+1} = {fname, fpath, has_xy, has_xz, has_yz};
        else
            none{end+1} = {fname, fpath};
        end
    catch
    end
end

fprintf('BOTH XY and XZ (COMPLETE):\n');
for i=1:length(both)
    fprintf('  %s\n', both{i}{1});
end

fprintf('\nINCOMPLETE (missing planes):\n');
for i=1:length(only_one)
    entry = only_one{i};
    planes = '';
    if entry{3}, planes = [planes 'XY']; end
    if entry{4}, if ~isempty(planes), planes = [planes '+']; end; planes = [planes 'XZ']; end
    if entry{5}, if ~isempty(planes), planes = [planes '+']; end; planes = [planes 'YZ']; end
    fprintf('  [%s] %s\n', planes, entry{1});
end

fprintf('\nNO FIELDS:\n');
for i=1:length(none)
    fprintf('  %s\n', none{i}{1});
end

fprintf('\nSUMMARY: %d with both, %d incomplete, %d with none\n', length(both), length(only_one), length(none));
