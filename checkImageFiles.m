function failed_files = checkImageFiles(folder_path)
    % 初始化文件列表
    img_files = dir(fullfile(folder_path, '**', '*.jpg'));
    
    % 初始化无法读取的文件列表
    failed_files = {}; 
    
    % 遍历文件
    for i = 1:length(img_files)
        file_path = fullfile(img_files(i).folder, img_files(i).name);
        try
            % 尝试读取图像
            img = imread(file_path);
        catch ME
            % 记录读取失败的文件路径和错误信息
            failed_files{end+1, 1} = file_path; %#ok<*AGROW>
            failed_files{end, 2} = ME.message;
            fprintf('文件 %s 无法读取，错误: %s\n', file_path, ME.message);
        end
    end
    
    % 打印统计结果
    fprintf('\n总文件数: %d\n', length(img_files));
    fprintf('无法读取的文件数: %d\n', length(failed_files));
    
    % 如果有无法读取的文件，将结果保存到日志文件
    if ~isempty(failed_files)
        log_file = 'failed_images_log.txt';
        fid = fopen(log_file, 'w');
        fprintf(fid, '无法读取的文件列表:\n');
        for i = 1:size(failed_files, 1)
            fprintf(fid, '文件: %s | 错误: %s\n', failed_files{i, 1}, failed_files{i, 2});
        end
        fclose(fid);
        fprintf('无法读取的文件已记录到 %s。\n', log_file);
    else
        fprintf('所有文件均成功读取，无需检查。\n');
    end
end

