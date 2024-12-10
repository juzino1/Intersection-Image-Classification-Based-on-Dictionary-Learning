% 主程序入口
clc; clear; close all;

%% 1. 数据准备
train_path = 'train'; % 训练图像文件夹路径，每类放在子文件夹中
test_folder = 'test3'; % 测试图像文件夹路径

% 提取训练集SURF特征
disp('提取训练集SURF特征...');
[all_descriptors, labels] = extractSURFFeatures(train_path);

% 检查提取的特征数量
if isempty(all_descriptors)
    error('未能提取到有效的训练集特征，请检查训练数据。');
end


%% 2. 字典学习
disp('进行字典学习...');
max_clusters = 200; % 最大字典大小
num_clusters = min(max_clusters, size(all_descriptors, 1)); % 确保簇数不超过特征点数量

if num_clusters < 2
    error('提取到的特征点数量不足以进行聚类，请检查训练数据。');
end

opts = statset('MaxIter', 3000); % 设置最大迭代次数
[~, dictionary] = kmeans(single(all_descriptors), num_clusters, ...
    'MaxIter', 3000, 'Replicates', 5, 'Options', opts);
fprintf('生成字典维度：%d x %d\n', size(dictionary, 1), size(dictionary, 2));

%% 3. 构建词袋模型特征并训练分类器
disp('训练分类器...');
training_features = buildHistogramFeatures(train_path, dictionary, num_clusters);
fprintf('训练特征维度：%d x %d\n', size(training_features, 1), size(training_features, 2));
training_features = normalize(training_features, 2); % 特征归一化

% 使用线性SVM分类器
svm_model = fitcecoc(training_features, labels, 'Learners', templateSVM('KernelFunction', 'linear'));

%% 4. 测试文件夹中的图片
disp('测试文件夹中的图片...');
test_files = dir(fullfile(test_folder, '*.jpg')); % 获取文件夹中所有图片文件
if isempty(test_files)
    error('测试文件夹中没有找到图片，请检查路径。');
end

% 对文件名进行自然排序
[~, order] = sort_nat({test_files.name});
test_files = test_files(order);

% 初始化结果保存
results = cell(length(test_files), 2); 

% 遍历文件夹中的所有图片
for i = 1:length(test_files)
    test_image_path = fullfile(test_files(i).folder, test_files(i).name);
    test_feature = buildHistogramFeature(test_image_path, dictionary, num_clusters);
    test_feature = normalize(test_feature, 2);
    predicted_label = predict(svm_model, test_feature);
    results{i, 1} = test_files(i).name;
    results{i, 2} = predicted_label;
   fprintf('测试图片: %s -> 路口编号 %d\n', test_files(i).name, predicted_label);
end

% 保存结果到文件
save('classification_results.mat', 'results');
disp('分类结果已保存到 classification_results.mat');

%% 功能函数

% 提取SURF特征点
function [all_descriptors, labels] = extractSURFFeatures(image_path)
    all_descriptors = [];
    labels = [];
    class_dirs = dir(image_path);
    class_dirs = class_dirs([class_dirs.isdir] & ~ismember({class_dirs.name}, {'.', '..'}));
    for i = 1:length(class_dirs)
        if ~isnan(str2double(class_dirs(i).name))
            img_files = dir(fullfile(image_path, class_dirs(i).name, '*.jpg'));
            for j = 1:length(img_files)
                img = imread(fullfile(img_files(j).folder, img_files(j).name));
                img = preprocessImage(img); 
                features = extractColorSURFFeatures(img); % 提取彩色图像SURF特征
                if ~isempty(features)
                    all_descriptors = [all_descriptors; features];
                    labels = [labels; str2double(class_dirs(i).name)];
                else
                    warning('图像 %s 未提取到有效特征，已跳过。', img_files(j).name);
                end
            end
        end
    end
end

% 提取彩色图像的 SURF 特征
function features = extractColorSURFFeatures(img)
    features = [];
    for channel = 1:3 % 分别处理 R, G, B 通道
        single_channel = img(:, :, channel);
        points = detectSURFFeatures(single_channel);
        [channel_features, ~] = extractFeatures(single_channel, points);
        features = [features; channel_features];
    end
end

% 构建词袋模型特征
function histogram_features = buildHistogramFeatures(image_path, dictionary, num_clusters)
    class_dirs = dir(image_path);
    class_dirs = class_dirs([class_dirs.isdir] & ~ismember({class_dirs.name}, {'.', '..'}));
    histogram_features = [];
    for i = 1:length(class_dirs)
        if ~isnan(str2double(class_dirs(i).name))
            img_files = dir(fullfile(image_path, class_dirs(i).name, '*.jpg'));
            for j = 1:length(img_files)
                img = imread(fullfile(img_files(j).folder, img_files(j).name));
                img = preprocessImage(img); 
                features = extractColorSURFFeatures(img);
                if ~isempty(features)
                    histogram_features = [histogram_features; buildHistogram(features, dictionary, num_clusters)];
                else
                    warning('图像 %s 未提取到有效特征，已跳过。', img_files(j).name);
                end
            end
        end
    end
end

% 生成单张图片的词袋模型特征
function histogram = buildHistogramFeature(image_path, dictionary, num_clusters)
    img = imread(image_path);
    img = preprocessImage(img); 
    features = extractColorSURFFeatures(img);
    if isempty(features)
        error('测试图像未提取到有效特征，请检查图像质量。');
    end
    histogram = buildHistogram(features, dictionary, num_clusters);
end

% 构建直方图
function histogram = buildHistogram(descriptors, dictionary, num_clusters)
    histogram = zeros(1, num_clusters);
    descriptors = single(descriptors);
    dictionary = single(dictionary');
    for i = 1:size(descriptors, 1)
        [~, idx] = min(vl_alldist2(dictionary, descriptors(i, :)')); 
        histogram(idx) = histogram(idx) + 1;
    end
    histogram = histogram / sum(histogram); 
end

% 图像预处理
function img = preprocessImage(img)
    img = imresize(img, [128, 128]); % 保留彩色图像，只调整大小
end

% 自然排序函数
function [sorted, idx] = sort_nat(strings)
    numbers = cellfun(@(c) str2double(regexp(c, '\d+', 'match', 'once')), strings);
    [~, idx] = sort(numbers);
    sorted = strings(idx);
end

