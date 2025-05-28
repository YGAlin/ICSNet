clc; clear all;

% 从文件中读取路径
fileID = fopen('E:\LY\work_two\ACSNet\steganography\input_path.txt', 'r');
if fileID == -1
    error('无法打开文件，请检查文件路径或权限。');
end
line = fgetl(fileID);
fclose(fileID);
parts = strsplit(line, {' ', '\t'});
input_path = parts{1};  % 第一部分为路径
payload = str2double(parts{2});  % 第二部分为 payload，并转换为数值
disp(['Received path: ', input_path]);
disp(['Received payload: ', payload]);
input_dir = fullfile(input_path,'cover');%需要转换格式的文件夹
output_dir = fullfile(input_path,'stego');%转换完格式图片保存的文件夹
files=dir([input_dir '/*.pgm']);%打开文件夹中pgm图片
tic
for w=1: length(files)
%     filename=[input_dir '\' num2str(w) '.pgm'];%对测试集进行隐写时w要加8000
%     numbers = [0.2,0.3,0.4];  % 定义数的数组
%     index = randi(length(numbers));  % 随机生成一个索引
%     payload = numbers(index);  % 根据随机索引选择一个数
%     disp(payload);
    filename=fullfile(input_dir,files(w).name);
    stego = HILL(filename, payload);
%     imshow(stego/256); % 将图像矩阵转化到0-1之间
%     imshow(stego,[]); % 自动调整数据的范围以便于显示
%     imshow(uint8(stego)); % 转成uint8
    %https://blog.csdn.net/u012162771/article/details/79901160
    %stego显示白色问题解决方案
%     a = num2str(w)%对测试集进行隐写时w要加8000
%     imageName=[output_dir a '.pgm'];
%     imwrite(uint8(stego),imageName);
    imwrite(uint8(stego),fullfile(output_dir,files(w).name));
    %imshow(stego)
end
toc