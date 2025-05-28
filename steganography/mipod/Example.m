% This example demonstrates how to use the MiPOD embedding function
clc
clear all
close all

% Read the input cover image
% Cover = double(imread ('1.pgm'));
% SamplePath = 'E:\LY\Bilinear\mipod\500-mipod0.4\m\test-r\cover\';
% % Set the payload to 0.4 bpp
% Payload = 0.4;
% % 获取文件
% files = dir(strcat(SamplePath,'*.pgm'));
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
% 获取文件长度
len = length(files);
% MiPOD embedding
tStart = tic;
for i=1:1:len
    fileName = fullfile(input_dir,files(i).name);
    Cover = double(imread(fileName));
    [Stego, pChange, ChangeRate] = MiPOD( Cover, payload );
    imwrite(uint8(Stego),fullfile(output_dir,files(i).name));
end
tEnd = toc(tStart);
fprintf('MiPOD embedding is done in: %f (sec)\n',tEnd);

%%
close all
% 
% figure;
% imshow (Cover,[]);
% title ('Cover image');
% 
% figure;
% imshow(1-pChange/0.3333);
% title('MiPOD - Embedding Change Probabilities');
% 
% figure;
% imshow(Stego-Cover,[]);
% title('MiPOD - Changed Pixels (+1 -> white ,-1 -> black)');