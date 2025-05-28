%           EXAMPLE - USING "S-UNIWARD" embedding distortion
%
% -------------------------------------------------------------------------
% Copyright (c) 2013 DDE Lab, Binghamton University, NY.
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software for
% educational, research and non-profit purposes, without fee, and without a
% written agreement is hereby granted, provided that this copyright notice
% appears in all copies. The program is supplied "as is," without any
% accompanying services from DDE Lab. DDE Lab does not warrant the
% operation of the program will be uninterrupted or error-free. The
% end-user understands that the program was developed for research purposes
% and is advised not to rely exclusively on the program for any reason. In
% no event shall Binghamton University or DDE Lab be liable to any party
% for direct, indirect, special, incidental, or consequential damages,
% including lost profits, arising out of the use of this software. DDE Lab
% disclaims any warranties, and has no obligations to provide maintenance,
% support, updates, enhancements or modifications.
% -------------------------------------------------------------------------
% Author: Vojtech Holub
% -------------------------------------------------------------------------
% Contact: vojtech_holub@yahoo.com
%          fridrich@binghamton.edu
%          http://dde.binghamton.edu
% -------------------------------------------------------------------------
clc; clear all;

% load cover image
% coverPath = fullfile('E:\LY\S-UNIWARD\S-UNIWARD_matlab\images_cover', '1.pgm');
% SamplePath = 'E:\LY\Bilinear\suni\500-suni0.4\b\a-b\cover\';
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
% 后缀名
% fileExt = '*.pgm';
% 获取文件
% files = dir(fullfile(input_dir,fileExt));
% 获取文件长度
len = size(files,1);

% set payload
% payload = single(0.4);

fprintf('Embedding using Matlab file');
MEXstart = tic;

%% Run default embedding

        
MEXend = toc(MEXstart);
fprintf(' - DONE');
for i=1 : len
    fileName = fullfile(input_dir,files(i).name);
    stego = S_UNIWARD(fileName, payload);
    imwrite(uint8(stego),fullfile(output_dir,files(i).name));
end
fprintf('finished');