%           EXAMPLE - USING "WOW" embedding distortion
%
% -------------------------------------------------------------------------
% Copyright (c) 2012 DDE Lab, Binghamton University, NY.
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
% cover = imread(fullfile('D:\ruanjian\Matlab2022a\Work\WOW_matlab\images_cover\1.pgm'));
% SamplePath = 'E:\LY\Bilinear\wow\500-wow0.4\m\test-r\cover\';
% fileExt = '*.pgm';
% files = dir(fullfile(SamplePath,fileExt));
fileID = fopen('E:\LY\work_two\ACSNet\steganography\input_path.txt', 'r');
if fileID == -1
    error('�޷����ļ��������ļ�·����Ȩ�ޡ�');
end
line = fgetl(fileID);
fclose(fileID);
parts = strsplit(line, {' ', '\t'});
input_path = parts{1};  % ��һ����Ϊ·��
payload = str2double(parts{2});  % �ڶ�����Ϊ payload����ת��Ϊ��ֵ
disp(['Received path: ', input_path]);
disp(['Received payload: ', payload]);
input_dir = fullfile(input_path,'cover');%��Ҫת����ʽ���ļ���
output_dir = fullfile(input_path,'stego');%ת�����ʽͼƬ������ļ���
files=dir([input_dir '/*.pgm']);%���ļ�����pgmͼƬ

len = size(files,1);
% set payload
payload = 0.4;

% set params
params.p = -1;  % holder norm parameter

fprintf('Embedding using matlab code');
MEXstart = tic;

for i=1:len
    fileName = fullfile(input_dir,files(i).name);
    cover = imread(fileName);
%     cover = imresize(cover,[256 256]);
    % 对每�?��图像进行隐写
    [stego, distortion] = WOW(cover, payload, params);
%     imwrite(cover,strcat('F:\LY\BOSSbase_1.01\BOSSbase_1.01-256\',int2str(i),'.pgm'));
    imwrite(uint8(stego),fullfile(output_dir,files(i).name))
end
% %% Run embedding simulation
% [stego, distortion] = WOW(cover, payload, params);
%         
% MEXend = toc(MEXstart);
% fprintf(' - DONE');
% 
% figure;
% subplot(1, 2, 1); imshow(cover); title('cover');
% % subplot(1, 2, 2); imshow((double(stego) - double(cover) + 1)/2); title('embedding changes: +1 = white, -1 = black');
% subplot(1, 2, 2); imshow(uint8(stego)); title('embedding changes: +1 = white, -1 = black');
fprintf('\n\n �����');
% fprintf('\n\nImage embedded in %.2f seconds, change rate: %.4f, distortion per pixel: %.6f\n', MEXend, sum(cover(:)~=stego(:))/numel(cover), distortion/numel(cover));