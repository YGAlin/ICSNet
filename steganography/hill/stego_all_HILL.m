clc; clear all;

% ���ļ��ж�ȡ·��
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
tic
for w=1: length(files)
%     filename=[input_dir '\' num2str(w) '.pgm'];%�Բ��Լ�������дʱwҪ��8000
%     numbers = [0.2,0.3,0.4];  % ������������
%     index = randi(length(numbers));  % �������һ������
%     payload = numbers(index);  % �����������ѡ��һ����
%     disp(payload);
    filename=fullfile(input_dir,files(w).name);
    stego = HILL(filename, payload);
%     imshow(stego/256); % ��ͼ�����ת����0-1֮��
%     imshow(stego,[]); % �Զ��������ݵķ�Χ�Ա�����ʾ
%     imshow(uint8(stego)); % ת��uint8
    %https://blog.csdn.net/u012162771/article/details/79901160
    %stego��ʾ��ɫ����������
%     a = num2str(w)%�Բ��Լ�������дʱwҪ��8000
%     imageName=[output_dir a '.pgm'];
%     imwrite(uint8(stego),imageName);
    imwrite(uint8(stego),fullfile(output_dir,files(w).name));
    %imshow(stego)
end
toc