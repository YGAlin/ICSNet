import os
import Unet
import torch
import cv2
import numpy as np
import opt
import TES
args = opt.opt()
tes = TES.TES().cuda()

def image2tensor(cover_path):
    image = cv2.imread(cover_path, -1)
    data = image.astype(np.float32)
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    return data,image

generator = Unet.UNet().cuda()

image_folder = os.path.join(args.data_path, args.data_path_target, args.tar)
output_filename = args.data_path_source+'_'+args.data_path_target
output_folder = os.path.join(args.data_path, args.data_path_target,output_filename,'cover')  # 保存输出图像的文件夹
os.makedirs(output_folder, exist_ok=True)  # 如果输出文件夹不存在，则创建
output_folder_stego = os.path.join(args.data_path, args.data_path_target,output_filename,'stego')
os.makedirs(output_folder_stego, exist_ok=True)
# pkl = 'result/'+image_folder.split('-')[1].split('\\')[0]+'\\'+image_folder.split('-')[1].split('\\')[1]+'\\'+'msePronetG_epoch_99.pkl'
# print(pkl)
pkl = os.path.join(args.unet_pkl, 'unetParams.pkl')
pretrained_net_dict = torch.load(pkl)
# new_state_dict = OrderedDict()
# for k, v in pretrained_net_dict.items():
#     name = k[7:] # remove "module."
#     new_state_dict[name] = v

generator.load_state_dict(pretrained_net_dict)
generator.eval()

# filename = r'E:\LY\Bilinear\suni\500-suni0.4\b\test\cover\1.pgm'
i = 1
for subname in os.listdir(image_folder):
    print(subname)
    if subname not in ('cover', 'stego'):
        break
    subname_1 = os.path.join(image_folder,subname)
    for filename in os.listdir(subname_1):
        if filename.lower().endswith(('.jpg', '.jpeg', '.pgm')):
            cover_path = os.path.join(subname_1, filename)                      # Path of the original cover
            _, extension = os.path.splitext(filename)
            if subname == 'stego':
                filename = 'stego'+filename
            Aug_cover_path = os.path.join(output_folder, filename)              # Path of the Augmented cover
            data,image = image2tensor(cover_path)
            y = generator(data.cuda())
            y = tes(y/2, y/2)
            y = y.reshape(256,256)
            y = y.detach().cpu().numpy()
            y = image + 8 * y                                  # The amplitude of noise is set to 16
            y[y > 255] = 255
            y[y < 0] = 0
            y = np.uint8(y)
            # if filename.lower().endswith(('.jpg', '.jpeg')):
            #     # 获取图像的质量因子
            #     quality_factor = getQF.getQF(cover_path)
            #     # 保存为JPEG格式并设置质量因子
            #     cv2.imwrite(Aug_cover_path, y, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            # else:
                # 非JPEG格式，按原来的方式保存
            cv2.imwrite(Aug_cover_path, y)
            i+=1
            # cv2.imshow('Combined Heatmaps',y)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
#运行.m文件对生成图像进行重嵌入
input_path = os.path.join(args.data_path, args.data_path_target,output_filename)
# 将路径写入一个文本文件
pathtxt = 'E:\LY\work_two\ACSNet\steganography\input_path.txt'
if os.path.exists(pathtxt):
    # 删除之前的文件
    os.remove(pathtxt)
with open(pathtxt, "w") as f:
    f.write(f'{input_path}\t{args.payload}')
    f.close()