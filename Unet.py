import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import TES
tes = TES.TES().cuda()

# 定义U-Net的编码器块
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
from padding_same_conv import Conv2d
from torch.autograd import Variable
import torch.nn.init as init


class Group1(nn.Module):
    def __init__(self, ):
        super(Group1, self).__init__()
        self.conv = Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(16)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group2(nn.Module):
    def __init__(self, ):
        super(Group2, self).__init__()
        self.conv = Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(32)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group3(nn.Module):
    def __init__(self, ):
        super(Group3, self).__init__()
        self.conv = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(64)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group4(nn.Module):
    def __init__(self, ):
        super(Group4, self).__init__()
        self.conv = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group5(nn.Module):
    def __init__(self, ):
        super(Group5, self).__init__()
        self.conv = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group6(nn.Module):
    def __init__(self, ):
        super(Group6, self).__init__()
        self.conv = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group7(nn.Module):
    def __init__(self, ):
        super(Group7, self).__init__()
        self.conv = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group8(nn.Module):
    def __init__(self, ):
        super(Group8, self).__init__()
        self.conv = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group9(nn.Module):
    def __init__(self, ):
        super(Group9, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2,
                                       output_padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group10(nn.Module):
    def __init__(self, ):
        super(Group10, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2,
                                       output_padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group11(nn.Module):
    def __init__(self, ):
        super(Group11, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2,
                                       output_padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group12(nn.Module):
    def __init__(self, ):
        super(Group12, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2,
                                       output_padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group13(nn.Module):
    def __init__(self, ):
        super(Group13, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=5, stride=2, padding=2,
                                       output_padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.actv = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group14(nn.Module):
    def __init__(self, ):
        super(Group14, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=5, stride=2, padding=2,
                                       output_padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.actv = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group15(nn.Module):
    def __init__(self, ):
        super(Group15, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=5, stride=2, padding=2,
                                       output_padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.actv = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group16(nn.Module):
    def __init__(self, ):
        super(Group16, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5, stride=2, padding=2,
                                       output_padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class Group17(nn.Module):
    def __init__(self, ):
        super(Group17, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.sigmoid(x) - 0.5
        out = self.ReLU(out)
        return out


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.block1 = Group1()
        self.block2 = Group2()
        self.block3 = Group3()
        self.block4 = Group4()
        self.block5 = Group5()
        self.block6 = Group6()
        self.block7 = Group7()
        self.block8 = Group8()
        self.block9 = Group9()
        self.block10 = Group10()
        self.block11 = Group11()
        self.block12 = Group12()
        self.block13 = Group13()
        self.block14 = Group14()
        self.block15 = Group15()
        self.block16 = Group16()
        self.block17 = Group17()

    def forward(self, x):
        x = self.block1(x)
        output1 = x

        x = self.block2(x)
        output2 = x

        x = self.block3(x)
        output3 = x

        x = self.block4(x)
        output4 = x

        x = self.block5(x)
        output5 = x

        x = self.block6(x)
        output6 = x

        x = self.block7(x)
        output7 = x

        x = self.block8(x)

        x = self.block9(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.cat([x, output7], dim=1)

        x = self.block10(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.cat([x, output6], dim=1)

        x = self.block11(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.cat([x, output5], dim=1)

        x = self.block12(x)
        x = torch.cat([x, output4], dim=1)

        x = self.block13(x)
        x = torch.cat([x, output3], dim=1)

        x = self.block14(x)
        x = torch.cat([x, output2], dim=1)

        x = self.block15(x)
        x = torch.cat([x, output1], dim=1)

        x = self.block16(x)

        x = self.block17(x)

        return x

def image2tensor(cover_path):
    image = cv2.imread(cover_path, -1)
    data = image.astype(np.float32)
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    return data,image

# 示例输入
if __name__ == "__main__":
    # 模型实例化
    model = UNet()
    # from collections import OrderedDict
    # pkl =r'E:\LY\Augcover\Distribution-preserving-based-Automatic-Data-Augmentation-for-Deep-Image-Steganalysis-main\result\hill0.4\b\msePronetG_epoch_99.pkl'
    # pretrained_net_dict = torch.load(pkl)
    # new_state_dict = OrderedDict()
    # for k, v in pretrained_net_dict.items():
    #     name = k[7:]  # remove "module."
    #     new_state_dict[name] = v
    model.load_state_dict(torch.load(r'unet/wow0.4/a_b/unetParams.pkl'))
    # model.load_state_dict(new_state_dict)
    model.eval()
    # 输入为 256x256 大小的灰度图，批次大小为 1
    x = r'E:\LY\Bilinear\suni\500-suni0.4\b\test\cover\1.pgm'  # 1 表示单通道灰度图
    z = r'E:\LY\Bilinear\wow\500-wow0.4\b\test\cover\40.pgm'
    data, img_s = image2tensor(z)
    y = model(data)
    y = tes(y / 2, y / 2)
    y = y.reshape(256, 256)
    y = y.detach().cpu().numpy()
    y = 8 * y  # The amplitude of noise is set to 16
    # y[y > 255] = 255
    # y[y < 0] = 0
    # y[y == 0] = 255
    y = np.uint8(y)

    # 添加噪声到原始图像
    img_original_float = img_s.astype(np.float32)
    # 添加噪声
    img_noisy = img_original_float + y
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)  # 确保像素值在有效范围内

    # 创建噪声掩码（假设 y > 0 表示有噪声）
    noise_mask = y > 0  # 根据 y 的定义调整条件

    # 将原始图像转换为RGB以便高亮显示噪声
    img_original_rgb = cv2.cvtColor(img_s, cv2.COLOR_GRAY2BGR)

    # 创建一个红色的高亮图层
    highlight = np.zeros_like(img_original_rgb)
    highlight[:, :, 2] = y  # 将噪声幅度映射到红色通道

    img_highlight = img_original_rgb.copy()
    img_highlight[noise_mask] = [0, 0, 255]  # 将噪声区域设置为红色
    # # 将高亮图层叠加到原始图像上
    # alpha = 0.5  # 透明度
    # img_highlight = img_original_rgb.copy()
    # img_highlight[noise_mask] = cv2.addWeighted(img_original_rgb, 1 - alpha, highlight, alpha, 0)[noise_mask]

    # 显示结果
    # 使用 OpenCV 显示
    # cv2.imshow('Original Image', img_s)
    # cv2.imshow('Noise Map', y)
    # cv2.imshow('Noisy Image', img_noisy)
    cv2.imshow('Highlighted Noise', img_highlight)
    cv2.imwrite(r'E:\LY\work_two\ACSNet\img\noise4.png', img_highlight)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # img_noisy = img_original_float + y
    # img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)  # 确保像素值在有效范围内
    # z = y.sum()
    # print(y,z)
    # cv2.imshow('Combined Heatmaps', y)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # output = np.uint8(output)
    # output_image = (y)[0, 0]
    # # output_image=(output_image1-img_s)
    # # 将像素值转换到 [0, 255] 范围
    # # output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())  # 归一化
    # output_image = (output_image * 255).astype(np.uint8)
    #
    # plt.imshow(output_image, cmap='gray')
    # plt.title('U-Net Output Image')
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
    # x = torch.randn((1, 1, 256, 256))  # 1 表示单通道灰度图
    # output = model(x)
    #
    # print(f"输入形状: {x.shape}")
    # print(f"输出形状: {output.shape}")
