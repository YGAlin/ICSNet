import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from model import SRNet
from LWENet import lwenet

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        # nn.init.xavier_normal_(m.weight)
        # nn.init.zeros_(m.bias)
        nn.init.normal_(m.weight, 0, 0.01)
        # if isinstance(m, nn.Linear):
        #     nn.init.normal_(m.weight, 0, 0.01)

class SRNetFC(nn.Module):
    def __init__(self,args):
        super(SRNetFC, self).__init__()
        self.args = args
        models = SRNet()
        # models.load_state_dict(torch.load(r"E:\LY\500-hill0.4\m-hill0.4\epoch418acc0.78875.pkl"))
        pkl_name = os.listdir(args.pkl)
        pkl_path = os.path.join(args.pkl, pkl_name[0])
        models.load_state_dict(torch.load(pkl_path))
        # print(args.pkl)
        # 特征提取层
        self.layer1 = models.layer1
        self.layer2 = models.layer2
        self.layer3 = models.layer3
        self.layer4 = models.layer4
        self.layer5 = models.layer5
        self.layer6 = models.layer6
        self.layer7 = models.layer7
        self.layer8 = models.layer8
        self.layer9 = models.layer9
        self.layer10 = models.layer10
        self.layer11 = models.layer11
        self.layer12 = models.layer12.block[:5]
        # self.pam = Pam(512)
        self.cca = Cca3(512)
        self.avepool = nn.AdaptiveAvgPool2d(1)
        # self.fc = models.layer13
        self.fc = nn.Linear(models.layer13.in_features, models.layer13.out_features, bias=True)

    def get_parameters(self):
        parameter_list = [
            {"params": self.layer1.parameters(), 'name': 'pre-trained',"lr_mult": 0.5},
            {"params": self.layer2.parameters(), 'name': 'pre-trained',"lr_mult": 0.5},
            {"params": self.layer3.parameters(), 'name': 'pre-trained',"lr_mult": 0.5},
            {"params": self.layer4.parameters(), 'name': 'pre-trained',"lr_mult": 0.5},
            {"params": self.layer5.parameters(), 'name': 'pre-trained',"lr_mult": 0.5},
            {"params": self.layer6.parameters(), 'name': 'pre-trained',"lr_mult": 0.5},
            {"params": self.layer7.parameters(), 'name': 'pre-trained',"lr_mult": 0.5},
            {"params": self.layer8.parameters(), 'name': 'pre-trained',"lr_mult": 0.5},
            {"params": self.layer9.parameters(), 'name': 'pre-trained',"lr_mult": 0.5},
            {"params": self.layer10.parameters(), 'name': 'pre-trained',"lr_mult": 0.5},
            {"params": self.layer11.parameters(), 'name': 'pre-trained',"lr_mult": 0.5},
            {"params": self.layer12.parameters(), 'name': 'fc',"lr_mult": 0.5},
            {"params": self.fc.parameters(), 'name': 'fc',"lr_mult": 1.0},
        {"params": self.cca.parameters(), 'name': 'fc',"lr_mult": 1.0},]
        # {"params": self.pam.parameters(), 'name': 'fc',"lr_mult": 1.0}]

        return parameter_list

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x2 = self.layer12(x)
        x2 = self.cca(x2)
        # x2 = self.pam(x2)
        x1 = self.avepool(x2).view(x.shape[0],-1)

        y = self.fc(x1)
        return x2,y,x1

class LWENetFC(nn.Module):
    def __init__(self,args):
        super(LWENetFC, self).__init__()
        self.args = args
        models = lwenet()
        # models.load_state_dict(torch.load(r"E:\LY\500-hill0.4\m-hill0.4\epoch418acc0.78875.pkl"))
        pkl_name = os.listdir(args.pkl)
        pkl_path = os.path.join(args.pkl, pkl_name[0])
        models.load_state_dict(torch.load(pkl_path))

        self.feature = nn.Sequential(
            models.Dense_layers,models.layer5,models.layer5_BN,models.layer5_AC,
            models.layer6,models.layer6_BN,models.layer6_AC,models.avgpooling2,
            models.layer7,models.layer7_BN,models.layer7_AC,
            models.layer8,models.layer8_BN,models.layer8_AC,models.avgpooling3,
            models.layer9, models.layer9_BN, models.layer9_AC,
            models.layer10, models.layer10_BN, models.layer10_AC,
        )
        self.cca = Cca3(256)
        self.pam = Pam(256)
        self.GAP = models.GAP
        self.L2_norm = models.L2_norm
        self.L1_norm = models.L1_norm
        # self.fc = models.layer13
        self.fc = nn.Linear(models.fc1.in_features, models.fc1.out_features, bias=True)

    def get_parameters(self):
        parameter_list = [
            {"params": self.feature.parameters(), 'name': 'pre-trained',"lr_mult": 0.5},
            {"params": self.fc.parameters(), 'name': 'fc', "lr_mult": 1.0},
            {"params": self.cca.parameters(), 'name': 'fc', "lr_mult": 1.0},
            {"params": self.pam.parameters(), 'name': 'fc', "lr_mult": 1.0},
        ]
        return parameter_list

    def forward(self,x):
        x = self.feature(x)
        x2 = self.cca(x)
        x = self.pam(x2)
        output_GAP = self.GAP(x).view( -1,256)
        output_L2 = self.L2_norm(x).view( -1,256)
        output_L1 = self.L1_norm(x).view( -1,256)

        Final_feat = torch.cat([output_GAP,output_L2,output_L1],dim=-1)
        y = self.fc(Final_feat)
        return x2,y,Final_feat

class Pam(nn.Module):
    def __init__(self, in_dim):
        super(Pam,self).__init__()
        self.chanel_in = in_dim

        self.q = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.k = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.v = nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if self.training:
            x_s, x_t= x.chunk(2,dim=0)
            b,c,h,w = x_t.size()
            # # A -> (N,C,HW)
            # s_q = x_s.view(b, c, -1)
            # # A -> (N,HW,C)
            # s_k = x_s.view(b, c, -1).permute(0, 2, 1)
            # # 矩阵乘积，通道注意图：X -> (N,C,C)
            # energy = torch.bmm(s_q, s_k)
            # attention = self.softmax(energy)
            # # A -> (N,C,HW)
            # proj_value = x_s.view(b, c, -1)
            # # XA -> （N,C,HW）
            # out = torch.bmm(attention, proj_value)
            # # output -> (N,C,H,W)
            # x_s = out.view(b, c, h, w)
            q = self.q(x_s).view(b,-1,h*w).permute(0,2,1)
            k = self.k(x_t).view(b,-1,h*w)
            v = self.v(x_t).view(b,-1,h*w)
            qk = torch.bmm(q,k)
            att = self.softmax(qk)
            out = torch.bmm(v,att.permute(0,2,1))
            out = out.view(b,c,h,w)
            out = x_t+out
            x = torch.cat((x_s,out),dim=0)
        else:
            x_s, x_t= x.chunk(2, dim=0)
            b, c, h, w = x_t.size()
            q = self.q(x_s).view(b, -1, h * w).permute(0, 2, 1)
            k = self.k(x_t).view(b, -1, h * w)
            v = self.v(x_t).view(b, -1, h * w)
            qk = torch.bmm(q, k)
            att = self.softmax(qk)
            out = torch.bmm(v, att.permute(0, 2, 1))
            out = out.view(b, c, h, w)
            x_t = x_t + out
            x = torch.cat((x_s, x_t), dim=0)
        return x


class Cca3(nn.Module):
    def __init__(self,in_dim):
        super(Cca3, self).__init__()
        self.q = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.k = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.v = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if self.training:
            x_s, x_t=x.chunk(2,dim=0)
            b,c,h,w=x_s.size()

            q_t = self.q(x_t).view(b, -1, h * w)# B,C,H*W
            k_s = self.k(x_s).view(b, -1, h * w).permute(0, 2, 1)# B,H*W,C
            v_s = self.v(x_s).view(b, -1, h * w)# B,C,H*W
            q_s = self.q(x_s).view(b, -1, h * w)# B,C,H*W
            k_t = self.k(x_t).view(b, -1, h * w).permute(0, 2, 1)# B,H*W,C
            v_t = self.v(x_t).view(b, -1, h * w)# B,C,H*W

            a_st = self.softmax(torch.bmm(q_t,k_s))# B,C,C
            a_ts = self.softmax(torch.bmm(q_s,k_t))# B,C,C
            att = self.softmax(torch.bmm(a_st,a_ts.permute(0,2,1)))

            out_s = torch.bmm(att,v_s).view(b,c,h,w)# B,C,H*W
            out_t = torch.bmm(att, v_t).view(b,c,h,w)  # B,C,H*W

            x_s = x_s+out_s
            x_t = x_t+out_t
            x = torch.cat((x_s,x_t),dim=0)
            return x
        else:
            # b,c,h,w = x.size()
            # if  b==1:
            #
            #     q_t = self.q(x).view(b, -1, h * w)  # B,C,H*W
            #     k_t = self.k(x).view(b, -1, h * w).permute(0, 2, 1)  # B,H*W,C
            #     v_t = self.v(x).view(b, -1, h * w)  # B,C,H*W
            #
            #     a_st = self.softmax(torch.bmm(q_t, k_t))  # B,C,C
            #
            #     out_t = torch.bmm(a_st, v_t).view(b, c, h, w)  # B,C,H*W
            #
            #     # x_s = x_s + out_s
            #     x = x + out_t
            #     # x = torch.cat((x_s, x_t), dim=0)
            # else:
            x_s, x_t = x.chunk(2, dim=0)
            b, c, h, w = x_s.size()

            q_s = self.q(x_s).view(b, -1, h * w)  # B,C,H*W
            k_s = self.k(x_s).view(b, -1, h * w).permute(0, 2, 1)  # B,H*W,C
            v_s = self.v(x_s).view(b, -1, h * w)  # B,C,H*W
            q_t = self.q(x_t).view(b, -1, h * w)  # B,C,H*W
            k_t = self.k(x_t).view(b, -1, h * w).permute(0, 2, 1)  # B,H*W,C
            v_t = self.v(x_t).view(b, -1, h * w)  # B,C,H*W

            a_st = self.softmax(torch.bmm(q_t, k_s))  # B,C,C
            a_ts = self.softmax(torch.bmm(q_s, k_t))  # B,C,C
            att = self.softmax(torch.bmm(a_st, a_ts.permute(0, 2, 1)))
            # att = torch.bmm(a_st, a_ts.permute(0, 2, 1))
            out_s = torch.bmm(att, v_s).view(b, c, h, w)  # B,C,H*W
            out_t = torch.bmm(att, v_t).view(b, c, h, w)  # B,C,H*W

            x_s = x_s + out_s
            x_t = x_t + out_t
            x = torch.cat((x_s, x_t), dim=0)
            return x
