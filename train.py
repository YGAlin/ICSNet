import math
import os
import numpy
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils
from opt import opt
from torch import optim
from torch.nn import functional as F
import torch.nn as nn
from net import SRNetFC,LWENetFC
from model import SRNet
# from nwd import NuclearWassersteinDiscrepancy
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np
args = opt()
if args.net == 'LWENet':
    model = LWENetFC(args).cuda()
elif args.net == 'SRNet':
    model = SRNetFC(args).cuda()
# modelB = SRNet().cuda()
# modelB.load_state_dict(torch.load(r"E:\LY\SRNet\RT_n-b\a\epoch99_acc0.612000.pkl"))
# domain = domain().cuda()
CUDA = True if torch.cuda.is_available() else False

def euclidean_distance(feature_map1, feature_map2):
    # 将特征图展平为一维向量
    b,c,h,w = feature_map1.size()
    feature_map1 = feature_map1.view(b,-1)
    feature_map2 = feature_map2.view(b,-1)

    # 计算欧氏距离
    distance = 1-F.cosine_similarity(feature_map1,feature_map2,dim=1)
    # print(distance.shape)
    return distance.mean()

def main(args):
    if args.net == 'SRNet':
        optimizer = optim.Adamax(model.get_parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    elif args.net == 'LWENet':
        optimizer = optim.SGD(model.get_parameters(), lr=0.001, momentum=args.momentum)
    # print(optimizer)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
    # dataloader
    args.batch_size_t = 4
    train_source_loader, train_target_loader, test_target_loader = utils.create_dataloader(args)
    args.tar = args.data_path_source + '_' + args.data_path_target
    train_inter = os.path.join(args.data_path,args.data_path_target,args.tar)
    train_inter_dataset = datasets.ImageFolder(
        train_inter,transform=transforms.Compose([transforms.Grayscale(),transforms.ToTensor()]))
    train_inter_loader = DataLoader(
        train_inter_dataset, batch_size=args.batch_size_t, shuffle=True, drop_last=False,
        pin_memory=True, sampler=None
    )
    iters_epoch = max(len(train_source_loader), len(train_target_loader))
    # 输出特征提取器的预训练权重路径
    pkl_name = os.listdir(args.pkl)
    pkl_path = os.path.join(args.pkl, pkl_name[0])
    print(pkl_path)
    pkl_logs = args.output_pkl
    acctxt_logs = args.output_dir
    if not os.path.isdir(pkl_logs):
        os.makedirs(pkl_logs)
    if not os.path.isdir(acctxt_logs):
        os.makedirs(acctxt_logs)
    filenames = '非线性参数.txt'
    filepath = os.path.join(acctxt_logs, filenames)
    with open(filepath, 'a') as f:
        f.write(f'\n')
        f.write(f'g_lr:{0.00025}\tf_loss:{0.001}\tloss_weight:0.8\tattentive:max\n')
        f.write(f'source_path:{args.data_path+args.data_path_source+args.src}\n'
                f'target_path:{args.data_path+args.data_path_target+args.tar}\n'
                f'pkl:{pkl_path}\n')

    torch.manual_seed(args.seed)
    if CUDA:
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpu = None
    best_acc = 0.
    for epoch in range(1, args.epochs + 1):
        source_iter = utils.IterDataLoader(train_source_loader)
        target_iter = utils.IterDataLoader(train_target_loader)
        inter_iter = utils.IterDataLoader(train_inter_loader)
        test_iter = utils.IterDataLoader(test_target_loader)
        lr_train = (optimizer.state_dict()['param_groups'][0]['lr'])
        print('epoch:{},lr:{:.6f}'.format(epoch,lr_train))
        print('tar_name:',args.data_path+'//'+args.data_path_target+'//'+args.tar)
        train(args, epoch, optimizer, scheduler, source_iter, target_iter,inter_iter, model,iters_epoch)
        tar_acc, src_acc = test(epoch, source_iter, test_iter,iters_epoch)
        print("The best_acc={:.2f}%".format(best_acc))
        if best_acc < tar_acc:
            best_acc = tar_acc
            a = 'epoch{}_acc{:.2f}.pkl'.format(epoch, tar_acc)
            dir = os.path.join(pkl_logs, a)
            torch.save(model.state_dict(), dir)

        with open(filepath, 'a') as f:
            tar_acc = "{:.4f}".format(tar_acc / 100)
            src_acc = "{:.4f}".format(src_acc / 100)
            f.write(f'Epoch{epoch} Target:{tar_acc}\tTest_r:{src_acc}\n')

def train(args, epoch, optimizer, scheduler, source_iter, target_iter, inter_iter,model,iters_epoch):
    model.train()
    # modelB.eval()
    lam_sum, gama_sum=0,0
    utils.adjust_learning_rate(optimizer, epoch, args)
    clc = nn.CrossEntropyLoss()

    # loss_nwd = NuclearWassersteinDiscrepancy(model.fc)
    for i in range(iters_epoch):
        src_img, src_label = next(source_iter)
        tar_img, tar_label = next(target_iter)
        inter_img, inter_label = next(inter_iter)
        if CUDA:
            src_img, src_label = src_img.cuda(), src_label.cuda()
            inter_img, inter_label = inter_img.cuda(), inter_label.cuda()
            tar_img,tar_label = tar_img.cuda(),tar_label.cuda()
        input = torch.cat((src_img,tar_img,inter_img),dim=0)

        # jlabel = src_label-tar_label
        x,y_all,_ = model(input)
        # _,t_out = modelB(tar_img)
        # y_s,y_inter = y_all.chunk(2,dim=0)
        y_s = y_all[:src_label.size(0)]
        y_inter = y_all[-inter_label.size(0):]
        # y_t,y_inter = y_inter.chunk(2,dim=0)
        x_s,x_t = x.chunk(2,dim=0)
        # p_t = F.softmax(y_t, dim=1)
        #
        # p = float(i + epoch * iters_epoch) / args.epochs / iters_epoch
        # lam = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1
        #
        # t_label = F.softmax(y_t, dim=1).max(1, keepdim=True)[1].view(-1)
        clc_s = clc(y_s,src_label)
        # domain_out = clc(y2,domain_label)
        loss = clc_s+clc(y_inter,inter_label)*0.5
        #loss = clc_s + loss_MI*0.1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            pred1 = y_inter.max(1, keepdim=True)[1]
            correct1 = pred1.eq(inter_label.view_as(pred1)).sum().item()
            acc1 = correct1 / inter_label.size(0)
            pred2 = y_s.max(1, keepdim=True)[1]
            correct2 = pred2.eq(src_label.view_as(pred2)).sum().item()
            acc2 = correct2 / src_label.size(0)
            print(
                'Train epoch:{} [{}/{} ({:.0f}%)]\ntar_acc:{:.2f}\tsrc_acc:{:.2f}\tloss_s:{:.4f}\tloss_t:{:.4f}\tloss_inter:{:.4f}\tloss_all:{:.4f}\tloss_tar1:{:.4f}'.format(
                    epoch, (i) * args.batch_size_s, iters_epoch*8,
                           (i) * 100. / iters_epoch,
                    acc1, acc2, clc(y_s, src_label).item(), clc(y_inter,inter_label).item(),
                    euclidean_distance(x_s,x_t).item(),
                    loss.item(), loss.item()
                ),flush=True)

    # scheduler.step()
    return lam_sum/(iters_epoch),gama_sum/(iters_epoch)


def test(epoch,source_iter, test_iter,iters_epoch):
    model.eval()
    test_loss1 = 0
    test_loss2 = 0
    correct1 = 0
    correct2 = 0
    cover_target =0
    cover_test = 0
    num = 0
    clc = nn.CrossEntropyLoss()
    ft = np.zeros((1000, 512))
    truelabels_t = np.zeros(1000)
    with torch.no_grad():
        for idx in range(iters_epoch):
            src_img, src_label = next(source_iter)
            tar_img, tar_label = next(test_iter)
            if CUDA:
                src_img, src_label = src_img.cuda(), src_label.cuda()
                tar_img, tar_label = tar_img.cuda(), tar_label.cuda()
            input = torch.cat((src_img, tar_img), dim=0)
            x, y, f = model(input)
            # y1, y2 = y.chunk(2, dim=0)
            y1 = y[:src_label.size(0)]
            y2 = y[src_label.size(0):]
            f_s,f_t=f.chunk(2,dim=0)
            # ft[8 * idx:8 * idx + 8] = f_t.cpu().detach().numpy()
            # truelabels_t[8 * idx:8 * idx +8] = tar_label.cpu().detach().numpy()
            test_loss1 += clc(y1, src_label).item()
            test_loss2 += clc(y2, tar_label).item()
            pred1 = y1.max(1, keepdim=True)[1]
            pred2 = y2.max(1, keepdim=True)[1]
            correct1 += pred1.eq(src_label.view_as(pred1)).sum().item()
            correct2 += pred2.eq(tar_label.view_as(pred2)).sum().item()
            cover_test += (src_label==0).sum()
            cover_target+=(tar_label==0).sum()
            num += tar_img.size(0)
    test_loss1 = test_loss1 / num
    test_loss2 = test_loss2 / num
    acc1 = correct1 * 100. / (num*2)
    acc2 = correct2 * 100. / num
    print(
        '\nTest1 set: loss_t: {:.4f}, tar_acc: {}/{} ({:.6f}%)\tTest2 set: loss_s: {:.4f}, tar_r_acc: {}/{} ({:.6f}%)\n'.format(
            cover_target, correct2, num, acc2, cover_test, correct1, num*2, acc1),flush=True)
    #
    # def visual(feat):
    #     # t-SNE的最终结果的降维与可视化
    #     ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    #     x_ts = ts.fit_transform(feat)
    #     x_min, x_max = x_ts.min(0), x_ts.max(0)
    #     x_final = (x_ts - x_min) / (x_max - x_min)
    #     return x_final
    # # 设置散点形状
    # maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    # # 设置散点颜色
    # colors = ['red', 'green', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
    #           'hotpink']
    # # 图例名称
    # Label_Com = ['a', 'b', 'c', 'd']
    # # 设置字体格式
    # font1 = {'family': 'Times New Roman',
    #          'weight': 'bold',
    #          'size': 32,
    #          }
    #
    # def plotlabels(S_lowDWeights, Trure_labels, name):
    #     plt.clf()
    #     True_labels = Trure_labels.reshape((-1, 1))
    #     S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    #     S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    #       # [num, 3]
    #
    #     for index in range(2):  # 假设总共有三个类别，类别的表示为0,1,2
    #         X = S_data.loc[S_data['label'] == index]['x']
    #         Y = S_data.loc[S_data['label'] == index]['y']
    #         plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index],
    #                     alpha=0.65)
    #         plt.xticks([])  # 去掉横坐标值
    #         plt.yticks([])  # 去掉纵坐标值
    #
    #     plt.title(name, fontsize=32, fontweight='normal', pad=20)
    #
    # fig = plt.figure(figsize=(10, 10))
    # plotlabels(visual(ft), truelabels_t, 'DMNet')
    # a = 'epoch{}_acc{:.2f}.png'.format(epoch, acc2)
    # t_sne_path = os.path.join(args.output_dir,'t_sne',a)
    # if not os.path.isdir(os.path.join(args.output_dir,'t_sne')):
    #     os.makedirs(os.path.join(args.output_dir,'t_sne'))
    # plt.savefig(t_sne_path)

    return acc2, acc1

def margin_loss(y,label):
    # one_hot = torch.zeros_like(y).scatter_(1,label.long(),1)
    y = F.softmax(y,dim=1)
    a = 1- (y[label]-y[1-label])
    loss = torch.sum(torch.clamp(a,min=0.0) - y[label])/y.shape[0]
    return loss

if __name__ == '__main__':
    main(args)
