import math
import os
import numpy
import numpy as np
import torch
import utils
from opt import opt
from torch import optim
import TES
from torch.nn import functional as F
import torch.nn as nn
from net import SRNetFC
from model import SRNet
from LWENet import lwenet
from Domain import DomainDiscriminator
from Unet import UNet
# from nwd import NuclearWassersteinDiscrepancy
args = opt()
if args.net == 'LWENet':
    modelB = lwenet().cuda()
elif args.net == 'SRNet':
    modelB = SRNet().cuda()
modelA = DomainDiscriminator().cuda()
model = UNet().cuda()
# modelB.load_state_dict(torch.load(r"E:\LY\SRNet\RT_n-b\a\epoch99_acc0.612000.pkl"))
# domain = domain().cuda()
CUDA = True if torch.cuda.is_available() else False
tes = TES.TES().cuda()



def main(args):
    # optimizer = optim.Adamax([{'params':model.parameters()},{'params':modelA.parameters(),'lr':0.001}], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    optimizer = optim.Adamax(model.parameters(), lr=0.01,
                             betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
    # dataloader
    train_source_loader, train_target_loader, test_target_loader = utils.create_dataloader(args)
    iters_epoch = min(len(train_source_loader), len(train_target_loader))
    # train_name = 'suni0.4'
    # pkl_logs = args.log + '\\' + train_name
    pkl_name = os.listdir(args.pkl)
    pkl_path = os.path.join(args.pkl, pkl_name[0])
    print(pkl_path)
    pkl_logs = args.unet_pkl
    # acctxt_logs = 'acctxt' + '\\' + args.data_path_source + '_' + args.data_path_target + '\\' + train_name
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
        test_iter = utils.IterDataLoader(test_target_loader)
        lr_train = (optimizer.state_dict()['param_groups'][0]['lr'])
        print('epoch:{},lr:{:.6f}'.format(epoch,lr_train))
        print('train_name:',args.data_path+args.data_path_source+'_'+args.data_path_target)
        print('weight:',args.weight)
        train(args, epoch, optimizer, scheduler, source_iter, target_iter, model,iters_epoch)
        print("The best_acc={:.2f}%".format(best_acc))
        tar_acc, src_acc = test(epoch, source_iter, test_iter,iters_epoch)
        if best_acc <= tar_acc:
            best_acc = tar_acc
        a = 'unetParams.pkl'
        dir = os.path.join(pkl_logs, a)
        torch.save(model.state_dict(), dir)

        with open(filepath, 'a') as f:
            tar_acc = "{:.4f}".format(tar_acc / 100)
            src_acc = "{:.4f}".format(src_acc / 100)
            f.write(f'Epoch{epoch} Target:{tar_acc}\tSource:{src_acc}\n')

def loss_noise(noise):                     #loss2
    loss = abs(noise).sum()/args.batch_size_t
    return abs(loss/8-400)

def train(args, epoch, optimizer, scheduler, source_iter, target_iter, model,iters_epoch):
    model.train()
    modelB.eval()
    modelA.eval()
    pkl_name = os.listdir(args.pkl)
    pkl_path = os.path.join(args.pkl, pkl_name[0])
    modelB.load_state_dict(torch.load(pkl_path))
    modelA.load_state_dict(torch.load(os.path.join(args.D_pkl,'D_best.pkl')))
    lam_sum, gama_sum=0,0
    # utils.adjust_learning_rate(optimizer, epoch, args)
    clc = nn.CrossEntropyLoss()

    # loss_nwd = NuclearWassersteinDiscrepancy(model.fc)
    for i in range(iters_epoch):
        src_img, src_label = next(source_iter)
        tar_img, tar_label = next(target_iter)
        if CUDA:
            src_img, src_label = src_img.cuda(), src_label.cuda()
            tar_img, tar_label = tar_img.cuda(), tar_label.cuda()

        p = model(tar_img)
        p1 = p / 2
        m1 = p / 2
        noise = tes(p1, m1) * 8
        tar_img1 = noise + tar_img

        input = torch.cat((src_img,tar_img1),dim=0)
        with torch.no_grad():
            f,_ = modelB(input)
            d_y = modelA(f)
            # x1_s,x1_t1,x1_t = x1.chunk(3,dim=0)
            # doamin_label = torch.cat((torch.zeros(tar_img1.size(0)),torch.zeros(tar_img1.size(0))),dim=0).long().cuda()
            # perm = torch.randperm(f.size(0))
            # f_shuffled = f[perm]
            # domain_label_shuffled = doamin_label[perm]
            # d_y = modelA(f_shuffled)
        d_y_s, d_y_t = d_y.chunk(2, dim=0)
        loss_d = - d_y_t.log().sum() / d_y_t.size(0)
        # loss_t = - (1 - d_y_t).log().sum() / d_y_t.size(0)
        # loss_d = loss_s+loss_t
        criterion = nn.L1Loss()

        loss = loss_d+loss_noise(noise)*0.8
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % 20 == 0:
            # pred1 = d_y.max(1, keepdim=True)[1]
            # correct1 = pred1.eq(domain_label_shuffled.view_as(pred1)).sum().item()
            # acc1 = correct1 / domain_label_shuffled.size(0)
            # pred2 = d_y.max(1, keepdim=True)[1]
            # correct2 = pred2.eq(domain_label_shuffled.view_as(pred2)).sum().item()
            # acc2 = correct2 / domain_label_shuffled.size(0)
            # print(
            #     'Train epoch:{} [{}/{} ({:.0f}%)]\ntar_acc:{:.2f}\tsrc_acc:{:.2f}\tloss_s:{:.4f}\tloss_t:{:.4f}\tloss_inter:{:.4f}\tloss_all:{:.4f}\tloss_tar1:{:.4f}'.format(
            #         epoch, (i) * args.batch_size_s, iters_epoch*8,
            #                (i) * 100. / iters_epoch,
            #         acc1, acc2, clc(d_y, domain_label_shuffled).item(), criterion(tar_img,tar_img1).item(),
            #         loss.item(),
            #         loss.item(), loss.item()
            #     ),flush=True)

    # scheduler.step()
    return lam_sum/(iters_epoch),gama_sum/(iters_epoch)


def test(epoch, source_iter, test_iter,iters_epoch):
    model.eval()
    modelB.eval()
    modelA.eval()
    test_loss1 = 0
    test_loss2 = 0
    correct1 = 0
    correct2 = 0
    num = 0
    clc = nn.CrossEntropyLoss()
    with torch.no_grad():
        for idx in range(iters_epoch):
            src_img, src_label = next(source_iter)
            tar_img, tar_label = next(test_iter)
            if CUDA:
                src_img, src_label = src_img.cuda(), src_label.cuda()
                tar_img, tar_label = tar_img.cuda(), tar_label.cuda()
            p = model(tar_img)
            p1 = p / 2
            m1 = p / 2
            noise = tes(p1, m1) * 8
            tar_img1 = noise + tar_img

            doamin_label = torch.zeros(tar_img1.size(0)).long().cuda()
            # input = torch.cat((tar_img1,tar_img), dim=0)
            x, _ = modelB(tar_img1)
            y = modelA(x)
            loss_s = - y.log().sum() / y.size(0)
            # y1, y2 = y.chunk(2, dim=0)
            test_loss1 += loss_s.item()
            # test_loss2 += clc(y2, tar_label).item()
            # pred1 = y.max(1, keepdim=True)[1]
            # pred2 = y2.max(1, keepdim=True)[1]
            correct1 += y.sum()/y.size(0)
            # correct2 += pred2.eq(tar_label.view_as(pred2)).sum().item()
            num += doamin_label.size(0)
    test_loss1 = test_loss1 / iters_epoch
    test_loss2 = test_loss2 / iters_epoch
    acc1 = correct1 * 100. / num
    acc2 = correct2 * 100. / num
    # print(
    #     '\nTest1 set: loss_t: {:.4f}, tar_acc: {}/{} ({:.6f}%)\tTest2 set: loss_s: {:.4f}, src_acc: {}/{} ({:.6f}%)\n'.format(
    #         test_loss2, correct2, num, acc2, test_loss1, correct1, num, acc1),flush=True)
    print(
        '\nTest1 set: loss_t: {:.4f}\tpred: {:.4f}\n'.format(
            test_loss1, correct1 / iters_epoch), flush=True)
    return acc1, acc2

if __name__ == '__main__':
    main(args)
