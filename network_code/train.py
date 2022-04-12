from __future__ import print_function
from __future__ import division

import argparse
import random
# from sched import scheduler
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time 
# from warpctc_pytorch import CTCLoss
import os
import utils
from tensorboardX import SummaryWriter

# from dataloader import create_dataloader
from dataloader import create_dataloader
# from models.doanet25 import DOANet25 as DOANet
from models.crnn10 import CRNN10 as DOANet

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(model, dataloader, opt, criterion, optimizer):
    model.train()
    loss = 0.0
    for inputs, label in dataloader:
        inputs = inputs.cuda()
        # label = torch.LongTensor(label)
        label = label.long()
        label = label.cuda()
        outputs = model(inputs)
        # print(outputs.shape)
        loss_batch = criterion(outputs, label)
        loss += loss_batch.item()
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
    return loss 


def valid(model, dataloader, opt, criterion):
    model.eval()
    loss = 0.0
    for inputs, label in dataloader:
        inputs = inputs.cuda()
        label = label.long()
        label = label.cuda()

        outputs = model(inputs)
        loss_batch = criterion(outputs, label)
        loss += loss_batch.item()
    return loss     

def trainNetwork(opt, traindataloader, cvdataloader):
    model = DOANet()
    model.cuda()
    # model.apply(weights_init)
    print(model)
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(opt.expr_dir)

    if opt.cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(opt.ngpu))
        criterion = criterion.cuda()
    if opt.adam:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                            betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(model.parameters())
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    best_cv_loss = np.inf 
    for epoch in range(opt.nepoch):
        print('time: ', time.asctime(time.localtime(time.time())))
        print('epoch: ', epoch)
        train_loss = train(model, traindataloader, opt, criterion, optimizer)
        print('time: ', time.asctime(time.localtime(time.time())))
        print('train loss: ', train_loss)
        cv_loss = valid(model, cvdataloader, opt, criterion)
        print('time: ', time.asctime(time.localtime(time.time())))
        print('epoch: {} train_loss: {}  cv_loss: {}'.format(epoch, train_loss, cv_loss))
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/cv', cv_loss, epoch)
        if cv_loss < best_cv_loss:
            best_cv_loss = cv_loss
            save_path = os.path.join(opt.expr_dir, 'best_model.pt')
            torch.save({
                'model': model.state_dict(), 
                'epoch': epoch,
            }, save_path)
        save_path = os.path.join(opt.expr_dir, 'last_model.pt')
        torch.save({
                'model': model.state_dict(), 
                'epoch': epoch,
                }, save_path)
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    # TODO(meijieru): epoch -> iter
    # parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--cuda', type=int, default=1, help='use cuda')
    parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
    parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
    parser.add_argument('--expr_dir', default='doanet_result_scheduler_normalize', help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
    parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadealta')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
    parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
    parser.add_argument('--train_path', type=str, default='/data/wyw/repos/0328/generate_dataset/train_dataset.lst', help='path for train dataset')
    parser.add_argument('--cv_path', type=str, default='/data/wyw/repos/0328/generate_dataset/cv_dataset.lst', help='path for cv dataset')
    parser.add_argument('--log_path', type=str, default='./log/', help='path for log record')
    parser.add_argument('--angle_resolution', type=int, default=5, help='angle resolution')
    opt = parser.parse_args()
    print(opt)

    if not os.path.exists(opt.expr_dir):
        os.makedirs(opt.expr_dir)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    # if torch.cuda.is_available() and not opt.cuda:
    #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    trainloader = create_dataloader(opt, opt.train_path)
    cvloader = create_dataloader(opt, opt.cv_path)
    trainNetwork(opt, trainloader, cvloader)

