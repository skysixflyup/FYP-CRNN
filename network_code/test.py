from __future__ import print_function
from __future__ import division

import argparse
import random
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


def testNetwork(opt, model_path, data_loader, data_cate=72):
    model = DOANet()
    print(model)
    # model.apply(weights_init)
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path)['model'])
    result_matrix = np.zeros((data_cate, data_cate), dtype=np.int)
    correct_label, total_label = 0, 0
    for inputs, label in data_loader:
        inputs = inputs.cuda()
        label = label.cuda()
        outputs = model(inputs)
        label_choice = label.cpu().numpy() 
        label_choice = int(label_choice[0])
        output_result = torch.argmax(outputs).cpu().numpy()
        # print(label_choice, output_result)
        result_matrix[label_choice][output_result] += 1
        if label_choice == output_result:
            result_matrix[label_choice][output_result] += 1
            correct_label += 1
        elif label_choice + output_result == 36:
            result_matrix[label_choice][36-output_result] += 1
            correct_label += 1
        else:
            result_matrix[label_choice][output_result] += 1
        total_label += 1
    
    print(correct_label, total_label, correct_label/total_label)
    np.save("test_result.npy", result_matrix)

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
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
    parser.add_argument('--expr_dir', default='doanet_result_scheduler', help='Where to store samples and models')
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
    parser.add_argument('--test_path', type=str, default='/data/wyw/repos/0328/generate_dataset/test_dataset.lst', help='path for cv dataset')
    opt = parser.parse_args()
    print(opt)

    if not os.path.exists(opt.expr_dir):
        os.makedirs(opt.expr_dir)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    testloader = create_dataloader(opt, opt.test_path)
    modelpath = '/data/wyw/repos/0328/network/doanet_result_scheduler_normalize/best_model.pt'
    testNetwork(opt, modelpath, testloader)

