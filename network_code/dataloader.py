import os
import glob
import torch
import librosa
import scipy.io as io
import math
import scipy.signal as signal 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import h5py 


def create_dataloader(opt, trainpath, shuffle=True):
    return DataLoader(dataset=VFDataset(opt, trainpath),
                        batch_size=opt.batch_size,
                        shuffle=shuffle,
                        num_workers=opt.workers,
                        pin_memory=True,
                        drop_last=True,
                        sampler=None)


class VFDataset(Dataset):
    def __init__(self, opt, trainpath):
        self.opt = opt
        self.trainpath = trainpath 
        self.trainlist = []
        self.trainlabel = []
        with open(self.trainpath, 'r') as wf:
            trainlists = wf.readlines()
            for traininfo in trainlists:
                traininfo = traininfo.strip().split(' ')
                self.trainlist.append(traininfo[0])
                self.trainlabel.append(int(traininfo[1]) / self.opt.angle_resolution)
        assert (len(self.trainlist) == len(self.trainlabel))
        self.total_len = len(self.trainlist)


    def __len__(self):
        return self.total_len


    def readmatfunc(self, mat_path):
        mat_info = h5py.File(mat_path)
        signal_input = np.array(mat_info['signal'], dtype=np.float32)
        return signal_input 

    def __getitem__(self, idx):
        matpath = self.trainlist[idx]
        matid = self.trainlabel[idx]
        raw_audio = self.readmatfunc(matpath)
        raw_audio = torch.from_numpy(raw_audio).float()
        spec = torch.stft(raw_audio, n_fft=256, hop_length=64, return_complex=False)
        rea = spec[:, :, :, 0]#实部
        imag = spec[:, :, :, 1]#虚部
        mag = torch.abs(torch.sqrt(torch.pow(rea, 2) + torch.pow(imag, 2)))
        phase = torch.atan2(imag.data, rea.data)
        mag_phase = torch.cat((mag, phase), 0) # [12, 129, 376]
        mag_phase /= torch.max(torch.abs(mag_phase))
        return mag_phase, matid
