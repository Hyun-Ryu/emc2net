import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

from util import *


class RBx2(nn.Module):
    def __init__(self, dim_hidden=2, ker_size=129):
        super(RBx2, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=dim_hidden, kernel_size=(1,ker_size), padding=(0,ker_size//2), bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim_hidden, out_channels=2, kernel_size=(1,ker_size), padding=(0,ker_size//2), bias=True),
            
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=dim_hidden, kernel_size=(1,ker_size), padding=(0,ker_size//2), bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim_hidden, out_channels=2, kernel_size=(1,ker_size), padding=(0,ker_size//2), bias=True),
        )

    def forward(self, x):
        # x: (b, 2, 1 8192)
        out = self.conv_1(x)
        out1 = out + x
        out2 = self.conv_2(out1)
        out_ = out2 + out1
        # out_: (b, 2, 1, 8192)
        return out_


class RRC(nn.Module):
    def __init__(self, N, alpha, OS, stride=8):
        super(RRC, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,N), stride=(1,stride), padding=(0,N//2), bias=False)
        _, self.rrc = self.rrcosfilter(N=N, alpha=alpha, Ts=OS, Fs=1)
        self.conv.weight = nn.Parameter(torch.Tensor(self.rrc).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda())

    def forward(self, x):
        return self.conv(x)

    def rrcosfilter(self, N, alpha, Ts, Fs):
        T_delta = 1/float(Fs)
        time_idx = ((np.arange(N)-N//2))*T_delta
        sample_num = np.arange(N)
        h_rrc = np.zeros(N, dtype=float)

        for x in sample_num:
            t = (x-N//2)*T_delta
            if t == 0.0:
                h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
            elif alpha != 0 and t == Ts/(4*alpha):
                h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                        (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
            elif alpha != 0 and t == -Ts/(4*alpha):
                h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                        (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
            else:
                h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Ts) + \
                        4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts))/ \
                        (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)
        return time_idx, h_rrc
