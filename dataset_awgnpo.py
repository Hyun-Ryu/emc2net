import numpy as np
from glob import glob
from scipy import io
import torch
from torch.utils.data import Dataset


class SignalSet(Dataset):
    def __init__(self, root='', mode='train', n_class=8, n_snr=10):
        self.input_sig_dict = dict()
        self.n_class = n_class
        self.n_snr = n_snr
        self.list_snr = list(28-2*np.arange(self.n_snr))
        self.list_snr.reverse()

        ref_point_1 = int(288*0.8)
        ref_point_2 = int(288*0.9)
        if mode == 'train':
            start, end = 0, ref_point_1
        elif mode == 'valid':
            start, end = ref_point_1, ref_point_2
        else:
            start, end = ref_point_2, 288
        self.n_inst = end - start

        mat_list = glob(root + '/*.mat')
        for mat in mat_list:
            mod_type = mat.split('/')[-1].split('_')[0]
            snr = int(mat.split('/')[-1].split('_')[1][:2])
            if snr in self.list_snr:
                tot_arr = io.loadmat(mat)['dataset']
                data_arr = tot_arr[start:end, :]
                self.input_sig_dict[(mod_type, snr)] = data_arr

    def __getitem__(self, index):
        ind_mod_type = index//(self.n_snr * self.n_inst)
        ind_snr = (index%(self.n_snr * self.n_inst))//self.n_inst
        ind_inst = (index%(self.n_snr * self.n_inst))%self.n_inst

        mod_type = self.num2class()[ind_mod_type]
        snr = self.list_snr[ind_snr]
        input_sig_array = self.input_sig_dict[(mod_type, snr)]
        input_ = input_sig_array[ind_inst, :]
        
        return {'input_i': input_.real, 'input_q': input_.imag, 'modtype': mod_type, 'snr': snr}

    def __len__(self):
        return self.n_class * self.n_snr * self.n_inst

    def class2num(self):
        CLASS2NUM = {
            'BPSK': 0,
            'QPSK': 1,
            '8PSK': 2,
            '16QAM': 3,
            '32QAM': 4,
            '64QAM': 5,
            '128QAM': 6,
            '256QAM': 7,
        }
        return CLASS2NUM

    def num2class(self):
        old_dict = self.class2num()
        NUM2CLASS = dict([(value, key) for key, value in old_dict.items()])
        return NUM2CLASS

