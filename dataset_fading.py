import numpy as np
from scipy import io
from glob import glob
import torch
from torch.utils.data import Dataset


class SignalSet(Dataset):
    def __init__(self, root='/', mode='train'):
        self.sig_dict = dict({'BPSK':[], 'QPSK':[], '8PSK':[], '16QAM':[], '32QAM':[], '64QAM':[], '128QAM':[], '256QAM':[]})
        
        # load .mat, construct sig_dict
        mat_list = glob(root + '/*.mat')
        for mat in mat_list:
            mod_type = mat.split('/')[-1].split('_')[1]
            input_ = io.loadmat(mat)['frame_input'][:,0]
            self.sig_dict[mod_type].append(input_)

        # devide into train/validation/test set
        ref_point_1 = int(len(self.sig_dict[list(self.sig_dict.keys())[0]]) * 0.8)
        ref_point_2 = int(len(self.sig_dict[list(self.sig_dict.keys())[0]]) * 0.9)
        for key in self.sig_dict.keys():
            if mode == 'train':
                self.sig_dict[key] = self.sig_dict[key][:ref_point_1]
            elif mode == 'valid':
                self.sig_dict[key] = self.sig_dict[key][ref_point_1:ref_point_2]
            else:
                self.sig_dict[key] = self.sig_dict[key][ref_point_2:]

    def __getitem__(self, index):
        num_per_class = len(self.sig_dict[list(self.sig_dict.keys())[0]])
        ind_mod_type = index // num_per_class
        ind_instance = index % num_per_class

        mod_type = self.num2class()[ind_mod_type]
        input_ = self.sig_dict[mod_type][ind_instance]
        input_ = np.array([input_.real, input_.imag])
        return {'input': input_, 'modtype': mod_type}

    def __len__(self):
        return sum([len(self.sig_dict[key]) for key in self.sig_dict.keys()])

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
