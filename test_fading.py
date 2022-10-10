import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model_classifier import *
from model_equalizer import *
from dataset_fading import *
from util import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument("--root", type=str, default="/home/user/amc", help='root directory')
parser.add_argument("--data_name", type=str, default="Rician_30dB_1024sym", help='name of the dataset')
parser.add_argument("--exp_name", type=str, default="rician_phase3", help='name of the experiment')
opt = parser.parse_args()
print(str(opt) + "\n")

os.makedirs(opt.root + "/experiments/" + opt.exp_name + "/scatter_plot", exist_ok=True)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# Load Models
eq = RBx2(dim_hidden=2, ker_size=65).cuda()
eq.load_state_dict(torch.load(opt.root+'/experiments/'+opt.exp_name+'/saved_models/eq_epoch_best.pth'))
print("[Equalizer] [# of parameters: %d]" % count_parameters(eq))

MF = RRC(N=33, alpha=.35, OS=8)

cl = SetTransformer(dim_output=8, dim_hidden=128, num_heads=4, num_inds=64, num_outputs=1).cuda()
cl.load_state_dict(torch.load(opt.root+'/experiments/%s/saved_models/cl_epoch_best.pth' % opt.exp_name))
print("[Classifier] [# of parameters: %d]" % count_parameters(cl))

# Loss
CE = torch.nn.CrossEntropyLoss().cuda()

# Dataset & Dataloader
dataset = SignalSet(root=opt.root+'/data/'+opt.data_name, mode='test')
dataloader = DataLoader(
    dataset,
    batch_size = 20,
    shuffle = False,
    num_workers = opt.n_cpu,
)

class2num = dataset.class2num()

# Test
loss_valid_tot = 0
num_correct_tot_valid, num_data_valid = 0, 0
num_correct_mod, num_data_mod = 0, 0

confusion = np.zeros((8,8))

for t, sigg in enumerate(dataloader):

    # Configure model input & GT
    input_ = Variable(sigg["input"].unsqueeze(2).type(Tensor))
    mod_ = Variable(torch.Tensor([class2num[jj] for jj in sigg["modtype"]]).type(LongTensor))

    # --------------------
    # Inferenece
    # --------------------

    eq.eval()
    cl.eval()

    # Equalizer
    inter_ = eq(input_)     # input_: (b, 2, 1, 8192)
        
    # Zero-mean equalizer output
    inter_ = inter_ - inter_.mean(dim=-1).unsqueeze(-1)

    # MF
    inter_real = MF(inter_[:,0,:,:].unsqueeze(1)).squeeze().squeeze()   # (b, 1024)
    inter_imag = MF(inter_[:,1,:,:].unsqueeze(1)).squeeze().squeeze()

    # Unit-Power Normalization
    avgpow = (inter_real.pow(2)+inter_imag.pow(2)).mean(dim=1).sqrt().unsqueeze(1)
    inter_real = torch.div(inter_real, avgpow)
    inter_imag = torch.div(inter_imag, avgpow)
    inter2_ = torch.cat((inter_real.unsqueeze(-1), inter_imag.unsqueeze(-1)), dim=-1)

    # Classifier
    output_ = cl(inter2_)   # inter2_: (b, 1024, 2)

    loss_valid = CE(output_, mod_)
    loss_valid_tot += loss_valid.item()

    num_correct = (torch.max(output_, dim=1)[1].data==mod_.data).sum()
    num_correct_tot_valid += num_correct
    num_data_valid += output_.data.shape[0]
 
    # --------------------
    # Log Progress
    # --------------------

    # Accuracy for each modtype
    num_correct_mod += num_correct
    num_data_mod += output_.data.shape[0]
    if t % 20 == 19:
        print(
            "\r[MOD: %s] [Acc: %.2f%%]"
            % (
                sigg["modtype"][0],
                num_correct_mod/num_data_mod * 100,
            )
        )
        num_correct_mod, num_data_mod = 0, 0
    
    # Confusion matrix
    gt = mod_.data.cpu().numpy()
    pred = torch.max(output_, dim=1)[1].data.cpu().numpy()
    for i in range(len(gt)):
        confusion[gt[i]][pred[i]] += 1
    
print(confusion)

print(
    "---Total Accuracy---\n\r[CE: %.4f, Acc: %.2f%%]"
    % (
        loss_valid_tot/len(dataloader),
        num_correct_tot_valid/num_data_valid * 100,
    )
)

