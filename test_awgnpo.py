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
from dataset_awgnpo import *
from util import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=str, default='best', help='choose which trained epoch to use')
parser.add_argument("--n_cpu", type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument("--dim", type=int, default=128, help='hidden dimension of ISAB in classifier')
parser.add_argument("--n_heads", type=int, default=4, help='number of attention heads of ISAB in classifier')
parser.add_argument("--n_anc", type=int, default=64, help='number of inducing points of ISAB in classifier')
parser.add_argument("--n_seeds", type=int, default=1, help='number of seed vectors of PMA in classifier')
parser.add_argument("--n_class", type=int, default=8, help='number of target modulation types')
parser.add_argument("--root", type=str, default="/home/user/amc", help='root directory')
parser.add_argument("--data_name", type=str, default="matlab_awgn_8class_1kpts", help='name of the dataset')
parser.add_argument("--exp_name", type=str, default="noise_curriculum_pretraining", help='name of the experiment')
opt = parser.parse_args()
print(str(opt) + "\n")

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# Load Pre-trained Model
cl = SetTransformer(dim_output=opt.n_class, dim_hidden=opt.dim, num_heads=opt.n_heads, num_inds=opt.n_anc, num_outputs=opt.n_seeds).cuda()
cl.load_state_dict(torch.load(opt.root+"/experiments/"+opt.exp_name+"/saved_models/cl_epoch_%s.pth" % opt.epoch))
print("[Classifier] [# of parameters: %d]" % count_parameters(cl))

# Loss
CE = torch.nn.CrossEntropyLoss().cuda()

# Dataset & Dataloader
dataset = SignalSet(root=opt.root+'/data/'+opt.data_name, mode='test', n_class=opt.n_class)
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

confusion = np.zeros((opt.n_class,opt.n_class))

for t, sigg in enumerate(dataloader):

    # Configure model input & GT
    in_i = sigg["input_i"].unsqueeze(-1).type(Tensor)
    in_q = sigg["input_q"].unsqueeze(-1).type(Tensor)
    input_ = torch.cat([in_i, in_q], dim=-1)
    input_ = Variable(input_)
    mod_ = Variable(torch.Tensor([class2num[jj] for jj in sigg["modtype"]]).type(LongTensor))

    # --------------------
    # Inferenece
    # --------------------

    cl.eval()
    output_ = cl(input_)

    loss_valid = CE(output_, mod_)
    loss_valid_tot += loss_valid.item()

    num_correct = (torch.max(output_, dim=1)[1].data==mod_.data).sum()
    num_correct_tot_valid += num_correct
    num_data_valid += output_.data.shape[0]

    # --------------------
    # Log Progress
    # --------------------

    if t % 1 == 0:
        print(
            "\r[Epoch %s] [MOD: %s, SNR: %ddB] [CE: %.4f, Acc: %.2f%%]"
            % (
                opt.epoch,
                sigg["modtype"][0],
                sigg["snr"][0],
                loss_valid.item(),
                num_correct/output_.data.shape[0] * 100,
            )
        )
    
    # Accuracy for each modtype
    num_correct_mod += num_correct
    num_data_mod += output_.data.shape[0]
    if t % 36 == 35:
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
        confusion[gt[i], pred[i]] += 1

print(confusion)

print(
    "---Total Accuracy---\n\r[Epoch %s] [CE: %.4f, Acc: %.2f%%]"
    % (
        opt.epoch,
        loss_valid_tot/len(dataloader),
        num_correct_tot_valid/num_data_valid * 100,
    )
)
