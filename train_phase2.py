import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import os
import time
import argparse
import datetime
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
parser.add_argument("--n_epochs", type=int, default=500, help='number of training epochs')
parser.add_argument("--batch_size", type=int, default=64, help='size of the batches')
parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
parser.add_argument("--n_cpu", type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument("--root", type=str, default="/home/user/amc", help='root directory')
parser.add_argument("--data_name", type=str, default="Rician_30dB_1024sym", help='name of the dataset')
parser.add_argument("--exp_name", type=str, default="rician_phase2", help='name of the experiment')
parser.add_argument("--pretrain_exp_name", type=str, default="noise_curriculum_pretraining", help='name of the experiment of pretrained classifier')
opt = parser.parse_args()
print(str(opt) + "\n")

os.makedirs(opt.root + "/experiments/" + opt.exp_name + "/saved_models", exist_ok=True)
os.makedirs(opt.root + "/experiments/" + opt.exp_name + "/loss_curve", exist_ok=True)
os.makedirs(opt.root + "/experiments/" + opt.exp_name + "/acc_curve", exist_ok=True)
os.makedirs(opt.root + "/experiments/" + opt.exp_name + "/scatter_plot", exist_ok=True)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# Pre-calculated R_2 value
MOD2R2 = {
    'BPSK': 1.0000,
    'QPSK': 1.0000,
    '8PSK': 1.0000,
    '16QAM': 1.3200,
    '32QAM': 1.3100,
    '64QAM': 1.3810,
    '128QAM': 1.3427,
    '256QAM': 1.3953,
}

# Load Models
eq = RBx2(dim_hidden=2, ker_size=65).cuda()

MF = RRC(N=33, alpha=.35, OS=8)
for para in MF.parameters():
    para.requires_grad = False

cl = SetTransformer(dim_output=8, dim_hidden=256, num_heads=4, num_inds=16, num_outputs=1).cuda()
cl.load_state_dict(torch.load(opt.root+'/experiments/%s/saved_models/cl_epoch_best.pth' % opt.pretrain_exp_name))
for para in cl.parameters():
    para.requires_grad = False

# Loss
MSE = torch.nn.MSELoss().cuda()
CE = torch.nn.CrossEntropyLoss().cuda()

# Optimizers
optimizer = torch.optim.Adam(eq.parameters(), lr=opt.lr)

# Dataset & Dataloader
dataset = SignalSet(root=opt.root+'/data/'+opt.data_name, mode='train')
dataloader = DataLoader(
    dataset,
    batch_size = opt.batch_size,
    shuffle = True,
    num_workers = opt.n_cpu,
)

dataset_valid = SignalSet(root=opt.root+'/data/'+opt.data_name, mode='valid')
dataloader_valid = DataLoader(
    dataset_valid,
    batch_size = 80,
    shuffle = False,
    num_workers = opt.n_cpu,
)

loss_epoch_list, loss_epoch_list_val = [], []
acc_epoch_list, acc_epoch_list_val = [], []
acc_top1 = 0
prev_time = time.time()
class2num = dataset.class2num()

for epoch in range(0, opt.n_epochs):

    # Train
    loss_tot = 0
    num_correct_tot, num_data = 0, 0

    for i, sig in enumerate(dataloader):

        # Configure model input & GT
        input_ = Variable(sig["input"].unsqueeze(2).type(Tensor))
        mod_ = Variable(torch.Tensor([class2num[jj] for jj in sig["modtype"]]).type(LongTensor))
        r2_ = Variable(torch.Tensor([MOD2R2[jj] for jj in sig["modtype"]]).type(Tensor))

        # --------------------
        # Train Model
        # --------------------

        eq.train()
        optimizer.zero_grad()

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

        # CMA Loss
        estr2_ = inter_real.pow(2)+inter_imag.pow(2)    # (b, 1024)
        gtr2_ = r2_.unsqueeze(1).repeat(1, estr2_.data.shape[-1])
        loss_mse = MSE(estr2_, gtr2_)

        # Classification Loss
        output_ = cl(inter2_)   # inter2_: (b, 1024, 2)
        loss_ce = CE(output_, mod_)

        # Total Loss
        loss = 0.01 * loss_mse + loss_ce
        loss_tot += loss.item()

        num_correct = (torch.max(output_, dim=1)[1].data==mod_.data).sum()
        num_correct_tot += num_correct
        num_data += output_.data.shape[0]

        # Backprop
        loss.backward()
        optimizer.step()
 
        # --------------------
        # Log Progress
        # --------------------

        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if i % 10 == 0:
            print(
                "\r[Epoch %d/%d, Batch %d/%d] [Tot: %.4f, MSE: %.4f, CE: %.4f] [Acc: %.2f%%] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss.item(),
                    loss_mse.item(),
                    loss_ce.item(),
                    num_correct/output_.data.shape[0] * 100,
                    time_left,
                )
            )
        
    loss_epoch_list.append(loss_tot/len(dataloader))
    acc_epoch_list.append(num_correct_tot/num_data * 100)

    # Validation
    loss_valid_tot = 0
    num_correct_tot_valid, num_data_valid = 0, 0

    for t, sigg in enumerate(dataloader_valid):

        # Configure model input & GT
        input_ = Variable(sigg["input"].unsqueeze(2).type(Tensor))
        mod_ = Variable(torch.Tensor([class2num[jj] for jj in sigg["modtype"]]).type(LongTensor))
        r2_ = Variable(torch.Tensor([MOD2R2[jj] for jj in sigg["modtype"]]).type(Tensor))

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

        # CMA Loss
        estr2_ = inter_real.pow(2)+inter_imag.pow(2)    # (b, 1024)
        gtr2_ = r2_.unsqueeze(1).repeat(1, estr2_.data.shape[-1])
        loss_valid_mse = MSE(estr2_, gtr2_)

        # Classification Loss
        output_ = cl(inter2_)   # inter2_: (b, 1024, 2)
        loss_valid_ce = CE(output_, mod_)

        # Total Loss
        loss_valid = 0.01 * loss_valid_mse + loss_valid_ce
        loss_valid_tot += loss_valid.item()

        num_correct_valid = (torch.max(output_, dim=1)[1].data==mod_.data).sum()
        num_correct_tot_valid += num_correct_valid
        num_data_valid += output_.data.shape[0]
        
        if epoch % 10 == 0 and t % 5 == 0:
            # Visualization
            scatter_plot_channelInverse(opt.root+'/experiments/'+opt.exp_name, MF, input_, inter_, 'epoch_%d_batch' % epoch, t)
 
        # --------------------
        # Log Progress
        # --------------------

        if t % 5 == 0:
            print(
                    "\r[Epoch %d/%d] [MOD: %s] [Tot: %.4f, MSE: %.4f, CE: %.4f] [Acc: %.2f%%]"
                % (
                    epoch,
                    opt.n_epochs,
                    sigg["modtype"][0],
                    loss_valid.item(),
                    loss_valid_mse.item(),
                    loss_valid_ce.item(),
                    num_correct_valid/output_.data.shape[0] * 100,
                )
            )
    
    print(
        "---Summary of Epoch %d/%d---\n\r[Train] [Tot: %.4f, Acc: %.2f%%]\n\r[Valid] [Tot: %.4f, Acc: %.2f%%]"
        % (
            epoch,
            opt.n_epochs,
            loss_tot/len(dataloader),
            num_correct_tot/num_data * 100,
            loss_valid_tot/len(dataloader_valid),
            num_correct_tot_valid/num_data_valid * 100,
        )
    )
    
    loss_epoch_list_val.append(loss_valid_tot/len(dataloader_valid))
    acc_epoch_list_val.append(num_correct_tot_valid/num_data_valid * 100)

    if (num_correct_tot_valid/num_data_valid * 100) > acc_top1:
        acc_top1 = num_correct_tot_valid/num_data_valid * 100
        loss_top1 = loss_valid_tot/len(dataloader_valid)
        epoch_top1 = epoch
        torch.save(eq.state_dict(), opt.root+'/experiments/'+opt.exp_name+'/saved_models/eq_epoch_best.pth')

    if epoch % 10 == 0:
        # save model checkpoint       
        torch.save(eq.state_dict(), opt.root+'/experiments/'+opt.exp_name+'/saved_models/eq_epoch_%d.pth' % epoch)

        # plot loss curves
        draw_loss_epoch_curve(opt.root+'/experiments/'+opt.exp_name, epoch, loss_epoch_list, loss_epoch_list_val, 'loss_TrainValid_epoch')

        # plot accuracy curves
        draw_acc_epoch_curve(opt.root+'/experiments/'+opt.exp_name, epoch, acc_epoch_list, acc_epoch_list_val, 'acc_TrainValid_epoch')
        
    print(
        "---Summary of TOP1---\n\r[Valid] [Epoch: %d/%d, Tot: %.4f, Acc: %.2f%%]"
        % (
            epoch_top1,
            opt.n_epochs,
            loss_top1,
            acc_top1,
        )
    )
