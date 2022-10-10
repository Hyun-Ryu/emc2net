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
from dataset_awgnpo import *
from util import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help='number of training epochs')
parser.add_argument("--switch_epochs", type=list, default=[0,10,20,30,40,50,60,70,80,90], help='list of epochs where noise curriculum change')
parser.add_argument("--batch_size", type=int, default=64, help='size of the batches')
parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
parser.add_argument("--n_cpu", type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument("--dim", type=int, default=128, help='hidden dimension of ISAB in classifier')
parser.add_argument("--n_heads", type=int, default=4, help='number of attention heads of ISAB in classifier')
parser.add_argument("--n_anc", type=int, default=64, help='number of inducing points of ISAB in classifier')
parser.add_argument("--n_seeds", type=int, default=1, help='number of seed vectors of PMA in classifier')
parser.add_argument("--n_class", type=int, default=8, help='number of target modulation types')
parser.add_argument("--n_snr", type=int, default=10, help='total number of steps of noise curriculum')
parser.add_argument("--root", type=str, default='/home/user/amc', help='root directory')
parser.add_argument("--data_name", type=str, default='matlab_awgn_8class_1kpts', help='name of the dataset')
parser.add_argument("--exp_name", type=str, default='noise_curriculum_pretraining', help='name of the experiment')
opt = parser.parse_args()
print(str(opt) + "\n")

os.makedirs(opt.root + "/experiments/" + opt.exp_name + "/saved_models", exist_ok=True)
os.makedirs(opt.root + "/experiments/" + opt.exp_name + "/loss_curve", exist_ok=True)
os.makedirs(opt.root + "/experiments/" + opt.exp_name + "/acc_curve", exist_ok=True)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# Load & Initialize Model
cl = SetTransformer(dim_output=opt.n_class, dim_hidden=opt.dim, num_heads=opt.n_heads, num_inds=opt.n_anc, num_outputs=opt.n_seeds).cuda()
print("[Classifier] [# of parameters: %d]" % count_parameters(cl))

# Loss
CE = torch.nn.CrossEntropyLoss().cuda()

# Optimizers
optimizer = torch.optim.Adam(cl.parameters(), lr=opt.lr)

# Dataset & Dataloader
dataset_valid = SignalSet(root=opt.root+'/data/'+opt.data_name, mode='valid', n_class=opt.n_class, n_snr=opt.n_snr)
dataloader_valid = DataLoader(
    dataset_valid,
    batch_size = 36,
    shuffle = False,
    num_workers = opt.n_cpu,
)

loss_epoch_list, loss_epoch_list_val = [], []
acc_epoch_list, acc_epoch_list_val = [], []
acc_top1 = 0
prev_time = time.time()
class2num = dataset_valid.class2num()
curriculum = 0

for epoch in range(0, opt.n_epochs):

    # Curriculum
    if epoch == opt.switch_epochs[curriculum]:
        print("----[Curriculum #%s] SNR=%s:2:28dB----" % (curriculum, 28-2*curriculum))

        dataset = SignalSet(root=opt.root+'/data/'+opt.data_name, mode='train', n_class=opt.n_class, n_snr=curriculum+1)
        dataloader = DataLoader(
            dataset,
            batch_size = opt.batch_size,
            shuffle = True,
            num_workers = opt.n_cpu,
        )

        if not curriculum == len(opt.switch_epochs)-1:
            curriculum += 1

    # Train
    loss_tot = 0
    num_correct_tot, num_data = 0, 0

    for i, sig in enumerate(dataloader):

        # Configure model input & GT
        in_i = sig["input_i"].unsqueeze(-1).type(Tensor)
        in_q = sig["input_q"].unsqueeze(-1).type(Tensor)
        input_ = torch.cat([in_i, in_q], dim=-1)
        input_ = Variable(input_)
        mod_ = Variable(torch.Tensor([class2num[jj] for jj in sig["modtype"]]).type(LongTensor))

        # --------------------
        # Train Model
        # --------------------

        cl.train()
        optimizer.zero_grad()

        output_ = cl(input_)

        loss = CE(output_, mod_)
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
                "\r[Epoch %d/%d, Batch %d/%d] [CE: %.4f, Acc: %.2f%%] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss.item(),
                    num_correct/output_.data.shape[0] * 100,
                    time_left,
                )
            )
        
    loss_epoch_list.append(loss_tot/len(dataloader))
    acc_epoch_list.append(num_correct_tot/num_data * 100)

    # Validation
    if num_correct_tot/num_data * 100 > 50:

        loss_valid_tot = 0
        num_correct_tot_valid, num_data_valid = 0, 0

        for t, sigg in enumerate(dataloader_valid):

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

            if t % 5 == 0:
                print(
                    "\r[Epoch %d/%d] [MOD: %s, SNR: %ddB] [CE: %.4f, Acc: %.2f%%]"
                    % (
                        epoch,
                        opt.n_epochs,
                        sigg["modtype"][0],
                        sigg["snr"][0],
                        loss_valid.item(),
                        num_correct/output_.data.shape[0] * 100,
                    )
                )

        loss_epoch_list_val.append(loss_valid_tot/len(dataloader_valid))
        acc_epoch_list_val.append(num_correct_tot_valid/num_data_valid * 100)

        if (num_correct_tot_valid/num_data_valid * 100) > acc_top1:
            acc_top1 = num_correct_tot_valid/num_data_valid * 100
            loss_top1 = loss_valid_tot/len(dataloader_valid)
            epoch_top1 = epoch
            torch.save(cl.state_dict(), opt.root+'/experiments/'+opt.exp_name+'/saved_models/cl_epoch_best.pth')
        
        print(
            "---Summary of Epoch %d/%d---\n\r[Train] [CE: %.4f, Acc: %.2f%%]\n\r[Valid] [CE: %.4f, Acc: %.2f%%]"
            % (
                epoch,
                opt.n_epochs,
                loss_tot/len(dataloader),
                num_correct_tot/num_data * 100,
                loss_valid_tot/len(dataloader_valid),
                num_correct_tot_valid/num_data_valid * 100,
            )
        )

        print(
            "---Summary of TOP1---\n\r[Valid] [Epoch: %d/%d, CE: %.4f, Acc: %.2f%%]"
            % (
                epoch_top1,
                opt.n_epochs,
                loss_top1,
                acc_top1,
            )
        )
    else:
        loss_epoch_list_val.append(0)
        acc_epoch_list_val.append(0)

    if epoch % 10 == 0:
       # save model checkpoint
        torch.save(cl.state_dict(), opt.root+'/experiments/'+opt.exp_name+'/saved_models/cl_epoch_%d.pth' % epoch)

        # plot loss curves
        draw_loss_epoch_curve(opt.root+'/experiments/'+opt.exp_name, epoch, loss_epoch_list, loss_epoch_list_val, 'loss_TrainValid_epoch')

        # plot accuracy curves
        draw_acc_epoch_curve(opt.root+'/experiments/'+opt.exp_name, epoch, acc_epoch_list, acc_epoch_list_val, 'acc_TrainValid_epoch')
        
