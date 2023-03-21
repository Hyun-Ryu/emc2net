import torch
import numpy as np
import matplotlib.pyplot as plt

def scatter_plot_channelInverse(root, MF, input_, output_, name, ind):
    # input tensors: (b, 2, 1, seqlen)
    in_real = MF(input_[0,0,:,:].unsqueeze(0).unsqueeze(0)).squeeze().squeeze().squeeze()
    in_imag = MF(input_[0,1,:,:].unsqueeze(0).unsqueeze(0)).squeeze().squeeze().squeeze()
    out_real = MF(output_[0,0,:,:].unsqueeze(0).unsqueeze(0)).squeeze().squeeze().squeeze()
    out_imag = MF(output_[0,1,:,:].unsqueeze(0).unsqueeze(0)).squeeze().squeeze().squeeze()

    # Unit-Power Normalization
    avgpow = (in_real.pow(2)+in_imag.pow(2)).mean(dim=0).sqrt()
    in_real, in_imag = torch.div(in_real, avgpow), torch.div(in_imag, avgpow)
    avgpoww = (out_real.pow(2)+out_imag.pow(2)).mean(dim=0).sqrt()
    out_real, out_imag = torch.div(out_real, avgpoww), torch.div(out_imag, avgpoww)
    in_real, in_imag = in_real.cpu().data.numpy(), in_imag.cpu().data.numpy()
    out_real, out_imag = out_real.cpu().data.numpy(), out_imag.cpu().data.numpy()

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(in_real, in_imag, '.', color='blue')
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(root + '/scatter_plot/input_con_%s_%d.png' % (name, ind))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(out_real, out_imag, '.', color='blue')
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(root + '/scatter_plot/output_con_%s_%d.png' % (name, ind))

def draw_loss_epoch_curve(root, epoch, loss, loss_val, name):
    assert len(loss) == len(loss_val)
    x = [i for i in range(1, len(loss)+1)]
    plt.clf()
    plt.plot(x, loss, lw=0.75, color='blue', label='Train')
    plt.plot(x, loss_val, lw=0.75, color='red', label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left', prop={'size': 6})
    plt.savefig(root + '/loss_curve/%s_%d.png' % (name, epoch))
    
def draw_acc_epoch_curve(root, epoch, acc, acc_val, name):
    assert len(acc) == len(acc_val)
    x = [i for i in range(1, len(acc)+1)]
    plt.clf()
    plt.plot(x, acc, lw=0.75, color='blue', label='Train')
    plt.plot(x, acc_val, lw=0.75, color='red', label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left', prop={'size': 6})
    plt.savefig(root + '/acc_curve/%s_%d.png' % (name, epoch))
