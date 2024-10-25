#%%

import os, sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
# basepath = '/misc/vlgscratch4/ChoGroup/imj/'

def all_in_one_plot():

    mlr = 0.000010
    lr = 0.001000
    l2 = 0.000000
    num_epoch = 100 
    batch_sz = 11
    batch_vl = 100
    fdir = './mnist/results/exp/mnist/mlr%f_lr%f_l2%f/bptt_%depoch_%dvlbz_sgd_1updatefreq_0resetfreq_fold0/' % (mlr, lr, l2, num_epoch, batch_vl)
    lr_list = np.load(fdir+'lr.npy')[:-1]
    l2_list = np.load(fdir+'l2.npy')[:-1]
    dFlr_list = np.load(fdir+'dFdlr_list.npy')[:-1]
    dFl2_list = np.load(fdir+'dFdl2_list.npy')[:-1]
    te_epoch = np.load(fdir+'te_epoch.npy')
    tr_data_list = np.load(fdir+'tr_loss.npy')
    te_data_list = np.load(fdir+'te_loss.npy')

    updates = np.arange(tr_data_list.shape[0])
    epochs = np.arange(num_epoch) * len(updates) / num_epoch
    te_epoch = te_epoch* len(updates) / num_epoch
    te_data_list = np.mean(te_data_list.reshape([batch_sz, -1]), axis=0)

    color='indianred'
    fig, axs = plt.subplots(2, 2, figsize=(16, 16), sharey=False)
    ax00 = axs[0,0]
    ax00.set_xlabel('Updates')
    ax00.set_ylabel('Loss', color=color)
    ax00.plot(updates, tr_data_list, color=color, label='Train', alpha=0.5)
    print(te_data_list)
    ax00.plot(te_epoch[::100], te_data_list, color=color, ls='--', label='Test')
    ax00.legend()

    color='skyblue'
    ax00a = ax00.twinx()
    ax00a.set_ylabel('Learning Rate', color=color)
    ax00a.plot(epochs, lr_list, color=color, lw=3)
    ax00a.grid(True)

    color='indianred'
    ax01 = axs[0,1]
    ax01.set_xlabel('Updates')
    ax01.set_ylabel('|dPdLr|', color=color)
    ax01.plot(epochs, dFlr_list, color=color, ls='-', label='dFdLr')

    color='skyblue'
    ax01a = ax01.twinx()
    ax01a.set_ylabel('Learning Rate', color=color)
    ax01a.plot(epochs, lr_list, color=color, lw=3)
    ax01a.grid(True)

    color='indianred'
    ax10 = axs[1,0]
    ax10.set_xlabel('Updates')
    ax10.set_ylabel('Loss', color=color)
    ax10.plot(updates, tr_data_list, color=color, label='Train', alpha=0.5)
    ax10.plot(te_epoch[::100], te_data_list, color=color, ls='--', label='Test')
    ax10.legend()

    color='mediumpurple'
    ax10a = ax10.twinx()
    ax10a.set_ylabel('L2 Weight Decay', color=color)
    ax10a.plot(epochs, l2_list, color=color, lw=3)
    ax10a.grid(True)

    color='indianred'
    ax11 = axs[1,1]
    ax11.set_xlabel('Updates')
    ax11.set_ylabel('|dPdL2|', color=color)
    ax11.plot(epochs, dFl2_list, color=color, ls='-', label='dFdL2')
    #ax1.legend()
    #ax1.plot(epochs, dFl2_list, color=color, ls='-', label='dFdL2')
    color='mediumpurple'
    ax11a = ax11.twinx()
    ax11a.set_ylabel('L2 Weight Decay', color=color)
    ax11a.plot(epochs, l2_list, color=color, lw=3)
    ax11a.grid(True)

    plt.tight_layout()
    plt.savefig('../figs/mnist/loss_lr.png', format='png')
    plt.close()


def lineplot(Xs, Ys, colours, labels, xlabel, ylabel, fname, ls='-', lw=3, logyscale=0, logxscale=0, vmax=None):

    fig, axs = plt.subplots(1, 1, figsize=(16, 16), sharey=False)
    ax1 = axs

    for i, (X,Y) in enumerate(zip(Xs,Ys)):
        lw_ = lw[i] if type(lw) == type([]) else lw
        ls_ = ls[i] if type(ls) == type([]) else ls
   
        if labels is not None:
            ax1.plot(X, Y, color=colours[i], label=labels[i], lw=lw_, ls=ls_, alpha=0.5)
        else:
            ax1.plot(X, Y, color=colours[i], lw=lw_, ls=ls_, alpha=0.5)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if labels is not None: ax1.legend()
    if vmax is not None: plt.ylim([0, vmax])

    if logyscale: plt.yscale('log')
    if logxscale: plt.xscale('log')
    plt.tight_layout()
    if 'pdf' in fname:
        plt.savefig('./figs/'+ fname, format='pdf')
    else:
        plt.savefig('./figs/'+ fname, format='png')
    plt.close()


def violinplot(Xticklabels, Y_list, colours, xlabel, ylabel, fname):

    fig, axs = plt.subplots(1, 1, figsize=(16, 16), sharey=False)
    ax1 = axs

    #for i, Ys in enumerate(Y_list):
    ax1.violinplot(Y_list, showmeans=False, showmedians=True)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.setp(axs, xticks=[y + 1 for y in range(len(Y_list))], xticklabels=Xticklabels)
    plt.tight_layout()
    if 'pdf' in fname:
        plt.savefig('./figs/'+ fname, format='pdf')
    else:
        plt.savefig('./figs/'+ fname, format='png')
    plt.close()


def boxfigure(Xticklabels, Y_list, colours, xlabel, ylabel, fname):

    fig, axs = plt.subplots(1, 1, figsize=(16, 16), sharey=False)
    ax1 = axs

    #for i, Ys in enumerate(Y_list):
    ax1.boxplot(Y_list, showmeans=False, showmedians=True)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.setp(axs, xticks=[y + 1 for y in range(len(Y_list))], xticklabels=Xticklabels)
    plt.tight_layout()
    if 'pdf' in fname:
        plt.savefig('./figs/'+ fname, format='pdf')
    else:
        plt.savefig('./figs/'+ fname, format='png')
    plt.close()





if __name__ == '__main__':

    if os.path.isdir("./results/exp/mnist/mlr0.000010_lr0.001000_l20.000000/bptt_100epoch_100vlbz_sgd_1updatefreq_0resetfreq_fold0/lr.npy"):
        print("yes")
    else:
        print(os.getcwd())
    all_in_one_plot()



# %%
