import matplotlib.pyplot as plt
import numpy as np 
import torch, os

def scatter_plot(x, y, xlabel='Epochs', ylabel='Average Reward', filename='reward', 
            save_dir=''):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.scatter(x, y)
    plt.savefig(os.path.join(save_dir, filename + '.png'))
    plt.close(fig)

def errorbar_plot(x, y, xlabel='Epochs', ylabel='Average Reward', filename='reward', error=None, 
            save_dir=''):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.errorbar(x, y, yerr=error, fmt='-o')
    plt.savefig(os.path.join(save_dir, filename + '.png'))
    plt.close(fig)

def save_noise(u, save_dir, epoch):
    """Plots an histogram of MAF noise.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    u = torch.cat((u, torch.normal(torch.zeros(u.size(0))).reshape(-1,1)),1)
    ax.hist(u.detach().numpy(),range=(-4,4))
    plt.savefig(os.path.join(save_dir,str(epoch)+'_noise.png'))
    plt.close(fig)