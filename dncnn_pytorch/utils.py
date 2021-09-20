import re
import os
import glob
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
from torch.nn.modules.loss import _Loss

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2) equivalent to MSE.loss() / 2 
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def divergence_loss(output, output_gauss, n):
    div = torch.sum(torch.matmul(n, output_gauss - output))
    return div

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1),cmap="Greys_r")


def show(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()
    
    
def create_noisy_tensor_minibatch(data):
    nb_elmts, img_size = data.shape[0], data.shape[1:]
    n = torch.unsqueeze(torch.normal(0., 1, size = tuple(img_size)),dim=0)
    for i in range(nb_elmts-1):
        n = torch.cat((n, torch.unsqueeze(torch.normal(0., 1, size = tuple(img_size)),dim=0)), dim=0)
    return n
    
    