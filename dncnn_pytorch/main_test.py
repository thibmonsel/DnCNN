# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# run this to test the model

import argparse
import os, time
import numpy as np

import torch
from utils import show, save_result, log
from dncnn import DnCNN
from skimage import measure
# from skimage.measure import peak_signal_noise_ratio, structural_similarity # compare_psnr, compare_ssim,
from skimage.io import imread, imsave


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/Test', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set12'], help='directory of test dataset') #['Set68', 'Set12']
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('models', 'DnCNN_sigma25'), help='directory of the model')
    parser.add_argument('--model_name', default='model_001.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DnCNN()
    model = torch.load(os.path.join(args.model_dir, 'model.pth'), map_location=torch.device('cpu') )
    model = model.to(device)
    model.eval()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:
        psnrs,ssims = [], []
        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))
       

        for im in os.listdir(os.path.join(args.set_dir, set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):

                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32)/255.0
                np.random.seed(seed=0) 
                y = x + np.random.normal(0, args.sigma/255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

                start_time = time.time()
                y_ = y_.to(device)
                x_ = model(y_)  
                x_ = x_.view(y.shape[0], y.shape[1])
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))

                # psnr_x_ = measure.peak_signal_noise_ratio(x, x_)
                # ssim_x_ = measure.structural_similarity(x, x_)
                # psnrs.append(psnr_x_)
                # ssims.append(ssim_x_)
                
                #saving denoised image
                if args.save_result:
                    name, ext = os.path.splitext(im)
                    # show(np.hstack((y, x_)))  
                    cleaned_img = y - x_
                    save_result(cleaned_img, path=os.path.join(args.result_dir, set_cur, name+'_denoised_dncnn'+ext))  
                    save_result(x_, path=os.path.join(args.result_dir, set_cur, name+'_dncnn'+ext))  
                    
                
                
        # psnr_avg = np.mean(psnrs)
        # ssim_avg = np.mean(ssims)
        # psnrs.append(psnr_avg)
        # ssims.append(ssim_avg)
        # if args.save_result:
        #     save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_cur, 'results.txt'))
        # log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))








