"""
-------------------------------------------------
   File Name:    train.py
   Author:       Zhonghao Huang
   Date:         2019/10/18
   Description:
-------------------------------------------------
"""

import argparse

import torch

from data import make_dataset
from models.GAN import StyleGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StyleGAN pytorch re-implement.")
    parser.add_argument('--config', default='./configs/sample.yaml')
    args = parser.parse_args()

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    # create the dataset for training
    dataset = make_dataset(opt.dataset)

    # Init the network
    style_gan = StyleGAN(structure=opt.structure,
                         resolution=opt.dataset.resolution,
                         num_channels=opt.dataset.channels,
                         g_args=opt.model.gen,
                         d_args=opt.model.dis,
                         g_opt_args=opt.model.g_optim,
                         d_opt_args=opt.model.d_optim,
                         device=torch.device('cuda:3'))

    # train the network
    style_gan.train(dataset=dataset,
                    epochs=opt.sched.epochs,
                    batch_sizes=opt.sched.batch_sizes,
                    fade_in_percentage=opt.sched.fade_in_percentage)

    print('Done.')
