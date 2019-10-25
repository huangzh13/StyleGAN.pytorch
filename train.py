"""
-------------------------------------------------
   File Name:    train.py
   Author:       Zhonghao Huang
   Date:         2019/10/18
   Description:
-------------------------------------------------
"""

import os
import argparse

import torch
from torch.backends import cudnn

from data import make_dataset
from logger import make_logger
from models.GAN import StyleGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StyleGAN pytorch re-implement.")
    parser.add_argument('--config', default='./configs/sample.yaml')
    args = parser.parse_args()

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    # set random seed
    # TODO

    # make output dir
    # if os.path.exists(output_dir):
    #     raise KeyError("Existing path: ", output_dir)
    # os.makedirs(output_dir)
    os.makedirs(opt.output_dir, exist_ok=True)

    # logger
    logger = make_logger("project", opt.output_dir, 'log')
    # logger.info('Random seed is {}'.format(SEED))

    # device
    if opt.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_id
        num_gpus = len(opt.device_id.split(','))
        logger.info("Using {} GPUs.\n".format(num_gpus))
    cudnn.benchmark = True
    device = torch.device(opt.device)

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
                         device=device)

    # train the network
    style_gan.train(dataset=dataset,
                    epochs=opt.sched.epochs,
                    batch_sizes=opt.sched.batch_sizes,
                    fade_in_percentage=opt.sched.fade_in_percentage,
                    logger=logger,
                    output=opt.output_dir)

    print('Done.')
