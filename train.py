"""
-------------------------------------------------
   File Name:    train.py
   Author:       Zhonghao Huang
   Date:         2019/10/18
   Description:
-------------------------------------------------
"""

import argparse

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
    dataset = make_dataset(opt)

    # Init the network
    style_gan = StyleGAN(opt, )

    # train the network
    style_gan.train(dataset=dataset,
                    epochs=,
                    fade_in_percentage=,
                    start_depth=,
                    batch_sizes=,
                    num_samples=, )

    print('Done.')
