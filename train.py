"""
-------------------------------------------------
   File Name:    train.py
   Author:       Zhonghao Huang
   Date:         2019/10/18
   Description:
-------------------------------------------------
"""

import argparse
import os
import shutil

import torch
from torch.backends import cudnn

from data import make_dataset
from models.GAN import StyleGAN
from utils import (copy_files_and_create_dirs,
                   list_dir_recursively_with_ignore, make_logger)


# Load fewer layers of pre-trained models if possible
def load(model, cpk_file):
    pretrained_dict = torch.load(cpk_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StyleGAN pytorch implementation.")
    parser.add_argument('--config', default='./configs/sample.yaml')
    
    parser.add_argument("--start_depth", action="store", type=int, default=0,
                        help="Starting depth for training the network")
    
    parser.add_argument("--generator_file", action="store", type=str, default=None,
                        help="pretrained Generator file (compatible with my code)")
    parser.add_argument("--gen_shadow_file", action="store", type=str, default=None,
                        help="pretrained gen_shadow file")
    parser.add_argument("--discriminator_file", action="store", type=str, default=None,
                        help="pretrained Discriminator file (compatible with my code)")
    parser.add_argument("--gen_optim_file", action="store", type=str, default=None,
                        help="saved state of generator optimizer")
    parser.add_argument("--dis_optim_file", action="store", type=str, default=None,
                        help="saved_state of discriminator optimizer")
    args = parser.parse_args()

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    # make output dir
    output_dir = opt.output_dir
    if os.path.exists(output_dir):
        raise KeyError("Existing path: ", output_dir)
    os.makedirs(output_dir)

    # copy codes and config file
    files = list_dir_recursively_with_ignore('.', ignores=['diagrams', 'configs'])
    files = [(f[0], os.path.join(output_dir, "src", f[1])) for f in files]
    copy_files_and_create_dirs(files)
    shutil.copy2(args.config, output_dir)

    # logger
    logger = make_logger("project", opt.output_dir, 'log')

    # device
    if opt.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_id
        num_gpus = len(opt.device_id.split(','))
        logger.info("Using {} GPUs.".format(num_gpus))
        logger.info("Training on {}.\n".format(torch.cuda.get_device_name(0)))
        cudnn.benchmark = True
    device = torch.device(opt.device)

    # create the dataset for training
    dataset = make_dataset(opt.dataset, conditional=opt.conditional)

    # init the network
    style_gan = StyleGAN(structure=opt.structure,
                         conditional=opt.conditional,
                         n_classes=opt.n_classes,
                         resolution=opt.dataset.resolution,
                         num_channels=opt.dataset.channels,
                         latent_size=opt.model.gen.latent_size,
                         g_args=opt.model.gen,
                         d_args=opt.model.dis,
                         g_opt_args=opt.model.g_optim,
                         d_opt_args=opt.model.d_optim,
                         loss=opt.loss,
                         drift=opt.drift,
                         d_repeats=opt.d_repeats,
                         use_ema=opt.use_ema,
                         ema_decay=opt.ema_decay,
                         device=device)

    # Resume training from checkpoints
    if args.generator_file is not None:
        logger.info("Loading generator from: %s", args.generator_file)
        # style_gan.gen.load_state_dict(torch.load(args.generator_file))
        # Load fewer layers of pre-trained models if possible
        load(style_gan.gen, args.generator_file)
    else:
        logger.info("Training from scratch...")

    if args.discriminator_file is not None:
        logger.info("Loading discriminator from: %s", args.discriminator_file)
        style_gan.dis.load_state_dict(torch.load(args.discriminator_file))

    if args.gen_shadow_file is not None and opt.use_ema:
        logger.info("Loading shadow generator from: %s", args.gen_shadow_file)
        # style_gan.gen_shadow.load_state_dict(torch.load(args.gen_shadow_file))
        # Load fewer layers of pre-trained models if possible
        load(style_gan.gen_shadow, args.gen_shadow_file)

    if args.gen_optim_file is not None:
        logger.info("Loading generator optimizer from: %s", args.gen_optim_file)
        style_gan.gen_optim.load_state_dict(torch.load(args.gen_optim_file))

    if args.dis_optim_file is not None:
        logger.info("Loading discriminator optimizer from: %s", args.dis_optim_file)
        style_gan.dis_optim.load_state_dict(torch.load(args.dis_optim_file))

    # train the network
    style_gan.train(dataset=dataset,
                  num_workers=opt.num_works,
                  epochs=opt.sched.epochs,
                  batch_sizes=opt.sched.batch_sizes,
                  fade_in_percentage=opt.sched.fade_in_percentage,
                  logger=logger,
                  output=output_dir,
                  num_samples=opt.num_samples,
                  start_depth=args.start_depth,
                  feedback_factor=opt.feedback_factor,
                  checkpoint_factor=opt.checkpoint_factor)
