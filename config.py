"""
-------------------------------------------------
   File Name:    config.py
   Author:       Zhonghao Huang
   Date:         2019/10/22
   Description:
-------------------------------------------------
"""

from yacs.config import CfgNode as CN

cfg = CN()

cfg.output_dir = ''
cfg.structure = 'fixed'
cfg.device = 'cuda'
cfg.device_id = '3'

# ---------------------------------------------------------------------------- #
# Options for scheduler
# ---------------------------------------------------------------------------- #
cfg.sched = CN()

# example for {depth:9,resolution:1024}
# res --> [4,8,16,32,64,128,256,512,1024]
cfg.sched.epochs = [2, 2, 2, 4, 8, 16, 32, 64, 64]
cfg.sched.batch_sizes = [512, 256, 256, 128, 64, 32, 8, 4, 2]
cfg.sched.fade_in_percentage = [50, 50, 50, 50, 50, 50, 50, 50, 50]

# cfg.sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
# cfg.sched.D_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.img_dir = ""
cfg.dataset.folder = True
cfg.dataset.resolution = 128
cfg.dataset.channels = 3

cfg.model = CN()
# ---------------------------------------------------------------------------- #
# Options for Generator
# ---------------------------------------------------------------------------- #
cfg.model.gen = CN()
cfg.model.gen.latent_size = 512
cfg.model.gen.mapping_layers = 4

# ---------------------------------------------------------------------------- #
# Options for Discriminator
# ---------------------------------------------------------------------------- #
cfg.model.dis = CN()
cfg.model.dis.use_wscale = True

# ---------------------------------------------------------------------------- #
# Options for Generator Optimizer
# ---------------------------------------------------------------------------- #
cfg.model.g_optim = CN()
cfg.model.g_optim.learning_rate = 0.003
cfg.model.g_optim.beta_1 = 0
cfg.model.g_optim.beta_2 = 0.99
cfg.model.g_optim.eps = 1e-8

# ---------------------------------------------------------------------------- #
# Options for Discriminator Optimizer
# ---------------------------------------------------------------------------- #
cfg.model.d_optim = CN()
cfg.model.d_optim.learning_rate = 0.003
cfg.model.d_optim.beta_1 = 0
cfg.model.d_optim.beta_2 = 0.99
cfg.model.d_optim.eps = 1e-8
