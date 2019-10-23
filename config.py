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

cfg.output_dir = ""
cfg.depth = 7

# ---------------------------------------------------------------------------- #
# Options for scheduler
# ---------------------------------------------------------------------------- #
cfg.sched = CN()
cfg.sched.epochs = [27, 54, 54, 54, 54, 54, 54]
cfg.sched.batch_sizes = [64, 64, 64, 64, 32, 32, 16]
cfg.sched.fade_in_percentage = [50, 50, 50, 50, 50, 50, 50]

# cfg.sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
# cfg.sched.D_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.img_dir = ""
cfg.dataset.folder = True

cfg.model = CN()
# ---------------------------------------------------------------------------- #
# Options for Generator
# ---------------------------------------------------------------------------- #
cfg.model.gen = CN()
cfg.model.gen.latent_size = 512

# ---------------------------------------------------------------------------- #
# Options for Discriminator
# ---------------------------------------------------------------------------- #
cfg.model.dis = CN()
cfg.model.dis.use_wscale = True
# ---------------------------------------------------------------------------- #
# Options for Generator Optimizer
# ---------------------------------------------------------------------------- #
cfg.model.g_optim = CN()
cfg.model.g_optim.learning_rate = 0.001
cfg.model.g_optim.beta_1 = 0
cfg.model.g_optim.beta_2 = 0.99
cfg.model.g_optim.eps = 1e-8

# ---------------------------------------------------------------------------- #
# Options for Discriminator Optimizer
# ---------------------------------------------------------------------------- #
cfg.model.d_optim = CN()
cfg.model.d_optim.learning_rate = 0.001
cfg.model.d_optim.beta_1 = 0
cfg.model.d_optim.beta_2 = 0.99
cfg.model.d_optim.eps = 1e-8
