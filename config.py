"""
-------------------------------------------------
   File Name:    config.py
   Author:       Zhonghao Huang
   Date:         2019/10/22
   Description:  Global Configure.
-------------------------------------------------
"""

from yacs.config import CfgNode as CN

cfg = CN()

cfg.output_dir = ''
cfg.device = 'cuda'
cfg.device_id = '3'

cfg.structure = 'fixed'
cfg.loss = "logistic"
cfg.drift = 0.001
cfg.d_repeats = 1
cfg.use_ema = True
cfg.ema_decay = 0.999

cfg.num_works = 4
cfg.num_samples = 36
cfg.feedback_factor = 10
cfg.checkpoint_factor = 10

# ---------------------------------------------------------------------------- #
# Options for scheduler
# ---------------------------------------------------------------------------- #
cfg.sched = CN()

# example for {depth:9,resolution:1024}
# res --> [4,8,16,32,64,128,256,512,1024]
cfg.sched.epochs = [4, 4, 4, 4, 8, 16, 32, 64, 64]
# batches for oen 1080Ti with 11G memory
cfg.sched.batch_sizes = [128, 128, 128, 64, 32, 16, 8, 4, 2]
cfg.sched.fade_in_percentage = [50, 50, 50, 50, 50, 50, 50, 50, 50]

# TODO
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
# 8 in original paper
cfg.model.gen.mapping_layers = 4
cfg.model.gen.blur_filter = [1, 2, 1]
cfg.model.gen.truncation_psi = 0.7
cfg.model.gen.truncation_cutoff = 8

# ---------------------------------------------------------------------------- #
# Options for Discriminator
# ---------------------------------------------------------------------------- #
cfg.model.dis = CN()
cfg.model.dis.use_wscale = True
cfg.model.dis.blur_filter = [1, 2, 1]

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
