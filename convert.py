"""
-------------------------------------------------
   File Name:    convert.py
   Date:         2019/11/20
   Description:  Modified from:
                 https://github.com/lernapparat/lernapparat
-------------------------------------------------
"""

import argparse
import pickle
import collections
import numpy as np

import torch

from models.GAN import Generator
from dnnlib import tflib


def load_weights(weights_dir):
    tflib.init_tf()
    weights = pickle.load(open(weights_dir, 'rb'))
    weights_pt = [collections.OrderedDict([(k, torch.from_numpy(v.value().eval()))
                                           for k, v in w.trainables.items()]) for w in weights]

    # dlatent_avg
    for k, v in weights[2].vars.items():
        if k == 'dlatent_avg':
            weights_pt.append(collections.OrderedDict([(k, torch.from_numpy(v.value().eval()))]))
    return weights_pt


def key_translate(k):
    k = k.lower().split('/')
    if k[0] == 'g_synthesis':
        if not k[1].startswith('torgb'):
            if k[1] != '4x4':
                k.insert(1, 'blocks')
                k[2] = str(int(np.log2(int(k[2].split('x')[0])) - 3))
            else:
                k[1] = 'init_block'
        k = '.'.join(k)
        k = (k.replace('const.const', 'const').replace('const.bias', 'bias')
             .replace('const.stylemod', 'epi1.style_mod.lin')
             .replace('const.noise.weight', 'epi1.top_epi.noise.weight')
             .replace('conv.noise.weight', 'epi2.top_epi.noise.weight')
             .replace('conv.stylemod', 'epi2.style_mod.lin')
             .replace('conv0_up.noise.weight', 'epi1.top_epi.noise.weight')
             .replace('conv0_up.stylemod', 'epi1.style_mod.lin')
             .replace('conv1.noise.weight', 'epi2.top_epi.noise.weight')
             .replace('conv1.stylemod', 'epi2.style_mod.lin')
             .replace('torgb_lod0', 'to_rgb.{}'.format(out_depth)))
    elif k[0] == 'g_mapping':
        k.insert(1, 'map')
        k = '.'.join(k)
    else:
        k = '.'.join(k)

    return k


def weight_translate(k, w):
    k = key_translate(k)
    if k.endswith('.weight'):
        if w.dim() == 2:
            w = w.t()
        elif w.dim() == 1:
            pass
        else:
            assert w.dim() == 4
            w = w.permute(3, 2, 0, 1)
    return w


def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./configs/sample.yaml')
    parser.add_argument("--input_file", action="store", type=str,
                        help="pretrained weights from official tensorflow repo.", required=True)
    parser.add_argument("--output_file", action="store", type=str, required=True,
                        help="path to the output weights.")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    print("Creating generator object ...")
    # create the generator object
    gen = Generator(resolution=opt.dataset.resolution,
                    num_channels=opt.dataset.channels,
                    structure=opt.structure,
                    **opt.model.gen)
    out_depth = gen.g_synthesis.depth - 1

    state_G, state_D, state_Gs, dlatent_avg = load_weights(args.input_file)

    # we delete the useless to_rgb filters
    params = {}
    for k, v in state_Gs.items():
        params[k] = v
    param_dict = {key_translate(k): weight_translate(k, v) for k, v in state_Gs.items()
                  if 'torgb_lod' not in key_translate(k)}

    for k, v in dlatent_avg.items():
        param_dict['truncation.avg_latent'] = v

    sd_shapes = {k: v.shape for k, v in gen.state_dict().items()}
    param_shapes = {k: v.shape for k, v in param_dict.items()}

    # check for mismatch
    for k in list(sd_shapes) + list(param_shapes):
        pds = param_shapes.get(k)
        sds = sd_shapes.get(k)
        if pds is None:
            print("sd only", k, sds)
        elif sds is None:
            print("pd only", k, pds)
        elif sds != pds:
            print("mismatch!", k, pds, sds)

    gen.load_state_dict(param_dict, strict=False)  # needed for the blur kernels
    torch.save(gen.state_dict(), args.output_file)
    print('Done.')
