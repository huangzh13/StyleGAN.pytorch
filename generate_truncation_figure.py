"""
-------------------------------------------------
   File Name:    generate_truncation_figure.py
   Author:       Zhonghao Huang
   Date:         2019/11/23
   Description:  
-------------------------------------------------
"""

import argparse
import numpy as np
from PIL import Image

import torch

from generate_grid import adjust_dynamic_range
from models.GAN import Generator


def draw_truncation_trick_figure(png, gen, out_depth, seeds, psis):
    w = h = 2 ** (out_depth + 2)
    latent_size = gen.g_mapping.latent_size

    with torch.no_grad():
        latents_np = np.stack([np.random.RandomState(seed).randn(latent_size) for seed in seeds])
        latents = torch.from_numpy(latents_np.astype(np.float32))
        dlatents = gen.g_mapping(latents).detach().numpy()  # [seed, layer, component]
        dlatent_avg = gen.truncation.avg_latent.numpy()  # [component]

        canvas = Image.new('RGB', (w * len(psis), h * len(seeds)), 'white')
        for row, dlatent in enumerate(list(dlatents)):
            row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(psis, [-1, 1, 1]) + dlatent_avg
            row_dlatents = torch.from_numpy(row_dlatents.astype(np.float32))
            row_images = gen.g_synthesis(row_dlatents, depth=out_depth, alpha=1)
            for col, image in enumerate(list(row_images)):
                image = adjust_dynamic_range(image)
                image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                canvas.paste(Image.fromarray(image, 'RGB'), (col * w, row * h))
        canvas.save(png)


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    print("Creating generator object ...")
    # create the generator object
    gen = Generator(resolution=opt.dataset.resolution,
                    num_channels=opt.dataset.channels,
                    structure=opt.structure,
                    **opt.model.gen)

    print("Loading the generator weights from:", args.generator_file)
    # load the weights into it
    gen.load_state_dict(torch.load(args.generator_file))

    draw_truncation_trick_figure('figure08-truncation-trick.png', gen, out_depth=5,
                                 seeds=[91, 388], psis=[1, 0.7, 0.5, 0, -0.5, -1])

    print('Done.')


def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./configs/sample.yaml')
    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", required=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_arguments())
