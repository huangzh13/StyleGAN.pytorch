"""
-------------------------------------------------
   File Name:    generate_samples.py
   Date:         2019/10/27
   Description:  Generate single image samples from a particular depth of a model
                 Modified from: https://github.com/akanimax/pro_gan_pytorch
-------------------------------------------------
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from models.GAN import Generator


def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./configs/sample.yaml')
    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", required=True)
    parser.add_argument("--num_samples", action="store", type=int,
                        default=300, help="number of synchronized grids to be generated")
    parser.add_argument("--output_dir", action="store", type=str,
                        default="output/",
                        help="path to the output directory for the frames")
    parser.add_argument("--input", action="store", type=str,
                        default=None, help="the dlatent code (W) for a certain sample")
    parser.add_argument("--output", action="store", type=str,
                        default="output.png", help="the output for the certain samples")

    args = parser.parse_args()

    return args


def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
    """
    adjust the dynamic colour range of the given input data
    :param data: input image data
    :param drange_in: original range of input
    :param drange_out: required range of output
    :return: img => colour range adjusted images
    """
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return torch.clamp(data, min=0, max=1)


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

    # path for saving the files:
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)
    latent_size = opt.model.gen.latent_size
    out_depth = int(np.log2(opt.dataset.resolution)) - 2

    if args.input is None:
        print("Generating scale synchronized images ...")
        for img_num in tqdm(range(1, args.num_samples + 1)):
            # generate the images:
            with torch.no_grad():
                point = torch.randn(1, latent_size)
                point = (point / point.norm()) * (latent_size ** 0.5)
                ss_image = gen(point, depth=out_depth, alpha=1)
                # color adjust the generated image:
                ss_image = adjust_dynamic_range(ss_image)

            # save the ss_image in the directory
            save_image(ss_image, os.path.join(save_path, str(img_num) + ".png"))

        print("Generated %d images at %s" % (args.num_samples, save_path))
    else:
        code = np.load(args.input)
        dlatent_in = torch.unsqueeze(torch.from_numpy(code), 0)
        ss_image = gen.g_synthesis(dlatent_in, depth=out_depth, alpha=1)
        # color adjust the generated image:
        ss_image = adjust_dynamic_range(ss_image)
        save_image(ss_image, args.output)


if __name__ == '__main__':
    main(parse_arguments())
