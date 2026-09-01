"""Generate a deterministic latent interpolation as PNG frames and a GIF."""

import argparse
import os

import numpy as np
from PIL import Image

import torch

from generate_samples import adjust_dynamic_range
from models.GAN import Generator


def slerp(start, end, amount):
    """Spherical interpolation for two batches of latent vectors."""
    start_norm = start / start.norm(dim=1, keepdim=True).clamp_min(1e-8)
    end_norm = end / end.norm(dim=1, keepdim=True).clamp_min(1e-8)
    dot = (start_norm * end_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    linear = start + amount * (end - start)
    spherical = (
        torch.sin((1.0 - amount) * theta) / sin_theta * start
        + torch.sin(amount * theta) / sin_theta * end
    )
    # Nearly parallel endpoints make the spherical formula ill-conditioned.
    return torch.where(sin_theta.abs() < 1e-6, linear, spherical)


def tensor_to_pil(image):
    """Convert one generator output in [-1, 1] to an RGB PIL image."""
    image = adjust_dynamic_range(image).mul(255).clamp(0, 255).byte()
    image = image.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(image, "RGB")


def generate_interpolation(gen, seed_a, seed_b, steps, out_depth,
                           space="w", device=None):
    """Return ``steps`` images interpolating between two deterministic seeds.

    W-space interpolation maps the endpoints first and linearly interpolates
    the resulting per-layer dlatents. Z-space interpolation uses spherical
    interpolation before the generator's mapping network.
    """
    device = device or next(gen.parameters()).device
    latent_size = gen.g_mapping.latent_size
    random_a = np.random.RandomState(seed_a).randn(1, latent_size)
    random_b = np.random.RandomState(seed_b).randn(1, latent_size)
    latent_a = torch.from_numpy(random_a.astype(np.float32)).to(device)
    latent_b = torch.from_numpy(random_b.astype(np.float32)).to(device)

    with torch.no_grad():
        if space == "w":
            start = gen.g_mapping(latent_a)
            end = gen.g_mapping(latent_b)
        else:
            start = latent_a
            end = latent_b

        images = []
        for index in range(steps):
            amount = index / float(steps - 1)
            t = torch.tensor(amount, device=device, dtype=start.dtype)
            if space == "w":
                current = start + t * (end - start)
                image = gen.g_synthesis(current, depth=out_depth, alpha=1.0)
            else:
                current = slerp(start, end, t)
                image = gen(current, depth=out_depth, alpha=1.0)
            images.append(tensor_to_pil(image[0]))
    return images


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate a deterministic StyleGAN latent interpolation.")
    parser.add_argument("--config", default="./configs/sample.yaml")
    parser.add_argument("--generator_file", required=True)
    parser.add_argument("--output_dir", default="output/interpolation")
    parser.add_argument("--seed_a", type=int, default=0)
    parser.add_argument("--seed_b", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--space", choices=("w", "z"), default="w")
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--random_noise", action="store_true",
                        help="Do not fix synthesis noise (not recommended for animation)")
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def main(args):
    from config import cfg as opt

    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    if args.fps < 1:
        raise ValueError("--fps must be at least 1")

    opt.merge_from_file(args.config)
    opt.freeze()
    device = torch.device(opt.device)

    gen = Generator(resolution=opt.dataset.resolution,
                    num_channels=opt.dataset.channels,
                    structure=opt.structure,
                    **opt.model.gen).to(device)
    gen.load_state_dict(torch.load(args.generator_file, map_location=device))
    gen.eval()
    gen.set_noise_seed(None if args.random_noise else args.noise_seed)

    out_depth = int(np.log2(opt.dataset.resolution)) - 2
    frames = generate_interpolation(gen, args.seed_a, args.seed_b, args.steps,
                                    out_depth, space=args.space, device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(os.path.join(args.output_dir, "frame_{:04d}.png".format(index)))

    gif_path = os.path.join(args.output_dir, "interpolation.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=1000 // args.fps, loop=0)
    print("Generated {} frames and {}".format(len(frames), gif_path))


if __name__ == "__main__":
    main(parse_arguments())
