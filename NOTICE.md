# Third-party notices and provenance

This repository is a historical research implementation assembled from original
work in this repository and code adapted from several upstream projects. The
repository-level license files do not by themselves establish that every file
has the same copyright holder or license.

## Upstream projects

- [NVIDIA StyleGAN](https://github.com/NVlabs/stylegan): the `dnnlib/` tree
  contains files with NVIDIA copyright and Creative Commons Attribution-
  NonCommercial 4.0 notices. The corresponding local text is preserved in
  [`LICENSE_ORIGINAL.txt`](LICENSE_ORIGINAL.txt).
- [akanimax/pro_gan_pytorch](https://github.com/akanimax/pro_gan_pytorch):
  referenced as an upstream source in parts of `models/`, `models/GAN.py`,
  `models/__init__.py`, and `generate_samples.py`.
- [lernapparat/lernapparat](https://github.com/lernapparat/lernapparat):
  referenced as an upstream source in `models/Blocks.py`,
  `models/CustomLayers.py`, `models/GAN.py`, and `convert.py`.

The exact upstream commit and applicable license for every adapted component have
not yet been established. See [`docs/license-provenance.md`](docs/license-provenance.md)
for the current evidence and open questions.

## Current license files

- [`LICENSE.txt`](LICENSE.txt) contains a Creative Commons Attribution-
  NonCommercial 4.0 license text with a copyright notice for `akanimax`. Its
  intended file scope has not been confirmed.
- [`LICENSE_ORIGINAL.txt`](LICENSE_ORIGINAL.txt) contains a Creative Commons
  Attribution-NonCommercial 4.0 license text with a copyright notice for NVIDIA.

Do not replace, remove, or reinterpret these notices without first completing the
provenance review and confirming the relevant rights. This notice is a factual
maintenance record, not a legal opinion or a relicensing statement.

## Redistribution guidance

When redistributing or modifying the repository, preserve the upstream copyright,
license, and attribution notices that accompany the relevant files. Do not claim
that the entire repository is OSI-licensed or that all files may be relicensed
under a single license until the review is complete.
