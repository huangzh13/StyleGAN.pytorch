# StyleGAN.pytorch

## \[:star: New :star:\] Please head over to [StyleGAN2.pytorch](https://github.com/huangzh13/StyleGAN2.pytorch) for my stylegan2 pytorch implementation.

<p align="center">
     <img src=diagrams/grid.png width=100% /> <br>
     <a align="center" href="http://www.seeprettyface.com/mydataset.html">[ChineseGirl Dataset]</a>
</p>

This repository contains the unofficial PyTorch implementation of the following paper:

> A Style-Based Generator Architecture for Generative Adversarial Networks <br>
> Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA) <br>
> http://stylegan.xyz/paper
> 
> Abstract: We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture. Finally, we introduce a new, highly varied and high-quality dataset of human faces.


## Features

- [x] Progressive Growing Training
- [x] Exponential Moving Average
- [x] Equalized Learning Rate
- [x] PixelNorm Layer
- [x] Minibatch Standard Deviation Layer
- [x] Style Mixing Regularization
- [x] Truncation Trick   
- [x] Using official tensorflow pretrained weights 
- [x] Gradient Clipping
- [ ] Multi-GPU Training
- [ ] FP-16 Support
- [ ] Conditional GAN

## How to use

### Requirements
- yacs
- tqdm
- numpy
- torch
- torchvision
- tensorflow(Optional, for ./convert.py)

### Running the training script:
Train from scratch:
```shell script
python train.py --config configs/sample.yaml
```

### Using trained model:
Resume training from a checkpoint (start form 128x128):
```shell script
python train.py --config config/sample.yaml --start_depth 5 --generator_file [] [--gen_shadow_file] --discriminator_file [] --gen_optim_file [] --dis_optim_file []
```
### Style Mixing

```shell script
python generate_mixing_figure.py --config config/sample.yaml --generator_file [] 
```

<p align="center">
     <img src=diagrams/figure03-style-mixing-mix.png width=90% /> <br>
</p>

> Thanks to dataset provider:Copyright(c) 2018, seeprettyface.com, BUPT_GWY contributes the dataset.

### Truncation trick

```shell script
python generate_truncation_figure.py --config configs/sample_cari2_128_truncation.yaml --generator_file cari2_128_truncation_gen.pth
```

<p align="center">
     <img src=diagrams/figure08-truncation-trick.png width=90% /> <br>
</p>

### Convert from official format
```shell script
python convert.py --config configs/sample_ffhq_1024.yaml --input_file PATH/karras2019stylegan-ffhq-1024x1024.pkl --output_file ffhq_1024_gen.pth
```

## Generated samples

<p align="center">
     <img src=diagrams/ffhq_128.png width=90% /> <br>
     <a align="center" href="https://github.com/NVlabs/ffhq-dataset">[FFHQ Dataset](128x128)</a>
</p>

Using weights tranferred from official tensorflow repo.
<p align="center">
     <img src=diagrams/ffhq_1024.png width=90% /> <br>
     <a align="center" href="https://github.com/NVlabs/ffhq-dataset">[FFHQ Dataset](1024x1024)</a><br>
</p>

<p align="center">
     <img src=diagrams/cari2_128.png width=90% /> <br>
     <a align="center" href="https://cs.nju.edu.cn/rl/WebCaricature.htm">[WebCaricatureDataset](128x128)</a><br>
</p>

## Reference

- **stylegan[official]**: https://github.com/NVlabs/stylegan
- **pro_gan_pytorch**: https://github.com/akanimax/pro_gan_pytorch
- **pytorch_style_gan**: https://github.com/lernapparat/lernapparat

## Thanks

Please feel free to open PRs / issues / suggestions here.

## Due Credit
This code heavily uses NVIDIA's original 
[StyleGAN](https://github.com/NVlabs/stylegan) code. We accredit and acknowledge their work here. The 
[Original License](/LICENSE_ORIGINAL.txt) 
is located in the base directory (file named `LICENSE_ORIGINAL.txt`).