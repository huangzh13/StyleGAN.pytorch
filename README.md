# StyleGAN.pytorch

<p align="center">
     <img src=diagrams/race_chinese_girl_256.png width=100% /> <br>
     <a align="center" href="http://www.seeprettyface.com/mydataset.html">[ChineseGirl Dataset]</a>
</p>

> Thanks to dataset provider:Copyright(c) 2018, seeprettyface.com, BUPT_GWY contributes the dataset.


## Features

- [x] Progressive Growing Training
- [x] Exponential Moving Average
- [x] Equalized Learning Rate
- [x] PixelNorm Layer
- [x] Minibatch Standard Deviation Layer
- [x] Style Mixing Regularization (Experimental)
- [ ] Truncation Trick   
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


## Thanks

Please feel free to open PRs / issues / suggestions here.

## Reference

- **stylegan**[official]: https://github.com/NVlabs/stylegan
- **pro_gan_pytorch**: https://github.com/akanimax/pro_gan_pytorch
- **pytorch_style_gan**: https://github.com/lernapparat/lernapparat