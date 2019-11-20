# StyleGAN.pytorch

<p align="center">
     <img src=diagrams/chinese_girl_256.png width=100% /> <br>
     <a align="center" href="http://www.seeprettyface.com/mydataset.html">[ChineseGirl Dataset]</a>
</p>

> Thanks to dataset provider:Copyright(c) 2018, seeprettyface.com, BUPT_GWY contributes the dataset.


## Features

- [x] Progressive Growing Training
- [x] Exponential Moving Average
- [x] Equalized Learning Rate
- [x] PixelNorm Layer
- [x] Minibatch Standard Deviation Layer
- [x] Style Mixing Regularization
- [x] Using official tensorflow pretrained weights 
- [ ] Gradient Clipping
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
     <img src=diagrams/Cari2_128.png width=90% /> <br>
     <a align="center" href="https://cs.nju.edu.cn/rl/WebCaricature.htm">[WebCaricatureDataset](128x128)</a><br>
</p>

## Thanks

Please feel free to open PRs / issues / suggestions here.

## Reference

- **stylegan**[official]: https://github.com/NVlabs/stylegan
- **pro_gan_pytorch**: https://github.com/akanimax/pro_gan_pytorch
- **pytorch_style_gan**: https://github.com/lernapparat/lernapparat