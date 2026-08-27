# Weight Conversion Research Notes

Status: **Pending validation; static analysis complete**<br>
Last updated: 2026-08-26

## Executive Summary

Converting an original NVIDIA StyleGAN TensorFlow checkpoint to this repository's PyTorch Generator remains feasible under defined conditions, but the current repository cannot claim that conversion is correct.

The feasible target should be limited to:

- Original NVIDIA StyleGAN TensorFlow `.pkl` files;
- A resolution, channel count, number of mapping layers, and generator configuration that exactly match the checkpoint;
- Reading the checkpoint in an isolated legacy TensorFlow 1.x/Python environment;
- Layer-by-layer or pixel-by-pixel validation after conversion using the same seed and the same noise.

StyleGAN2, StyleGAN2-ADA, and StyleGAN3 checkpoints should not be converted directly to this repository's original StyleGAN architecture. They use different networks and operator systems and should be handled by their corresponding official implementations or by separate adapters.

## Related Issues

The following issues are likely part of the same problem chain:

- [#3 Possible mistake in pretrained weights porting](https://github.com/huangzh13/StyleGAN.pytorch/issues/3)
- [#5 Error while converting weights](https://github.com/huangzh13/StyleGAN.pytorch/issues/5)
- [#7 Official pretrained weights generate noisy faces](https://github.com/huangzh13/StyleGAN.pytorch/issues/7)
- [#12 Generate samples colors all messed up](https://github.com/huangzh13/StyleGAN.pytorch/issues/12)

See [issue-triage.md](issue-triage.md) for the complete issue classification. “Confirmed” in this document primarily means supported by static code evidence; it does not mean that reproduction with an official checkpoint has been completed.

## Discovered Problems

### 1. Incomplete ToRGB Mapping

The official StyleGAN synthesis network creates multiple output layers, such as ToRGB_lod0 and ToRGB_lod1, for different resolutions. Reference: [NVIDIA networks_stylegan.py](https://raw.githubusercontent.com/NVlabs/stylegan/master/training/networks_stylegan.py).

The current `key_translate()` in [convert.py](../convert.py) explicitly replaces only `torgb_lod0`:

~~~
torgb_lod0 -> to_rgb.<out_depth>
~~~

Other `torgb_lodN` keys are not converted to the corresponding indices of the target `to_rgb` module and are subsequently excluded by the filtering conditions. As a result, intermediate ToRGB layers are missing from the target model.

The correct relationship should be:

~~~
target_index = out_depth - source_lod
~~~

This is especially important for low-resolution generation and progressive fade-in paths. At the full resolution with alpha=1, some unloaded ToRGB layers may temporarily not participate in the output, but that cannot be used as evidence that the conversion is correct.

### 2. strict=False Masks Conversion Failures

The current implementation uses:

~~~
gen.load_state_dict(param_dict, strict=False)
~~~

This leaves missing keys, unexpected keys, and some configuration problems only in the logs while the program continues to save the `.pth` file. `dlatent_avg` is also unconditionally written to `truncation.avg_latent`; when the target configuration disables truncation, it becomes an unexpected key.

The converter should explicitly collect and check the source-key count, target-key count, missing keys, unexpected keys, shape mismatches, and whether all required layers are covered. Unless a key is explicitly recorded as non-parameter state, validation failure should stop the save operation.

### 3. Bias and Noise Execution Order Differs

The official implementation's epilogue order is:

~~~
conv -> noise -> bias -> activation -> instance norm -> style
~~~

In this repository, the convolution layer already adds bias in its own forward pass, followed by:

~~~
conv(with bias) -> noise -> activation -> instance norm -> style
~~~

Related code:

- [models/CustomLayers.py](../models/CustomLayers.py)'s EqualizedConv2d;
- [models/CustomLayers.py](../models/CustomLayers.py)'s LayerEpilogue;
- [models/Blocks.py](../models/Blocks.py)'s InputBlock and GSynthesisBlock.

This difference cannot be fixed by transposing weights alone. Strict equivalence with the official TensorFlow network requires a compatibility path for the official operator order. Because this may change the behavior of existing checkpoints in this repository, an explicit `tf_compatible` mode is recommended instead of overwriting the historical default behavior.

### 4. Inference Scripts Default to Training Mode

After constructing a Generator, [generate_samples.py](../generate_samples.py), `generate_grid.py`, and similar scripts do not call `eval()`. When `self.training` is true, [models/GAN.py](../models/GAN.py) performs style mixing and may update the truncation mean.

Therefore, the current generation scripts do not produce output equivalent to the official inference workflow. Every inference entry point should explicitly call:

~~~
gen.eval()
~~~

Then fix the random inputs, noise, and truncation parameters before making comparisons.

### 5. Fixed Noise Is Not Included in the PyTorch Checkpoint

In the official network, fixed noise is non-trainable state stored in `vars` rather than `trainables`. The current converter extracts `dlatent_avg` as an additional value but does not import variables such as `noise0` and `noise1`.

This may not affect visual quality in random-noise mode, but it prevents strict reproduction and same-seed TensorFlow/PyTorch comparisons. Explicit fixed-noise import and inference options should be added.

## Unverified Items

1. Whether the bias/noise order difference is the primary visual cause of the noisy faces in #3/#7;
2. The output error of a full-resolution FFHQ checkpoint after fixing the ToRGB mapping;
3. Whether RGB/BGR order, dynamic range, and PNG saving constitute an independent cause of #12;
4. Output consistency across different resolutions, alpha values, and depths;
5. Consistency when restoring training checkpoints, EMA, optimizer, and progressive state.

## Recommended Validation Protocol

### Stage 1: Unit Tests Without an External Checkpoint

- Construct a small synthetic TensorFlow-style key set;
- Check the target indices for all ToRGB_lodN layers;
- Check the transposition rules for 2D linear and 4D convolution weights;
- Check key behavior with truncation enabled and disabled;
- Add failure tests for missing, unexpected, and shape-mismatched keys.

### Stage 2: Conversion in a Legacy Environment

Create an isolated TensorFlow 1.x environment or container solely for reading the original `.pkl` files. Do not make TensorFlow 1.x a default dependency for modern installations.

The recommended records include the checkpoint filename and hash, resolution, number of mapping layers, fmap configuration, Python/TensorFlow/CUDA/PyTorch versions, conversion log, source/target key coverage, and whether fixed noise was used.

### Stage 3: Same-Seed Comparison

Use the same latent, the same `dlatent_avg`, and the same fixed noise, and compare layer by layer from 4x4, 8x8, and 16x16 through the final resolution:

- Tensor shape;
- min/max/mean/std;
- Maximum absolute error;
- Mean absolute error;
- Final image error and visual result.

Only after this protocol passes should the README claim support for obtaining supported models through official weight conversion.

## Reference Implementations and Scope

- [NVIDIA original StyleGAN](https://github.com/NVlabs/stylegan): source of the original TensorFlow network and checkpoints.
- [NVIDIA StyleGAN2-ADA PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch): provides a validated approach to TensorFlow/PyTorch-compatible migration, but its target architecture is not this repository's original StyleGAN.
- [NVIDIA StyleGAN3](https://github.com/NVlabs/stylegan3): supports legacy StyleGAN2 checkpoints, but obtaining StyleGAN3 architectural capabilities still requires retraining.

## Current Assessment

This is a “fixable, but requiring layered validation” problem:

- ToRGB, strict loading, inference mode, and fixed noise: can be fixed directly at the engineering level;
- Bias/noise order: requires a new compatibility operator path;
- TensorFlow 1.x: requires an isolated legacy environment;
- StyleGAN2/3 checkpoints: outside the supported scope of this converter.
