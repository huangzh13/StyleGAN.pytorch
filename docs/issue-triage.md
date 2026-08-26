# Issue triage and maintenance queue

Status: **initial classification**. This document records the 15 open GitHub
Issues visible on 2026-08-26. The classifications are working priorities, not
claims that the reported behavior has been reproduced. Issues should remain
open until their behavior, scope, and resolution have been verified.

## Working principles

- Reproduce the report before changing model behavior or closing an Issue.
- Preserve historical behavior unless a correctness or compatibility problem is
  demonstrated.
- Record the environment, checkpoint, configuration, and command used for each
  verification.
- Treat pretrained-weight conversion and pickle-based checkpoints as a separate
  legacy compatibility boundary.
- Combine duplicate requests only after their relationship has been confirmed;
  do not close an Issue solely because it is old.

## Priority queue

| Priority | Issue | Category | Current assessment | Next verification |
| --- | --- | --- | --- | --- |
| P1 | [#3 Possible mistake in pretrained weights porting](https://github.com/huangzh13/StyleGAN.pytorch/issues/3) | Correctness / conversion | Reports visible artifacts when porting official TensorFlow weights; root cause is unverified | Establish a trusted TensorFlow reference, convert one checkpoint, compare outputs and state-dict coverage |
| P1 | [#5 Error while converting weights](https://github.com/huangzh13/StyleGAN.pytorch/issues/5) | Bug / conversion | Reports missing synthesis keys and an unexpected truncation key | Reproduce with the referenced checkpoint/config and inspect the converter's key mapping |
| P1 | [#7 Official pretrained weights generate noisy faces](https://github.com/huangzh13/StyleGAN.pytorch/issues/7) | Bug / conversion | Concrete symptom overlapping with #3 and #5 | Run a same-seed TensorFlow/PyTorch comparison after conversion |
| P1 | [#12 Generate samples colors all messed up](https://github.com/huangzh13/StyleGAN.pytorch/issues/12) | Bug / generation | Training images reportedly look normal while generated files have incorrect colors | Reproduce with the original checkpoint and inspect output range, channel order, and image serialization |
| P1 | [#9 CUDNN_STATUS_NOT_SUPPORTED error](https://github.com/huangzh13/StyleGAN.pytorch/issues/9) | Bug / compatibility | CUDA/cuDNN-specific failure; local CPU baseline cannot validate it | Capture a current GPU environment and test contiguous inputs plus the linked convolution path |
| P1 | [#10 Continue training from checkpoints reduced quality](https://github.com/huangzh13/StyleGAN.pytorch/issues/10) | Bug / checkpointing | Quality changes after resume are reported but the checkpoint/config pair is unavailable | Compare optimizer, EMA, progressive depth, alpha, and random-state restoration |
| P2 | [#1 Gradient sanitizing (clipping) helps a lot](https://github.com/huangzh13/StyleGAN.pytorch/issues/1) | Enhancement / training stability | The repository currently lists gradient clipping as implemented; coverage and effect are not independently verified | Locate the active clipping path and add a regression test before responding |
| P2 | [#4 GAN not training](https://github.com/huangzh13/StyleGAN.pytorch/issues/4) | Support / training | Only 50 training images and no reproducible configuration are reported | Request or reconstruct dataset layout, config, device, and training logs |
| P2 | [#8 Smooth interpolation between outputs](https://github.com/huangzh13/StyleGAN.pytorch/issues/8) | Documentation / generation | Likely involves stochastic noise inputs and evaluation mode, but behavior is unverified | Trace generation randomness and document a deterministic interpolation recipe |
| P2 | [#11 Where is the latent vector stored?](https://github.com/huangzh13/StyleGAN.pytorch/issues/11) | Documentation / generation | The generation interface exposes a dlatent input but the workflow is not documented clearly | Verify the expected W-space tensor shape and add a minimal example |
| P2 | [#15 Train with conditional set to True](https://github.com/huangzh13/StyleGAN.pytorch/issues/15) | Documentation / conditional GAN | Conditional mode exists, but the repository lacks a complete training example | Validate the conditional config, dataset labels, and command line |
| P2 | [#6 Provide pretrained models](https://github.com/huangzh13/StyleGAN.pytorch/issues/6) | Artifact / distribution | Request is blocked by missing maintained weights and unresolved redistribution/licensing questions | Check provenance, conversion status, storage options, and whether a safe artifact can be published |
| P2 | [#17 Pre-trained weights](https://github.com/huangzh13/StyleGAN.pytorch/issues/17) | Artifact / distribution | Overlaps with #6 | Confirm the overlap and handle both requests consistently after the licensing review |
| P3 | [#2 SomeQuestion](https://github.com/huangzh13/StyleGAN.pytorch/issues/2) | Support / training | Broad questions about training duration, hardware, and StyleGAN2 are not tied to a reproducible failure | Provide current project scope and link to the modern StyleGAN2 recommendation |
| P3 | [#16 Multi-GPU](https://github.com/huangzh13/StyleGAN.pytorch/issues/16) | Enhancement / distributed training | Multi-GPU training is explicitly not implemented or validated | Defer until single-device training and compatibility boundaries are documented |

## Recommended work packages

### 1. Weight-conversion correctness: #3, #5, and #7

These reports describe one likely workstream. First determine whether the
converter produces a complete and numerically plausible generator state. Only
then should the project document converted checkpoints or promise compatibility
with official TensorFlow weights.

### 2. Generation output integrity: #12

This can be investigated independently from TensorFlow conversion if the
reporter checkpoint or a small reproduction fixture is available. The first
checks should cover tensor value ranges, RGB/BGR ordering, normalization, and
the image-writing path.

### 3. Checkpoint and GPU compatibility: #9 and #10

These require environment-specific evidence. The CPU CI baseline is useful for
model construction and finite forwards, but it does not establish CUDA or
resume-training compatibility.

### 4. Documentation quick wins: #8, #11, and #15

Once the command lines and tensor shapes are verified, these can be addressed
with small README examples without changing the model implementation.

## Labels

The Issues use the repository's existing standard labels. Labels indicate the
current triage view only; they do not mean that a bug has been reproduced or
that a requested feature is committed to the roadmap.
