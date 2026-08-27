# StyleGAN.pytorch Iteration Roadmap

Status: **Planning**<br>
Last updated: 2026-08-26

## Project Positioning

This project should not be positioned as a general-purpose modern image-generation framework, nor should it directly compete with large-scale diffusion/DiT systems.

The recommended positioning is:

> Reproducible, portable, and editable research infrastructure for the original StyleGAN.

This positioning covers historical reproduction, weight conversion, latent manipulation, and domain research, while preserving the current repository's value as a reference for legacy StyleGAN code.

## Roadmap Overview

~~~
P0  Weight-conversion correctness
 ↓
P1  Reproducible benchmarks and a modern runtime baseline
 ↓
P2  Latent inversion/editing tools
 ↓
P3  Small-data domain transfer or an alias-free/video research branch
~~~

## P0: Weight-Conversion Correctness

Goal: Resolve the shared foundational problems behind #3, #5, and #7, and provide a verifiable generation-output baseline for #12.

### Work Items

- Fix the complete ToRGB_lodN mapping;
- Change out_depth from a global variable to an explicit parameter;
- Add source/target key coverage reports;
- Prohibit unaudited strict=False usage;
- Correctly handle truncation and dlatent_avg;
- Import fixed noise;
- Explicitly call eval() from every generation entry point;
- Add synthetic checkpoint tests;
- Add reproduction records for official checkpoints in a legacy environment.

### Acceptance Criteria

- The target model has no unexplained missing, unexpected, or mismatched keys;
- All resolution/depth/alpha paths run successfully;
- With fixed latent and noise, TensorFlow/PyTorch output errors remain below a clearly defined threshold;
- The README clearly lists supported and unsupported checkpoint types.

## P1: Reproducible Runtime Baseline

Goal: Establish a stable measurement foundation for future fixes and paper experiments.

### Work Items

- Establish a modern CPU smoke test;
- Document the GPU/legacy-TF environment;
- Complete currently empty or weakly asserted tests;
- Fix random seeds, noise, configuration, and checkpoint metadata;
- Record image range, RGB order, and save paths;
- Add minimal benchmarks for FID, interpolation continuity, truncation, and style mixing;
- Record the depth, alpha, EMA, and optimizer states required for progressive resume.

### Acceptance Criteria

- A new environment can reproduce model construction and a CPU forward pass;
- Key behaviors have automated regression tests;
- Every experimental result can be traced to its configuration, code version, and checkpoint hash.

## P2: Latent Inversion and Editing

Compared with general-purpose diffusion models, this remains a direction where the original StyleGAN has clear research value.

### Work Items

- Unify the Z, W, W+, and StyleSpace APIs;
- Provide projector and image-inversion examples;
- Support loading, saving, and composing semantic directions;
- Add visualizations for latent interpolation, style mixing, and truncation;
- Evaluate reconstruction, identity, perceptual, and editability metrics;
- Provide both fixed-noise and optimized-noise inversion modes.

### Reference Directions

- [Third Time's the Charm? Image and Video Editing with StyleGAN3](https://arxiv.org/abs/2201.13433) demonstrates how different latent spaces affect editing capability.
- A large number of community StyleGAN tools still focus on inversion, editing, attribute manipulation, and domain adaptation.

### Acceptance Criteria

- An inversion result can be reproduced for a given input image;
- Identity and target attributes before and after editing are recorded quantitatively;
- The same latent, noise, and parameters produce the same result.

## P3-A: Small-Data and Domain Transfer

Goal: Extend the project beyond “reproducing FFHQ only” to more practical research datasets.

### Work Items

- Introduce ADA-style adaptive discriminator augmentation;
- Support transfer learning from an existing generator;
- Complete the data and label interfaces for conditional GANs;
- Add diagnostics for class imbalance, long-tail distributions, and data alignment;
- Provide training benchmarks for hundreds, thousands, and tens of thousands of samples;
- Evaluate FID, precision/recall, attribute coverage, and memorization risk.

StyleGAN2-ADA demonstrated the practical value of ADA for small-data training and transfer learning: [paper](https://arxiv.org/abs/2006.06676). StyleGAN-Human showed that dataset size, data distribution, and alignment are important factors in StyleGAN quality: [paper](https://arxiv.org/abs/2204.11823).

## P3-B: Alias-Free, Video, and 3D

This is a research-innovation track and should not be started at the same time as P3-A.

### Work Items

- Study StyleGAN3's alias-free signal processing;
- Add equivariance evaluations for translation, rotation, subpixel shifts, and related transformations;
- Support generating videos from latent trajectories;
- Explore 3D-aware representations, multi-view consistency, and volumetric data generation;
- Design dedicated benchmarks for medical images, multi-slice data, or video data.

The official StyleGAN3 implementation emphasizes alias-free synthesis, equivariance metrics, and video tools: [StyleGAN3](https://github.com/NVlabs/stylegan3). However, its architecture differs from this repository's. If this direction is adopted, it should be implemented as an independent backend or research branch rather than silently changing the original implementation.

## Directions Not Recommended as Near-Term Priorities

### 1. Fully Rewriting the Project Directly as StyleGAN2/3

This would simultaneously introduce new architectures, operators, checkpoint formats, CUDA extensions, licensing issues, and reproduction problems, while undermining the project's historical positioning. It is better suited to an independent backend or a new repository.

### 2. Adding General-Purpose Text-to-Image Generation Directly

DiT and related work represent a scaling path for diffusion/Transformer-based general-purpose generation: [DiT](https://arxiv.org/abs/2212.09748). This project is better suited to leveraging StyleGAN's fast inference, latent editability, and domain-transfer capabilities, and studying GAN/diffusion hybrids when necessary, rather than competing with general-purpose text-to-image generation from scratch.

### 3. Starting with Large-Scale Engineering Refactoring

Before weight conversion and provenance are clearly established, broad renaming, formatting, or copying of new upstream implementations should be avoided. Refactoring should proceed in small patches, behavior tests, and explicitly defined compatibility modes.

## Recommended Implementation Order

### Milestone 1: Converter Audit

- Complete key-mapping unit tests;
- Fix the ToRGB mapping;
- Add strict coverage checks;
- Fix inference mode;
- Write the official-checkpoint validation command and record template.

### Milestone 2: Official Checkpoint Validation

- Prepare an isolated TensorFlow 1.x environment;
- Select one low-resolution checkpoint and one FFHQ checkpoint;
- Complete a fixed latent/noise, same-seed comparison;
- Decide whether to implement a tf_compatible operator path based on the error results.

### Milestone 3: Release a Trusted Baseline

- Update the README's support boundaries;
- Publish conversion reports and checkpoint hashes;
- Do not directly redistribute weights with licensing or provenance questions;
- Add a modern CPU smoke test and minimal regression tests.

### Milestone 4: Select One Research Track

Latent editing/inversion plus reproducibility is the top recommendation. If maintainers are more interested in training research, choose ADA plus small-data transfer; if video or geometry is the priority, choose alias-free/video/3D.

## Decision Principles

- Preserve original StyleGAN behavior; use explicit configuration for new behavior;
- Every compatibility claim must have checkpoint and environment records;
- Every model-quality conclusion should report the data, metrics, and randomness controls together;
- Do not publish unvalidated conversion results as officially equivalent weights;
- Before introducing NVIDIA or other upstream code, complete provenance and license verification.
