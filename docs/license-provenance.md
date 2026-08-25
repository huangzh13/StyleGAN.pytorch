# License and provenance audit

Status: preliminary maintenance record. This document records repository evidence
and open questions; it is not legal advice and does not change any license terms.

## Scope and method

The audit began with the repository's license files, source headers, explicit
upstream references, and Git history. A high-confidence origin means that the
file itself contains an explicit copyright or source notice. A medium-confidence
origin means that the source is declared in a file header or strongly indicated by
the implementation, but the exact upstream version or contribution boundary is
not yet confirmed. A low-confidence entry requires source comparison or rights
confirmation before any licensing conclusion is made.

The current repository contains two root license files:

- [`LICENSE.txt`](../LICENSE.txt) starts with `Copyright (c) 2019, akanimax` and
  contains the CC BY-NC 4.0 text.
- [`LICENSE_ORIGINAL.txt`](../LICENSE_ORIGINAL.txt) starts with `Copyright (c)
  2019, NVIDIA CORPORATION` and contains the same CC BY-NC 4.0 text.

Neither file clearly states which paths it covers. They should therefore be
treated as historical notices until the component boundaries are verified.

## Component inventory

| Path or component | Likely origin | Evidence in this repository | Confidence | Follow-up |
| --- | --- | --- | --- | --- |
| `dnnlib/` | NVIDIA StyleGAN | Files in `dnnlib/` carry NVIDIA copyright and CC BY-NC notices; `LICENSE_ORIGINAL.txt` carries the NVIDIA notice | High | Compare with the exact NVIDIA StyleGAN commit and preserve file-level notices |
| `models/Blocks.py` | `lernapparat/lernapparat`, with local changes possible | Header says `Copy from: https://github.com/lernapparat/lernapparat` | Medium | Identify the source commit and document the local modifications |
| `models/CustomLayers.py` | `lernapparat/lernapparat`, with local changes possible | Header says `Copy from: https://github.com/lernapparat/lernapparat` | Medium | Identify the source commit and applicable upstream license |
| `models/GAN.py` | Mixed adaptation | Header names `akanimax/pro_gan_pytorch`, `lernapparat/lernapparat`, and NVIDIA StyleGAN | Medium | Perform a line-level comparison and separate upstream portions from local work |
| `models/__init__.py` | `akanimax/pro_gan_pytorch`, with local changes possible | Header says `Copy from: https://github.com/akanimax/pro_gan_pytorch` | Medium | Identify the source commit and local changes |
| `models/Losses.py` | Unresolved; likely local and/or upstream-derived portions | Header says `Copy from: https://github.com/akanimax/pro_gan_pytorch`; file contains multiple loss implementations | Medium | Compare against the upstream source and check later contributor rights |
| `generate_samples.py` | `akanimax/pro_gan_pytorch`, with local changes possible | Header says `Modified from: https://github.com/akanimax/pro_gan_pytorch` | Medium | Identify the source commit and local modifications |
| `convert.py` | `lernapparat/lernapparat` plus NVIDIA checkpoint conventions | Header says `Modified from: https://github.com/lernapparat/lernapparat`; imports local `dnnlib.tflib` | Medium | Audit the converter and its relationship to NVIDIA's checkpoint tooling |
| `train.py`, `config.py`, `data/`, `utils/` | Likely repository-specific, but contributor rights are not yet confirmed | No explicit third-party license header; Git history attributes most early commits to the repository maintainer | Low | Confirm authorship, employment/affiliation constraints, and later contributions |
| `generate_grid.py`, `generate_mixing_figure.py`, `generate_truncation_figure.py` | Unresolved mixture of local and adapted research code | No complete component-level license map | Low | Compare history and upstream implementations before assigning a license |
| `diagrams/` and checked-in sample images | Separate media/data rights may apply | Images reference ChineseGirl, FFHQ, and WebCaricature datasets in README | Low | Determine image authorship, dataset terms, and redistribution permissions |

## Current conclusions

1. The repository should be treated as a mixed-provenance research repository.
2. The NVIDIA-derived `dnnlib/` files should retain their existing NVIDIA notices.
3. The root `LICENSE.txt` should not be assumed to cover every file, especially
   files that identify NVIDIA or other upstream sources.
4. The repository should not currently advertise a single OSI-approved software
   license or promise that the maintainer can relicense all code.
5. Changing the root license to MIT, Apache-2.0, BSD, or another permissive
   license would be premature without upstream and contributor rights clearance.

## Recommended next actions

### P0 — Preserve notices and clarify scope

- Keep `LICENSE.txt` and `LICENSE_ORIGINAL.txt` unchanged for now.
- Keep the file-level NVIDIA notices in `dnnlib/` unchanged.
- Add or improve file-level attribution only when the source and scope are
  supported by evidence.

### P1 — Complete source comparison

- Identify the historical upstream commits for `pro_gan_pytorch`,
  `lernapparat/lernapparat`, and NVIDIA StyleGAN.
- Compare the high-risk files line by line.
- Record the source commit, local modifications, and applicable license for each
  component.

### P1 — Confirm contribution rights

- Review all contributors in Git history, including later feature contributors.
- Confirm that contributions were authorized for redistribution under the terms
  claimed by the repository.
- Record unresolved authorship or institutional ownership questions explicitly.

### P2 — Decide on a long-term licensing model

After the above work, choose one of:

1. retain a clearly documented non-commercial mixed-license arrangement;
2. separate repository-authored code from third-party components and license each
   component appropriately; or
3. obtain written permissions or replace incompatible code before adopting a
   single software license.

Do not make this decision solely to improve repository metadata or an application
profile.
