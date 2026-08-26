# Maintenance audit — StyleGAN.pytorch

Status: **working document**. Compiled at the start of the modern maintenance
cycle. This document records observable repository state, a verified execution
baseline, a failure inventory, and prioritized recommendations. It is a factual
maintenance record, not a design proposal.

---

## A. Repository map

The repository is a single-package Python project (no `setup.py`/`pyproject.toml`).
Modules import each other as top-level packages (`models.*`, `data.*`, `dnnlib.*`),
so entry points must be run from the repository root.

| Component | Path(s) | Role |
|---|---|---|
| Training entry point | `train.py` | Progressive-growing training loop over a `yacs` config |
| Configuration | `config.py`, `configs/*.yaml` | Global `yacs` `CfgNode` (`config.cfg`) merged with per-run YAML |
| Model implementation | `models/GAN.py` | `GMapping`, `GSynthesis`, `Generator`, `Discriminator`, `StyleGAN`, losses wiring |
| Custom layers | `models/CustomLayers.py` | `EqualizedLinear`, `EqualizedConv2d`, `PixelNormLayer`, `StddevLayer`, `BlurLayer`, `LayerEpilogue`, `Truncation`, etc. |
| Synthesis blocks | `models/Blocks.py` | `InputBlock`, `GSynthesisBlock`, `DiscriminatorBlock`, `DiscriminatorTop` |
| Losses | `models/Losses.py` | Legacy GAN losses incl. logistic |
| Data pipeline | `data/datasets.py`, `data/transforms.py` | Dataset loading, transforms, `create_dataset` |
| Utilities | `utils/` | EMA copy, logger |
| TF→PyTorch conversion | `convert.py` | Legacy conversion via NVIDIA `dnnlib.tflib` + pickle checkpoints |
| NVIDIA utilities | `dnnlib/` | NVIDIA-origin tree (`util.py`, `tflib/`, `submission/`), CC BY-NC notices |
| Generation scripts | `generate_samples.py`, `generate_grid.py`, `generate_mixing_figure.py`, `generate_truncation_figure.py` | Checkpoint-driven sample/figure generation |
| Tests | `test/` | Legacy `unittest` stubs with no assertions (`test_Blocks.py`, `test_CustomLayers.py`) |

### Relationships

`train.py` ⇒ `config.cfg` + `models.GAN.StyleGAN` ⇒ `Generator`/`Discriminator` ⇒
`models.Blocks` ⇒ `models.CustomLayers`. Data flows from `data.datasets` into the
training loop. Generation scripts construct `Generator` directly from config.
`convert.py` bypasses the model code, translating NVIDIA pickle weights into
this repo's state-dict format through `dnnlib.tflib`.

---

## B. Environment baseline

### Historical assumptions (inferred, not verified)

- Python: written for ~Python 3.6/3.7 era (2019).
- PyTorch: `torch` was unpinned; code uses pre-`amp` APIs and is syntax-compatible
  with torch 2.x so far (see C).
- TensorFlow: required **only** for `convert.py`/`dnnlib.tflib`; must be a legacy
  TF 1.x-compatible installation for the NVIDIA checkpoint tooling.
- CUDA: training configs default `device='cuda'`; original environment used an
  NVIDIA GPU (a 1080Ti is referenced in `config.py` comments).

### Verified modern baseline (this audit)

Established in a controlled environment **outside the repository** (to keep the
git tree clean):

```
Location:    ~/Workspace/.venvs/stylegan-py311 (uv-created, not committed)
Python:      3.11.15
PyTorch:     2.13.0+cpu        (CPU build; no CUDA)
numpy:       2.4.6
torchvision: 0.28.0+cpu
yacs:        0.1.8
tqdm:        4.70.0
requests:    installed (needed by dnnlib.util, see C.6)
```

Installation used `uv` with the official CPU wheel index; no GPU is required for
the smoke-testable subset.

### What runs today (verified this cycle)

| Check | Result |
|---|---|
| `import models.CustomLayers/Blocks/GAN` | ✅ |
| Generator construction (res=32, defaults) | ✅ 22.8M parameters |
| Generator full-depth forward → `(1, 3, 32, 32)` | ✅ finite |
| Discriminator construction + forward → `(1, 1)` | ✅ 20.7M parameters, finite |
| Fade-in (`alpha=0.5`) path | ✅ finite |
| Config parse (`components/sample.yaml` merged into `config.cfg`) | ✅ |
| Import `generate_samples/grid/mixing_figure/truncation_figure`, `train` | ✅ |
| Syntax check all entry scripts incl. `convert.py` | ✅ |
| `import dnnlib.tflib` / `convert` | ❌ requires TensorFlow (by design, legacy) |

---

## C. Failure inventory

| # | Check | Result | Classification |
|---|---|---|---|
| C.1 | `import dnnlib.util` | fails — `requests` missing | **Environment gap**; fixed for local probing by installing `requests` (not part of `requirements.txt`) |
| C.2 | `import dnnlib.tflib` / `convert.py` | fails — `tensorflow` not installed | **Legacy boundary, by design**: converter is a TF 1.x-only path; not part of modern default env |
| C.3 | Original tests | `test/test_Blocks.py`, `test/test_CustomLayers.py` contain empty `pass` bodies from 2019 commit `90d1d1e` | **No test value**; replace/populate, do not treat as coverage |
| C.4 | `requirements.txt` | `torch`, `torchvision` unpinned with no CPU/GPU index pin; `requests` missing; no Python version bounds | **Reproducibility gap**; adequate for historical record, not for CI |
| C.5 | CUDA | not available locally; `config.cfg.device` defaults `'cuda'` | **Compatibility**: CPU run requires explicit device override; unverified on modern CUDA |
| C.6 | TF weight conversion | not runnable in this env; correctness of conversion is **unverified** (README already states this) | **Open correctness question** |

---

## D. Maintenance debt

| Class | Findings |
|---|---|
| Correctness | Conversion correctness unverified; gmap broadcast/truncation semantics unverified against NVIDIA; image-artifact issue reports open in tracker |
| Compatibility | Torch 2.x smoke OK; unverified: training loop, multi-GPU paths, old CUDA, torchvision dataset transforms with new PIL normalization |
| Dependency | Unpinned `requirements.txt`; missing `requests`; TF legacy is separate env |
| Documentation | README modernized; needs compatibility matrix + verified install; legacy conversion appetite documented as legacy-only |
| Reproducibility | No locked env (uv/Pipfile) committed; no CI; no test commands documented |
| Licensing | Mixed-provenance (see `docs/license-provenance.md`); root `LICENSE.txt` scope unconfirmed |
| Security | Pickle-based checkpoint loading (`convert.py`, official TF `.pkl`), untrusted-weight guidance exists in README; no sandbox mechanism |
| Tooling | No `pytest`, no CI, no lint/format pins |
| Issue backlog | 15 open issues initially classified in [`docs/issue-triage.md`](issue-triage.md); reproduction and maintainer review remain pending |

---

## E. Recommended priorities

> P0 = blocks responsible maintenance or legal/correctness risk; P1 = prevents
> basic usage; P2 = important improvement; P3 = optional.

- **P0 — Provenance/licensing** (in progress: `docs/license-provenance.md`
  exists; line-level comparison and contributor-rights confirmation pending).
- **P0 — Reproducible baseline** ✅ **done this cycle**: controlled venv +
  verified CPU smoke path (this document). Next: wire it into committed scripts/CI.
- **P1 — Populate real unit tests** (pytest): PixelNorm, StddevLayer,
  EqualizedConv/Linear shapes, Generator/Discriminator construction + determinism,
  configura parsing. Fixestthe empty 2019 test stubs.
- **P1 — Commit env/CI**: pin dev-env recipe (uv), add CPU GitHub Actions run of
  the test suite; document `EXPERIMENTAL`-free command lines in README.
- **P1 — Compatibility matrix** (`docs/compatibility.md`): torch 2.x CPU verified,
  GPU/legacy TF column written as "not verified".
- **P1 — Issue triage** (`docs/issue-triage.md`): verify the initial classification
  of the 15 open issues; do not close/respond to GitHub until maintainer review.
- **P2 — Security review** of pickle/checkpoint load paths + download scripts.
- **P2 — Downstream-usage verification** (ECCV 2024 lead) into `docs/downstream-usage.md`.
- **P3 — Optional**: release process, package metadata, licensing cleanup decision.

---

## E. Open questions logged (for follow-up docs)

1. Exact upstream commits for `pro_gan_pytorch`, `lernapparat/lernapparat`,
   `NVIDIA/stylegan` and the line-level diff for `models/*` (provenance doc).
2. Whether `Generator.forward(depth)` semantics and `truncation_psi=0.7` default
   match NVIDIA's original behavior.
3. Whether official FFHQ pretrained weights can be converted and validated
   (requires legacy TF env) — currently unverified.
4. Whether a multi-lateral CPU test can cover train loop without a GPU.
5. Contributor confirmation for later feature commits (conditional GAN: fit).

> Confidence notes: all "✅" results above were produced **in the venv described
> in §B this cycle**. Anything not executed is not claimed as verified.
