# pose_estimation_models

A unified Python API for pose estimation and image localization, wrapping five state-of-the-art model types:

| Model | Type | Paper |
|-------|------|-------|
| DUSt3R (`duster`) | Dense 3D reconstruction + pose | [CVPR 2024](https://arxiv.org/abs/2312.14132) |
| MASt3R (`master`) | Dense matching + metric pose | [ArXiv 2024](https://arxiv.org/abs/2406.09756) |
| HLoc (`hloc_*`) | Sparse feature-based localization | [CVPR 2020](https://arxiv.org/abs/1812.03506) |
| Reloc3r (`reloc3r`) | Relative pose from dense features | [ArXiv 2024](https://arxiv.org/abs/2412.08376) |
| VPR (`vpr_*`) | Global place recognition | multiple |

## Install

```bash
git clone --recursive https://github.com/gogojjh/pose_estimation_models
cd pose_estimation_models
pip install .
```

Required submodules (initialized by `--recursive`):
- `estimator/third_party/duster` — DUSt3R backbone (gogojjh fork)
- `estimator/third_party/mast3r` — MASt3R backbone (gogojjh fork)
- `estimator/third_party/Hierarchical-Localization` — HLoc pipeline
- `estimator/third_party/reloc3r` — Reloc3r model

## Available Models

```python
from estimator import available_models, get_estimator
print(available_models)
# [
#   "hloc_disk_dilg", "hloc_superpoint_splg",
#   "vpr_cosplace_resnet18_256", "vpr_netvlad_resnet18_4096",
#   "duster", "master", "reloc3r",
#   "duster_{nocalib,calib}_pretrain", "master_{nocalib,calib}_pretrain"
# ]
```

## Usage

```python
from estimator import get_estimator

# Pose estimation with MASt3R
estimator = get_estimator("master", device="cuda")
result = estimator.forward(img0, img1)
# result keys: focal, im_pose, loss

# Global place recognition with CosPlace
vpr = get_estimator("vpr_cosplace_resnet18_256", device="cuda")
similarity = vpr.get_similarity(img0, img1)

# Sparse feature-based localization
hloc = get_estimator("hloc_disk_dilg", device="cuda")
pose = hloc.forward(img0, img1)
```

See `main_estimator.py` for a complete end-to-end localization example.

## Adding a New Model

See [CONTRIBUTING.md](CONTRIBUTING.md) and [TEMPLATE.py](TEMPLATE.py).

## License

See [LICENSE](LICENSE). Check individual submodule licenses before production use.
