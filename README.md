# pose_estimation_models

A unified Python API for pose estimation and image localization, wrapping six state-of-the-art model types:

| Model | Type | Paper |
|-------|------|-------|
| DUSt3R (`duster`) | Feed-forward dense 3D reconstruction + pose (pairwise) | [CVPR 2024](https://arxiv.org/abs/2312.14132) |
| MASt3R (`master`) | Feed-forward dense matching + metric pose (pairwise) | [ECCV 2024](https://arxiv.org/abs/2406.09756) |
| VGGT (`vggt`) | Feed-forward multi-view geometry (joint, many views) | [CVPR 2025](https://arxiv.org/abs/2503.11651) |
| HLoc (`hloc_*`) | Sparse feature-based localization | [CVPR 2020](https://arxiv.org/abs/1812.03506) |
| Reloc3r (`reloc3r`) | Relative pose from dense features | [CVPR 2025](https://arxiv.org/abs/2412.08376) |
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
- `estimator/third_party/vggt` — VGGT backbone (facebookresearch)
- `estimator/third_party/Hierarchical-Localization` — HLoc pipeline
- `estimator/third_party/reloc3r` — Reloc3r model

VGGT weights (`facebook/VGGT-1B`, ~1.3 B parameters) are pulled from the HuggingFace Hub on first use
and cached under `WEIGHTS_DIR` — no manual download step.

## Test Data

`main_estimator.py` expects a scene directory in the Map-free layout (`poses.txt`, `intrinsics.txt`, `seq/`). Two ready-to-use sources:
- **Map-free dataset** (real-world scenes): download from [nianticspatial.com/research/map-free/dataset](https://nianticspatial.com/research/map-free/dataset).
- **Simulated Matterport3D + others**: download from [this Google Drive folder](https://drive.google.com/drive/folders/1j1QxQeJOfk4pLWeKFLm6WBWD7zb9Y0Rn?usp=sharing).

After downloading and extracting a scene (e.g. `<scene_root>/{poses.txt,intrinsics.txt,seq/}`), run:

```bash
python main_estimator.py --model master --scene_root <scene_root> --device cuda --out_dir outputs_master
```

- `--model` — any entry from `available_models` (see below), e.g. `duster`, `master`, `reloc3r`.
- `--scene_root` — path to the downloaded scene directory.
- `--out_dir` — where visualizations/results are written (defaults to `outputs_<model>`).

## Available Models

```python
from estimator import available_models, get_estimator
print(available_models)
# [
#   "hloc_disk_dilg", "hloc_superpoint_splg",
#   "vpr_cosplace_resnet18_256", "vpr_netvlad_resnet18_4096",
#   "duster", "master", "reloc3r", "vggt",
#   "duster_{nocalib,calib}_pretrain", "master_{nocalib,calib}_pretrain"
# ]
```

## VGGT

DUSt3R, MASt3R, and VGGT are all feed-forward multi-view geometry networks: each regresses dense
pointmaps (and, for VGGT, camera poses/depth directly) in a single forward pass, with no test-time
optimization such as bundle adjustment. They differ in how many views that forward pass covers, and
that's what sets VGGT apart from the other estimators in how it consumes input and what it returns.

**Joint multi-view pass vs. pairwise + alignment.** DUSt3R and MASt3R are feed-forward but *pairwise* —
each pass takes exactly two images and directly regresses their pointmaps, with no optimization needed
for that pair. Reconstructing more than two views means running many such pairs and then stitching the
results together with a separate iterative global-alignment optimization. VGGT instead is feed-forward
across *all* input views at once: it predicts every camera pose, depth map, and point map for the full
view set in one pass, with no pairwise stitching or alignment stage, which makes it roughly an order of
magnitude faster on multi-view scenes.

**All reference images at once.** The estimator takes the full reference list plus the query image in a
single call, rather than one reference/query pair at a time.

**Predicted poses live in an arbitrary-scale frame** anchored at the first input image, so they are not
directly comparable to the scene's absolute poses. To recover an absolute query pose, the predicted
reference poses are aligned to their known absolute poses with a rotation-constrained similarity
transform, and that same transform is applied to the query prediction. VGGT also predicts its own
intrinsics, so the `*_intr` arguments are ignored.

```bash
python main_estimator.py --model vggt --scene_root <scene_root> --device cuda --out_dir outputs_vggt
```

`show_reconstruction(conf_thres=50.0)` opens an interactive viewer with the aligned point cloud and
camera frustums — references in blue, query in red. `conf_thres` drops that percentage of the
lowest-confidence points (global percentile across all views).

> **Viewer troubleshooting.** The viewer needs a working GL context. If it fails with
> `MESA-LOADER: failed to open iris/swrast` or `Could not create GL context`, a shell-level
> `LD_PRELOAD` of the system libGL is likely colliding with conda's, and the system Mesa may be too
> old for recent Intel iGPUs. Run with the preload masked for that process only:
> ```bash
> env -u LD_PRELOAD LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri LIBGL_ALWAYS_SOFTWARE=1 \
>     python main_estimator.py --model vggt --scene_root <scene_root>
> ```
> The viewer additionally requires `pyglet<2` — trimesh's windowed viewer never adopted the pyglet 2.x
> API. Inference and pose estimation are unaffected by any of this.

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
