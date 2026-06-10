# pose_estimation_models 清理计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 移除所有未被 `available_models` 使用的 git submodule、对应 model 文件、无用资产和冗余文档，并重写 README.md 和 CONTRIBUTING.md 以反映当前仓库实际功能。

**Architecture:** 以 `estimator/__init__.py` 的 `available_models` 列表为权威依据，确定实际使用的 submodule（`duster`、`mast3r`、`Hierarchical-Localization`、`reloc3r`）；对其余 17 个 submodule 执行 `git submodule deinit + git rm`，同步清理 `.gitmodules`、对应 model 文件、未用 assets；最后重写文档。

**Tech Stack:** git submodule, Python 3.8+

---

## 文件映射

### 保留的 submodule（被 `available_models` 直接依赖）

| submodule | 依赖方 |
|---|---|
| `estimator/third_party/duster` | `duster.py`, `reloc3r.py` |
| `estimator/third_party/mast3r` | `master.py` |
| `estimator/third_party/Hierarchical-Localization` | `hloc.py` |
| `estimator/third_party/reloc3r` | `reloc3r.py` |

### 移除的 submodule（17 个）

`LightGlue`（hloc 的嵌套依赖，不是直接 submodule；但此处是 top-level submodule，独立存在）、`RoMa`、`DeDoDe`、`imatch-toolbox`、`Steerers`、`accelerated_features`、`omniglue`、`gim`、`EfficientLoFTR`、`Se2_LoFTR`、`aspanformer`、`MatchFormer`、`mickey`、`keypt2subpx`、`pixel-perfect-sfm`、`CF-3DGS`、`NoPoSplat`

> **注意：** `LightGlue` 是 `Hierarchical-Localization` 的**嵌套** submodule，会随 hloc 递归初始化。顶层 `.gitmodules` 中的独立 `LightGlue` 条目属于冗余，可移除。

### 移除的 model 文件（对应未使用 submodule）

`estimator/models/roma.py`、`dedode.py`、`matching_toolbox.py`、`steerers.py`、`omniglue.py`、`gim.py`、`efficient_loftr.py`、`se2loftr.py`、`aspanformer.py`、`matchformer.py`、`mickey.py`、`keypt2subpx.py`、`lightglue.py`、`loftr.py`、`silk.py`、`xfeat.py`、`kornia.py`、`handcrafted.py`

### 移除的根目录文件

- `benchmark.py`（仅引用 `assets/`，与 `available_models` 无关）
- `demo.ipynb`（引用旧 `matching` API，已不适用）
- `hloc_demo.ipynb`（demo 文件）
- `hloc_known_param.ipynb`（demo 文件）
- `VISUALIZATION_GUIDE.md`（用户明确要求移除）

### 移除的 assets

整个 `assets/` 目录（仅被 `benchmark.py` 引用，且已在 `.gitignore` 中排除版本控制）

### 保留的根目录文件

- `main_estimator.py`（主入口脚本）
- `TEMPLATE.py`（新模型贡献模板，有参考价值）
- `requirements.txt`、`pyproject.toml`、`LICENSE`、`.gitignore`、`.gitmodules`

### 修改的文件

- `README.md`（重写，反映实际支持的模型和使用方式）
- `CONTRIBUTING.md`（重写，反映 `estimator` 包结构和新模型添加流程）

---

## Task 1: 移除 17 个未使用的 git submodule

**Files:**
- Modify: `.gitmodules`
- Delete dirs: `estimator/third_party/{LightGlue,RoMa,DeDoDe,imatch-toolbox,Steerers,accelerated_features,omniglue,gim,EfficientLoFTR,Se2_LoFTR,aspanformer,MatchFormer,mickey,keypt2subpx,pixel-perfect-sfm,CF-3DGS,NoPoSplat}`

- [ ] **Step 1: 批量 deinit 并 git rm 所有未使用 submodule**

```bash
cd /Titan/code/robohike_ws/src/opennavmap/third_party/pose_estimation_models

REMOVE=(
  "estimator/third_party/LightGlue"
  "estimator/third_party/RoMa"
  "estimator/third_party/DeDoDe"
  "estimator/third_party/imatch-toolbox"
  "estimator/third_party/Steerers"
  "estimator/third_party/accelerated_features"
  "estimator/third_party/omniglue"
  "estimator/third_party/gim"
  "estimator/third_party/EfficientLoFTR"
  "estimator/third_party/Se2_LoFTR"
  "estimator/third_party/aspanformer"
  "estimator/third_party/MatchFormer"
  "estimator/third_party/mickey"
  "estimator/third_party/keypt2subpx"
  "estimator/third_party/pixel-perfect-sfm"
  "estimator/third_party/CF-3DGS"
  "estimator/third_party/NoPoSplat"
)

for sub in "${REMOVE[@]}"; do
  git submodule deinit -f "$sub"
  git rm -f "$sub"
done
```

- [ ] **Step 2: 清理 .git/modules 缓存**

```bash
for sub in LightGlue RoMa DeDoDe imatch-toolbox Steerers accelerated_features \
           omniglue gim EfficientLoFTR Se2_LoFTR aspanformer MatchFormer mickey \
           keypt2subpx pixel-perfect-sfm CF-3DGS NoPoSplat; do
  rm -rf .git/modules/estimator/third_party/$sub
done
```

- [ ] **Step 3: 验证 .gitmodules 只剩 4 个 submodule**

```bash
grep "^\[submodule" .gitmodules
```

期望输出（4 行）：
```
[submodule "estimator/third_party/duster"]
[submodule "estimator/third_party/mast3r"]
[submodule "estimator/third_party/Hierarchical-Localization"]
[submodule "estimator/third_party/reloc3r"]
```

- [ ] **Step 4: 验证 third_party 目录内容**

```bash
ls estimator/third_party/
```

期望：只有 `duster/  mast3r/  Hierarchical-Localization/  reloc3r/  vpr_models/`（`vpr_models` 为本地目录，非 submodule）

- [ ] **Step 5: Commit**

```bash
git add .gitmodules
git commit -m "chore(submodule): remove 17 unused third-party submodules"
```

---

## Task 2: 移除未使用的 model 文件

**Files:**
- Delete: `estimator/models/roma.py`, `estimator/models/dedode.py`, `estimator/models/matching_toolbox.py`, `estimator/models/steerers.py`, `estimator/models/omniglue.py`, `estimator/models/gim.py`, `estimator/models/efficient_loftr.py`, `estimator/models/se2loftr.py`, `estimator/models/aspanformer.py`, `estimator/models/matchformer.py`, `estimator/models/mickey.py`, `estimator/models/keypt2subpx.py`, `estimator/models/lightglue.py`, `estimator/models/loftr.py`, `estimator/models/silk.py`, `estimator/models/xfeat.py`, `estimator/models/kornia.py`, `estimator/models/handcrafted.py`

- [ ] **Step 1: 删除所有未使用的 model 文件**

```bash
cd /Titan/code/robohike_ws/src/opennavmap/third_party/pose_estimation_models

REMOVE_MODELS=(
  estimator/models/roma.py
  estimator/models/dedode.py
  estimator/models/matching_toolbox.py
  estimator/models/steerers.py
  estimator/models/omniglue.py
  estimator/models/gim.py
  estimator/models/efficient_loftr.py
  estimator/models/se2loftr.py
  estimator/models/aspanformer.py
  estimator/models/matchformer.py
  estimator/models/mickey.py
  estimator/models/keypt2subpx.py
  estimator/models/lightglue.py
  estimator/models/loftr.py
  estimator/models/silk.py
  estimator/models/xfeat.py
  estimator/models/kornia.py
  estimator/models/handcrafted.py
)

git rm "${REMOVE_MODELS[@]}"
```

- [ ] **Step 2: 验证保留的 model 文件**

```bash
ls estimator/models/
```

期望（7 个文件）：
```
__init__.py
base_estimator.py
duster.py
hloc.py
master.py
reloc3r.py
vpr.py
```

- [ ] **Step 3: Commit**

```bash
git commit -m "chore(models): remove model files for unused submodules"
```

---

## Task 3: 移除冗余根目录文件和 assets

**Files:**
- Delete: `benchmark.py`, `demo.ipynb`, `hloc_demo.ipynb`, `hloc_known_param.ipynb`, `VISUALIZATION_GUIDE.md`
- Delete dir: `assets/`（如在版本控制中）

- [ ] **Step 1: 检查 assets 是否在 git track 中**

```bash
git ls-files assets/
```

若有输出（文件被 track），执行 Step 2；若无输出（已被 `.gitignore` 排除），跳到 Step 3。

- [ ] **Step 2: 移除 assets（如在 git track 中）**

```bash
git rm -r assets/
```

若不在 git track 中，直接：

```bash
rm -rf assets/
```

- [ ] **Step 3: 移除根目录冗余文件**

```bash
git rm benchmark.py demo.ipynb hloc_demo.ipynb hloc_known_param.ipynb VISUALIZATION_GUIDE.md
```

若某文件不在 git track 中，用 `rm` 代替 `git rm`。

- [ ] **Step 4: 验证根目录**

```bash
ls
```

期望（10 项）：
```
.git  .gitignore  .gitmodules  CONTRIBUTING.md  LICENSE
README.md  TEMPLATE.py  docs/
estimator/  main_estimator.py  pyproject.toml  requirements.txt
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove benchmark.py, demo notebooks, VISUALIZATION_GUIDE.md, and assets"
```

---

## Task 4: 将 docs/plans 加入 .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: 在 .gitignore 末尾追加 docs/plans/**

当前 `.gitignore` 内容（10 行）：
```
.spyproject
.idea
__pycache__
model_weights
outputs_*
*.egg-info/*
assets/*
build/*
tmp/
test/
```

修改后（追加一行）：
```
.spyproject
.idea
__pycache__
model_weights
outputs_*
*.egg-info/*
assets/*
build/*
tmp/
test/
docs/plans/
```

- [ ] **Step 2: 验证 docs/plans/ 未被 track**

```bash
git status docs/
```

期望：`docs/plans/` 下的文件不出现在 `Changes not staged` 或 `Untracked files` 中（被忽略）。

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore(gitignore): ignore docs/plans/ directory"
```

---

## Task 5: 重写 README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 将 README.md 替换为以下内容**

```markdown
# pose_estimation_models

A unified Python API for pose estimation and image localization, wrapping four state-of-the-art models:

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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): rewrite to reflect actual supported models and structure"
```

---

## Task 6: 重写 CONTRIBUTING.md

**Files:**
- Modify: `CONTRIBUTING.md`

- [ ] **Step 1: 将 CONTRIBUTING.md 替换为以下内容**

```markdown
# Contributing: Adding a New Pose Estimation Model

This guide explains how to add a new estimator to `pose_estimation_models`.

## Structure

```
estimator/
├── __init__.py              # available_models list + get_estimator() factory
├── models/
│   ├── base_estimator.py    # BaseEstimator base class — read this first
│   ├── duster.py            # DUSt3R example
│   ├── master.py            # MASt3R example
│   ├── hloc.py              # HLoc example
│   ├── reloc3r.py           # Reloc3r example
│   └── vpr.py               # VPR example
├── third_party/             # git submodules for external model code
│   ├── duster/              # gogojjh/dust3r (branch: lora_finetune)
│   ├── mast3r/              # gogojjh/mast3r (branch: main)
│   ├── Hierarchical-Localization/
│   └── reloc3r/
└── utils.py                 # add_to_path, resize_to_divisible, etc.
```

## Steps to Add a New Model

### 1. Add a git submodule (if the model has external dependencies)

```bash
git submodule add <repo_url> estimator/third_party/<model_name>
git submodule update --init --recursive estimator/third_party/<model_name>
```

### 2. Create the estimator class

Create `estimator/models/<model_name>.py`. Use [TEMPLATE.py](../TEMPLATE.py) as a starting point.

Your class must:
- Inherit from `BaseEstimator` (`from estimator import BaseEstimator`)
- Add the external repo to `sys.path` via `add_to_path`:

```python
from estimator.utils import add_to_path
from estimator import THIRD_PARTY_DIR, WEIGHTS_DIR

add_to_path(THIRD_PARTY_DIR / "<model_name>")
```

- Implement `_forward(self, img0, img1)` returning `(focal, im_pose, loss)`
- Implement `download_weights(self)` to fetch model weights to `WEIGHTS_DIR`

### 3. Register in `estimator/__init__.py`

Add to `available_models`:

```python
available_models = [
    ...
    "your_model_name",
]
```

Add a branch in `get_estimator()`:

```python
elif "your_model" in estimator_name:
    from estimator.models.your_model import YourEstimator
    return YourEstimator(device=device, **kwargs)
```

### 4. Update `requirements.txt`

Add any new pip dependencies (one per line with a comment):

```
# your_model
your-dependency>=1.0
```

### 5. Code style

Format with Black (`--line-length 120`):

```bash
black --line-length 120 estimator/models/<model_name>.py
```

### 6. Test manually

```python
from estimator import get_estimator
est = get_estimator("your_model_name", device="cpu")
# load two test images and run forward
```

### 7. Commit and open a PR

```bash
git add estimator/models/<model_name>.py estimator/__init__.py .gitmodules requirements.txt
git commit -m "feat(estimator): add <ModelName> estimator"
git push origin <your-branch>
```
```

- [ ] **Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs(contributing): rewrite for estimator package structure"
```

---

## Task 7: 推送 main 到 remote

- [ ] **Step 1: 确认所有提交已就位**

```bash
git log --oneline -8
```

期望看到 6 条新提交（来自 Task 1-6）：
```
xxxxxxx docs(contributing): rewrite for estimator package structure
xxxxxxx docs(readme): rewrite to reflect actual supported models and structure
xxxxxxx chore(gitignore): ignore docs/plans/ directory
xxxxxxx chore: remove benchmark.py, demo notebooks, VISUALIZATION_GUIDE.md, and assets
xxxxxxx chore(models): remove model files for unused submodules
xxxxxxx chore(submodule): remove 17 unused third-party submodules
```

- [ ] **Step 2: 推送到 origin main**

```bash
git push origin main
```

- [ ] **Step 3: 验证远端**

```bash
git log --oneline origin/main -5
```

期望与本地 `main` 一致。

---

## Self-Review

### Spec Coverage 检查

| 需求 | 对应 Task |
|---|---|
| 移除未使用 submodule | Task 1 |
| 移除未使用 model 文件 | Task 2 |
| 移除未用 assets | Task 3 |
| 移除冗余文件（benchmark, notebooks） | Task 3 |
| 检查 TEMPLATE.py 是否必要 → 保留 | 分析结论（TEMPLATE.py 保留，作为贡献模板） |
| 移除 VISUALIZATION_GUIDE.md | Task 3 |
| 将 docs/plans/ 加入 .gitignore | Task 4 |
| 重写 README.md | Task 5 |
| 重写 CONTRIBUTING.md | Task 6 |
| 推送到远端 | Task 7 |

### Placeholder 扫描

无 TBD / TODO / placeholder。

### 类型一致性

所有引用的文件路径和命令均基于实际探索结果，无交叉引用不一致问题。
