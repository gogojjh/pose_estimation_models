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
