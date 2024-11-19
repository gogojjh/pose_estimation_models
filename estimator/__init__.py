"""
File to import matchers. The module's import are within the functions, so that
a module is imported only iff needed, reducing the number of raised errors and
warnings due to unused modules.
"""
from pathlib import Path
from .utils import supress_stdout, add_to_path
from .models.base_estimator import BaseEstimator

# add viz2d from lightglue to namespace - thanks lightglue!
THIRD_PARTY_DIR = Path(__file__).parent.joinpath('third_party')

# add_to_path(THIRD_PARTY_DIR.joinpath('LightGlue'))
# from lightglue import viz2d  # for quick import later 'from matching import viz2d'

WEIGHTS_DIR = Path(__file__).parent.joinpath("model_weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

__version__ = "1.0.0"

available_models = [
    "hloc_disk_dilg",
    "duster",
    "master",
]

def get_version(pkg):
    version_num = pkg.__version__.split("-")[0]
    major, minor, patch = [int(num) for num in version_num.split(".")]
    return major, minor, patch

# @supress_stdout
def get_estimator(estimator_name="master", device="cpu", max_num_keypoints=2048, out_dir='/tmp', *args, **kwargs):
    if 'hloc' in estimator_name:
        from estimator.models import hloc
        
        # ['superpoint_max', 'superpoint_inloc', 'r2d2', 'd2net-ss', 'sift', 'sosnet', 'disk', 'aliked-n16']
        name1 = estimator_name.split('_')[1]
        if name1 == 'suerpoint':
            feature_name = 'superpoint_max'
        elif name1 == 'r2d2':
            feature_name = 'r2d2'
        elif name1 == 'd2net':
            feature_name = 'd2net-ss'
        elif name1 == 'sift':
            feature_name = 'sift'
        elif name1 == 'sosnet':
            feature_name = 'sosnet'
        elif name1 == 'disk':
            feature_name = 'disk'
        elif name1 == 'aliked':
            feature_name = 'aliked-n16'
        else:
            raise RuntimeError(
                f"Feature {name1} for hloc not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
            )

        # ['superpoint+lightglue', 'disk+lightglue', 'aliked+lightglue', 'superglue', 'superglue-fast', 'NN-superpoint', 'NN-ratio', 'NN-mutual', 'adalam']
        name2 = estimator_name.split('_')[2]
        if name2 == 'splg':
            matcher_name = 'superpoint+lightglue'
        elif name2 == 'dilg':
            matcher_name = 'disk+lightglue'
        elif name2 == 'alilg':
            matcher_name = 'aliked+lightglue'
        elif name2 == 'sg':
            matcher_name = 'superglue'
        elif name2 == 'sgfast':
            matcher_name = 'superglue-fast'
        elif name2 == 'nnsp':
            matcher_name = 'NN-superpoint'
        elif name2 == 'nnratio':
            matcher_name = 'NN-ratio'
        elif name2 == 'nnmutual':
            matcher_name = 'NN-mutual'
        elif name2 == 'adalam':
            matcher_name = 'adalam'
        else:
            raise RuntimeError(
                f"Matcher {name2} for hloc not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
            )

        return hloc.HlocEstimator(device, feature_name, matcher_name, max_num_keypoints, out_dir, *args, **kwargs)

    if estimator_name in ["duster", "dust3r"]:
        from estimator.models import duster
        return duster.Dust3rEstimator(device, *args, **kwargs)

    elif estimator_name in ["master", "mast3r"]:
        from estimator.models import master
        return master.Mast3rEstimator(device, *args, **kwargs)

    else:
        raise RuntimeError(
            f"Estimator {estimator_name} not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
        )
