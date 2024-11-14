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
    "duster",
    "master",
]

def get_version(pkg):
    version_num = pkg.__version__.split("-")[0]
    major, minor, patch = [int(num) for num in version_num.split(".")]
    return major, minor, patch

@supress_stdout
def get_estimator(estimator_name="master", device="cpu", *args, **kwargs):
    # if isinstance(estimator_name, list):
        # from matching.im_models.base_matcher import EnsembleMatcher

        # return EnsembleMatcher(estimator_name, device, *args, **kwargs)
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
